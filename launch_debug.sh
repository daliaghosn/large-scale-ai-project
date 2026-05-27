#!/bin/bash
#
# Usage: ./launch.sh <mode> <model_size> [steps] [nodes]
#
# Modes:     throughput  (50 steps, with W&B)
#            train       (N steps, with W&B and Tensorboard)
#
# Sizes:     125m, 350m, 760m, 1.5b, 3b, 8b
#
# Steps:     required for train mode (e.g., 1000, 5000, 15000)
# Nodes:     optional, default 4 (max 8)
#
# Examples:  ./launch_debug.sh throughput 760m
#            DTYPE=fp8 ./launch_debug.sh throughput 760m 50 1
#            DTYPE=bf16 ./launch_debug.sh throughput 760m 50 1
#            DTYPE=fp8 ./launch_debug.sh train 760m 400 1   # ~20 min smoke on debug (30m cap)
#            DTYPE=fp8 ./launch.sh train 760m 1030 1        # full ~1h budget: use launch.sh
#
# Perf overrides (throughput tuning):
#   DTYPE=fp8 NUM_WORKERS=8 MANUAL_GC=0 ./launch_debug.sh throughput 760m 50 1
#   MBS=6 DTYPE=fp8 ...   # override default micro-batch for a model size
#
# Recipe overrides (training):
#   OPTIMIZER=muon LR=6e-4 LR_DECAY=cosine MIN_LR=6e-5 LR_WARMUP=50 \
#     DTYPE=bf16 NUM_WORKERS=8 MANUAL_GC=0 MBS=4 ./launch_debug.sh train 760m 400 1
#   GBS=128 WEIGHT_DECAY=0.01 ./launch_debug.sh train 350m 400 1

set -euo pipefail

source "$(dirname "$0")/config.sh"

# Mixed precision: bf16 (default) or fp8 (TransformerEngine hybrid recipe).
DTYPE=${DTYPE:-bf16}
case $DTYPE in
    bf16|fp8) ;;
    *)
        echo "Unknown DTYPE: $DTYPE. Use: bf16, fp8"
        exit 1
        ;;
esac

MODE=${1:?Usage: ./launch.sh <mode> <model_size> [steps] [nodes]}
MODEL_SIZE=${2:?Usage: ./launch.sh <mode> <model_size> [steps] [nodes]}

################ Mode config ################
case $MODE in
    throughput)
        TRAINING_STEPS=${3:-50}
        NODES=${4:-4}
        TIME=00:30:00
        EVAL_INTERVAL=$TRAINING_STEPS
        EVAL_ITERS=0
        LR_WARMUP_ITERS=10
        LOGGING_EXTRA=""
        WANDB=true
        ;;
    train)
        TRAINING_STEPS=${3:?Usage: ./launch.sh train <model_size> <steps> [nodes]}
        NODES=${4:-4}
        TIME=00:30:00
        EVAL_ITERS=10
        # Debug partition is capped at 30 min; scale warmup/eval for short smoke runs.
        if [ "$TRAINING_STEPS" -lt 500 ]; then
            LR_WARMUP_ITERS=$(( TRAINING_STEPS / 5 ))
            [ "$LR_WARMUP_ITERS" -lt 10 ] && LR_WARMUP_ITERS=10
            EVAL_INTERVAL=100
        else
            LR_WARMUP_ITERS=200
            EVAL_INTERVAL=200
        fi
        LOGGING_EXTRA="
    --tensorboard-dir \$TENSORBOARD_DIR
    --log-timers-to-tensorboard
    --log-memory-to-tensorboard"
        WANDB=true
        ;;
    *)
        echo "Unknown mode: $MODE. Choose: throughput, train"
        exit 1
        ;;
esac

################ Model config ################
REQ_MBS=${MBS:-}
case $MODEL_SIZE in
    125m)
        NUM_LAYERS=12;  HIDDEN=768;  FFN=2048;  HEADS=12; KV_HEADS=4
        MBS=16
        ;;
    350m)
        NUM_LAYERS=24; HIDDEN=1024; FFN=2816;  HEADS=16; KV_HEADS=4
        MBS=8
        ;;
    760m)
        NUM_LAYERS=24; HIDDEN=1536; FFN=4096;  HEADS=16; KV_HEADS=4
        MBS=4
        ;;
    1.5b)
        NUM_LAYERS=48; HIDDEN=1600; FFN=4352;  HEADS=20; KV_HEADS=4
        MBS=4
        ;;
    3b)
        NUM_LAYERS=32; HIDDEN=3072; FFN=8192;  HEADS=24; KV_HEADS=8
        MBS=4
        ;;
    8b)
        NUM_LAYERS=32; HIDDEN=4096; FFN=14336; HEADS=32; KV_HEADS=8
        MBS=2
        ;;
    *)
        echo "Unknown model size: $MODEL_SIZE. Choose: 125m, 350m, 760m, 1.5b, 3b, 8b"
        exit 1
        ;;
esac

# Optional perf overrides (see scripts/throughput_sweep_c1.sh).
NUM_WORKERS=${NUM_WORKERS:-4}
MANUAL_GC=${MANUAL_GC:-1}
[ -n "$REQ_MBS" ] && MBS=$REQ_MBS

OPTIMIZER=${OPTIMIZER:-adam}
LR=${LR:-3e-4}
LR_DECAY=${LR_DECAY:-constant}
MIN_LR=${MIN_LR:-0}
LR_WARMUP=${LR_WARMUP:-${LR_WARMUP_ITERS}}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
GBS=${GBS:-256}
SEQ_LEN=4096

LR_WARMUP_ITERS=$LR_WARMUP

case $OPTIMIZER in
    adam|muon) ;;
    *)
        echo "Unknown OPTIMIZER: $OPTIMIZER. Use: adam, muon"
        exit 1
        ;;
esac

PERF_TAG=""
[ "$NUM_WORKERS" != "4" ] && PERF_TAG="${PERF_TAG}-w${NUM_WORKERS}"
[ "$MANUAL_GC" = "0" ] && PERF_TAG="${PERF_TAG}-nogc"
[ "$OPTIMIZER" != "adam" ] && PERF_TAG="${PERF_TAG}-${OPTIMIZER}"
[ "$LR_DECAY" != "constant" ] && PERF_TAG="${PERF_TAG}-${LR_DECAY}"
[ "$LR" != "3e-4" ] && PERF_TAG="${PERF_TAG}-lr${LR}"
[ "$GBS" != "256" ] && PERF_TAG="${PERF_TAG}-gbs${GBS}"
JOB_NAME="gipfel-${MODE}-${MODEL_SIZE}-${DTYPE}${PERF_TAG}-${TRAINING_STEPS}s-${NODES}n"

if [ "$MANUAL_GC" = "1" ]; then
    TRAINING_GC_BLOCK='    --manual-gc
    --manual-gc-interval 50'
else
    TRAINING_GC_BLOCK=''
fi

################ Mixed precision (Megatron + TE) ################
case $DTYPE in
    bf16)
        MIXED_PRECISION_BLOCK='MIXED_PRECISION_ARGS=(
    --bf16
)'
        if [ "$OPTIMIZER" = "muon" ]; then
            TRANSFORMER_ENGINE_BLOCK='TRANSFORMER_ENGINE_ARGS=(
    --transformer-impl transformer_engine
)'
        else
            TRANSFORMER_ENGINE_BLOCK='TRANSFORMER_ENGINE_ARGS=(
    --transformer-impl transformer_engine
    --use-precision-aware-optimizer
    --main-grads-dtype bf16
)'
        fi
        ;;
    fp8)
        if [ "$OPTIMIZER" = "muon" ]; then
            echo "Invalid combo: OPTIMIZER=muon with DTYPE=fp8 is unsupported in this launcher."
            echo "Use OPTIMIZER=muon with DTYPE=bf16, or OPTIMIZER=adam with DTYPE=fp8."
            exit 1
        fi
        # Keep --bf16 for master weights; TE uses --fp8-format for compute (see Megatron FP8 tests).
        # Do not use --use-precision-aware-optimizer with --fp8-param-gather (TE Adam expects fp32 master).
        MIXED_PRECISION_BLOCK='MIXED_PRECISION_ARGS=(
    --bf16
    --fp8-format hybrid
    --fp8-amax-history-len 1024
    --fp8-amax-compute-algo max
    --fp8-param-gather
)'
        TRANSFORMER_ENGINE_BLOCK='TRANSFORMER_ENGINE_ARGS=(
    --transformer-impl transformer_engine
)'
        ;;
esac

################ W&B block ################
if [ "$WANDB" = true ]; then
    WANDB_BLOCK='
# WANDB
if [ -n "$WANDB_API_KEY" ]; then
    echo "[$(date)] WANDB enabled."
    TRAINING_CMD="$TRAINING_CMD \
        --wandb-save-dir $LOG_DIR \
        --wandb-project $PROJECT_NAME \
        --wandb-exp-name $EXP_NAME-$SLURM_JOB_ID"
else
    export WANDB_MODE=disabled
    echo "[$(date)] WANDB disabled."
fi'
else
    WANDB_BLOCK='export WANDB_MODE=disabled'
fi

################ Generate script ################
mkdir -p logs

SCRIPT="logs/${JOB_NAME}.sbatch"

cat > "$SCRIPT" << 'HEADER'
#!/bin/bash
HEADER

cat >> "$SCRIPT" << SBATCH_DIRECTIVES
#SBATCH --account=${SBATCH_ACCOUNT}
#SBATCH --time=${TIME}
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log
#SBATCH --nodes=${NODES}
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --mem=460000
#SBATCH --no-requeue
#SBATCH --partition=debug
#SBATCH --time=00:30:00
SBATCH_DIRECTIVES

cat >> "$SCRIPT" << 'BODY_HEAD'

echo "START TIME: \$(date)"

################ Configs ################
BODY_HEAD

cat >> "$SCRIPT" << BODY_WORKDIR
WORKDIR=${WORKDIR}
MEGATRON_LM_DIR=\$WORKDIR/Megatron-LM
DATA_PREFIX=/capstor/store/cscs/swissai/infra01/datasets/nvidia/Nemotron-ClimbMix/climbmix_small_megatron/climbmix_small
DATASET_CACHE_DIR=/iopsstor/scratch/cscs/\$USER/gipfelsturm/cache
BODY_WORKDIR

cat >> "$SCRIPT" << CONFIGS

# Training config
MBS=${MBS}
GBS=${GBS}
SEQ_LEN=${SEQ_LEN}
TRAINING_STEPS=${TRAINING_STEPS}
DTYPE=${DTYPE}

# Logging
PROJECT_NAME=gipfelsturm
EXP_NAME=${MODE}-${MODEL_SIZE}-${DTYPE}-\${SLURM_NNODES}n
LOG_DIR=/iopsstor/scratch/cscs/\$USER/gipfelsturm/\$PROJECT_NAME/\$EXP_NAME
TENSORBOARD_DIR=\$LOG_DIR/tensorboard
CONFIGS

cat >> "$SCRIPT" << 'SETUP'

#########################################

mkdir -p logs $LOG_DIR $TENSORBOARD_DIR $DATASET_CACHE_DIR

cd $MEGATRON_LM_DIR
flock $MEGATRON_LM_DIR/.git-lock bash -c "cd $MEGATRON_LM_DIR && git checkout -- . && git apply $WORKDIR/patches/*.patch"
export PYTHONPATH=$MEGATRON_LM_DIR:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TRITON_CACHE_DIR=/iopsstor/scratch/cscs/$USER/gipfelsturm/.triton_cache
export TORCHINDUCTOR_CACHE_DIR=/iopsstor/scratch/cscs/$USER/gipfelsturm/.inductor_cache
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK/SLURM_GPUS_PER_NODE))
echo "[\$(date)] DTYPE=\${DTYPE}"
MASTER_ADDR=$(hostname)
MASTER_PORT=25678

SETUP

cat >> "$SCRIPT" << TRANSFORMER_ENGINE
${TRANSFORMER_ENGINE_BLOCK}

TRANSFORMER_ENGINE

cat >> "$SCRIPT" << MODEL
NETWORK_SIZE_ARGS=(
    --num-layers ${NUM_LAYERS}
    --hidden-size ${HIDDEN}
    --ffn-hidden-size ${FFN}
    --num-attention-heads ${HEADS}
    --group-query-attention
    --num-query-groups ${KV_HEADS}
    --max-position-embeddings \$SEQ_LEN
    --position-embedding-type rope
    --normalization RMSNorm
    --swiglu
    --untie-embeddings-and-output-weights
    --seq-length \$SEQ_LEN
)
MODEL

cat >> "$SCRIPT" << TRAINING

TRAINING_ARGS=(
    --micro-batch-size \$MBS
    --global-batch-size \$GBS
    --train-iters \$TRAINING_STEPS
    --log-interval 1
    --eval-interval ${EVAL_INTERVAL}
    --eval-iters ${EVAL_ITERS}
    --cross-entropy-loss-fusion
    --disable-bias-linear
    --optimizer ${OPTIMIZER}
    --dataloader-type single
    --no-check-for-nan-in-loss-and-grad
${TRAINING_GC_BLOCK}
    --use-flash-attn
)

REGULARIZATION_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --weight-decay ${WEIGHT_DECAY}
    --clip-grad 1.0
    --adam-beta1 0.9
    --adam-beta2 0.95
)

LEARNING_RATE_ARGS=(
    --lr ${LR}
    --lr-decay-style ${LR_DECAY}
    --lr-warmup-iters ${LR_WARMUP_ITERS}
    --lr-decay-iters ${TRAINING_STEPS}
    --min-lr ${MIN_LR}
)
TRAINING

cat >> "$SCRIPT" << 'REST'

INITIALIZATION_ARGS=(
    --seed 42
    --init-method-std 0.02
)

REST

cat >> "$SCRIPT" << MIXED_PRECISION
${MIXED_PRECISION_BLOCK}

MIXED_PRECISION

if [ "$OPTIMIZER" = "muon" ]; then
cat >> "$SCRIPT" << 'REST'

DISTRIBUTED_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
)
REST
else
cat >> "$SCRIPT" << 'REST'

DISTRIBUTED_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
)
REST
fi

cat >> "$SCRIPT" << 'REST'

LOGGING_ARGS=(
    --log-throughput
    --log-progress
REST

cat >> "$SCRIPT" << LOGGING_EXTRA
${LOGGING_EXTRA}
)
LOGGING_EXTRA

cat >> "$SCRIPT" << 'TOKENIZER'

TOKENIZER_ARGS=(
    --tokenizer-type GPT2BPETokenizer
    --vocab-file $WORKDIR/data/gpt2-vocab.json
    --merge-file $WORKDIR/data/gpt2-merges.txt
)

DATA_ARGS=(
    --data-path $DATA_PREFIX
    --data-cache-path $DATASET_CACHE_DIR
    --split 99,1,0
    --num-workers ${NUM_WORKERS}
)


TORCHRUN_ARGS=(
    --nproc-per-node $SLURM_GPUS_PER_NODE
    --nnodes $SLURM_NNODES
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT
    --rdzv_backend c10d
    --max_restarts 0
    --tee 3
)

TRAINING_CMD="torchrun ${TORCHRUN_ARGS[@]} $MEGATRON_LM_DIR/pretrain_gpt.py \
    ${TRANSFORMER_ENGINE_ARGS[@]} \
    ${NETWORK_SIZE_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${REGULARIZATION_ARGS[@]} \
    ${LEARNING_RATE_ARGS[@]} \
    ${INITIALIZATION_ARGS[@]} \
    ${MIXED_PRECISION_ARGS[@]} \
    ${DISTRIBUTED_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    ${TOKENIZER_ARGS[@]} \
    ${DATA_ARGS[@]}"

TOKENIZER

cat >> "$SCRIPT" << 'WANDB_PLACEHOLDER'
WANDB_PLACEHOLDER

# Replace placeholder with actual W&B block
sed -i '/^WANDB_PLACEHOLDER$/d' "$SCRIPT"
cat >> "$SCRIPT" << WANDB_INSERT
${WANDB_BLOCK}
WANDB_INSERT

cat >> "$SCRIPT" << 'FOOTER'

echo "CMD: $TRAINING_CMD"
srun -lu --mpi=pmix --network=disable_rdzv_get --environment=alps3 --cpus-per-task $SLURM_CPUS_PER_TASK --wait 60 bash -c "numactl --membind=0-3 $TRAINING_CMD"

echo "END TIME: $(date)"
FOOTER

chmod +x "$SCRIPT"

echo "Generated: $SCRIPT (DTYPE=${DTYPE} NUM_WORKERS=${NUM_WORKERS} MANUAL_GC=${MANUAL_GC} MBS=${MBS} GBS=${GBS} OPT=${OPTIMIZER} LR=${LR} SCHED=${LR_DECAY})"
sbatch "$SCRIPT"
