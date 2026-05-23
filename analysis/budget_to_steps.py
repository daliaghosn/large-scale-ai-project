# !/usr/bin/env python3
# python3 /iopsstor/scratch/cscs/$USER/large-scale-ai-project/analysis/budget_to_steps.py \
#  --throughput 57000 --model 760m --nodes 8
"""
Budget -> steps converter for Gipfelsturm Challenge 1.

Inputs:
    --throughput   tokens/sec/GPU (median of last 50 steps, post-warmup)
    --model        125m | 350m | 760m | 1.5b | 3b | 8b
    --nodes        number of nodes (4 GPUs each on Clariden)
    --budgets      comma-separated wall-clock budgets (default: 30m,1h,2h)

Outputs a table: budget -> steps, tokens, tokens-per-param ratio.

Assumptions (match launch.sh defaults):
    - GBS = 256
    - SEQ_LEN = 4096
    - GPUS_PER_NODE = 4
    - 60s startup penalty
    - 3% eval overhead (with EVAL_INTERVAL=200, EVAL_ITERS=50 this is ballpark)
"""

import argparse
import sys

# Model param counts (approximate, from standard transformer arithmetic
# given the configs in launch.sh). These are body params only; embedding
# params differ from the "name" by a small amount.
MODEL_PARAMS = {
    "125m": 124_000_000,
    "350m": 354_000_000,
    "760m": 760_000_000,
    "1.5b": 1_500_000_000,
    "3b":   3_000_000_000,
    "8b":   8_000_000_000,
}

GBS = 256
SEQ_LEN = 4096
GPUS_PER_NODE = 4
STARTUP_PENALTY_S = 60
EVAL_OVERHEAD_FRAC = 0.03

BUDGETS_DEFAULT = {
    "30min": 30 * 60,
    "1h":    60 * 60,
    "2h":   120 * 60,
}


def parse_budgets(arg):
    if not arg:
        return BUDGETS_DEFAULT
    out = {}
    for tok in arg.split(","):
        tok = tok.strip()
        if tok.endswith("h"):
            out[tok] = int(float(tok[:-1]) * 3600)
        elif tok.endswith("min") or tok.endswith("m"):
            n = tok.rstrip("min").rstrip("m")
            out[tok] = int(float(n) * 60)
        else:
            print(f"Bad budget token: {tok}", file=sys.stderr)
            sys.exit(1)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--throughput", type=float, required=True,
                   help="tok/s/GPU, median post-warmup")
    p.add_argument("--model", required=True, choices=list(MODEL_PARAMS.keys()))
    p.add_argument("--nodes", type=int, default=4)
    p.add_argument("--budgets", type=str, default="",
                   help="comma-separated, e.g. '30min,1h,2h'")
    args = p.parse_args()

    budgets = parse_budgets(args.budgets)
    n_gpus = args.nodes * GPUS_PER_NODE
    params = MODEL_PARAMS[args.model]

    tok_per_step = GBS * SEQ_LEN
    tok_per_sec_total = args.throughput * n_gpus

    print(f"Model:       {args.model} ({params/1e6:.0f}M params)")
    print(f"Throughput:  {args.throughput:,.0f} tok/s/GPU "
          f"x {n_gpus} GPUs = {tok_per_sec_total/1e3:,.0f}k tok/s total")
    print(f"GBS x SEQ:   {GBS} x {SEQ_LEN} = {tok_per_step:,} tok/step")
    print(f"Startup:     -{STARTUP_PENALTY_S}s | Eval overhead: -{EVAL_OVERHEAD_FRAC*100:.0f}%")
    print()
    print(f"{'budget':>8} {'steps':>8} {'tokens':>14} {'tok/param':>10} {'verdict':>20}")
    print("-" * 64)

    for label, seconds in budgets.items():
        usable = (seconds - STARTUP_PENALTY_S) * (1 - EVAL_OVERHEAD_FRAC)
        steps = int((usable * tok_per_sec_total) // tok_per_step)
        tokens = steps * tok_per_step
        tok_per_param = tokens / params

        # Chinchilla-ish heuristic. Below 10: severely undertrained.
        # 10-20: undertrained but workable. 20-30: ~optimal. >30: over.
        if tok_per_param < 10:
            verdict = "severely undertrained"
        elif tok_per_param < 20:
            verdict = "undertrained"
        elif tok_per_param < 30:
            verdict = "near-optimal"
        else:
            verdict = "over-trained (ok)"

        print(f"{label:>8} {steps:>8,} {tokens/1e9:>12.2f}B "
              f"{tok_per_param:>9.1f}x {verdict:>20}")

    print()
    print("Notes:")
    print("- Chinchilla optimum is ~20 tok/param for LONG training horizons.")
    print("  At sub-optimal horizons the optimum shifts toward SMALLER models")
    print("  (closer to 30-50 tok/param). Treat verdicts as rough guidance.")
    print("- 'steps' assumes constant throughput. Real runs vary +/-10-30%.")
    print("- Update --throughput when Frederico locks in the optimized config.")


if __name__ == "__main__":
    main()

