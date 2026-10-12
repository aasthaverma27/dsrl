"""One-line status per training run: last eval, last episode stats, health.

Usage: python scripts/run_status.py [runs_dir]   (default: runs/)
"""
import json, sys, glob, os

root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs")
dirs = sorted(d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d))

for d in dirs:
    mp = os.path.join(d, "metrics.jsonl")
    if not os.path.exists(mp):
        print(f"{os.path.basename(d):<45} (no metrics yet)")
        continue
    evals, eps = [], []
    with open(mp) as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "eval_correct" in rec:
                evals.append(rec)
            elif "episode_reward" in rec:
                eps.append(rec)
    last_step = eps[-1]["step"] if eps else 0
    n_ep = len(eps)
    recent = eps[-20:]
    avg_r = sum(e["episode_reward"] for e in recent) / max(len(recent), 1)
    avg_c = sum(e["correct"] for e in recent) / max(len(recent), 1)
    cal = next((e.get("cal_corr") for e in reversed(eps) if e.get("cal_corr") is not None), None)
    alpha = next((e.get("alpha") for e in reversed(eps) if e.get("alpha") is not None), None)
    q = next((e.get("q") for e in reversed(eps) if e.get("q") is not None), None)
    line = (f"{os.path.basename(d):<45} step {last_step:>6} eps {n_ep:>4} "
            f"| last20: R={avg_r:6.1f} correct={avg_c:.2f} "
            f"| cal_corr={cal} alpha={alpha} q={q}")
    if evals:
        ev = evals[-1]
        line += (f" | EVAL@{ev['step']}: correct={ev['eval_correct']:.2f}"
                 f" ({ev['eval_correct'] / 3 * 100:.0f}% per-cube), falls={ev['eval_falls']}")
    print(line)
