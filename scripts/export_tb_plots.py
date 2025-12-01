import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_scalar(path: Path, tag: str):
    ea = EventAccumulator(str(path))
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return None, None
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events], dtype=np.int64)
    vals = np.array([e.value for e in events], dtype=np.float64)
    return steps, vals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="runs/ppo_tiny_bakery")
    parser.add_argument("--outdir", type=str, default="reports/figs")
    args = parser.parse_args()

    logdir = Path(args.logdir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Find latest TB run subdir
    runs = sorted(logdir.glob("**/events.*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        print("No TensorBoard event files found.")
        return
    event_file = runs[0]

    # Export only reward-focused plots by default
    plots = [
        ("rollout/ep_rew_mean", "Episode reward (mean)"),
        ("eval/mean_reward", "Eval mean reward"),
    ]

    for tag, title in plots:
        steps, vals = load_scalar(event_file, tag)
        if steps is None:
            print(f"Tag missing: {tag}")
            continue
        plt.figure(figsize=(6, 4))
        plt.plot(steps, vals, color="#8a5fbf")
        plt.title(title)
        plt.xlabel("steps")
        plt.ylabel(tag)
        plt.grid(alpha=0.2)
        outfile = outdir / (tag.replace("/", "_") + ".png")
        plt.tight_layout()
        plt.savefig(outfile, dpi=160)
        plt.close()
        print(f"Saved {outfile}")


if __name__ == "__main__":
    main()


