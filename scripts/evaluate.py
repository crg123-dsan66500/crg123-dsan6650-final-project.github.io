import argparse
import csv
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.qrdqn import QRDQN
from sb3_contrib.common.wrappers import ActionMasker

from bellmans_bakery import BellmansBakeryEnv
from .pricing_bandit import PricingBanditWrapper, ParBanditWrapper
from .baselines import BakeToParPolicy, GreedyQueuePolicy, NewsvendorPolicy


def _mask(env):
    return env.unwrapped._action_mask()


DEFAULT_CFG = dict(
    day_ticks=240,
    num_ovens=2,
    oven_capacity=4.0,
    avg_customers_per_day=60,
    first_arrival_delay_ticks=2,
    enable_nonstationarity=True,
    daily_drift_pct=0.10,
    weekly_item_swing_pct=0.10,
    queue_cap=12,
    observe_k=5,
    wait_penalty_per_tick=0.01,
    abandon_penalty=0.7,
    serve_bonus=0.1,
    idle_penalty=0.0,
    balk_penalty=0.1,
    serve_per_tick=3,
)


def eval_baselines(seeds: int = 5, days: int = 10, out: str = "reports/baselines.csv", cfg=None):
    cfg = dict(DEFAULT_CFG if cfg is None else cfg)
    Path(Path(out).parent).mkdir(parents=True, exist_ok=True)

    def _run(policy, seed, config):
        env = ActionMasker(BellmansBakeryEnv(config=config), _mask)
        obs, info = env.reset(seed=seed)
        if hasattr(policy, "on_reset"):
            policy.on_reset(env.unwrapped)
        done = False
        last = info
        while not done:
            action = policy.choose(env, info["action_mask"])
            obs, reward, terminated, truncated, info = env.step(action)
            last = info
            done = terminated or truncated
        return last

    policies = [
        ("bake_to_par", BakeToParPolicy()),
        ("greedy_queue", GreedyQueuePolicy()),
        ("newsvendor", NewsvendorPolicy()),
    ]

    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["policy", "seed", "day", "profit", "served", "arrivals", "abandoned", "balked", "avg_wait_seconds", "leftover_units", "leftover_cost"]
        )
        for name, pol in policies:
            for s in range(seeds):
                for d in range(days):
                    info = _run(pol, seed=s * 1000 + d, config=cfg)
                    served = info.get("served", 0) or 0
                    arrivals = info.get("arrivals", 0) or 0
                    wait_ticks = info.get("wait_ticks", 0) or 0
                    avg_wait = (wait_ticks * 10.0) / max(1, arrivals)
                    w.writerow([name, s, d, f"{info.get('profit', 0.0):.2f}", served, arrivals, info.get("abandoned", 0) or 0, info.get("balked", 0) or 0, f"{avg_wait:.2f}", info.get("leftover_units"), f"{(info.get('leftover_cost') or 0.0):.2f}"])


def eval_model(model_path: str, out: str = "reports/ppo_eval.csv", seeds: int = 5, days: int = 10, bandit: bool = False, par_bandit: bool = False, cfg=None, bandit_metric: str = "composite", bandit_arms: list[float] | None = None):
    cfg = dict(DEFAULT_CFG if cfg is None else cfg)
    Path(Path(out).parent).mkdir(parents=True, exist_ok=True)

    _env = BellmansBakeryEnv(config=cfg)
    if par_bandit:
        _env = ParBanditWrapper(_env)
    if bandit:
        _env = PricingBanditWrapper(_env, arms=(bandit_arms or [0.9, 1.0, 1.1]), metric_mode=bandit_metric)
    env = ActionMasker(_env, _mask)
    model = MaskablePPO.load(model_path, env=env)

    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["model", "bandit", "seed", "day", "profit", "served", "arrivals", "abandoned", "balked", "avg_wait_seconds", "leftover_units", "leftover_cost"]
        )
        for s in range(seeds):
            for d in range(days):
                obs, info = env.reset(seed=s * 1000 + d)
                done = False
                last = info
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    last = info
                    done = terminated or truncated
                served = last.get("served", 0) or 0
                arrivals = last.get("arrivals", 0) or 0
                wait_ticks = last.get("wait_ticks", 0) or 0
                avg_wait = (wait_ticks * 10.0) / max(1, arrivals)
                w.writerow([model_path, int(bandit), s, d, f"{last.get('profit', 0.0):.2f}", served, arrivals, last.get("abandoned", 0) or 0, last.get("balked", 0) or 0, f"{avg_wait:.2f}", last.get("leftover_units"), f"{(last.get('leftover_cost') or 0.0):.2f}"])


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("baselines")
    p1.add_argument("--seeds", type=int, default=5)
    p1.add_argument("--days", type=int, default=10)
    p1.add_argument("--out", type=str, default="reports/baselines.csv")

    p2 = sub.add_parser("model")
    p2.add_argument("--model", type=str, required=True)
    p2.add_argument("--out", type=str, default="reports/ppo_eval.csv")
    p2.add_argument("--seeds", type=int, default=5)
    p2.add_argument("--days", type=int, default=10)
    p2.add_argument("--bandit", action="store_true")
    p2.add_argument("--par_bandit", action="store_true")
    p2.add_argument("--price_metric", type=str, default="composite", choices=["composite", "profit"])
    p2.add_argument("--price_arms", type=str, default=None, help="Comma-separated floats, e.g. 0.7,0.85,1.0,1.15,1.3")

    args = parser.parse_args()
    if args.cmd == "baselines":
        eval_baselines(args.seeds, args.days, args.out)
    else:
        arms = None
        if args.price_arms:
            try:
                arms = [float(x) for x in args.price_arms.split(",")]
            except Exception:
                arms = None
        eval_model(args.model, args.out, args.seeds, args.days, args.bandit, args.par_bandit, None, args.price_metric, arms)


def eval_model_qrdqn(model_path: str, out: str = "reports/qrdqn_eval.csv", seeds: int = 5, days: int = 10, cfg=None):
    cfg = dict(DEFAULT_CFG if cfg is None else cfg)
    Path(Path(out).parent).mkdir(parents=True, exist_ok=True)

    env = ActionMasker(BellmansBakeryEnv(config=cfg), _mask)
    model = QRDQN.load(model_path, env=env)

    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["model", "seed", "day", "profit", "served", "arrivals", "abandoned", "balked", "avg_wait_seconds", "leftover_units", "leftover_cost"]
        )
        for s in range(seeds):
            for d in range(days):
                obs, info = env.reset(seed=s * 1000 + d)
                done = False
                last = info
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    last = info
                    done = terminated or truncated
                served = last.get("served", 0) or 0
                arrivals = last.get("arrivals", 0) or 0
                wait_ticks = last.get("wait_ticks", 0) or 0
                avg_wait = (wait_ticks * 10.0) / max(1, arrivals)
                w.writerow([model_path, s, d, f"{last.get('profit', 0.0):.2f}", served, arrivals, last.get("abandoned", 0) or 0, last.get("balked", 0) or 0, f"{avg_wait:.2f}", last.get("leftover_units"), f"{(last.get('leftover_cost') or 0.0):.2f}"])


if __name__ == "__main__":
    main()


