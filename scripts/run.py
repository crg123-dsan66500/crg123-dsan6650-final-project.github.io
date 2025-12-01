import argparse

from .training import train
from .evaluate import eval_baselines, eval_model
from .export_tb_plots import main as export_tb_plots_main


def main():
    parser = argparse.ArgumentParser(description="Bellman's Bakery runner")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("train", help="Train PPO")
    p1.add_argument("--steps", type=int, default=300_000)
    p1.add_argument("--bandit", action="store_true")
    p1.add_argument("--par_bandit", action="store_true")
    p1.add_argument("--logdir", type=str, default=None)
    p1.add_argument("--model_out", type=str, default=None)
    # Optional env overrides
    p1.add_argument("--serve_per_tick", type=int, default=None)
    p1.add_argument("--avg_customers", type=int, default=None)
    p1.add_argument("--nonstationary_off", action="store_true")
    # Price bandit options
    p1.add_argument(
        "--price_metric", type=str, default="composite", choices=["composite", "profit"]
    )
    p1.add_argument(
        "--price_arms",
        type=str,
        default=None,
        help="Comma-separated floats, e.g. 0.7,0.85,1.0,1.15,1.3",
    )

    p3 = sub.add_parser("eval-baselines", help="Run heuristic baselines")
    p3.add_argument("--seeds", type=int, default=5)
    p3.add_argument("--days", type=int, default=10)
    p3.add_argument("--out", type=str, default="reports/baselines.csv")

    p4 = sub.add_parser("eval-model", help="Evaluate a trained model to CSV")
    p4.add_argument("--model", type=str, required=True)
    p4.add_argument("--out", type=str, default="reports/ppo_eval.csv")
    p4.add_argument("--seeds", type=int, default=5)
    p4.add_argument("--days", type=int, default=10)
    p4.add_argument("--bandit", action="store_true")
    p4.add_argument("--par_bandit", action="store_true")
    # Optional env overrides
    p4.add_argument("--serve_per_tick", type=int, default=None)
    p4.add_argument("--avg_customers", type=int, default=None)
    p4.add_argument("--nonstationary_off", action="store_true")
    # Price bandit options
    p4.add_argument(
        "--price_metric", type=str, default="composite", choices=["composite", "profit"]
    )
    p4.add_argument(
        "--price_arms",
        type=str,
        default=None,
        help="Comma-separated floats, e.g. 0.7,0.85,1.0,1.15,1.3",
    )

    p5 = sub.add_parser("export-plots", help="Export TB plots to PNG")
    p5.add_argument("--logdir", type=str, required=True)
    p5.add_argument("--outdir", type=str, required=True)

    # QRDQN: train/eval
    p6 = sub.add_parser("train-qrdqn", help="Train QRDQN")
    p6.add_argument("--steps", type=int, default=1_000_000)
    p6.add_argument("--n_envs", type=int, default=1)
    p6.add_argument("--logdir", type=str, default=None)
    p6.add_argument("--model_out", type=str, default=None)
    p6.add_argument("--checkpoint_freq", type=int, default=100_000)
    p6.add_argument("--checkpoint_prefix", type=str, default="qrdqn_ckpt")
    p6.add_argument("--no_save_replay_buffer", action="store_true")

    p7 = sub.add_parser("eval-qrdqn", help="Evaluate a QRDQN model to CSV")
    p7.add_argument("--model", type=str, required=True)
    p7.add_argument("--out", type=str, default="reports/qrdqn_eval.csv")
    p7.add_argument("--seeds", type=int, default=5)
    p7.add_argument("--days", type=int, default=10)
    args = parser.parse_args()
    if args.cmd == "train":
        cfg = None
        if (
            args.serve_per_tick is not None
            or args.avg_customers is not None
            or args.nonstationary_off
        ):
            cfg = {}
            if args.serve_per_tick is not None:
                cfg["serve_per_tick"] = int(args.serve_per_tick)
            if args.avg_customers is not None:
                cfg["avg_customers_per_day"] = int(args.avg_customers)
            if args.nonstationary_off:
                cfg["enable_nonstationarity"] = False
        arms = None
        if args.price_arms:
            try:
                arms = [float(x) for x in args.price_arms.split(",")]
            except Exception:
                arms = None
        train(
            use_bandit=args.bandit,
            use_par_bandit=args.par_bandit,
            steps=args.steps,
            logdir=args.logdir,
            model_out=args.model_out,
            cfg=cfg,
            bandit_metric=args.price_metric,
            bandit_arms=arms,
        )
    elif args.cmd == "eval-baselines":
        eval_baselines(args.seeds, args.days, args.out)
    elif args.cmd == "eval-model":
        cfg = None
        if (
            args.serve_per_tick is not None
            or args.avg_customers is not None
            or args.nonstationary_off
        ):
            cfg = {}
            if args.serve_per_tick is not None:
                cfg["serve_per_tick"] = int(args.serve_per_tick)
            if args.avg_customers is not None:
                cfg["avg_customers_per_day"] = int(args.avg_customers)
            if args.nonstationary_off:
                cfg["enable_nonstationarity"] = False
        arms = None
        if args.price_arms:
            try:
                arms = [float(x) for x in args.price_arms.split(",")]
            except Exception:
                arms = None
        eval_model(
            args.model,
            args.out,
            args.seeds,
            args.days,
            args.bandit,
            args.par_bandit,
            cfg,
            args.price_metric,
            arms,
        )
    elif args.cmd == "export-plots":
        export_tb_plots_main()
    elif args.cmd == "train-qrdqn":
        from .training import train_qrdqn
        train_qrdqn(
            steps=args.steps,
            n_envs=args.n_envs,
            logdir=args.logdir,
            model_out=args.model_out,
            checkpoint_freq=args.checkpoint_freq,
            checkpoint_prefix=args.checkpoint_prefix,
            save_replay_buffer=not args.no_save_replay_buffer,
        )
    elif args.cmd == "eval-qrdqn":
        from .evaluate import eval_model_qrdqn
        eval_model_qrdqn(args.model, args.out, args.seeds, args.days)


if __name__ == "__main__":
    main()
