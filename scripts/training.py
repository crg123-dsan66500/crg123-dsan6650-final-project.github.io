import argparse
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.qrdqn import QRDQN
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList

from bellmans_bakery import BellmansBakeryEnv
from .pricing_bandit import PricingBanditWrapper, ParBanditWrapper


def _action_mask(env):
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


def _make_env(use_price_bandit: bool, use_par_bandit: bool, cfg: dict, bandit_metric: str, bandit_arms: list[float] | None):
    def _thunk():
        env = BellmansBakeryEnv(config=cfg)
        if use_par_bandit:
            env = ParBanditWrapper(env)
        if use_price_bandit:
            env = PricingBanditWrapper(env, arms=(bandit_arms or [0.9, 1.0, 1.1]), metric_mode=bandit_metric)
        return ActionMasker(env, _action_mask)

    return _thunk


def train(
    use_bandit: bool = False,
    use_par_bandit: bool = False,
    steps: int = 300_000,
    n_envs: int = 8,
    logdir: str | None = None,
    model_out: str | None = None,
    cfg: dict | None = None,
    bandit_metric: str = "composite",
    bandit_arms: list[float] | None = None,
):
    cfg = dict(DEFAULT_CFG if cfg is None else cfg)
    tag = "ppo_bandit" if use_bandit else "ppo"
    if logdir is None:
        logdir = f"runs/{tag}"
    if model_out is None:
        model_out = f"models/{tag}_{steps}.zip"
    Path(logdir).mkdir(parents=True, exist_ok=True)
    Path(Path(model_out).parent).mkdir(parents=True, exist_ok=True)

    vec_env = make_vec_env(_make_env(use_bandit, use_par_bandit, cfg, bandit_metric, bandit_arms), n_envs=n_envs, seed=42)

    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=logdir,
        n_steps=2048,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.995,
        gae_lambda=0.95,
        ent_coef=0.01,
        clip_range=0.2,
        device="auto",
    )

    _eval_env = BellmansBakeryEnv(config=cfg)
    if use_par_bandit:
        _eval_env = ParBanditWrapper(_eval_env)
    if use_bandit:
        _eval_env = PricingBanditWrapper(_eval_env, arms=(bandit_arms or [0.9, 1.0, 1.1]), metric_mode=bandit_metric)
    eval_env = ActionMasker(_eval_env, _action_mask)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="models",
        log_path=logdir,
        eval_freq=10_000 // n_envs,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=steps, progress_bar=True, callback=eval_cb)
    model.save(model_out)
    # Also keep a readable best-model name without subdirectories
    try:
        from pathlib import Path as _P

        best_named = _P("models") / f"best_{tag}.zip"
        model.save(str(best_named))
    except Exception:
        pass

    # quick eval
    _eval_env2 = BellmansBakeryEnv()
    if use_par_bandit:
        _eval_env2 = ParBanditWrapper(_eval_env2)
    if use_bandit:
        _eval_env2 = PricingBanditWrapper(_eval_env2, arms=(bandit_arms or [0.9, 1.0, 1.1]), metric_mode=bandit_metric)
    eval_env = ActionMasker(_eval_env2, _action_mask)
    returns, profits = [], []
    for _ in range(5):
        obs, info = eval_env.reset()
        ret, done, last = 0.0, False, info
        while not done:
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            ret += reward
            last = info
        returns.append(ret)
        profits.append(float(last.get("profit", 0.0)))
    tag = "Bandit" if use_bandit else "PPO"
    print(f"[{tag} Eval] mean return: {np.mean(returns):.3f} ± {np.std(returns):.3f}")
    print(f"[{tag} Eval] mean profit: ${np.mean(profits):.2f} ± {np.std(profits):.2f}")


def train_qrdqn(
    steps: int = 1_000_000,
    n_envs: int = 1,
    logdir: str | None = None,
    model_out: str | None = None,
    cfg: dict | None = None,
    checkpoint_freq: int = 100_000,
    checkpoint_prefix: str = "qrdqn_ckpt",
    save_replay_buffer: bool = True,
):
    """
    Train QRDQN as a third method (discrete off-policy).
    Note: action masking is not natively supported; env penalizes illegal actions softly.
    """
    cfg = dict(DEFAULT_CFG if cfg is None else cfg)
    tag = "qrdqn"
    if logdir is None:
        logdir = f"runs/{tag}"
    if model_out is None:
        model_out = f"models/{tag}_{steps}.zip"
    Path(logdir).mkdir(parents=True, exist_ok=True)
    Path(Path(model_out).parent).mkdir(parents=True, exist_ok=True)

    def _env_thunk():
        return ActionMasker(BellmansBakeryEnv(config=cfg), _action_mask)

    vec_env = make_vec_env(_env_thunk, n_envs=n_envs, seed=42)

    model = QRDQN(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        learning_starts=20_000,
        batch_size=256,
        gamma=0.995,
        target_update_interval=1_000,
        train_freq=1,
        gradient_steps=1,
        tensorboard_log=logdir,
        verbose=1,
        device="auto",
    )

    eval_env = ActionMasker(BellmansBakeryEnv(config=cfg), _action_mask)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="models",
        log_path=logdir,
        eval_freq=25_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )
    # Periodic checkpointing (model + optional replay buffer)
    try:
        ckpt_cb = CheckpointCallback(
            save_freq=int(checkpoint_freq),
            save_path="models",
            name_prefix=str(checkpoint_prefix),
            save_replay_buffer=bool(save_replay_buffer),
        )
    except TypeError:
        # Older SB3 fallback without save_replay_buffer parameter
        ckpt_cb = CheckpointCallback(
            save_freq=int(checkpoint_freq),
            save_path="models",
            name_prefix=str(checkpoint_prefix),
        )
    callbacks = CallbackList([eval_cb, ckpt_cb])

    model.learn(total_timesteps=steps, progress_bar=True, callback=callbacks)
    model.save(model_out)

    # quick eval
    returns, profits = [], []
    obs, info = eval_env.reset()
    done, last = False, info
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        done = terminated or truncated
        returns.append(float(reward))
        last = info
    print(f"[QRDQN Eval] profit=${float(last.get('profit', 0.0)):.2f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bandit", action="store_true")
    parser.add_argument("--steps", type=int, default=300_000)
    parser.add_argument("--n_envs", type=int, default=8)
    parser.add_argument("--par_bandit", action="store_true")
    parser.add_argument("--logdir", type=str, default=None)
    parser.add_argument("--model_out", type=str, default=None)
    args = parser.parse_args()
    train(
        use_bandit=args.bandit,
        steps=args.steps,
        use_par_bandit=args.par_bandit,
        n_envs=args.n_envs,
        logdir=args.logdir,
        model_out=args.model_out,
    )


if __name__ == "__main__":
    main()
