from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import gymnasium as gym


@dataclass
class ThompsonBandit:
    """
    Very small TS bandit over fixed arms. We model per-arm rewards with a normal prior,
    keep running mean and count and sample from N(mean, 1/(count+1)) to select.
    Reward is any scalar; here we use profit minus penalties.
    """

    arms: List[float]

    def __post_init__(self):
        self.n = np.zeros(len(self.arms), dtype=np.int32)
        self.mu = np.zeros(len(self.arms), dtype=np.float64)

    def select(self) -> int:
        sigma = 1.0 / (self.n + 1.0)  # simple variance heuristic
        samples = np.random.normal(self.mu, sigma)
        return int(np.argmax(samples))

    def update(self, arm_idx: int, reward: float) -> None:
        k = arm_idx
        self.n[k] += 1
        # online mean update
        self.mu[k] += (reward - self.mu[k]) / float(self.n[k])


class PricingBanditWrapper(gym.Wrapper):
    """
    Wraps BellmansBakeryEnv and, at each reset (day), chooses a global price
    multiplier via a small Thompson-sampling bandit. On episode termination,
    updates the bandit using a scalar reward:
        metric = profit - 2.0*abandoned - 1.0*balked
    """

    def __init__(self, env: gym.Env, arms: Optional[List[float]] = None, metric_mode: str = "composite"):
        super().__init__(env)
        self.arms = list(arms or [0.9, 1.0, 1.1])
        self.bandit = ThompsonBandit(self.arms)
        self._current_arm = 1
        # metric_mode: "composite" (default) or "profit"
        self.metric_mode = metric_mode

    def reset(self, **kwargs):
        self._current_arm = self.bandit.select()
        m = float(self.arms[self._current_arm])
        # VecEnv may pass options already; merge cleanly
        options = dict(kwargs.pop("options", {}) or {})
        options["price_multiplier"] = m
        obs, info = self.env.reset(options=options, **kwargs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            profit = float(info.get("profit", 0.0))
            if self.metric_mode == "profit":
                metric = profit
            else:
                abandoned = float(info.get("abandoned", 0) or 0)
                balked = float(info.get("balked", 0) or 0)
                wait_ticks = float(info.get("wait_ticks", 0) or 0)
                metric = profit - 0.02 * wait_ticks - 5.0 * abandoned - 2.0 * balked
            self.bandit.update(self._current_arm, metric)
        return obs, reward, terminated, truncated, info


class ParBanditWrapper(gym.Wrapper):
    """
    Morning par (pre-bake) bandit.
    On reset, chooses one of a small set of par vectors and tops up inventory to that level.
    On episode end, updates bandit with net profit (env.profit already has leftover cost subtracted).
    """

    def __init__(self, env: gym.Env, par_arms: Optional[List[List[int]]] = None):
        super().__init__(env)
        # Default three par strategies: more serve-focused toward demand skew
        self.par_arms = par_arms or [
            [3, 6, 10, 1, 8],
            [4, 8, 12, 2, 10],
            [5, 10, 14, 2, 12],
        ]
        self.bandit = ThompsonBandit([0.0] * len(self.par_arms))
        self._current_arm = 1

    def reset(self, **kwargs):
        self._current_arm = self.bandit.select()
        obs, info = self.env.reset(**kwargs)
        # Top up inventory to chosen par levels
        par = np.array(self.par_arms[self._current_arm], dtype=np.int32)
        inv = getattr(self.env.unwrapped, "inventory")
        delta = np.maximum(par - inv, 0)
        if delta.sum() > 0:
            self.env.unwrapped.inventory = inv + delta
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            net = float(info.get("profit", 0.0))  # already net after leftover subtraction
            self.bandit.update(self._current_arm, net)
        return obs, reward, terminated, truncated, info

