from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from bellmans_bakery import BellmansBakeryEnv


def _earliest_servable_item(env) -> Optional[int]:
    # Find earliest customer's item that is in stock
    inv = env.unwrapped.inventory
    for c in env.unwrapped.queue:
        if inv[c.desired_item] > 0:
            return int(c.desired_item)
    return None


def _first_true_index(mask: np.ndarray, start: int, end: int) -> Optional[int]:
    for i in range(start, end):
        if mask[i]:
            return i
    return None


@dataclass
class BakeToParPolicy:
    # Target inventory levels to “keep on shelf”
    par: List[int] = None

    def __post_init__(self):
        if self.par is None:
            # slice, matcha roll, red velvet, drip cake, orange roll (ordered by env indices)
            # env order: [red_velvet, matcha_roll, strawberry_slice, drip_cake, orange_roll]
            # Convert to env order:
            # red_velvet=1, matcha=2, slice=6, drip=1, orange=2  -> [1,2,6,1,2]
            self.par = [1, 2, 6, 1, 2]

    def on_reset(self, env: BellmansBakeryEnv):
        pass

    def choose(self, env: BellmansBakeryEnv, mask: np.ndarray) -> int:
        # Prefer serving the earliest matching customer
        item = _earliest_servable_item(env)
        if item is not None:
            return item  # serve_i index is exactly 0..4
        # Otherwise, bake items with largest deficit to par, if space
        inv = env.unwrapped.inventory
        deficit = np.array(self.par, dtype=np.int32) - inv
        order = np.argsort(-deficit)  # descending deficit
        for item_idx in order:
            a = 5 + int(item_idx)  # bake_i
            if mask[a]:
                return a
        # Idle
        return len(mask) - 1


@dataclass
class GreedyQueuePolicy:
    def on_reset(self, env: BellmansBakeryEnv):
        pass

    def choose(self, env: BellmansBakeryEnv, mask: np.ndarray) -> int:
        item = _earliest_servable_item(env)
        if item is not None:
            return item
        # Bake the most wanted item in the queue; fallback to highest demand mix
        wants = np.zeros(5, dtype=np.int32)
        for c in env.unwrapped.queue:
            wants[c.desired_item] += 1
        if wants.sum() > 0:
            order = np.argsort(-wants)
        else:
            order = np.argsort(-env.unwrapped.demand_mix_today)
        for item_idx in order:
            a = 5 + int(item_idx)
            if mask[a]:
                return a
        return len(mask) - 1


@dataclass
class NewsvendorPolicy:
    service_fraction: float = 0.35  # plan to cover ~35% of expected demand
    targets: Optional[np.ndarray] = None

    def on_reset(self, env: BellmansBakeryEnv):
        expected = env.unwrapped.avg_customers_per_day * env.unwrapped.demand_mix_today
        self.targets = np.round(self.service_fraction * expected).astype(np.int32)

    def choose(self, env: BellmansBakeryEnv, mask: np.ndarray) -> int:
        item = _earliest_servable_item(env)
        if item is not None:
            return item
        if self.targets is None:
            self.on_reset(env)
        inv = env.unwrapped.inventory
        gap = self.targets - inv
        order = np.argsort(-gap)
        for item_idx in order:
            a = 5 + int(item_idx)
            if mask[a] and gap[int(item_idx)] > 0:
                return a
        return len(mask) - 1


