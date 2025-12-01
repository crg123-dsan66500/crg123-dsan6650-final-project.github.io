import argparse
import time

import pygame
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from bellmans_bakery import BellmansBakeryEnv, PastelViewer


def mask_fn(env):
    return env._action_mask()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/ppo_quick.zip")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    env = ActionMasker(BellmansBakeryEnv(), mask_fn)
    viewer = PastelViewer()

    # Load model if present
    try:
        model = MaskablePPO.load(args.model, env=env)
    except Exception:
        model = None
        print(f"Model not found at {args.model}; running a random policy for preview.")

    obs, info = env.reset()
    done = False
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        if not done:
            if model:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        else:
            obs, info = env.reset()
            done = False
        viewer.render(env)
        time.sleep(1.0 / args.fps)


if __name__ == "__main__":
    main()


