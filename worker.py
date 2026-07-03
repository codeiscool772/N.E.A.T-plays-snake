from __future__ import annotations

import random
from typing import Any

from . import config
from .neat_brain import Genome
from .snake_env import SnakeEnv


def evaluate_genome_worker(genome_dict: dict[str, Any], seed: int, episodes: int = 1) -> float:
    """Subprocess-safe genome evaluation.

    IMPORTANT: This module must not import tkinter / GUI code.
    """
    rng = random.Random(seed)
    total = 0.0

    genome = Genome.from_dict(genome_dict, rng=rng)
    net = genome.build_network()

    for _ in range(episodes):
        env_seed = rng.randint(0, 2**31 - 1)
        env = SnakeEnv(seed=env_seed)
        obs = env.reset(seed=env_seed)
        done = False
        ep_reward = 0.0
        ate_count = 0

        steps = 0
        while not done and steps < config.MAX_STEPS:
            outputs = net.forward(obs)
            action = max(range(4), key=lambda i: outputs[i])
            res = env.step(action)
            ep_reward += res.reward
            ate_count += 1 if res.ate_food else 0
            obs = res.obs
            done = res.done
            steps += 1

        # Small bonus just to break ties early.
        ep_reward += ate_count * 0.5
        total += ep_reward

    return total / max(1, episodes)
