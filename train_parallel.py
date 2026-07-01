from __future__ import annotations

import json
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from . import config
from .neat_brain import Genome
from .snake_env import SnakeEnv


def evaluate_genome(genome_dict: dict[str, Any], seed: int, episodes: int = 1) -> float:
    """Evaluate a genome.

    Returns average fitness across episodes.

    NOTE: Keep this function self-contained because it runs in subprocesses.
    """
    rng = random.Random(seed)
    total = 0.0

    genome = Genome.from_dict(genome_dict, rng=rng)
    net = genome.build_network()

    for ep in range(episodes):
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

        # Small bonus just to break ties early (if reward shaping is weak).
        # This is still consistent with learning toward eating.
        ep_reward += ate_count * 0.5
        total += ep_reward

    return total / max(1, episodes)



@dataclass
class TrainResult:
    best_fitness: float
    best_genome: Genome


def tournament_select(pop: list[Genome], fitness: list[float], rng: random.Random) -> Genome:
    best_i = None
    best_fit = float('-inf')
    for _ in range(config.TOURNAMENT_K):
        i = rng.randrange(0, len(pop))
        if fitness[i] > best_fit:
            best_fit = fitness[i]
            best_i = i
    assert best_i is not None
    return pop[best_i]


def save_best(genome: Genome) -> None:
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    with open(config.BEST_GENOME_PATH, 'w', encoding='utf-8') as f:
        json.dump(genome.to_dict(), f)


def train(seed: int = 0, generations: int | None = None, parallel_eval: int | None = None) -> TrainResult:
    if generations is None:
        generations = config.GENERATIONS
    if parallel_eval is None:
        parallel_eval = config.PARALLEL_EVAL

    # Periodically save the current best so you can stop anytime and still resume.
    # (User requested single-file overwrite.)
    checkpoint_every_gens = 50


    rng = random.Random(seed)


    pop: list[Genome] = [Genome(rng=random.Random(rng.randint(0, 2**31 - 1)), input_size=config.OBS_SIZE, output_size=4) for _ in range(config.POP_SIZE)]
    best_fit = float('-inf')
    best_genome: Genome | None = None

    # Reuse the same process pool across generations.
    # Spawning/tearing down workers every generation is a major slowdown.
    with ProcessPoolExecutor(max_workers=config.PARALLEL_EVAL) as ex:
        for gen in range(generations):
            # Evaluate in parallel
            genome_dicts = [g.to_dict() for g in pop]
            seeds = [rng.randint(0, 2**31 - 1) for _ in range(len(pop))]

            fitness = [0.0] * len(pop)

            futures = {}
            for i, gd in enumerate(genome_dicts):
                fut = ex.submit(evaluate_genome, gd, seeds[i], config.EPISODES_PER_GENOME)
                futures[fut] = i

            for fut in as_completed(futures):
                i = futures[fut]
                fitness[i] = fut.result()


            # Sort by fitness
            order = sorted(range(len(pop)), key=lambda i: fitness[i], reverse=True)
            elites = [pop[i] for i in order[: config.ELITES]]
            gen_best = fitness[order[0]]

            if gen_best > best_fit:
                best_fit = gen_best
                best_genome = pop[order[0]].copy()

            # Periodic checkpoint save (single-file overwrite)
            if best_genome is not None and (gen + 1) % checkpoint_every_gens == 0:
                save_best(best_genome)

            if gen_best > best_fit:
                save_best(best_genome)


            print(f"Gen {gen:03d} | best={gen_best:.3f} | avg={sum(fitness)/len(fitness):.3f} | global_best={best_fit:.3f}")

            # Build next population
            next_pop: list[Genome] = []
            for e in elites:
                next_pop.append(e)

            while len(next_pop) < config.POP_SIZE:
                parent1 = tournament_select(pop, fitness, rng)
                parent2 = tournament_select(pop, fitness, rng)
                # Use fitness to decide which parent is fitter for crossover; estimate by comparing their indices is costly.
                # We'll just bias by using a deterministic ordering based on genome length.
                # For this quick prototype, crossover+mutate still works.
                child = parent1.crossover(parent2) if rng.random() < 0.75 else parent2.crossover(parent1)
                child.mutate()
                next_pop.append(child)

            pop = next_pop


    assert best_genome is not None
    return TrainResult(best_fitness=best_fit, best_genome=best_genome)

