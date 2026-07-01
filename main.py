from __future__ import annotations

import json
import os
import random
import time

import neat_snake.config as config
from neat_snake.neat_brain import Genome
from neat_snake.snake_env import SnakeEnv
from neat_snake.train_parallel import train


def render_episode(genome: Genome, seed: int | None = None) -> None:
    env = SnakeEnv(seed=seed)
    obs = env.reset(seed=seed)

    net = genome.build_network()

    done = False
    steps = 0

    # Prefer turtle rendering for visibility.
    # Turtle/Tk can crash with: _tkinter.TclError: invalid command name ".!canvas"
    # so we must recover gracefully.
    try:
        import turtle

        screen = turtle.Screen()
    except Exception:
        turtle = None
        screen = None

    if screen is None:
        # Text-only fallback if turtle/tk fails to initialize.
        for _ in range(config.MAX_STEPS):
            outs = net.forward(obs)
            action = max(range(4), key=lambda i: outs[i])
            res = env.step(action)
            obs = res.obs
            if res.ate_food:
                print(f"Ate food | score={env.score}")
            if res.done:
                print(f"Died | score={env.score} steps={_ + 1}")
                obs = env.reset(seed=(seed or 0) + _ + 1)
        return

    screen.title("NEAT Snake")
    screen.tracer(0, 0)  # manual updates

    cell = 20
    margin = 20
    screen.setup(
        config.GRID_W * cell + margin * 2,
        config.GRID_H * cell + margin * 2,
    )

    t = turtle.Turtle(visible=False)
    t.speed(0)
    t.hideturtle()

    def to_screen(x: int, y: int) -> tuple[float, float]:
        # Grid origin (0,0) is top-left in our env.
        sx = -config.GRID_W * cell / 2 + x * cell + cell / 2
        sy = config.GRID_H * cell / 2 - y * cell - cell / 2
        return sx, sy

    def draw_square(x: int, y: int, color: str) -> None:
        sx, sy = to_screen(x, y)
        t.penup()
        t.goto(sx - cell / 2, sy - cell / 2)
        t.color(color)
        t.fillcolor(color)
        t.pendown()
        t.begin_fill()
        for _ in range(4):
            t.forward(cell)
            t.left(90)
        t.end_fill()
        t.penup()

    episode_idx = 0
    while steps < config.MAX_STEPS:
        # If turtle/tk backend gets invalidated mid-run, fall back to text rendering.
        try:
            outs = net.forward(obs)
            action = max(range(4), key=lambda i: outs[i])
            res = env.step(action)
            obs = res.obs
            done = res.done
            steps += 1

            t.clear()
            screen.update()

            fx, fy = env.food
            draw_square(fx, fy, "red")

            for si, (sx, sy) in enumerate(env.snake):
                color = "green" if si == len(env.snake) - 1 else "lightgreen"
                draw_square(sx, sy, color)

            screen.title(
                "NEAT Snake"
                f" | genome={type(genome).__name__}"
                f" | episode={episode_idx}"
                f" | score={env.score}"
                f" | steps={steps}"
                f" | done={done}"
            )
            screen.update()
            turtle.update()
            time.sleep(config.RENDER_SLEEP_SEC)
        except Exception as e:
            # Switch to text mode for the remainder.
            print(f"Turtle render failed ({type(e).__name__}). Switching to text mode.")
            for _ in range(config.MAX_STEPS - steps):
                outs = net.forward(obs)
                action = max(range(4), key=lambda i: outs[i])
                res = env.step(action)
                obs = res.obs
                if res.ate_food:
                    print(f"Ate food | score={env.score}")
                if res.done:
                    print(f"Died | score={env.score} steps={_ + 1}")
                    break
            break

        if done:
            episode_idx += 1
            next_seed = (seed or 0) + episode_idx * 1000 + steps
            obs = env.reset(seed=next_seed)

    if screen is not None:
        screen.update()
    print(f"Episode finished. Score={env.score}")
    time.sleep(1.0)


def main() -> None:
    # Args (simple):
    #   python -m neat_snake.main                -> train then render best
    #   python -m neat_snake.main --render       -> render best from checkpoint
    #   python -m neat_snake.main --watch-render -> continuously render latest checkpoint
    #   python -m neat_snake.main --smoke        -> render quick demo (no training)
    import sys

    args = set(sys.argv[1:])

    def parse_int_flag(name: str, default: int) -> int:
        for a in sys.argv[1:]:
            if a.startswith(name + "="):
                return int(a.split("=", 1)[1])
        return default

    if any(a.startswith("--parallel=") for a in sys.argv[1:]):
        config.PARALLEL_EVAL = parse_int_flag("--parallel", config.PARALLEL_EVAL)
    if any(a.startswith("--gens=") for a in sys.argv[1:]):
        config.GENERATIONS = parse_int_flag("--gens", config.GENERATIONS)
    if any(a.startswith("--episodes=") for a in sys.argv[1:]):
        config.EPISODES_PER_GENOME = parse_int_flag("--episodes", config.EPISODES_PER_GENOME)
    if any(a.startswith("--pop=") for a in sys.argv[1:]):
        config.POP_SIZE = parse_int_flag("--pop", config.POP_SIZE)

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    watch_sec = parse_int_flag("--watch-sec", 10)
    watch_seed = parse_int_flag("--watch-seed", 111)

    if "--watch-render" in args:
        if not os.path.exists(config.BEST_GENOME_PATH):
            print(f"No checkpoint found yet at {config.BEST_GENOME_PATH}. Waiting...")

        last_hash: str | None = None
        episode_seed = watch_seed

        while True:
            try:
                if not os.path.exists(config.BEST_GENOME_PATH):
                    time.sleep(watch_sec)
                    continue

                # Read file and detect changes.
                with open(config.BEST_GENOME_PATH, "rb") as f:
                    raw = f.read()

                import hashlib

                h = hashlib.sha256(raw).hexdigest()
                if last_hash is not None and h == last_hash:
                    time.sleep(watch_sec)
                    continue

                last_hash = h

                d = json.loads(raw.decode("utf-8"))
                best = Genome.from_dict(d, rng=random.Random(0))

                render_episode(best, seed=episode_seed)
                episode_seed += 1
            except KeyboardInterrupt:
                print("watch-render stopped")
                return
            except Exception as e:
                print(f"watch-render error ({type(e).__name__}): {e}")
                time.sleep(watch_sec)

    if "--render" in args:

        if os.path.exists(config.BEST_GENOME_PATH):
            with open(config.BEST_GENOME_PATH, "r", encoding="utf-8") as f:
                d = json.load(f)
            best = Genome.from_dict(d, rng=random.Random(0))

            seeds = [111, 222, 333]
            for s in seeds:
                render_episode(best, seed=s)
        else:
            print(f"No checkpoint found at {config.BEST_GENOME_PATH}")
        return

    if "--smoke" in args:
        g = Genome(rng=random.Random(0), input_size=config.OBS_SIZE, output_size=4)
        render_episode(g, seed=42)
        return

    # Train
    result = train(seed=0)

    # Load the best from disk (ensures it's saved correctly)
    if os.path.exists(config.BEST_GENOME_PATH):
        with open(config.BEST_GENOME_PATH, "r", encoding="utf-8") as f:
            d = json.load(f)
        best = Genome.from_dict(d, rng=random.Random(0))
        render_episode(best, seed=123)
    else:
        render_episode(result.best_genome, seed=123)


if __name__ == "__main__":
    main()

