from __future__ import annotations

# Keep everything small and fast; this is a quick NEAT prototype.

# --- Snake ---
GRID_W = 12
GRID_H = 12
MAX_STEPS = 250

# Observation layout (floats in [0,1] typically except distance sign terms)
# We use:
# - danger_straight, danger_left, danger_right (3)
# - dir one-hot (4)
# - food relative dx_sign, dy_sign (2)  (each in {-1,0,1} then converted to float)
# - normalized Manhattan distance to food (1)
# Total = 10
OBS_SIZE = 10

# --- NEAT ---
POP_SIZE = 40  # population size per generation (we will still evaluate in parallel)

# How many genomes to evaluate in parallel per generation
# (You asked for ~20 running at once; this is our worker batch size.)
PARALLEL_EVAL = 20

GENERATIONS = 1000

# Optional: allow multiple consecutive generations without quitting
# (kept for compatibility with the training loop; rendering does not affect training).
ELITES = 2

# Mutation rates
ADD_NODE_PROB = 0.08
ADD_CONN_PROB = 0.20
WEIGHT_MUT_PROB = 0.80

# Mutation parameters
WEIGHT_PERTURB_STD = 0.5
WEIGHT_REPLACE_PROB = 0.10
WEIGHT_REPLACE_RANGE = 1.0

# Speciation is omitted in this quick prototype (simple tournament selection instead)
TOURNAMENT_K = 5

# --- Fitness ---
# Base rewards
REWARD_FOOD = 20.0

# Step penalty / time cost (small)
STEP_PENALTY = 0.01

# Death penalty
PENALTY_DEATH = 30.0

# Shaping: reward moving closer to food, small.
# We'll use delta_distance = prev_dist - new_dist (Manhattan)
# - if positive (closer): +REWARD_CLOSER * delta
# - if negative (farther): -REWARD_FARTHER * (-delta)
REWARD_CLOSER = 1.0
REWARD_FARTHER = 0.5

# --- Training ---
CHECKPOINT_DIR = "checkpoints"
BEST_GENOME_PATH = f"{CHECKPOINT_DIR}/best_genome.json"

# Episodes per genome (use >1 to reduce reward noise; still fast-ish)
EPISODES_PER_GENOME = 3

# If you want to run faster for smoke tests
SMOKE_GENERATIONS = 2
SMOKE_POP_SIZE = 10
SMOKE_PARALLEL_EVAL = 4




# --- Rendering ---
RENDER_TEXT = True

# Turtle render pacing (human visible lag). 
# Higher = slower rendering.
RENDER_SLEEP_SEC = 0.1


