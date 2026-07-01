from __future__ import annotations

import random
from dataclasses import dataclass

from . import config


DIRS = [
    (0, -1),  # up
    (1, 0),  # right
    (0, 1),  # down
    (-1, 0),  # left
]


def clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


@dataclass
class StepResult:
    obs: list[float]
    reward: float
    done: bool
    ate_food: bool


class SnakeEnv:
    """Small grid snake with fixed-size observation vector.

    Observation (10 floats):
      0-2: danger_straight, danger_left, danger_right
      3-6: direction one-hot (up,right,down,left)
      7-8: food relative dx_sign, dy_sign as floats in {-1,0,1}
      9: normalized Manhattan distance to food in [0,1] (closer => smaller)

    Outputs (NEAT): 4 floats; we take argmax to choose direction.
    """

    def __init__(
        self,
        seed: int | None = None,
    ) -> None:
        self.rng = random.Random(seed)
        self.reset(seed=seed)

    def reset(self, seed: int | None = None) -> list[float]:
        if seed is not None:
            self.rng.seed(seed)

        cx = self.rng.randrange(0, config.GRID_W)
        cy = self.rng.randrange(0, config.GRID_H)

        # Start length 3, heading right
        self.dir_idx = 1
        self.snake: list[tuple[int, int]] = [
            (cx - 1, cy),
            (cx, cy),
            (cx + 1, cy),
        ]

        # Real walls mode: no teleport wrapping.
        # Ensure the initial snake fits inside the grid.
        if self.snake[0][0] < 0:
            self.snake = [(x + 1, y) for (x, y) in self.snake]
        if self.snake[-1][0] >= config.GRID_W:
            self.snake = [(x - 1, y) for (x, y) in self.snake]

        self.score = 0
        self.steps = 0

        self.food = self._spawn_food()
        self.prev_dist = manhattan(self.snake[1], self.food)

        # Anti-loop bookkeeping
        self.last_head: tuple[int, int] | None = None
        self.last_action_dir_idx: int | None = None

        return self._get_obs()


    def _spawn_food(self) -> tuple[int, int]:
        empty = [(x, y) for x in range(config.GRID_W) for y in range(config.GRID_H) if (x, y) not in self.snake]
        if not empty:
            # Snake filled grid (rare); just pick head position
            return self.snake[0]
        return self.rng.choice(empty)

    def _would_collide(self, next_head: tuple[int, int]) -> bool:
        x, y = next_head

        # In this prototype we want "real walls" behavior (no teleport wrap).
        if x < 0 or x >= config.GRID_W or y < 0 or y >= config.GRID_H:
            return True

        # Collision with body (excluding tail since it moves) is complicated; keep it simple:
        # We'll treat any overlap with current body as collision.
        return (x, y) in self.snake


    def _move(self, action_dir_idx: int) -> tuple[bool, bool]:
        # Disallow reversing direction for stability.
        if (action_dir_idx - self.dir_idx) % 4 == 2:
            action_dir_idx = self.dir_idx

        self.dir_idx = action_dir_idx
        dx, dy = DIRS[self.dir_idx]
        hx, hy = self.snake[-1]
        new_head = (hx + dx, hy + dy)

        ate_food = new_head == self.food
        done = self._would_collide(new_head)

        # Update snake
        self.snake.append(new_head)
        if not ate_food:
            self.snake.pop(0)
        else:
            self.score += 1
            self.food = self._spawn_food()

        self.steps += 1
        if self.steps >= config.MAX_STEPS:
            done = True

        return done, ate_food

    def _danger_ahead_left_right(self) -> tuple[int, int, int]:
        # relative directions: straight, left, right from current heading
        straight = self.dir_idx
        left = (self.dir_idx - 1) % 4
        right = (self.dir_idx + 1) % 4

        hx, hy = self.snake[-1]

        def collides_for(di: int) -> int:
            dx, dy = DIRS[di]
            nx, ny = hx + dx, hy + dy

            # Real wall collision
            if nx < 0 or nx >= config.GRID_W or ny < 0 or ny >= config.GRID_H:
                return 1

            return 1 if (nx, ny) in self.snake else 0

        return collides_for(straight), collides_for(left), collides_for(right)

    def _get_obs(self) -> list[float]:
        danger_straight, danger_left, danger_right = self._danger_ahead_left_right()

        dir_one_hot = [0.0] * 4
        dir_one_hot[self.dir_idx] = 1.0

        head = self.snake[-1]
        dx = self.food[0] - head[0]
        dy = self.food[1] - head[1]

        # Convert to signs in {-1,0,1}
        dx_sign = 0.0 if dx == 0 else (1.0 if dx > 0 else -1.0)
        dy_sign = 0.0 if dy == 0 else (1.0 if dy > 0 else -1.0)

        dist = manhattan(head, self.food)
        # Normalize: max manhattan on grid ~ (w-1)+(h-1)
        max_dist = (config.GRID_W - 1) + (config.GRID_H - 1)
        dist_norm = clamp01(dist / max_dist)

        return [
            float(danger_straight),
            float(danger_left),
            float(danger_right),
            *dir_one_hot,
            dx_sign,
            dy_sign,
            dist_norm,
        ]

    def get_inputs(self) -> list[float]:
        return self._get_obs()

    def step(self, action_dir_idx: int) -> StepResult:
        prev_dist = self.prev_dist


        done, ate_food = self._move(action_dir_idx)

        new_head = self.snake[-1]
        self.prev_dist = manhattan(new_head, self.food)

        # Anti-loop reward shaping: penalize oscillation / stalling.
        anti_loop_penalty = 0.0
        if self.last_head is not None and new_head == self.last_head:
            # Going back to the same head position frequently indicates cycling.
            anti_loop_penalty += config.PENALTY_DEATH * 0.05

        if self.last_action_dir_idx is not None and action_dir_idx == self.last_action_dir_idx:
            # Keep moving in same direction into a loop can also happen; slight penalty.
            anti_loop_penalty += config.STEP_PENALTY * 2

        # Update bookkeeping for next step.
        self.last_head = new_head
        self.last_action_dir_idx = action_dir_idx

        reward = 0.0


        # Shaping: closer/farther based on Manhattan delta
        delta = prev_dist - self.prev_dist
        if delta > 0:
            reward += config.REWARD_CLOSER * delta
        elif delta < 0:
            reward -= config.REWARD_FARTHER * (-delta)

        # Base time penalty
        reward -= config.STEP_PENALTY

        if ate_food:
            reward += config.REWARD_FOOD

        if done and not ate_food:
            reward -= config.PENALTY_DEATH

        return StepResult(
            obs=self._get_obs(),
            reward=reward,
            done=done,
            ate_food=ate_food,
        )

