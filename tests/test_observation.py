"""Tests for the 10-dim observation refactor in snake_env.py.

Dims 0-7: negative distance to nearest obstacle along 8 rays (-1 = adjacent).
Dim 8: vertical food sign (-1 above / 0 same row / +1 below).
Dim 9: horizontal food sign (-1 left / 0 same column / +1 right).
"""
import numpy as np
import pytest

from envs import SnakeEnv


@pytest.fixture
def env():
    e = SnakeEnv(grid_size=20)
    # Deterministic state: one-cell snake at the center, food set per-test
    e.snake = [(10, 10)]
    return e


def observe(e, head, food):
    e.snake = [head]
    e.food = food
    return e._get_observation()


def test_observation_has_ten_dimensions(env):
    obs = observe(env, (10, 10), (0, 0))
    assert obs.shape == (10,)


@pytest.mark.parametrize("food,expected", [
    ((5, 10), -1.0),   # food above head (smaller row)
    ((15, 10), +1.0),  # food below head
    ((10, 4), 0.0),    # same row boundary
])
def test_vertical_food_sign_equivalence_classes(env, food, expected):
    obs = observe(env, (10, 10), food)
    assert obs[8] == expected


@pytest.mark.parametrize("food,expected", [
    ((10, 5), -1.0),   # food left of head (smaller column)
    ((10, 15), +1.0),  # food right of head
    ((4, 10), 0.0),    # same column boundary
])
def test_horizontal_food_sign_equivalence_classes(env, food, expected):
    obs = observe(env, (10, 10), food)
    assert obs[9] == expected


@pytest.mark.parametrize("food,v,h", [
    ((5, 5),   -1.0, -1.0),  # up-left quadrant
    ((5, 15),  -1.0, +1.0),  # up-right
    ((15, 5),  +1.0, -1.0),  # down-left
    ((15, 15), +1.0, +1.0),  # down-right
])
def test_diagonal_food_positions_encode_both_axes(env, food, v, h):
    obs = observe(env, (10, 10), food)
    assert obs[8] == v
    assert obs[9] == h


def test_food_sign_matches_action_space_geometry(env):
    # Action 0 Up = (-1, 0), action 2 Left = (0, -1): the observation sign
    # must agree with the direction the matching action moves the head.
    obs = observe(env, (10, 10), (10 - 1, 10))
    new_head = (10 + env.directions[0][0], 10 + env.directions[0][1])
    assert obs[8] < 0 and new_head == (9, 10)

    obs = observe(env, (10, 10), (10, 10 - 1))
    new_head = (10 + env.directions[2][0], 10 + env.directions[2][1])
    assert obs[9] < 0 and new_head == (10, 9)


def test_wall_distances_at_corner_boundary(env):
    # Head in the top-left corner of a 20-grid: adjacent walls are 1 step
    # away, the opposite walls 20 steps (grid boundary values)
    obs = observe(env, (0, 0), (19, 19))
    assert obs[0] == -1.0   # North wall adjacent
    assert obs[2] == -1.0   # West wall adjacent
    assert obs[1] == -20.0  # South wall far boundary
    assert obs[3] == -20.0  # East wall far boundary
    assert obs[4] == -1.0   # NE diagonal: leaves the grid at step 1


def test_food_sign_invariant_on_random_placements(env):
    # Property: for any head/food pair, obs[8] == sign(food.row - head.row)
    # and obs[9] == sign(food.col - head.col)
    rng = np.random.default_rng(42)
    for _ in range(200):
        head = (int(rng.integers(0, 20)), int(rng.integers(0, 20)))
        food = (int(rng.integers(0, 20)), int(rng.integers(0, 20)))
        if food == head:
            continue
        obs = observe(env, head, food)
        assert obs[8] == np.sign(food[0] - head[0])
        assert obs[9] == np.sign(food[1] - head[1])
        assert np.all(obs[:8] <= 0)
