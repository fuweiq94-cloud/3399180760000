"""Tests for reward shaping in SnakeEnv.

Mechanisms (adapted from the SnakeAI reference project, rescaled to our ±10
reward regime):
- size-dependent food/death rewards ('scaled' mode): the longer the snake,
  the bigger the food reward and the milder the death penalty;
- a 1.5× self-death factor plus death-cause tracking;
- always-on per-step guidance: moves toward the food earn +0.3/len, moves
  away lose the same — the signal fades as the snake grows.
"""
import pytest

from envs import SnakeEnv


def make_env(grid=10, shaping='flat'):
    return SnakeEnv(grid_size=grid, observation_type='vision',
                    reward_shaping=shaping)


def die_at_wall(env, length):
    """Grow the snake to `length` cells at the top wall, then step up."""
    env.reset()
    env.snake = [(0, c) for c in range(length)]     # head at (0, 0)
    env.food = (5, 5)                               # out of the way
    _, reward, terminated, _, _ = env.step(0)       # up → wall
    assert terminated
    return reward


def die_at_self(env):
    """Head runs straight into its own neck (not the tail — the tail vacates)."""
    env.reset()
    env.snake = [(5, 5), (5, 6), (5, 7)]  # head, neck, tail to the right
    env.food = (0, 0)
    _, reward, terminated, _, _ = env.step(3)       # right → neck
    assert terminated
    return reward


def eat_with_length(env, length):
    """Grow the snake to `length`, place food above the head, eat it."""
    env.reset()
    env.snake = [(5, c) for c in range(length)]     # head at (5, 0)
    env.food = (4, 0)                               # just above the head
    _, reward, terminated, _, _ = env.step(0)       # up → food
    assert not terminated
    return reward


# ------------------------------------------------------------ flat (legacy)

def test_default_env_ships_full_mechanism_set():
    """No backward-compat mode anymore: a bare env defaults to scaled
    shaping, the 1.5× self-death factor, and per-step food guidance."""
    env = SnakeEnv(grid_size=8, observation_type='vision')
    assert env.reward_shaping == 'scaled'
    assert env.self_death_factor == 1.5


def test_flat_mode_rewards_unchanged():
    env = make_env()
    assert die_at_wall(env, length=1) == -10.0
    assert eat_with_length(env, length=1) == 10.0


# ------------------------------------------------------------ scaled mode

def test_scaled_death_penalty_at_length_one_is_near_minus_10():
    # grid 10, len 1: -(1 + 9 * (1 - 1/10)) = -9.1
    assert die_at_wall(make_env(shaping='scaled'), length=1) == pytest.approx(-9.1)


def test_scaled_death_penalty_at_board_edge_is_minus_1():
    assert die_at_wall(make_env(shaping='scaled'), length=10) == pytest.approx(-1.0)


def test_scaled_death_penalty_beyond_edge_clamps_at_minus_1():
    assert die_at_wall(make_env(shaping='scaled'), length=25) == pytest.approx(-1.0)


def test_scaled_food_reward_grows_with_length():
    env = make_env(shaping='scaled')
    early = eat_with_length(env, length=1)     # len 2 after eating: 1 + 9*0.2
    late = eat_with_length(env, length=10)     # len 11 → frac clamps to 1
    assert early == pytest.approx(2.8)
    assert late == pytest.approx(10.0)
    assert late > early


def test_scaled_rewards_monotone_and_bounded():
    env = make_env(shaping='scaled')
    prev_death = prev_food = None
    for length in range(1, env.grid_size + 5):
        env.snake = [(0, 0)] * length          # helpers only read len()
        death = env._death_reward('wall')
        food = env._food_reward()
        assert -10.0 <= death <= -1.0
        assert 1.0 <= food <= 10.0
        if prev_death is not None:
            assert death >= prev_death         # penalty keeps shrinking
            assert food >= prev_food           # food keeps growing
        prev_death, prev_food = death, food


def test_scaled_timeout_stays_minus_1():
    env = make_env(shaping='scaled')
    env.reset()
    env.snake = [(5, 5)]
    env.food = (0, 0)
    env.steps_without_food = env.max_steps
    _, reward, terminated, _, _ = env.step(1)  # free move, not food → timeout
    assert terminated
    assert reward == -1.0


# ------------------------------------------------- step guidance (always on)

def single_cell_env(food):
    """Fresh env with a 1-cell snake at (5, 5) and food placed by the test."""
    env = make_env()
    env.reset()
    env.snake = [(5, 5)]
    env.food = food
    return env


def test_step_guidance_rewards_closer_move():
    env = single_cell_env(food=(5, 8))
    _, reward, done, _, _ = env.step(3)        # right → closer: -0.1 + 0.3/1
    assert not done
    assert reward == pytest.approx(0.2)


def test_step_guidance_penalizes_away_move():
    env = single_cell_env(food=(5, 8))
    _, reward, done, _, _ = env.step(2)        # left → farther: -0.1 - 0.3/1
    assert not done
    assert reward == pytest.approx(-0.4)


def test_step_guidance_fades_with_snake_length():
    env = make_env()
    env.reset()
    env.snake = [(r, 5) for r in range(10)]    # len 10, head (0, 5)
    env.food = (0, 8)                          # to the right, off-body
    _, reward, done, _, _ = env.step(3)        # closer: -0.1 + 0.3/10
    assert not done
    assert reward == pytest.approx(-0.07)


def test_step_guidance_applies_in_both_shaping_modes():
    for shaping in ('flat', 'scaled'):
        env = single_cell_env(food=(5, 8))
        env.reward_shaping = shaping
        _, reward, done, _, _ = env.step(3)
        assert not done
        assert reward == pytest.approx(0.2)


def test_step_guidance_never_zero_on_normal_move():
    """Every 4-directional move changes the Manhattan distance by ±1, so a
    surviving move always carries a decisive guidance sign."""
    env = make_env()
    for action in range(4):
        env.reset()
        env.snake = [(5, 5), (6, 5)]           # head (5,5); (6,5) is the
        env.food = (5, 9)                      # tail, which vacates in time
        _, reward, done, _, _ = env.step(action)
        assert not done
        assert reward in (pytest.approx(-0.1 + 0.15),   # toward the food
                          pytest.approx(-0.1 - 0.15))   # away from it


# ------------------------------------------------- death-cause differentiation

def test_factor_1_keeps_causes_indistinguishable():
    env = SnakeEnv(grid_size=10, observation_type='vision',
                   reward_shaping='flat', self_death_factor=1.0)
    assert die_at_wall(env, length=1) == die_at_self(env) == -10.0


def test_self_death_costs_more_than_wall_death():
    env = make_env(shaping='flat')
    env.self_death_factor = 1.5
    assert die_at_wall(env, length=1) == -10.0
    assert die_at_self(env) == -15.0


def test_scaled_mode_also_applies_factor_to_self_death():
    env = make_env(shaping='scaled')
    env.self_death_factor = 2.0
    wall = die_at_wall(env, length=1)                 # len 1 → -(1 + 9*0.9)
    assert wall == pytest.approx(-9.1)
    env.reset()
    env.snake = [(5, 5), (5, 6), (5, 7)]              # len 3 → -(1 + 9*0.7)
    env.food = (0, 0)
    _, reward, terminated, _, _ = env.step(3)
    assert terminated
    assert reward == pytest.approx(-7.3 * 2.0)


def test_last_death_cause_tracks_the_actual_cause():
    env = make_env(shaping='scaled')

    env.reset()
    env.snake = [(0, 0)]
    env.food = (5, 5)
    env.step(0)                                       # up → wall
    assert env.last_death_cause == 'wall'

    env.reset()
    env.snake = [(5, 5), (5, 6), (5, 7)]
    env.food = (0, 0)
    env.step(3)                                       # right → neck
    assert env.last_death_cause == 'self'

    env.reset()
    env.snake = [(5, 5)]
    env.food = (0, 0)
    env.steps_without_food = env.max_steps
    env.step(1)
    assert env.last_death_cause == 'timeout'

    env.reset()                                       # survives: cleared
    env.snake = [(5, 5)]
    env.food = (4, 5)
    env.step(0)                                       # eats food
    assert env.last_death_cause is None
    env.snake = [(5, 5)]
    env.food = (0, 0)
    env.step(1)                                       # normal move
    assert env.last_death_cause is None


def test_reset_clears_death_cause():
    env = make_env()
    env.reset()
    env.snake = [(0, 0)]
    env.food = (5, 5)
    env.step(0)
    assert env.last_death_cause == 'wall'
    env.reset()
    assert env.last_death_cause is None


# ------------------------------------------------------------- wiring

def test_trainer_forwards_reward_shaping_to_env(tmp_path):
    from train import Trainer
    t = Trainer(model_path=str(tmp_path / "none.pth"), obs_type='vision',
                n_step=1, grid_size=8, reward_shaping='flat')
    assert t.env.reward_shaping == 'flat'
    assert t.env.self_death_factor == 1.5             # trainer default
    assert t.env._death_reward('wall') == -10.0
    assert t.env._death_reward('self') == -15.0


def test_trainer_defaults_to_scaled(tmp_path):
    from train import Trainer
    t = Trainer(model_path=str(tmp_path / "none.pth"), obs_type='vision',
                n_step=1, grid_size=8)
    assert t.env.reward_shaping == 'scaled'


def test_trainer_counts_death_causes_and_writes_history(tmp_path, monkeypatch):
    """Every episode ends in exactly one cause; the tally and the per-episode
    history lines must reflect what actually happened."""
    import json
    import train as train_mod
    monkeypatch.setattr(train_mod, 'PROJECT_ROOT', tmp_path)
    t = train_mod.Trainer(n_episodes=3, max_steps_per_episode=4, log_interval=100,
                          save_interval=1000, resume=False, obs_type='vision',
                          n_step=1, grid_size=8, preview_interval=0)
    t.train()
    with open(t._run_dir() / 'history.jsonl', encoding='utf-8') as f:
        rows = [json.loads(line) for line in f if line.strip()]
    causes = [row['death'] for row in rows]
    assert len(causes) == 3
    # tiny step-capped episodes can also end without any env-level death
    assert all(c in ('wall', 'self', 'timeout', None) for c in causes)
    real = [c for c in causes if c]
    assert sum(t.death_counts.values()) == len(real)
    if real:
        assert t.death_counts[real[0]] >= 1
