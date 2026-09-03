"""Tests for the advanced upgrade: full-board CNN observation + n-step returns."""
import numpy as np
import pytest
import torch

from envs import SnakeEnv
from agents import D3QNAgent
from models import D3QNCNN


# ---------- grid observation ----------

def test_grid_observation_shape_and_channel_semantics():
    env = SnakeEnv(grid_size=20, observation_type='grid')
    env.snake = [(10, 10), (10, 11), (11, 11)]
    env.food = (0, 5)
    obs = env._get_observation()
    assert obs.shape == (3, 20, 20)
    assert obs[2, 10, 10] == 1.0                            # head channel
    assert obs[0, 10, 10] == 0.0                            # head excluded from body
    assert obs[0, 10, 11] == 1.0 and obs[0, 11, 11] == 1.0  # body cells
    assert obs[1, 0, 5] == 1.0                              # food channel
    assert obs.sum() == 4.0                                 # 2 body + 1 food + 1 head


def test_grid_observation_flows_through_reset_and_step():
    env = SnakeEnv(grid_size=20, observation_type='grid')
    obs, _ = env.reset()
    assert obs.shape == (3, 20, 20)
    obs2, _, _, _, _ = env.step(0)
    assert obs2.shape == (3, 20, 20)


# ---------- CNN network ----------

@pytest.mark.parametrize("grid", [20, 30])
def test_cnn_forward_shapes_batch_and_single(grid):
    net = D3QNCNN(num_actions=4, grid_size=grid)
    assert net(torch.zeros(8, 3, grid, grid)).shape == (8, 4)
    assert net(torch.zeros(3, grid, grid)).shape == (1, 4)


# ---------- agent with grid input ----------

def test_grid_agent_select_action_and_optimize():
    agent = D3QNAgent(obs_type='grid', n_step=3, grid_size=20,
                      buffer_size=1000, batch_size=8, device='cpu')
    state = np.zeros((3, 20, 20), dtype=np.float32)
    assert agent.select_action(state) in (0, 1, 2, 3)
    for i in range(20):
        agent.store_transition(state, i % 4, 1.0, state, 0)
    agent.end_episode()  # flush remaining partial n-step transitions
    assert len(agent.memory) >= 8
    loss = agent.optimize_model()
    assert loss is not None and np.isfinite(loss)


# ---------- n-step return math (hand-computed) ----------

def test_nstep_composition_emits_on_nth_plus_one_push():
    # n=3, gamma=0.9, rewards 1,2,3 -> R = 1 + 0.9*2 + 0.81*3 = 5.23
    agent = D3QNAgent(n_step=3, buffer_size=1000, batch_size=8, device='cpu')
    agent.gamma = 0.9
    states = [np.full(10, i, dtype=np.float32) for i in range(5)]
    for i in range(3):
        agent.store_transition(states[i], 0, float(i + 1), states[i + 1], 0)
    assert len(agent.memory) == 0  # nothing until the window slides past n

    agent.store_transition(states[3], 0, 4.0, states[4], 0)  # 4th push emits T0
    assert len(agent.memory) == 1
    s, a, R, s_next, done = agent.memory[0]
    assert s[0] == 0 and a == 0
    assert R == pytest.approx(1 + 0.9 * 2 + 0.81 * 3)
    assert s_next[0] == states[3][0]  # bootstrap from s_{t+3}
    assert done == 0


def test_nstep_flush_emits_remaining_without_duplicates_or_loss():
    agent = D3QNAgent(n_step=3, buffer_size=1000, batch_size=8, device='cpu')
    agent.gamma = 0.9
    states = [np.full(10, i, dtype=np.float32) for i in range(7)]
    for i in range(5):  # pushes 4 and 5 emit T0 and T1
        agent.store_transition(states[i], i % 4, 1.0, states[i + 1], 0)
    assert len(agent.memory) == 2

    agent.end_episode()  # must emit exactly T2, T3, T4 — no duplicates
    assert len(agent.memory) == 5
    assert len(agent._nstep_buf) == 0

    starts = [t.state[0] for t in agent.memory]
    assert starts == [0, 1, 2, 3, 4]
    s, a, R, s_next, done = agent.memory[-1]  # T4: k=1 partial, R = r4
    assert s[0] == 4
    assert R == pytest.approx(1.0)
    assert s_next[0] == states[5][0]


def test_nstep_cut_by_terminal_state_keeps_done_flag():
    agent = D3QNAgent(n_step=3, buffer_size=1000, batch_size=8, device='cpu')
    states = [np.full(10, i, dtype=np.float32) for i in range(4)]
    agent.store_transition(states[0], 0, 1.0, states[1], 0)
    agent.store_transition(states[1], 0, -10.0, states[2], 0)
    agent.store_transition(states[2], 0, -10.0, states[3], 1)  # fatal step
    agent.store_transition(states[3], 0, 0.0, states[3], 1)    # pushes T0 out
    s, a, R, s_next, done = agent.memory[0]
    assert done == 1
    assert R == pytest.approx(1.0 + 0.99 * -10.0 + 0.99 ** 2 * -10.0)
