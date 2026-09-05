"""Tests for d3qn_agent.py: 10-dim input wiring, loss shapes, save/load."""
import warnings

import numpy as np
import pytest
import torch

from agents import D3QNAgent


@pytest.fixture
def agent():
    a = D3QNAgent(input_dim=10, num_actions=4, device="cpu")
    return a


def test_default_input_dim_is_ten():
    a = D3QNAgent(device="cpu")
    first_linear = a.policy_net.fc_shared[0]
    assert first_linear.in_features == 32 * 10


def test_q_values_shape_single_and_batch(agent):
    single = agent.policy_net(torch.zeros(1, 10))
    assert single.shape == (1, 4)
    batch = agent.policy_net(torch.zeros(16, 10))
    assert batch.shape == (16, 4)


def test_select_action_returns_valid_index(agent):
    state = np.zeros(10, dtype=np.float32)
    for _ in range(20):
        assert agent.select_action(state, train=True) in (0, 1, 2, 3)


def test_optimize_model_loss_finite_and_no_shape_broadcast(agent):
    # Regression: rewards/dones must stay [B,1] columns so the TD target
    # cannot broadcast with next_q into a wrong [B,B] matrix
    state = np.zeros(10, dtype=np.float32)
    for i in range(70):
        agent.store_transition(state, i % 4, 1.0, state, 0)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loss = agent.optimize_model()

    assert loss is not None and np.isfinite(loss)
    size_warnings = [w for w in caught
                     if "target size" in str(w.message) or "broadcast" in str(w.message)]
    assert not size_warnings, f"shape broadcast leaked back: {size_warnings}"


def test_save_load_roundtrip_restores_policy_target_and_epsilon(agent, tmp_path):
    # Regression: load() must also sync target_net (Double DQN target values)
    agent.epsilon = 0.42
    path = tmp_path / "d3qn_snake_episode_378.pth"
    agent.save(str(path))

    restored = D3QNAgent(input_dim=10, num_actions=4, device="cpu")
    restored.load(str(path))

    assert restored.epsilon == pytest.approx(0.42)
    for (name, p), (_, q) in zip(agent.policy_net.state_dict().items(),
                                 restored.policy_net.state_dict().items()):
        assert torch.allclose(p, q), f"policy weight mismatch in {name}"
    for (name, p), (_, q) in zip(restored.policy_net.state_dict().items(),
                                 restored.target_net.state_dict().items()):
        assert torch.allclose(p, q), f"target_net not synced on load: {name}"


def test_epsilon_floor_defaults_higher_for_pixel_cnn():
    """Pixel CNNs freeze into wall-avoiding circles if exploration dies
    early (observed: floor 0.05 at ~ep 3000 on 30×30, avg score stuck at
    ~0.1 for the remaining 17k episodes). They keep a 0.10 floor; the
    proven-fast vision model keeps 0.05."""
    assert D3QNAgent(obs_type='grid', grid_size=20, device='cpu').epsilon_end == 0.10
    assert D3QNAgent(input_dim=10, device='cpu').epsilon_end == 0.05
    assert D3QNAgent(obs_type='grid', grid_size=20,
                     epsilon_end=0.07, device='cpu').epsilon_end == 0.07


def test_target_net_soft_updates_and_never_hard_swaps(agent):
    """Regression: the target net must track the policy softly (Polyak,
    tau=0.005) every step instead of a hard copy every 1000 steps — hard
    swaps left the bootstrap unanchored between jumps and the policy
    forgot food-seeking once exploration thinned (0.92 -> 0.05 on 20x20)."""
    import copy
    before = copy.deepcopy(list(agent.target_net.parameters()))
    # nudge the policy so a soft update has something to track
    with torch.no_grad():
        for p in agent.policy_net.parameters():
            p.add_(0.1)
    agent.steps = 999          # old hard-swap boundary must NOT trigger a copy
    agent.step()
    tau = agent.tau
    for old, new, pol in zip(before, agent.target_net.parameters(),
                             agent.policy_net.parameters()):
        assert not torch.equal(old, new)            # moved toward policy
        expected = old * (1 - tau) + pol * tau
        assert torch.allclose(new, expected, atol=1e-6)
