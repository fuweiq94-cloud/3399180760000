"""Tests for train.py checkpoint discovery and resume bookkeeping."""
from agents import D3QNAgent
from train import Trainer, find_latest_checkpoint


def test_find_latest_checkpoint_missing_dir_returns_none(tmp_path):
    assert find_latest_checkpoint(str(tmp_path / "no_such_dir")) is None


def test_find_latest_checkpoint_empty_dir_returns_none(tmp_path):
    assert find_latest_checkpoint(str(tmp_path)) is None


def test_find_latest_checkpoint_picks_numeric_max_not_string_max(tmp_path):
    # 99 vs 100: a string comparison would rank "99" > "100"
    for ep in (30, 99, 100):
        (tmp_path / f"d3qn_snake_episode_{ep}.pth").touch()
    (tmp_path / "unrelated.pth").touch()
    episode, path = find_latest_checkpoint(str(tmp_path))
    assert episode == 100
    assert path.endswith("d3qn_snake_episode_100.pth")


def test_trainer_resume_continues_at_next_episode_no_off_by_one(tmp_path):
    agent = D3QNAgent(input_dim=10, num_actions=4, device="cpu")
    agent.epsilon = 0.5
    path = tmp_path / "d3qn_snake_episode_378.pth"
    agent.save(str(path))

    trainer = Trainer(n_episodes=5000, model_path=str(path))
    assert trainer.start_episode == 379
    assert trainer.agent.epsilon == 0.5


def test_trainer_missing_checkpoint_starts_fresh_without_crash(tmp_path):
    trainer = Trainer(model_path=str(tmp_path / "gone.pth"))
    assert trainer.start_episode == 1
