"""Tests for train.py checkpoint discovery and resume bookkeeping."""
from agents import D3QNAgent
from train import (Trainer, find_latest_checkpoint,
                   resolve_training_config)


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


# ------------------------------------------- start-mode config (CLI / GUI)

def test_resolve_training_config_auto_without_checkpoint_uses_requested_grid(
        monkeypatch, tmp_path):
    monkeypatch.setattr("train.PROJECT_ROOT", tmp_path)   # empty models dir
    cfg = resolve_training_config(grid_size=25)
    assert cfg == {"resume": True, "model_path": None,
                   "obs_type": "grid", "grid_size": 25}


def test_resolve_training_config_auto_checkpoint_grid_wins(
        monkeypatch, tmp_path):
    """Auto-resume must match the newest checkpoint's architecture, not the
    requested grid size — otherwise load_state_dict fails."""
    (tmp_path / "models").mkdir()
    agent = D3QNAgent(obs_type="grid", grid_size=30, buffer_size=100,
                      batch_size=8, device="cpu")
    agent.save(str(tmp_path / "models" / "d3qn_snake_episode_10.pth"))
    monkeypatch.setattr("train.PROJECT_ROOT", tmp_path)
    cfg = resolve_training_config(grid_size=20)
    assert cfg["resume"] is True
    assert (cfg["obs_type"], cfg["grid_size"]) == ("grid", 30)


def test_resolve_training_config_fresh_uses_requested_grid():
    cfg = resolve_training_config(fresh=True, grid_size=22)
    assert cfg["resume"] is False
    assert cfg["model_path"] is None
    assert cfg["grid_size"] == 22


def test_resolve_training_config_matches_cnn_architecture(tmp_path):
    agent = D3QNAgent(obs_type="grid", grid_size=30, buffer_size=100,
                      batch_size=8, device="cpu")
    p = tmp_path / "cnn30.pth"
    agent.save(str(p))
    cfg = resolve_training_config(resume_from=str(p), grid_size=20)
    assert cfg["resume"] is False
    assert cfg["model_path"] == str(p)
    assert (cfg["obs_type"], cfg["grid_size"]) == ("grid", 30)  # weights win


def test_resolve_training_config_matches_legacy_vision_architecture(tmp_path):
    agent = D3QNAgent(input_dim=10, obs_type="vision", buffer_size=100,
                      batch_size=8, device="cpu")
    p = tmp_path / "legacy.pth"
    agent.save(str(p))
    cfg = resolve_training_config(resume_from=str(p))
    assert (cfg["obs_type"], cfg["grid_size"]) == ("vision", 20)


def test_trainer_accepts_custom_epsilon_decay(tmp_path):
    trainer = Trainer(model_path=str(tmp_path / "none.pth"), obs_type="grid",
                      grid_size=10, epsilon_decay=0.995)
    assert trainer.agent.epsilon_decay == 0.995


def test_trainer_grid_default_epsilon_decay_is_slow(tmp_path):
    """The CNN branch must default to a long exploration tail (0.9997):
    with 0.999 the policy hit the floor at ~ep 3000 before learning to
    eat and never recovered."""
    from train import Trainer
    t = Trainer(model_path=str(tmp_path / "none.pth"), obs_type='grid',
                n_step=1, grid_size=10)
    assert t.agent.epsilon_decay == 0.9997
    assert t.agent.epsilon_end == 0.10
