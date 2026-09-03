"""Tests for the per-run model folder export (best + last in ONE folder,
auto-save during training + exit-save at the end).

Covers:
- best-model eligibility (minimum episodes before a best is trustworthy)
- best-model criterion is the trailing-window average, not the global mean
- exit-save does NOT overwrite the best snapshot with the final weights
- auto-save updates the run folder DURING training (before any exit)
- a full short run produces ONE folder with all four files
- a run too short to record a best falls back to the final weights
"""
import json

import pytest
import torch

import train as train_mod
from train import Trainer


@pytest.fixture
def cpu_only(monkeypatch):
    """Keep test trainers on CPU — faster and device-independent."""
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)


@pytest.fixture
def redirected(tmp_path, monkeypatch):
    """Point train.py's PROJECT_ROOT at a temp dir so tests never touch
    the real models/ and output/ folders."""
    monkeypatch.setattr(train_mod, 'PROJECT_ROOT', tmp_path)
    return tmp_path


def make_trainer(n_episodes, max_steps=6, **kw):
    return Trainer(n_episodes=n_episodes, max_steps_per_episode=max_steps,
                   log_interval=100, save_interval=1000, resume=False,
                   obs_type='vision', n_step=1, grid_size=8, preview_interval=0, **kw)


# ---------------------------------------------------------------- eligibility

def test_best_requires_min_episodes(cpu_only):
    t = make_trainer(1)
    t.best_min_episodes, t.best_avg_window = 3, 100

    t.scores = [5, 4]
    assert t._maybe_update_best(2) is False          # only 2 episodes — too noisy

    t.scores.append(6)
    assert t._maybe_update_best(3) is True
    assert t.best_avg == 5.0
    assert t.best_episode == 3


def test_best_requires_actual_improvement(cpu_only):
    t = make_trainer(1)
    t.best_min_episodes, t.best_avg_window = 3, 100

    t.scores = [5, 4, 6]
    assert t._maybe_update_best(3) is True

    t.scores.append(0)
    assert t._maybe_update_best(4) is False          # trailing avg dropped
    assert t.best_episode == 3                       # best unchanged


def test_best_uses_trailing_window_not_global_mean(cpu_only):
    t = make_trainer(1)
    t.best_min_episodes, t.best_avg_window = 2, 2

    t.scores = [10, 0]
    assert t._maybe_update_best(2) is True
    assert t.best_avg == 5.0

    # Global mean here would be 6.67 > 5, but the trailing window [0, 10]
    # averages 5.0 — no improvement, so the best must not move.
    t.scores.append(10)
    assert t._maybe_update_best(3) is False

    t.scores.append(10)
    assert t._maybe_update_best(4) is True           # window [10, 10] -> 10
    assert t.best_avg == 10.0
    assert t.best_episode == 4


# ------------------------------------------------------------ weight handling

def test_exit_save_preserves_best_snapshot(redirected, cpu_only):
    """The exit-save must not overwrite the best snapshot (auto-saved at the
    peak) with the final (post-training) weights."""
    t = make_trainer(1)
    t.best_min_episodes, t.best_avg_window = 2, 100
    t.scores = [3, 9]
    t.rewards = [1.0, 2.0]
    t.epsilon_values = [1.0, 0.99]
    t.last_episode = 2

    # Auto-save at the best moment (weights v1)
    assert t._maybe_update_best(2) is True
    t._save_best_to_run_dir()
    d = t._run_dir()
    v1 = torch.load(d / 'best_model.pth', map_location='cpu', weights_only=False)

    # Simulate continued training: every parameter shifts
    with torch.no_grad():
        for p in t.agent.policy_net.parameters():
            p.add_(1.0)

    t._finalize_run('test')
    v2 = torch.load(d / 'best_model.pth', map_location='cpu', weights_only=False)
    last = torch.load(d / 'last_model.pth', map_location='cpu', weights_only=False)

    for key, tensor in v1['model_dict'].items():
        assert torch.equal(tensor, v2['model_dict'][key]), \
            "exit-save overwrote the best snapshot"
    assert not torch.equal(v1['model_dict'][key], last['model_dict'][key]), \
        "best and last should hold different weights after continued training"

    # The finalized folder carries params + curves for BOTH models
    params = json.loads((d / 'params.json').read_text(encoding='utf-8'))
    assert params['export_status'] == 'final'
    assert params['stop_reason'] == 'test'
    assert params['best']['episode'] == 2
    assert params['best']['fallback_to_last'] is False
    assert params['last']['episode'] == 2
    assert (d / 'training_metrics.png').is_file()


# --------------------------------------------------------------- integration

def test_auto_save_updates_folder_during_run(redirected, cpu_only):
    """Every save_interval episodes the run folder refreshes — BEFORE any
    exit — so a crash still leaves best_model.pth + last_model.pth behind."""
    seen_statuses = []

    def hook(episode, score, reward, eps):
        d = t._run_dir()
        if (d / 'params.json').is_file():
            p = json.loads((d / 'params.json').read_text(encoding='utf-8'))
            seen_statuses.append(p['export_status'])
            assert (d / 'last_model.pth').is_file() or p['export_status'] == 'auto_save'

    t = Trainer(n_episodes=3, max_steps_per_episode=6, log_interval=100,
                save_interval=2, resume=False, obs_type='vision', n_step=1,
                grid_size=8, preview_interval=0, on_episode=hook)
    t.best_min_episodes = 2
    t.train()

    assert 'auto_save' in seen_statuses          # folder existed mid-run
    d = t._run_dir()
    assert (d / 'best_model.pth').is_file()
    assert (d / 'last_model.pth').is_file()
    assert (d / 'training_metrics.png').is_file()
    final = json.loads((d / 'params.json').read_text(encoding='utf-8'))
    assert final['export_status'] == 'final'     # overwritten by exit-save


def test_short_run_exports_single_folder(redirected, cpu_only):
    t = make_trainer(n_episodes=3)
    t.best_min_episodes = 2      # so a best gets recorded within 3 episodes
    t.train()

    models = redirected / 'models'
    run_dirs = list(models.glob('run_*'))
    assert len(run_dirs) == 1                   # ONE folder per training session
    assert run_dirs[0].name == f'run_{t.run_id}'

    d = run_dirs[0]
    assert (d / 'best_model.pth').is_file()
    assert (d / 'last_model.pth').is_file()
    assert (d / 'params.json').is_file()
    assert (d / 'training_metrics.png').is_file()

    p = json.loads((d / 'params.json').read_text(encoding='utf-8'))
    assert p['export_status'] == 'final'
    assert p['run']['end_episode'] == 3
    assert p['run']['episodes_this_run'] == 3
    assert p['run']['resumed_from'] is None
    assert p['best']['episode'] is not None
    assert p['best']['scope'] == 'this training run'
    assert p['last']['episode'] == 3
    assert p['hyperparams']['n_step'] == 1
    assert p['hyperparams']['lr'] == 1e-4
    assert p['hyperparams']['obs_type'] == 'vision'
    assert p['files']['best_model'] == 'best_model.pth'
    assert p['metrics']['max_score'] == max(t.scores)


def test_short_run_without_best_falls_back_to_final_weights(redirected, cpu_only):
    t = make_trainer(n_episodes=1)
    # best_min_episodes stays 20 → a 1-episode run never records a best
    t.train()

    d = redirected / 'models' / f'run_{t.run_id}'
    p = json.loads((d / 'params.json').read_text(encoding='utf-8'))
    assert p['best']['fallback_to_last'] is True
    assert p['best']['episode'] is None
    # The fallback best_model.pth still exists alongside everything else
    assert (d / 'best_model.pth').is_file()
    assert (d / 'last_model.pth').is_file()
    assert (d / 'training_metrics.png').is_file()
