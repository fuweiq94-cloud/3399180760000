"""Tests for scripts/studio.py (the ZCode-style PyQt6 studio GUI).

Pure helpers (log parsing, architecture inference, model discovery, history
loading) are tested directly; the GUI itself gets a build-and-destroy smoke
test rendered offscreen.
"""
import json
import os

import pytest

# headless Qt for the whole test process (no window flash on the desktop)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

torch = pytest.importorskip("torch")
pytest.importorskip("PyQt6.QtWidgets")

import studio as studio_mod                  # noqa: E402
from studio import (GameCanvas, StudioWindow, TrainingRunner, classify_death,
                    infer_model_arch, list_model_candidates, load_history,
                    load_settings, parse_train_line, resolve_eval_grid,
                    save_settings)

from agents import D3QNAgent


@pytest.fixture
def cpu_only(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)


@pytest.fixture(scope="session")
def qapp():
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


# ------------------------------------------------------------ log parsing

def test_parse_episode_line():
    ev = parse_train_line("[Episode  535/20000] Score:   0 | Reward:  -14.20 | ε: 0.392")
    assert ev == {"kind": "episode", "episode": 535, "total": 20000,
                  "score": 0, "reward": -14.2, "epsilon": 0.392}


def test_parse_episode_line_with_score():
    ev = parse_train_line("[Episode  356/20000] Score:   2 | Reward:    -7.80 | ε: 0.469")
    assert ev["score"] == 2
    assert ev["reward"] == -7.8


def test_parse_resumed_and_continue_lines():
    assert parse_train_line("📂 Resumed from: D:\\zm\\D3QN\\models\\d3qn_snake_episode_490.pth") == \
        {"kind": "resumed", "path": "D:\\zm\\D3QN\\models\\d3qn_snake_episode_490.pth"}
    assert parse_train_line("   Continuing at episode 491, ε=0.4101, agent steps=130090") == \
        {"kind": "continue", "episode": 491}


def test_parse_autosave_line():
    ev = parse_train_line("⏳ Auto-save → D:\\zm\\D3QN\\models\\run_20260903_231928")
    assert ev == {"kind": "autosave", "path": "D:\\zm\\D3QN\\models\\run_20260903_231928"}


def test_parse_ignores_other_lines():
    assert parse_train_line("Model saved to D:/zm/D3QN/models/d3qn_snake_episode_500.pth") is None
    assert parse_train_line("random garbage") is None
    assert parse_train_line("") is None


# ----------------------------------------------------- architecture inference

def test_infer_arch_cnn_grid_30(cpu_only, tmp_path):
    agent = D3QNAgent(obs_type='grid', grid_size=30, buffer_size=100, batch_size=8,
                      device='cpu')
    p = tmp_path / "cnn30.pth"
    agent.save(str(p))
    assert infer_model_arch(str(p)) == ('grid', 30)


def test_infer_arch_cnn_grid_20(cpu_only, tmp_path):
    agent = D3QNAgent(obs_type='grid', grid_size=20, buffer_size=100, batch_size=8,
                      device='cpu')
    p = tmp_path / "cnn20.pth"
    agent.save(str(p))
    assert infer_model_arch(str(p)) == ('grid', 20)


def test_infer_arch_legacy_vision(cpu_only, tmp_path):
    agent = D3QNAgent(input_dim=10, obs_type='vision', buffer_size=100, batch_size=8,
                      device='cpu')
    p = tmp_path / "legacy.pth"
    agent.save(str(p))
    assert infer_model_arch(str(p)) == ('vision', 20)


# --------------------------------------------------------- model discovery

def test_list_model_candidates_covers_all_sources(tmp_path):
    models = tmp_path / "models"
    run = models / "run_20260903_231928"
    run.mkdir(parents=True)
    legacy = models / "legacy_10dim_v1"
    legacy.mkdir()
    best = run / "best_model.pth"
    last = run / "last_model.pth"
    root_ck = models / "d3qn_snake_episode_100.pth"
    old = legacy / "d3qn_snake_episode_5000.pth"
    for i, p in enumerate((old, root_ck, last, best)):   # oldest → newest
        p.write_bytes(b"x")
        stamp = 1700000000 + i
        os.utime(p, (stamp, stamp))

    labels = [label for label, _ in list_model_candidates(models)]
    assert "run_20260903_231928/best" in labels
    assert "run_20260903_231928/last" in labels
    assert "d3qn_snake_episode_100.pth" in labels
    assert "legacy_10dim_v1/d3qn_snake_episode_5000.pth" in labels
    # newest file (run best) ranks first
    assert labels[0] == "run_20260903_231928/best"


def test_list_model_candidates_empty_dir(tmp_path):
    assert list_model_candidates(tmp_path) == []
    assert list_model_candidates(tmp_path / "missing") == []


# ------------------------------------------------------------- history I/O

def test_load_history_reads_jsonl(tmp_path):
    run = tmp_path / "run_x"
    run.mkdir()
    with open(run / "history.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"episode": 1, "score": 0, "reward": -5.0,
                            "epsilon": 1.0}) + "\n")
        f.write(json.dumps({"episode": 2, "score": 3, "reward": 10.0,
                            "epsilon": 0.99}) + "\n")
        f.write("not json\n")
    rows = load_history(run)
    assert len(rows) == 2
    assert rows[1]["score"] == 3


def test_load_history_missing_file_returns_empty(tmp_path):
    assert load_history(tmp_path) == []


def test_history_jsonl_written_during_training(tmp_path, monkeypatch, cpu_only):
    """train.py appends one history line per episode — the GUI chart source."""
    import train as train_mod
    monkeypatch.setattr(train_mod, 'PROJECT_ROOT', tmp_path)
    t = train_mod.Trainer(n_episodes=2, max_steps_per_episode=4, log_interval=100,
                          save_interval=1000, resume=False, obs_type='vision',
                          n_step=1, grid_size=8, preview_interval=0)
    t.train()
    hist = load_history(t._run_dir())
    assert len(hist) == 2
    assert [r["episode"] for r in hist] == [1, 2]
    assert set(hist[0]) == {"episode", "score", "reward", "epsilon", "death"}
    assert hist[0]["death"] in ("wall", "self", "timeout", None)


# ------------------------------------------------------------ death classify

def test_classify_death_wall_and_self():
    from envs import SnakeEnv
    env = SnakeEnv(grid_size=8, observation_type='vision')
    env.reset()
    env.snake = [(0, 0)]
    assert classify_death(env, 0) == 'wall'      # up from the corner
    assert classify_death(env, 2) == 'wall'      # left from the corner
    assert classify_death(env, 1) is None        # down / right stay inside
    assert classify_death(env, 3) is None
    env.snake = [(2, 2), (2, 3), (3, 3)]
    assert classify_death(env, 3) == 'self'      # head runs into its neck


# ------------------------------------------------------------- settings I/O

def test_settings_roundtrip_and_merge(tmp_path, monkeypatch):
    monkeypatch.setattr(studio_mod, 'PROJECT_ROOT', tmp_path)
    assert load_settings() == studio_mod.DEFAULT_SETTINGS      # missing file
    s = {sect: dict(v) for sect, v in studio_mod.DEFAULT_SETTINGS.items()}
    s['train']['grid_size'] = 25
    save_settings(s)
    assert load_settings()['train']['grid_size'] == 25
    # partial file: only eval changed, train falls back to defaults
    (tmp_path / 'studio_settings.json').write_text(
        '{"eval": {"grid_size": 15}, "junk": 1}', encoding='utf-8')
    merged = load_settings()
    assert merged['eval']['grid_size'] == 15
    assert merged['train']['grid_size'] == 30
    assert 'junk' not in merged
    # corrupt file ignored entirely
    (tmp_path / 'studio_settings.json').write_text('not json', encoding='utf-8')
    assert load_settings() == studio_mod.DEFAULT_SETTINGS


def test_resolve_eval_grid():
    # CNN models are locked to their trained board size
    assert resolve_eval_grid('grid', 30, 20) == 30
    assert resolve_eval_grid('grid', 20, 30) == 20
    # vision models use the configured board
    assert resolve_eval_grid('vision', 20, 14) == 14


def test_training_runner_command_includes_train_settings():
    r = TrainingRunner()
    train = {'grid_size': 25, 'episodes': 5000, 'n_step': 2,
             'epsilon_decay': 0.995, 'save_interval': 50,
             'reward_shaping': 'scaled', 'self_death_factor': 1.5}
    cmd = r._command(train=train)
    for flag, value in (('--grid-size', '25'), ('--episodes', '5000'),
                        ('--n-step', '2'), ('--epsilon-decay', '0.995'),
                        ('--save-interval', '50'),
                        ('--reward-shaping', 'scaled'),
                        ('--self-death-factor', '1.5')):
        assert cmd[cmd.index(flag) + 1] == value
    # legacy settings dict without the key still builds a valid command
    assert '--reward-shaping' not in r._command(
        train={'grid_size': 25, 'episodes': 5000, 'n_step': 2,
               'epsilon_decay': 0.995, 'save_interval': 50})


# --------------------------------------------------------------- GUI smoke

def test_settings_view_saves_and_resets(qapp, monkeypatch, tmp_path):
    """Settings page is embedded (no dialog), reflects the loaded file, saves
    edits to studio_settings.json and restores defaults."""
    monkeypatch.setattr(studio_mod, 'PROJECT_ROOT', tmp_path)
    w = StudioWindow()
    assert w.stack.currentIndex() == 0
    w._show_view(2)                       # settings page, no log tab to switch
    qapp.processEvents()
    assert w.spin_set_grid.value() == 30  # defaults (no settings file yet)
    assert w.combo_set_rshape.currentData() == 'scaled'   # new default mode
    w.combo_set_rshape.setCurrentIndex(1)                 # switch to flat
    w.spin_set_grid.setValue(25)
    w.spin_set_eps.setValue(5000)
    w.spin_set_egrid.setValue(14)
    w._save_settings()
    on_disk = load_settings(tmp_path / 'studio_settings.json')
    assert on_disk['train']['grid_size'] == 25
    assert on_disk['train']['episodes'] == 5000
    assert on_disk['train']['reward_shaping'] == 'flat'
    assert on_disk['eval']['grid_size'] == 14
    assert w.settings['train']['grid_size'] == 25
    w._reset_settings()
    assert w.spin_set_grid.value() == 30
    assert w.spin_set_egrid.value() == 20
    assert w.combo_set_rshape.currentData() == 'scaled'
    w.close()


def test_training_runner_command_variants():
    r = TrainingRunner()
    base = r._command()
    assert base[-1].endswith("train.py")
    assert "--resume-from" not in base and "--fresh" not in base
    assert r._command(resume_from="models/run_x/best_model.pth")[-2:] == \
        ["--resume-from", "models/run_x/best_model.pth"]
    assert "--fresh" in r._command(fresh=True)
    assert "--resume-from" not in r._command(fresh=True)


def test_resume_combo_lists_start_modes_and_models(qapp, monkeypatch, tmp_path):
    """起始模型 picker: auto-latest first, fresh start second, then every
    loadable checkpoint — and the selection survives a refresh."""
    models = tmp_path / "models"
    run = models / "run_x"
    run.mkdir(parents=True)
    best = run / "best_model.pth"
    best.write_bytes(b"x")
    monkeypatch.setattr(studio_mod, "PROJECT_ROOT", tmp_path)
    w = StudioWindow()
    assert w.combo_resume.count() == 3
    assert w.combo_resume.itemData(0) is None            # 自动 · 最新检查点
    assert w.combo_resume.itemData(1) == "fresh"         # 从头训练
    assert w.combo_resume.itemData(2) == str(best)       # 续训 · run_x/best
    assert w.combo_resume.currentIndex() == 0            # default: auto
    w.combo_resume.setCurrentIndex(2)
    w._refresh_resume_choices()
    assert w.combo_resume.currentData() == str(best)     # selection kept
    w.close()


def test_history_combo_populated_and_selectable(qapp, monkeypatch, tmp_path):
    """Regression: the 历史图表 combo must list run_* folders at startup
    (it used to stay empty until a user picked an item — impossible), show
    the newest run's curve on launch, redraw on selection, and keep the
    selection across refreshes."""
    models = tmp_path / "models"
    for name, scores in (("run_20260903_120000", (0, 3)),
                         ("run_20260903_130000", (5, 2))):
        run = models / name
        run.mkdir(parents=True)
        with open(run / "history.jsonl", "w", encoding="utf-8") as f:
            for ep, score in enumerate(scores, 1):
                f.write(json.dumps({"episode": ep, "score": score,
                                    "reward": 0.0, "epsilon": 1.0}) + "\n")
    monkeypatch.setattr(studio_mod, 'PROJECT_ROOT', tmp_path)
    w = StudioWindow()
    qapp.processEvents()
    assert w.combo_history.count() == 2
    assert w.combo_history.currentText() == "run_20260903_130000"  # newest
    assert w.chart_scores == [5, 2]                                # auto-drawn
    w.combo_history.setCurrentIndex(1)                             # older run
    w._draw_history()
    assert w.chart_scores == [0, 3]
    w._refresh_history_runs()                                      # keep pick
    assert w.combo_history.currentText() == "run_20260903_120000"
    w.close()


def test_svg_icons_render(qapp):
    """Every nav SVG rasterizes to a non-empty pixmap in each state tint."""
    from PyQt6.QtGui import QIcon
    for name in ("train", "eval", "settings"):
        icon = studio_mod.nav_icon(name)
        assert not icon.isNull()
        for mode in (QIcon.Mode.Normal, QIcon.Mode.Active):
            for state in (QIcon.State.Off, QIcon.State.On):
                img = icon.pixmap(18, 18, mode, state).toImage()
                assert not img.isNull()
                assert any(img.pixelColor(x, y).alpha() > 0     # has ink
                           for y in range(img.height())
                           for x in range(img.width()))


def test_game_canvas_colors_head_body_food(qapp):
    """The eval renderer must keep head, body and food in three clearly
    distinct hues: amber head vs green body vs red food, so the head is
    instantly tellable from the tail even on a 30×30 board."""
    from PyQt6.QtGui import QPixmap
    canvas = GameCanvas()
    canvas.resize(340, 340)
    canvas.set_frame([(5, 5), (5, 6), (5, 7)], (2, 2), 3, 1, 10)
    pix = QPixmap(340, 340)
    canvas.render(pix)
    img = pix.toImage()

    def cell_color(r, c):                      # 340px / 10 cells → 34px cells
        return img.pixelColor(c * 34 + 17, r * 34 + 17)

    head, body, tail = (cell_color(5, 5), cell_color(5, 6), cell_color(5, 7))
    food = cell_color(2, 2)
    assert body == tail                                    # uniform body
    assert head != body
    assert head.red() > 200 and head.green() > 140 and head.blue() < 100  # amber
    assert body.green() > 120 and body.red() < 60          # green
    assert food.red() > 200 and food.green() < 100         # red


def test_game_canvas_grid_lines(qapp):
    """The board draws inner grid lines when cells are readable, and hides
    them on huge boards where 1px lines would blur into mud."""
    from PyQt6.QtGui import QPixmap

    def render(n):
        canvas = GameCanvas()
        canvas.resize(340, 340)
        canvas.set_frame([(5, 5), (5, 6), (5, 7)], (2, 2), 3, 1, n)
        pix = QPixmap(340, 340)
        canvas.render(pix)
        return pix.toImage()

    # n=10 → 34px cells: line between rows 0/1 must differ from the empty
    # cells on both sides of it
    img = render(10)
    line = img.pixelColor(17, 34)
    assert img.pixelColor(17, 17) == img.pixelColor(17, 51)   # both empty
    assert line != img.pixelColor(17, 17)                     # a line is there

    # n=60 → 5px cells: below the readability threshold, no lines drawn
    img = render(60)                                          # ox=oy=20
    assert img.pixelColor(22, 25) == img.pixelColor(22, 22)   # boundary ==
    # ... background                                          # empty cell


def test_studio_window_builds_and_closes(qapp):
    """Build the full studio window offscreen, switch views both ways, feed
    one parsed episode event, paint one game frame, then close. Catches
    layout/traceback crashes without showing a window."""
    pytest.importorskip("matplotlib.backends.backend_qtagg")
    w = StudioWindow()
    qapp.processEvents()
    assert len(w.nav_buttons) == 3
    for btn in w.nav_buttons:                     # sidebar has SVG icons
        assert not btn.icon().isNull()
    w._show_view(1)          # eval view
    qapp.processEvents()
    w._show_view(2)          # settings view (no matching log tab)
    qapp.processEvents()
    w._show_view(0)          # training view
    qapp.processEvents()
    # a parsed episode line flows into chart data + status bar
    w._handle_train_event(parse_train_line(
        "[Episode  535/20000] Score:   2 | Reward:  -7.80 | ε: 0.392"))
    qapp.processEvents()
    assert w.chart_scores[-1] == 2
    # one frame paints on the embedded game canvas without error
    w.game_canvas.set_frame([(4, 4), (4, 5)], (10, 10), 1, 3, 20)
    w.game_canvas.repaint()
    qapp.processEvents()
    # graceful shutdown path with nothing running
    w.close()
    qapp.processEvents()


def test_game_canvas_paints_offscreen(qapp):
    canvas = GameCanvas()
    canvas.resize(400, 400)
    canvas.set_frame([(10, 10), (10, 11), (11, 11)], (5, 5), 2, 7, 20)
    canvas.repaint()         # forces paintEvent synchronously
