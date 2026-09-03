"""
D3QN Snake Studio — ZCode 风格的一体化训练 / 验证界面（PyQt6，单窗口，无弹窗）

布局（仿 IDE）:
    ┌────────────────────────────────────────────┐
    │ 顶栏: 标题 + 状态胶囊                         │
    │ ┌──────┬─────────────────────────────────┐ │
    │ │ 侧边 │  内容区（训练视图 / 验证视图切换）   │ │
    │ │ 导航 │        ↕ 可拖动分隔条              │ │
    │ ├──────┴─────────────────────────────────┤ │
    │ │ 底部日志面板（训练日志 / 验证日志 标签页）  │ │
    │ ├─────────────────────────────────────────┤ │
    │ │ 状态栏                                   │ │
    │ └─────────────────────────────────────────┘ │
    └────────────────────────────────────────────┘

训练视图: 子进程运行 scripts/train.py（stdout 实时流入日志面板），
          内嵌 matplotlib 实时曲线（得分/奖励/ε），可回看历史 run 的图表。
验证视图: 自动识别模型架构（CNN 30×30 / 旧 10 维），贪心策略连续对局，
          游戏画面直接画在窗口内的画布上（不弹出独立窗口）。

Run:
    python scripts/studio.py        （或双击 启动界面.bat）
"""

import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch

# Add src directory to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from envs import SnakeEnv                     # noqa: E402
from agents import D3QNAgent                  # noqa: E402
from ckpt_utils import infer_model_arch       # noqa: E402

# matplotlib embedded canvas (Figure API — no pyplot, no global backend fight)
import matplotlib                              # noqa: E402
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei',
                                          'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
from matplotlib.figure import Figure          # noqa: E402
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas  # noqa: E402

from PyQt6.QtCore import QSize, Qt, QTimer       # noqa: E402
from PyQt6.QtGui import QColor, QFont, QIcon, QPainter, QPixmap  # noqa: E402
from PyQt6.QtSvg import QSvgRenderer             # noqa: E402
from PyQt6.QtWidgets import (QApplication, QButtonGroup, QComboBox, QDoubleSpinBox,
                             QFrame, QGridLayout, QHBoxLayout, QLabel, QMainWindow,
                             QPlainTextEdit, QPushButton, QSizePolicy, QSlider,
                             QSpinBox, QSplitter, QStackedWidget, QTabWidget,
                             QVBoxLayout, QWidget)         # noqa: E402

# ---------------------------------------------------------------- palette
BG = "#0f172a"        # window background (dark slate)
PANEL = "#1e293b"     # card / sidebar background
PANEL2 = "#334155"    # hover / active
FG = "#e2e8f0"        # main text
MUTED = "#94a3b8"     # secondary text
ACCENT = "#22c55e"    # start / running
DANGER = "#ef4444"    # stop
WARN = "#f59e0b"      # stopping
BLUE = "#3b82f6"      # finished / links

CREATE_NO_WINDOW = 0x08000000 if os.name == 'nt' else 0

TRAIN_STATUS = {
    "idle": ("● 空闲", MUTED),
    "running": ("● 训练中", ACCENT),
    "stopping": ("● 停止中…", WARN),
    "finished": ("● 已完成", BLUE),
}
EVAL_STATUS = {
    "idle": ("● 空闲", MUTED),
    "running": ("● 验证中", ACCENT),
    "stopping": ("● 停止中…", WARN),
    "finished": ("● 已结束", BLUE),
}

QSS = f"""
QMainWindow, QWidget {{ background: {BG}; color: {FG};
    font-family: "Microsoft YaHei UI"; font-size: 10pt; }}
QFrame#topbar {{ background: {BG}; border: none; }}
QFrame#sidebar {{ background: {PANEL}; border: none; }}
QFrame#card {{ background: {PANEL}; border-radius: 6px; }}
QLabel#muted {{ color: {MUTED}; font-size: 9pt; }}
QLabel#title {{ font-size: 14pt; font-weight: bold; }}
QLabel#pill {{ font-size: 11pt; font-weight: bold; }}
QLabel#cardValue {{ font-family: Consolas; font-size: 16pt; font-weight: bold; }}
QPushButton {{ background: {PANEL}; color: {FG}; border: 1px solid {PANEL2};
    padding: 6px 14px; border-radius: 4px; }}
QPushButton:hover {{ background: {PANEL2}; }}
QPushButton:disabled {{ color: #64748b; background: {PANEL}; }}
QPushButton#startBtn {{ background: {ACCENT}; color: #052e16;
    font-weight: bold; border: none; }}
QPushButton#startBtn:disabled {{ background: {PANEL}; color: #64748b; }}
QPushButton#stopBtn {{ background: {DANGER}; color: #450a0a;
    font-weight: bold; border: none; }}
QPushButton#stopBtn:disabled {{ background: {PANEL}; color: #64748b; }}
QPushButton#nav {{ background: transparent; border: none; text-align: left;
    padding: 9px 12px; border-radius: 6px; font-size: 11pt; color: {MUTED}; }}
QPushButton#nav:hover {{ background: {PANEL2}; }}
QPushButton#nav:checked {{ background: {PANEL2}; color: {FG}; }}
QLabel#sidehead {{ color: #64748b; font-size: 8pt; padding: 0 10px; }}
QLabel#sideval {{ font-family: Consolas; }}
QLabel#sideBest {{ color: {ACCENT}; font-family: Consolas;
    font-size: 16pt; font-weight: bold; }}
QPlainTextEdit {{ background: #020617; color: {MUTED};
    font-family: Consolas; font-size: 9pt; border: none; }}
QTabWidget::pane {{ border: 1px solid {PANEL2}; border-radius: 4px; }}
QTabBar::tab {{ background: {PANEL}; color: {MUTED}; padding: 5px 16px; }}
QTabBar::tab:selected {{ background: {PANEL2}; color: {FG}; }}
QComboBox, QSpinBox, QDoubleSpinBox {{ background: {PANEL}; color: {FG};
    border: 1px solid {PANEL2}; padding: 4px 8px; border-radius: 4px; }}
QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{ color: #64748b; }}
QComboBox QAbstractItemView {{ background: {PANEL}; color: {FG};
    selection-background-color: {PANEL2}; }}
QLabel#secthead {{ font-size: 11pt; font-weight: bold; }}
QSlider::groove:horizontal {{ height: 4px; background: {PANEL2};
    border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {ACCENT}; width: 12px;
    margin: -5px 0; border-radius: 6px; }}
QStatusBar {{ background: {PANEL}; color: {MUTED}; }}
QStatusBar::item {{ border: none; }}
QSplitter::handle {{ background: {BG}; }}
QSplitter::handle:horizontal {{ width: 3px; }}
QSplitter::handle:vertical {{ height: 3px; }}
"""


# ------------------------------------------------------------ sidebar icons
# Inline SVG (lucide-style, 24×24 stroke icons). "{color}" placeholders are
# substituted at raster time so one template yields muted/hover/checked tint
# without external asset files.
NAV_ICONS = {
    # neural network: one input node feeding two output nodes
    "train": (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none"'
        ' stroke="{color}" stroke-width="2" stroke-linecap="round"'
        ' stroke-linejoin="round">'
        '<circle cx="5" cy="12" r="2.2"/>'
        '<circle cx="19" cy="5.5" r="2.2"/>'
        '<circle cx="19" cy="18.5" r="2.2"/>'
        '<path d="M7 10.8 17 6.7"/>'
        '<path d="M7 13.2 17 17.3"/>'
        '</svg>'),
    # bullseye: greedy evaluation against a target
    "eval": (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none"'
        ' stroke="{color}" stroke-width="2">'
        '<circle cx="12" cy="12" r="8.5"/>'
        '<circle cx="12" cy="12" r="4.8"/>'
        '<circle cx="12" cy="12" r="1.4" fill="{color}" stroke="none"/>'
        '</svg>'),
    # sliders: training / eval tunables
    "settings": (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none"'
        ' stroke="{color}" stroke-width="2" stroke-linecap="round">'
        '<line x1="4" y1="6" x2="20" y2="6"/>'
        '<line x1="4" y1="12" x2="20" y2="12"/>'
        '<line x1="4" y1="18" x2="20" y2="18"/>'
        '<line x1="14" y1="3.5" x2="14" y2="8.5"/>'
        '<line x1="8" y1="9.5" x2="8" y2="14.5"/>'
        '<line x1="17" y1="15.5" x2="17" y2="20.5"/>'
        '</svg>'),
}


def svg_pixmap(svg, color, size=36):
    """Colorize an SVG template and rasterize it (2× for crisp HiDPI)."""
    renderer = QSvgRenderer(svg.replace("{color}", color).encode("utf-8"))
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    pm.setDevicePixelRatio(2.0)
    painter = QPainter(pm)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    renderer.render(painter)
    painter.end()
    return pm


def nav_icon(name):
    """State-aware sidebar icon: muted idle, bright on hover, green checked."""
    icon = QIcon()
    for mode in (QIcon.Mode.Normal, QIcon.Mode.Active, QIcon.Mode.Selected):
        icon.addPixmap(svg_pixmap(NAV_ICONS[name], MUTED), mode, QIcon.State.Off)
        icon.addPixmap(svg_pixmap(NAV_ICONS[name], ACCENT), mode, QIcon.State.On)
    return icon


# ------------------------------------------------------------- pure helpers
EPISODE_RE = re.compile(
    r"\[Episode\s+(\d+)/(\d+)\]\s+Score:\s*(-?\d+)\s+\|\s+Reward:\s*(-?[\d.]+)\s+\|\s+ε:\s*([\d.]+)")
RESUMED_RE = re.compile(r"Resumed from:\s*(.+)")
CONTINUE_RE = re.compile(r"Continuing at episode\s+(\d+)")
AUTOSAVE_RE = re.compile(r"Auto-save →\s*(.+)")


def parse_train_line(line):
    """Parse one stdout line of scripts/train.py into a small event dict
    (or None). Pure function — unit-tested."""
    m = EPISODE_RE.search(line)
    if m:
        return {"kind": "episode",
                "episode": int(m.group(1)), "total": int(m.group(2)),
                "score": int(m.group(3)), "reward": float(m.group(4)),
                "epsilon": float(m.group(5))}
    m = RESUMED_RE.search(line)
    if m:
        return {"kind": "resumed", "path": m.group(1).strip()}
    m = CONTINUE_RE.search(line)
    if m:
        return {"kind": "continue", "episode": int(m.group(1))}
    m = AUTOSAVE_RE.search(line)
    if m:
        return {"kind": "autosave", "path": m.group(1).strip()}
    return None


def list_model_candidates(models_dir=None):
    """All loadable checkpoints, newest first. Covers run-folder exports
    (best/last), root resume checkpoints and the legacy archive."""
    models_dir = Path(models_dir or PROJECT_ROOT / 'models')
    found = []  # (mtime, label, path)

    def add(path, label):
        try:
            found.append((path.stat().st_mtime, label, str(path)))
        except OSError:
            pass

    if models_dir.is_dir():
        for run_dir in sorted(models_dir.glob('run_*')):
            for kind in ('best_model.pth', 'last_model.pth'):
                p = run_dir / kind
                if p.is_file():
                    add(p, f"{run_dir.name}/{kind.replace('_model.pth', '')}")
            for p in run_dir.glob('d3qn_snake_episode_*.pth'):
                add(p, f"{run_dir.name}/{p.name}")
        for p in models_dir.glob('d3qn_snake_episode_*.pth'):
            add(p, p.name)
        for p in (models_dir / 'legacy_10dim_v1').glob('d3qn_snake_episode_*.pth'):
            add(p, f"legacy_10dim_v1/{p.name}")

    found.sort(key=lambda t: t[0], reverse=True)
    return [(label, path) for _, label, path in found]


def classify_death(env, action):
    """Peek at what the chosen action would hit: 'wall', 'self' or None."""
    head = env.snake[0]
    d = env.directions[action]
    new_head = (head[0] + d[0], head[1] + d[1])
    if not (0 <= new_head[0] < env.grid_size and 0 <= new_head[1] < env.grid_size):
        return 'wall'
    if new_head in env.snake[:-1]:
        return 'self'
    return None


def load_history(run_dir):
    """Read a run folder's history.jsonl → list of per-episode dicts."""
    out = []
    path = Path(run_dir) / 'history.jsonl'
    if path.is_file():
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return out


# ------------------------------------------------------------- settings I/O
SETTINGS_FILE = 'studio_settings.json'

DEFAULT_SETTINGS = {
    'train': {
        'grid_size': 30,        # board edge in cells (CNN full-board obs)
        'episodes': 20000,      # total target episodes
        'n_step': 3,            # n-step returns
        'epsilon_decay': 0.999, # per-episode exploration decay
        'save_interval': 100,   # checkpoint + auto-save every N episodes
        'reward_shaping': 'scaled',  # 'scaled' = size-dependent rewards,
                                     # 'flat' = fixed +10 / -10
        'self_death_factor': 1.5,    # self-collision penalty vs wall death
    },
    'eval': {
        'grid_size': 20,        # board for legacy vision models (CNN models
                                # are locked to the size they were trained on)
    },
}


def load_settings(path=None):
    """Defaults merged with studio_settings.json — unknown keys dropped,
    missing/corrupt file falls back to defaults. Pure, unit-tested."""
    path = Path(path or PROJECT_ROOT / SETTINGS_FILE)
    out = {section: dict(values) for section, values in DEFAULT_SETTINGS.items()}
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return out
    if not isinstance(data, dict):
        return out
    for section, values in data.items():
        if isinstance(values, dict) and section in out:
            out[section].update({k: v for k, v in values.items()
                                 if k in out[section]})
    return out


def save_settings(settings, path=None):
    path = Path(path or PROJECT_ROOT / SETTINGS_FILE)
    path.write_text(json.dumps(settings, indent=2, ensure_ascii=False),
                    encoding='utf-8')
    return path


def resolve_eval_grid(obs_type, inferred_grid, configured_grid):
    """CNN models only accept the board they were trained on; the configured
    grid applies to vision models whose 10-dim input is board-independent."""
    return inferred_grid if obs_type == 'grid' else configured_grid


# ------------------------------------------------------------ train runner
class TrainingRunner:
    """Owns the train.py subprocess. No Qt here — reusable and testable."""

    def __init__(self, on_exit=None):
        self.proc = None
        self.lines = queue.Queue()
        self.on_exit = on_exit          # callback() fired from reader thread
        self.stop_file = PROJECT_ROOT / 'STOP'

    def _command(self, resume_from=None, fresh=False, train=None):
        """train.py argv for the requested start mode + train settings."""
        cmd = [sys.executable, str(PROJECT_ROOT / 'scripts' / 'train.py')]
        if resume_from:
            cmd += ['--resume-from', resume_from]
        if fresh:
            cmd += ['--fresh']
        if train:
            cmd += ['--grid-size', str(train['grid_size']),
                    '--episodes', str(train['episodes']),
                    '--n-step', str(train['n_step']),
                    '--epsilon-decay', str(train['epsilon_decay']),
                    '--save-interval', str(train['save_interval'])]
            if train.get('reward_shaping'):
                cmd += ['--reward-shaping', str(train['reward_shaping'])]
            if train.get('self_death_factor') is not None:
                cmd += ['--self-death-factor', str(train['self_death_factor'])]
        return cmd

    def start(self, resume_from=None, fresh=False, train=None):
        if self.proc and self.proc.poll() is None:
            return False
        env = dict(os.environ,
                   PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8',
                   MPLBACKEND='Agg')
        kwargs = dict(cwd=str(PROJECT_ROOT), env=env,
                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                      text=True, encoding='utf-8', errors='replace')
        if CREATE_NO_WINDOW:
            kwargs['creationflags'] = CREATE_NO_WINDOW
        self.proc = subprocess.Popen(self._command(resume_from, fresh, train),
                                     **kwargs)
        threading.Thread(target=self._read, daemon=True, name='train-reader').start()
        return True

    def _read(self):
        for line in self.proc.stdout:
            self.lines.put(line.rstrip('\n'))
        self.proc.wait()
        self.lines.put(None)            # EOF sentinel
        if self.on_exit:
            self.on_exit()

    @property
    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def request_stop(self):
        """Graceful: STOP file → train.py saves checkpoints and exits."""
        if self.running:
            try:
                self.stop_file.write_text('stop', encoding='utf-8')
            except OSError:
                pass

    def force_stop(self):
        if self.running:
            self.proc.terminate()


# ------------------------------------------------------------- eval worker
class EvalWorker:
    """Greedy evaluation loop in a thread. Sends ('frame'|'stats'|'log'|'done')
    messages through a queue; stoppable between steps. The game is rendered
    by the GUI on an embedded canvas — env.render() is never called, so no
    pygame window ever opens."""

    def __init__(self, model_path, fps_getter, msgs, stop_event,
                 eval_grid_size=DEFAULT_SETTINGS['eval']['grid_size']):
        self.model_path = model_path
        self.fps_getter = fps_getter
        self.msgs = msgs
        self.stop_event = stop_event
        self.eval_grid_size = eval_grid_size

    def run(self):
        try:
            obs_type, inferred_grid = infer_model_arch(self.model_path)
            grid_size = resolve_eval_grid(obs_type, inferred_grid,
                                          self.eval_grid_size)
            agent = D3QNAgent(obs_type=obs_type, grid_size=grid_size,
                              buffer_size=1000, batch_size=32)
            agent.load(self.model_path)
            agent.policy_net.eval()   # deterministic policy (dropout off)
        except Exception as exc:
            self.msgs.put(("log", f"❌ 模型加载失败：{exc!r}"))
            self.msgs.put(("done", 0))
            return
        arch = ('CNN 全盘' if obs_type == 'grid' else '旧版10维')
        extra = ('（按模型锁定）' if obs_type == 'grid'
                 else '（来自设置）' if grid_size != inferred_grid else '')
        self.msgs.put(("log", f"已加载：{self.model_path}"
                              f"（{arch} {grid_size}×{grid_size}{extra}，贪心模式）"))

        env = SnakeEnv(grid_size=grid_size, observation_type=obs_type)
        episodes, best, scores = 0, 0, []
        deaths = {'wall': 0, 'self': 0, 'timeout': 0}

        while not self.stop_event.is_set():
            state, _ = env.reset()
            done, r, cause = False, 0.0, None
            while not done:
                if self.stop_event.is_set():
                    break
                action = agent.select_action(state, train=False)
                cause = classify_death(env, action)
                state, r, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                self.msgs.put(("frame", list(env.snake), env.food,
                               env.score, episodes + 1, grid_size))
                time.sleep(1.0 / max(1.0, self.fps_getter()))

            if self.stop_event.is_set():
                break

            episodes += 1
            scores.append(env.score)
            best = max(best, env.score)
            if r == -1.0:
                deaths['timeout'] += 1
            elif cause:
                deaths[cause] += 1
            self.msgs.put(("stats", episodes, env.score, best,
                           sum(scores) / len(scores), dict(deaths)))

        self.msgs.put(("done", episodes))
        self.msgs.put(("log", f"验证结束：共 {episodes} 局，"
                              f"平均 {sum(scores) / max(1, len(scores)):.2f} 分，"
                              f"最高 {best} 分。"))


# ---------------------------------------------------------------- widgets
class GameCanvas(QWidget):
    """Embedded snake renderer — draws env state with QPainter. No pygame,
    no separate window."""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(340, 340)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)
        self.frame = None      # (snake, food, score, episode, grid_size)

    def set_frame(self, snake, food, score, episode, grid_size):
        self.frame = (snake, (int(food[0]), int(food[1])), score, episode,
                      grid_size)
        self.update()

    def paintEvent(self, event):
        if not self.frame:
            return
        snake, food, score, episode, n = self.frame
        painter = QPainter(self)
        w, h = self.width(), self.height()
        cell = max(3, min(w, h) // n)
        ox, oy = (w - cell * n) // 2, (h - cell * n) // 2

        painter.setPen(QColor(PANEL2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(ox, oy, cell * n, cell * n)
        if cell >= 6:                                # inner grid lines, only
            for i in range(1, n):                    # when cells are big
                x = ox + i * cell                    # enough to read them
                painter.drawLine(x, oy, x, oy + cell * n)
                y = oy + i * cell
                painter.drawLine(ox, y, ox + cell * n, y)

        for r, c in snake[1:]:                       # body — green
            painter.fillRect(ox + c * cell + 1, oy + r * cell + 1,
                             cell - 2, cell - 2, QColor("#16a34a"))
        if snake:                                    # head — amber, distinct
            r, c = snake[0]                          # hue from body & food
            painter.fillRect(ox + c * cell + 1, oy + r * cell + 1,
                             cell - 2, cell - 2, QColor("#fbbf24"))
        painter.fillRect(ox + food[1] * cell + 1, oy + food[0] * cell + 1,   # food
                         cell - 2, cell - 2, QColor(DANGER))

        painter.setPen(QColor(MUTED))
        painter.drawText(ox, max(painter.fontMetrics().height(), oy - 8),
                         f"第 {episode} 局    得分 {score}    蛇长 {len(snake)}")


# -------------------------------------------------------------------- app
class StudioWindow(QMainWindow):
    """ZCode-style single-window studio: sidebar nav + content views +
    docked log panel + status bar. Nothing ever opens a second window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("D3QN Snake Studio")
        self.resize(1180, 760)
        self.setMinimumSize(980, 640)

        # training state
        self.runner = TrainingRunner()
        self.chart_episodes, self.chart_scores = [], []
        self.chart_rewards, self.chart_epsilons = [], []
        self._last_draw = 0.0
        self._train_started_at = None
        self._train_status = "idle"

        # eval state
        self.eval_msgs = queue.Queue()
        self.eval_stop = None
        self.eval_thread = None
        self.fps_value = 15.0
        self._eval_gen = 0
        self._eval_episodes_done = 0

        # closing state
        self._closing = False
        self._close_deadline = 0.0

        # persisted settings (settings view edits these)
        self.settings = load_settings()

        self._build_ui()

        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._poll)
        self._poll_timer.start(120)

        self._close_timer = QTimer(self)
        self._close_timer.timeout.connect(self._finish_close)

    # ---------- UI scaffolding ----------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # top bar
        topbar = QFrame(objectName="topbar")
        top = QHBoxLayout(topbar)
        top.setContentsMargins(14, 9, 14, 9)
        top.addWidget(QLabel("🐍 D3QN Snake Studio", objectName="title"))
        top.addStretch(1)
        self.lbl_train_state = QLabel(TRAIN_STATUS["idle"][0], objectName="pill")
        self._style_pill("idle")
        top.addWidget(self.lbl_train_state)
        root.addWidget(topbar)

        # main body: [sidebar | views] / logs, both resizable like an IDE
        body = QSplitter(Qt.Orientation.Vertical)
        main = QSplitter(Qt.Orientation.Horizontal)

        main.addWidget(self._build_sidebar())
        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_training_view())   # index 0
        self.stack.addWidget(self._build_eval_view())       # index 1
        self.stack.addWidget(self._build_settings_view())   # index 2
        main.addWidget(self.stack)
        main.setStretchFactor(1, 1)
        main.setSizes([150, 1000])
        body.addWidget(main)

        self.log_notebook = QTabWidget()
        self.log_train = self._make_log_tab("训练日志")
        self.log_eval = self._make_log_tab("验证日志")
        body.addWidget(self.log_notebook)
        body.setStretchFactor(0, 3)
        body.setSizes([520, 200])
        root.addWidget(body, 1)

        # status bar
        self.lbl_status_left = QLabel("就绪")
        self.lbl_status_right = QLabel("")
        status = self.statusBar()
        status.addWidget(self.lbl_status_left)
        status.addPermanentWidget(self.lbl_status_right)

        self._show_view(0)
        if self.combo_history.count():      # show the newest run's curve
            self.combo_history.setCurrentIndex(0)
            self._draw_history()

    def _build_sidebar(self):
        sidebar = QFrame(objectName="sidebar")
        sidebar.setMinimumWidth(170)
        sidebar.setMaximumWidth(230)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(10, 14, 10, 12)
        layout.setSpacing(2)

        layout.addWidget(QLabel("导航", objectName="sidehead"))
        nav_group = QButtonGroup(self)
        nav_group.setExclusive(True)
        self.nav_buttons = []
        for i, (text, icon) in enumerate((("训练模型", "train"),
                                          ("验证模型", "eval"),
                                          ("设置", "settings"))):
            btn = QPushButton(text, objectName="nav")
            btn.setIcon(nav_icon(icon))
            btn.setIconSize(QSize(18, 18))
            btn.setCheckable(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            nav_group.addButton(btn, i)
            btn.clicked.connect(lambda _c, idx=i: self._show_view(idx))
            layout.addWidget(btn)
            self.nav_buttons.append(btn)

        layout.addSpacing(10)
        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background:{PANEL2};")
        layout.addWidget(sep)
        layout.addSpacing(10)

        layout.addWidget(QLabel("概览", objectName="sidehead"))

        info = QFrame(objectName="card")
        grid = QGridLayout(info)
        grid.setContentsMargins(12, 10, 12, 10)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        n_models = len(list_model_candidates())
        t = self.settings['train']
        for r, (key, val) in enumerate((
                ("可用模型", str(n_models)),
                ("网络", f"CNN {t['grid_size']}×{t['grid_size']}"),
                ("n-step", str(t['n_step'])))):
            grid.addWidget(QLabel(key, objectName="muted"), r, 0)
            grid.addWidget(QLabel(val, objectName="sideval"), r, 1,
                           Qt.AlignmentFlag.AlignRight)
        layout.addWidget(info)

        best = QFrame(objectName="card")
        best_layout = QVBoxLayout(best)
        best_layout.setContentsMargins(12, 10, 12, 10)
        best_layout.addWidget(QLabel("最近100轮均分", objectName="muted"))
        self.lbl_side_best = QLabel("—", objectName="sideBest")
        self.lbl_side_best.setAlignment(Qt.AlignmentFlag.AlignCenter)
        best_layout.addWidget(self.lbl_side_best)
        layout.addWidget(best)

        layout.addStretch(1)
        return sidebar

    def _make_log_tab(self, title):
        widget = QPlainTextEdit()
        widget.setReadOnly(True)
        widget.setMaximumBlockCount(3000)
        self.log_notebook.addTab(widget, title)
        return widget

    def _show_view(self, index):
        for i, btn in enumerate(self.nav_buttons):
            btn.setChecked(i == index)
        self.stack.setCurrentIndex(index)
        if index < self.log_notebook.count():   # settings view has no log tab
            self.log_notebook.setCurrentIndex(index)
        if index == 0:
            self._refresh_history_runs()
            self._refresh_resume_choices()

    # ---------- training view ----------
    def _build_training_view(self):
        view = QWidget()
        layout = QVBoxLayout(view)
        layout.setContentsMargins(12, 10, 12, 8)
        layout.setSpacing(6)

        bar = QHBoxLayout()
        self.btn_train_start = QPushButton("▶ 开始训练", objectName="startBtn")
        self.btn_train_start.clicked.connect(self._start_training)
        bar.addWidget(self.btn_train_start)
        self.btn_train_stop = QPushButton("■ 停止训练", objectName="stopBtn")
        self.btn_train_stop.clicked.connect(self._stop_training)
        self.btn_train_stop.setEnabled(False)
        bar.addWidget(self.btn_train_stop)
        bar.addStretch(1)
        bar.addWidget(QLabel("历史图表", objectName="muted"))
        self.combo_history = QComboBox()
        self.combo_history.setMinimumWidth(220)
        self.combo_history.activated.connect(self._draw_history)
        bar.addWidget(self.combo_history)
        layout.addLayout(bar)

        start_bar = QHBoxLayout()
        start_bar.addWidget(QLabel("起始模型", objectName="muted"))
        self.combo_resume = QComboBox()
        self.combo_resume.setToolTip(
            "选择训练起点：自动 = 最新检查点续训；从头训练 = 随机初始化；"
            "或指定任意已保存模型（自动匹配网络架构）")
        self._refresh_resume_choices()
        start_bar.addWidget(self.combo_resume, 1)
        start_bar.addWidget(QLabel(
            "每100轮自动保存；结束时导出 best/last 到 run_* 文件夹",
            objectName="muted"))
        layout.addLayout(start_bar)

        self.fig = Figure(figsize=(9, 4.6), dpi=100, facecolor=BG)
        self.chart_canvas = FigureCanvas(self.fig)
        self.chart_canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                        QSizePolicy.Policy.Expanding)
        layout.addWidget(self.chart_canvas, 1)
        self._refresh_history_runs()
        self._draw_chart(title="等待训练 — 点击「开始训练」后这里实时绘制曲线")
        return view

    def _start_training(self):
        if self.runner.running:
            return
        data = self.combo_resume.currentData()
        fresh = data == "fresh"
        resume_from = data if (data is not None and not fresh) else None
        if self.runner.start(resume_from=resume_from, fresh=fresh,
                             train=self.settings['train']):
            self._train_started_at = time.time()
            self.btn_train_start.setEnabled(False)
            self.btn_train_stop.setEnabled(True)
            self.btn_train_stop.setText("■ 停止训练")
            self.combo_resume.setEnabled(False)
            self._set_train_status("running")
            self.chart_episodes, self.chart_scores = [], []
            self.chart_rewards, self.chart_epsilons = [], []
            mode = ("从头训练（随机初始化）" if fresh else
                    f"从 {Path(resume_from).name} 续训" if resume_from else
                    "自动从最新检查点续训")
            self._log(self.log_train,
                      f"▶ 训练子进程已启动 — {mode}")
            self._show_view(0)

    def _stop_training(self):
        if not self.runner.running:
            return
        if self._train_status == "stopping":      # second click = force
            self.runner.force_stop()
            self._log(self.log_train, "⚠ 已强制终止训练子进程。")
            return
        self.runner.request_stop()
        self._set_train_status("stopping")
        self.btn_train_stop.setText("⟳ 再点一次强制终止")
        self._log(self.log_train,
                  "■ 已请求停止：等待本轮结束，模型会自动保存并导出 run 文件夹…")

    def _refresh_resume_choices(self):
        """(Re)build the 起始模型 picker: auto-latest / fresh start / every
        loadable checkpoint. Keeps the current selection across refreshes."""
        prev = self.combo_resume.currentData()
        combo = self.combo_resume
        combo.clear()
        combo.addItem("自动 · 从最新检查点续训", None)
        combo.addItem("从头训练 · 随机初始化", "fresh")
        for label, path in list_model_candidates():
            combo.addItem(f"续训 · {label}", path)
        if prev is not None:
            i = combo.findData(prev)
            if i >= 0:
                combo.setCurrentIndex(i)

    def _refresh_history_runs(self):
        runs = sorted((PROJECT_ROOT / 'models').glob('run_*'), reverse=True)
        names = [r.name for r in runs]
        prev = self.combo_history.currentText()   # keep the user's pick
        self.combo_history.clear()
        self.combo_history.addItems(names)
        if prev in names:
            self.combo_history.setCurrentText(prev)

    def _draw_history(self, *_):
        name = self.combo_history.currentText()
        if not name:
            return
        data = load_history(PROJECT_ROOT / 'models' / name)
        if not data:
            self._log(self.log_train, f"⚠ {name} 没有逐轮历史数据（history.jsonl 缺失）")
            return
        self.chart_episodes = [d['episode'] for d in data]
        self.chart_scores = [d['score'] for d in data]
        self.chart_rewards = [d['reward'] for d in data]
        self.chart_epsilons = [d['epsilon'] for d in data]
        self._draw_chart(title=f"{name}（历史，共 {len(data)} 轮）")
        self._log(self.log_train, f"📈 已加载 {name} 的训练曲线（{len(data)} 轮）")

    def _draw_chart(self, title=""):
        self.fig.clear()
        gs = self.fig.add_gridspec(2, 2, hspace=0.45, wspace=0.25)

        def moving_avg(values, window=100):
            out = []
            for i in range(len(values)):
                s = max(0, i + 1 - window)
                out.append(sum(values[s:i + 1]) / (i + 1 - s))
            return out

        ax1 = self.fig.add_subplot(gs[0, 0])
        ax2 = self.fig.add_subplot(gs[0, 1])
        ax3 = self.fig.add_subplot(gs[1, 0])
        for ax, series, label, color in (
                (ax1, self.chart_scores, 'Score', '#22c55e'),
                (ax2, self.chart_rewards, 'Reward', '#3b82f6'),
                (ax3, self.chart_epsilons, 'Epsilon', '#a78bfa')):
            ax.set_facecolor("#0b1220")
            ax.tick_params(colors=MUTED, labelsize=8)
            for spine in ax.spines.values():
                spine.set_color(PANEL2)
            if series:
                ax.plot(range(len(series)), series, color=color, alpha=0.35,
                        linewidth=0.8)
                if label != 'Epsilon':
                    avg = moving_avg(series)
                    ax.plot(range(len(avg)), avg, color='#f59e0b',
                            linewidth=1.6, label='avg100')
                    ax.legend(facecolor=BG, labelcolor=FG, fontsize=7,
                              frameon=False)
            ax.set_title(f"{label}  ({len(series)})", color=FG, fontsize=10)
        ax4 = self.fig.add_subplot(gs[1, 1])
        ax4.set_facecolor("#0b1220")
        ax4.axis("off")
        if title:
            ax4.text(0.5, 0.6, title, ha="center", va="center",
                     color=MUTED, fontsize=10, wrap=True,
                     transform=ax4.transAxes)
        if self.chart_episodes:
            recent = self.chart_scores[-100:]
            ax4.text(0.5, 0.25,
                     f"最近100轮平均分: {sum(recent) / len(recent):.2f}",
                     ha="center", va="center", color=ACCENT, fontsize=12,
                     transform=ax4.transAxes)
        self.chart_canvas.draw_idle()
        self._last_draw = time.time()

    # ---------- eval view ----------
    def _build_eval_view(self):
        view = QWidget()
        layout = QVBoxLayout(view)
        layout.setContentsMargins(12, 10, 12, 8)
        layout.setSpacing(6)

        bar = QHBoxLayout()
        bar.addWidget(QLabel("模型"))
        self.candidates = list_model_candidates()
        self.combo_model = QComboBox()
        self.combo_model.setMinimumWidth(320)
        self.combo_model.addItems([p for _, p in self.candidates])
        bar.addWidget(self.combo_model, 1)
        btn_refresh = QPushButton("刷新")
        btn_refresh.clicked.connect(self._refresh_models)
        bar.addWidget(btn_refresh)

        bar.addSpacing(10)
        bar.addWidget(QLabel("局数", objectName="muted"))
        self.spin_episodes = QSpinBox()
        self.spin_episodes.setRange(1, 1000)
        self.spin_episodes.setValue(20)
        bar.addWidget(self.spin_episodes)

        bar.addSpacing(10)
        bar.addWidget(QLabel("速度", objectName="muted"))
        self.scale_fps = QSlider(Qt.Orientation.Horizontal)
        self.scale_fps.setRange(1, 60)
        self.scale_fps.setValue(15)
        self.scale_fps.valueChanged.connect(self._on_fps)
        self.scale_fps.setMaximumWidth(120)
        bar.addWidget(self.scale_fps)
        self.lbl_fps = QLabel("15 FPS")
        bar.addWidget(self.lbl_fps)

        bar.addSpacing(10)
        self.btn_eval_start = QPushButton("▶ 开始验证", objectName="startBtn")
        self.btn_eval_start.clicked.connect(self._start_eval)
        bar.addWidget(self.btn_eval_start)
        self.btn_eval_stop = QPushButton("■ 停止", objectName="stopBtn")
        self.btn_eval_stop.clicked.connect(self._stop_eval)
        self.btn_eval_stop.setEnabled(False)
        bar.addWidget(self.btn_eval_stop)
        layout.addLayout(bar)

        body = QSplitter(Qt.Orientation.Horizontal)
        game_wrap = QFrame(objectName="card")
        game_layout = QVBoxLayout(game_wrap)
        game_layout.setContentsMargins(10, 8, 10, 10)
        game_layout.addWidget(QLabel("验证画面（内嵌）", objectName="muted"))
        self.game_canvas = GameCanvas()
        game_layout.addWidget(self.game_canvas, 1)
        body.addWidget(game_wrap)

        cards = QWidget()
        card_layout = QGridLayout(cards)
        card_layout.setContentsMargins(0, 0, 0, 0)
        card_layout.setSpacing(6)
        self.values = {}
        for i, (key, label, init) in enumerate((
                ("ep", "验证局数", "0"), ("score", "最近得分", "—"),
                ("best", "最高分", "—"), ("avg", "平均分", "—"),
                ("death", "死因 己/墙/超时", "0/0/0"))):
            card = QFrame(objectName="card")
            inner = QVBoxLayout(card)
            inner.setContentsMargins(12, 8, 12, 8)
            inner.addWidget(QLabel(label, objectName="muted"))
            value = QLabel(init, objectName="cardValue")
            inner.addWidget(value)
            card_layout.addWidget(card, i // 3, i % 3)
            self.values[key] = value
        self.lbl_eval_state = QLabel(EVAL_STATUS["idle"][0], objectName="pill")
        card_layout.addWidget(self.lbl_eval_state, 2, 0, 1, 3)
        card_layout.setRowStretch(3, 1)
        body.addWidget(cards)
        body.setSizes([700, 380])
        layout.addWidget(body, 1)
        return view

    # ---------- settings view ----------
    def _build_settings_view(self):
        view = QWidget()
        outer = QVBoxLayout(view)
        outer.setContentsMargins(12, 10, 12, 8)
        outer.setSpacing(10)

        # ---- training parameters ----
        tcard = QFrame(objectName="card")
        tgrid = QGridLayout(tcard)
        tgrid.setContentsMargins(16, 12, 16, 14)
        tgrid.setHorizontalSpacing(14)
        tgrid.setVerticalSpacing(10)
        tgrid.setColumnStretch(2, 1)
        tgrid.addWidget(QLabel("训练参数", objectName="secthead"), 0, 0, 1, 3)

        self.spin_set_grid = QSpinBox()
        self.spin_set_grid.setRange(8, 60)
        self.spin_set_eps = QSpinBox()
        self.spin_set_eps.setRange(100, 1000000)
        self.spin_set_eps.setSingleStep(1000)
        self.spin_set_nstep = QSpinBox()
        self.spin_set_nstep.setRange(1, 10)
        self.spin_set_edecay = QDoubleSpinBox()
        self.spin_set_edecay.setRange(0.9, 0.9999)
        self.spin_set_edecay.setDecimals(4)
        self.spin_set_edecay.setSingleStep(0.001)
        self.spin_set_saveint = QSpinBox()
        self.spin_set_saveint.setRange(10, 1000)
        self.spin_set_saveint.setSingleStep(10)
        self.spin_set_selfdeath = QDoubleSpinBox()
        self.spin_set_selfdeath.setRange(1.0, 3.0)
        self.spin_set_selfdeath.setDecimals(1)
        self.spin_set_selfdeath.setSingleStep(0.1)

        rows = (
            ("地图大小", self.spin_set_grid, "格（正方形棋盘边长）",
             "训练棋盘边长（格子数）。从头训练或无检查点时生效；"
             "续训时自动按检查点的尺寸，避免结构不匹配。"),
            ("总局数 (ep)", self.spin_set_eps, "局",
             "训练的目标总局数。续训时起始局数来自检查点，请保证大于它。"),
            ("n-step 回报", self.spin_set_nstep, "步",
             "多步回报：把奖励向前追认的步数，帮助学到远距离因果。"),
            ("ε 衰减率", self.spin_set_edecay, "/局",
             "每局结束后探索率 ε 的衰减系数，越接近 1 探索越久。"),
            ("自动保存间隔", self.spin_set_saveint, "局",
             "每 N 局保存根目录检查点并自动导出 run 文件夹。"),
            ("自撞惩罚倍率", self.spin_set_selfdeath, "×",
             "撞到自己而死相对撞墙而死的惩罚倍率（>1 更重视躲开自己的"
             "身体；1.0 = 两种死法惩罚相同）。"),
        )
        for r, (label, field, unit, tip) in enumerate(rows, start=1):
            field.setToolTip(tip)
            field.setFixedWidth(150)
            tgrid.addWidget(QLabel(label), r, 0)
            tgrid.addWidget(field, r, 1)
            tgrid.addWidget(QLabel(unit, objectName="muted"), r, 2)

        self.combo_set_rshape = QComboBox()
        self.combo_set_rshape.addItem("按蛇长缩放", "scaled")
        self.combo_set_rshape.addItem("固定 +10 / -10", "flat")
        self.combo_set_rshape.setFixedWidth(220)
        self.combo_set_rshape.setToolTip(
            "按蛇长缩放：蛇越长，吃到食物的奖励越高（约 +1 → +10），"
            "死亡的惩罚越轻（约 -10 → -1），后期学习信号更强。")
        r = len(rows) + 1
        tgrid.addWidget(QLabel("奖励机制"), r, 0)
        tgrid.addWidget(self.combo_set_rshape, r, 1)
        tgrid.addWidget(QLabel("长蛇吃果奖励更高、死亡惩罚更轻",
                               objectName="muted"), r, 2)
        outer.addWidget(tcard)

        # ---- eval parameters ----
        ecard = QFrame(objectName="card")
        egrid = QGridLayout(ecard)
        egrid.setContentsMargins(16, 12, 16, 14)
        egrid.setHorizontalSpacing(14)
        egrid.setColumnStretch(2, 1)
        egrid.addWidget(QLabel("验证参数", objectName="secthead"), 0, 0, 1, 3)
        self.spin_set_egrid = QSpinBox()
        self.spin_set_egrid.setRange(8, 60)
        self.spin_set_egrid.setFixedWidth(150)
        self.spin_set_egrid.setToolTip(
            "旧版 10 维视觉模型的对局棋盘边长。")
        egrid.addWidget(QLabel("验证地图大小"), 1, 0)
        egrid.addWidget(self.spin_set_egrid, 1, 1)
        egrid.addWidget(QLabel("格", objectName="muted"), 1, 2)
        egrid.addWidget(QLabel(
            "CNN 模型固定使用其训练时的棋盘尺寸，此项只对旧版视觉模型生效。",
            objectName="muted"), 2, 0, 1, 3)
        outer.addWidget(ecard)

        # ---- buttons ----
        btns = QHBoxLayout()
        btn_save = QPushButton("💾 保存设置", objectName="startBtn")
        btn_save.clicked.connect(self._save_settings)
        btns.addWidget(btn_save)
        btn_default = QPushButton("恢复默认")
        btn_default.clicked.connect(self._reset_settings)
        btns.addWidget(btn_default)
        self.lbl_settings_note = QLabel("", objectName="muted")
        btns.addWidget(self.lbl_settings_note)
        btns.addStretch(1)
        outer.addLayout(btns)
        outer.addWidget(QLabel(
            "设置保存在项目根目录 studio_settings.json，"
            "修改保存后于下次「开始训练 / 开始验证」时生效。", objectName="muted"))
        outer.addStretch(1)

        self._apply_settings_to_widgets()
        return view

    def _apply_settings_to_widgets(self):
        t = self.settings['train']
        self.spin_set_grid.setValue(t['grid_size'])
        self.spin_set_eps.setValue(t['episodes'])
        self.spin_set_nstep.setValue(t['n_step'])
        self.spin_set_edecay.setValue(t['epsilon_decay'])
        self.spin_set_saveint.setValue(t['save_interval'])
        self.spin_set_selfdeath.setValue(t.get('self_death_factor', 1.5))
        idx = self.combo_set_rshape.findData(t.get('reward_shaping'))
        self.combo_set_rshape.setCurrentIndex(idx if idx >= 0 else 0)
        self.spin_set_egrid.setValue(self.settings['eval']['grid_size'])

    def _collect_settings(self):
        return {
            'train': {
                'grid_size': self.spin_set_grid.value(),
                'episodes': self.spin_set_eps.value(),
                'n_step': self.spin_set_nstep.value(),
                'epsilon_decay': self.spin_set_edecay.value(),
                'save_interval': self.spin_set_saveint.value(),
                'reward_shaping': self.combo_set_rshape.currentData(),
                'self_death_factor': round(self.spin_set_selfdeath.value(), 1),
            },
            'eval': {
                'grid_size': self.spin_set_egrid.value(),
            },
        }

    def _save_settings(self):
        self.settings = self._collect_settings()
        path = save_settings(self.settings)
        self.lbl_settings_note.setText(
            f"已保存 → {path.name}（{time.strftime('%H:%M:%S')}）")
        self._log(self.log_train,
                  f"⚙️ 设置已保存：{json.dumps(self.settings, ensure_ascii=False)}")

    def _reset_settings(self):
        self.settings = {s: dict(v) for s, v in DEFAULT_SETTINGS.items()}
        self._apply_settings_to_widgets()
        self.lbl_settings_note.setText("已恢复默认值（尚未保存）")

    def _refresh_models(self):
        self.candidates = list_model_candidates()
        self.combo_model.clear()
        self.combo_model.addItems([p for _, p in self.candidates])
        self._log(self.log_eval, f"已刷新：发现 {len(self.candidates)} 个模型。")

    def _on_fps(self, value):
        self.fps_value = float(value)
        self.lbl_fps.setText(f"{int(self.fps_value)} FPS")

    def _start_eval(self):
        if self.eval_thread and self.eval_thread.is_alive():
            return
        model = self.combo_model.currentText()
        if not model or not os.path.exists(model):
            self._log(self.log_eval, f"❌ 找不到模型文件：{model}")
            return
        self.eval_stop = threading.Event()
        worker = EvalWorker(model_path=model,
                            fps_getter=lambda: self.fps_value,
                            msgs=self.eval_msgs, stop_event=self.eval_stop,
                            eval_grid_size=self.settings['eval']['grid_size'])
        self.eval_thread = threading.Thread(target=worker.run, daemon=True,
                                            name='eval-worker')
        # episode cap implemented via stop_event once reached (generation-
        # tagged so a stale cap check can't stop a newer run)
        self._eval_gen += 1
        target_episodes = self.spin_episodes.value()

        def cap_check(gen=self._eval_gen):
            if gen != self._eval_gen:
                return
            if self._eval_episodes_done >= target_episodes:
                self.eval_stop.set()
                return
            QTimer.singleShot(300, cap_check)

        self._eval_episodes_done = 0
        self.eval_thread.start()
        QTimer.singleShot(300, cap_check)
        self.btn_eval_start.setEnabled(False)
        self.btn_eval_stop.setEnabled(True)
        self._set_eval_status("running")
        self._log(self.log_eval, "▶ 验证已开始。")
        self._show_view(1)

    def _stop_eval(self):
        if self.eval_stop:
            self.eval_stop.set()
            self._set_eval_status("stopping")
            self._log(self.log_eval, "■ 停止中：当前步完成后退出…")

    # ---------- shared plumbing ----------
    def _log(self, widget, text):
        widget.appendPlainText(text)

    def _style_pill(self, key):
        text, color = TRAIN_STATUS[key]
        self.lbl_train_state.setText(text)
        self.lbl_train_state.setStyleSheet(f"color:{color};")

    def _set_train_status(self, key):
        self._train_status = key
        self._style_pill(key)

    def _set_eval_status(self, key):
        text, color = EVAL_STATUS[key]
        self.lbl_eval_state.setText(text)
        self.lbl_eval_state.setStyleSheet(f"color:{color};")

    def _poll(self):
        if self._closing:
            return
        # --- training subprocess output ---
        drained = 0
        while drained < 200:
            try:
                line = self.runner.lines.get_nowait()
            except queue.Empty:
                break
            drained += 1
            if line is None:                      # EOF
                self._handle_train_exit()
                continue
            self._log(self.log_train, line)
            ev = parse_train_line(line)
            if ev:
                self._handle_train_event(ev)
        if (self.chart_episodes
                and time.time() - self._last_draw > 2.0
                and self.runner.running):
            self._draw_chart()

        # --- eval worker messages ---
        while True:
            try:
                msg = self.eval_msgs.get_nowait()
            except queue.Empty:
                break
            self._handle_eval_msg(msg)

    def _handle_train_event(self, ev):
        if ev["kind"] == "episode":
            self.chart_episodes.append(ev["episode"])
            self.chart_scores.append(ev["score"])
            self.chart_rewards.append(ev["reward"])
            self.chart_epsilons.append(ev["epsilon"])
            elapsed = int(time.time() - self._train_started_at) if self._train_started_at else 0
            self.lbl_status_right.setText(
                f"Episode {ev['episode']}/{ev['total']}   "
                f"ε={ev['epsilon']:.3f}   用时 {elapsed // 60}:{elapsed % 60:02d}")
            recent = self.chart_scores[-100:]
            self.lbl_side_best.setText(f"{sum(recent) / len(recent):.2f}")
            self.lbl_status_left.setText("训练运行中")
        elif ev["kind"] == "resumed":
            self._log(self.log_train, f"📂 续训自 {ev['path']}")
        elif ev["kind"] == "autosave":
            self.lbl_status_left.setText(f"已自动保存 → {Path(ev['path']).name}")

    def _handle_train_exit(self):
        self._train_started_at = None
        self.btn_train_start.setEnabled(True)
        self.btn_train_stop.setEnabled(False)
        self.btn_train_stop.setText("■ 停止训练")
        self.combo_resume.setEnabled(True)
        self._refresh_resume_choices()
        self._set_train_status("finished")
        self._draw_chart(title="训练已结束（本次会话曲线）")
        self.lbl_status_left.setText("训练已结束")
        self._log(self.log_train, "— 训练子进程已退出 —")
        self._refresh_history_runs()    # this run's folder just finished exporting

    def _handle_eval_msg(self, msg):
        kind = msg[0]
        if kind == "frame":
            self.game_canvas.set_frame(*msg[1:])
        elif kind == "stats":
            _, ep, score, best, avg, deaths = msg
            self._eval_episodes_done = ep
            self.values["ep"].setText(str(ep))
            self.values["score"].setText(str(score))
            self.values["best"].setText(str(best))
            self.values["avg"].setText(f"{avg:.2f}")
            self.values["death"].setText(
                f"{deaths['self']}/{deaths['wall']}/{deaths['timeout']}")
        elif kind == "log":
            self._log(self.log_eval, msg[1])
        elif kind == "done":
            self.btn_eval_start.setEnabled(True)
            self.btn_eval_stop.setEnabled(False)
            self._set_eval_status("finished")

    # ---------- shutdown ----------
    def closeEvent(self, event):
        if self.runner.running:
            event.ignore()
            self._closing = True
            self._log(self.log_train, "窗口关闭：已请求训练保存并退出…")
            self.runner.request_stop()
            self._close_deadline = time.time() + 12
            self._close_timer.start(200)
            return
        if self.eval_stop:
            self.eval_stop.set()
        self._poll_timer.stop()
        event.accept()

    def _finish_close(self):
        if not self.runner.running or time.time() > self._close_deadline:
            if self.runner.running:
                self.runner.force_stop()
            self._close_timer.stop()
            self._closing = False
            self.close()          # runner no longer running → accepted


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Microsoft YaHei UI", 9))
    app.setStyleSheet(QSS)
    window = StudioWindow()
    window.show()
    if "--selftest" in sys.argv:
        QTimer.singleShot(2500, app.quit)   # build UI, pump events, exit
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
