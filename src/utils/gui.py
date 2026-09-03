"""
D3QN Snake - 模型验证控制台 (Tkinter GUI, stdlib only)

加载已训练的 checkpoint，用贪心策略（不探索、关 Dropout）连续运行游戏。
开始 / 停止按钮控制，实时显示得分与死因统计。

Run:
    python gui.py
"""

import glob
import os
import queue
import re
import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk

from envs import SnakeEnv
from agents import D3QNAgent

DEFAULT_MODEL = os.path.join('models', 'd3qn_snake_episode_4800.pth')

BG = "#0f172a"        # window background (dark slate)
PANEL = "#1e293b"     # card background
FG = "#e2e8f0"        # main text
MUTED = "#94a3b8"     # secondary text
ACCENT = "#22c55e"    # start / running
DANGER = "#ef4444"    # stop
WARN = "#f59e0b"      # stopping
BLUE = "#3b82f6"      # finished

STATUS = {
    "idle": ("● 待机", MUTED),
    "running": ("● 验证中", ACCENT),
    "stopping": ("● 停止中…", WARN),
    "finished": ("● 已结束", BLUE),
}


def list_checkpoints(directory='models'):
    """All d3qn_snake_episode_N.pth paths sorted by episode number."""
    out = []
    if os.path.isdir(directory):
        for f in glob.glob(os.path.join(directory, 'd3qn_snake_episode_*.pth')):
            m = re.search(r'episode_(\d+)\.pth$', f)
            if m:
                out.append((int(m.group(1)), f))
    return [path for _, path in sorted(out)]


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


class EvalWorker:
    """Greedy evaluation loop. Pure logic, runs inside a thread,
    reports everything through a queue; stoppable between steps."""

    def __init__(self, model_path, show, fps_getter, msgs, stop_event):
        self.model_path = model_path
        self.show = show
        self.fps_getter = fps_getter    # callable -> current slider FPS
        self.msgs = msgs
        self.stop_event = stop_event

    def run(self):
        try:
            agent = D3QNAgent()
            agent.load(self.model_path)
            agent.policy_net.eval()  # disable dropout for a deterministic policy
        except Exception as exc:
            self.msgs.put(("log", f"❌ 模型加载失败：{exc!r}"))
            self.msgs.put(("done", None))
            return
        self.msgs.put(("log", f"已加载模型：{self.model_path}（贪心模式，无探索/无 Dropout）"))

        env = SnakeEnv(grid_size=20)
        episodes, best = 0, 0
        scores = []
        deaths = {'wall': 0, 'self': 0, 'timeout': 0}

        while not self.stop_event.is_set():
            env.render_fps = max(1, int(self.fps_getter()))
            state, _ = env.reset()
            done = False
            r = 0.0
            cause = None
            while not done:
                if self.stop_event.is_set() or env.close_requested:
                    break
                action = agent.select_action(state, train=False)
                cause = classify_death(env, action)
                state, r, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                if self.show:
                    env.render()

            if self.stop_event.is_set() or env.close_requested:
                if env.close_requested:
                    self.msgs.put(("log", "检测到游戏窗口被关闭，停止验证。"))
                break

            # finished a full episode
            episodes += 1
            scores.append(env.score)
            best = max(best, env.score)
            if r == -1.0:  # timeout penalty is the only -1.0
                deaths['timeout'] += 1
            elif cause:
                deaths[cause] += 1
            avg = sum(scores) / len(scores)
            self.msgs.put(("stats", episodes, env.score, best, avg, dict(deaths)))

        env.close()
        self.msgs.put(("done", episodes))
        self.msgs.put(("log", f"验证结束：共 {episodes} 局，"
                              f"平均 {sum(scores) / max(1, len(scores)):.2f} 分，最高 {best} 分。"))


class TrainerGUI:
    """Thin UI layer; the eval worker runs in a thread and reports via queue."""

    def __init__(self, root):
        self.root = root
        self.worker = None
        self.worker_thread = None
        self.stop_event = None
        self.msgs = queue.Queue()
        self.fps_value = 20.0

        root.title("D3QN 模型验证控制台")
        root.configure(bg=BG)
        root.geometry("660x580")
        root.minsize(580, 500)

        self._build_style()
        self._build_ui()
        self._poll()
        root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------- UI ----------

    def _build_style(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background=BG)
        style.configure("Card.TFrame", background=PANEL)
        style.configure("TLabel", background=BG, foreground=FG, font=("Microsoft YaHei UI", 10))
        style.configure("Card.TLabel", background=PANEL, foreground=MUTED, font=("Microsoft YaHei UI", 9))
        style.configure("Value.TLabel", background=PANEL, foreground=FG, font=("Consolas", 15, "bold"))
        style.configure("Title.TLabel", background=BG, foreground=FG, font=("Microsoft YaHei UI", 13, "bold"))
        style.configure("TCheckbutton", background=BG, foreground=FG)
        style.configure("Green.TButton", font=("Microsoft YaHei UI", 10, "bold"), padding=(14, 8))
        style.configure("Red.TButton", font=("Microsoft YaHei UI", 10, "bold"), padding=(14, 8))
        style.map("Green.TButton",
                  foreground=[("disabled", "#64748b"), ("!disabled", "#052e16")],
                  background=[("disabled", PANEL), ("!disabled", ACCENT)])
        style.map("Red.TButton",
                  foreground=[("disabled", "#64748b"), ("!disabled", "#450a0a")],
                  background=[("disabled", PANEL), ("!disabled", DANGER)])

    def _build_ui(self):
        head = ttk.Frame(self.root)
        head.pack(fill="x", padx=16, pady=(14, 6))
        ttk.Label(head, text="🐍 模型验证台", style="Title.TLabel").pack(side="left")
        self.lbl_status = tk.Label(head, text=STATUS["idle"][0], fg=STATUS["idle"][1],
                                   bg=BG, font=("Microsoft YaHei UI", 11, "bold"))
        self.lbl_status.pack(side="right")

        # --- model selection ---
        sel = ttk.Frame(self.root)
        sel.pack(fill="x", padx=16, pady=4)
        ttk.Label(sel, text="模型").pack(side="left")
        self.models = list_checkpoints()
        default = DEFAULT_MODEL if DEFAULT_MODEL in self.models else (
            self.models[-1] if self.models else "")
        self.var_model = tk.StringVar(value=default)
        self.combo = ttk.Combobox(sel, textvariable=self.var_model, values=self.models,
                                  state="readonly", width=42)
        self.combo.pack(side="left", padx=(8, 8))
        ttk.Button(sel, text="刷新", width=5, command=self._refresh_models).pack(side="left")

        # --- control bar ---
        bar = ttk.Frame(self.root)
        bar.pack(fill="x", padx=16, pady=6)
        self.btn_start = ttk.Button(bar, text="▶ 开始", style="Green.TButton",
                                    command=self._start)
        self.btn_start.pack(side="left")
        self.btn_stop = ttk.Button(bar, text="■ 停止", style="Red.TButton",
                                   command=self._stop, state="disabled")
        self.btn_stop.pack(side="left", padx=(8, 20))

        self.var_show = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="显示游戏画面", variable=self.var_show).pack(side="left")

        ttk.Label(bar, text="速度").pack(side="left", padx=(16, 4))
        self.scale_fps = ttk.Scale(bar, from_=2, to=60, value=20,
                                   command=self._on_fps)
        self.scale_fps.pack(side="left", fill="x", expand=True)
        self.lbl_fps = ttk.Label(bar, text="20 FPS", font=("Consolas", 9))
        self.lbl_fps.pack(side="left", padx=(6, 0))

        # --- metric cards ---
        cards = ttk.Frame(self.root)
        cards.pack(fill="x", padx=16, pady=8)
        self.values = {}
        for i, (key, label, init) in enumerate([("ep", "验证局数", "0"),
                                                ("score", "最近得分", "—"),
                                                ("best", "最高分", "—"),
                                                ("avg", "平均分", "—"),
                                                ("death", "死因 己/墙/超时", "0/0/0")]):
            card = ttk.Frame(cards, style="Card.TFrame")
            card.grid(row=0, column=i, padx=3, sticky="nsew")
            cards.columnconfigure(i, weight=1)
            ttk.Label(card, text=label, style="Card.TLabel").pack(padx=8, pady=(8, 0))
            self.values[key] = ttk.Label(card, text=init, style="Value.TLabel")
            self.values[key].pack(padx=8, pady=(0, 8))

        # --- log ---
        self.log = scrolledtext.ScrolledText(self.root, height=11, bg="#020617",
                                             fg=MUTED, insertbackground=FG,
                                             font=("Consolas", 9), state="disabled",
                                             relief="flat")
        self.log.pack(fill="both", expand=True, padx=16, pady=(2, 14))
        self._append_log("就绪。选择模型后点「开始」，游戏画面会单独弹出；停止可随时中断。")

    def _append_log(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        if int(self.log.index("end-1c").split(".")[0]) > 2000:
            self.log.delete("1.0", "1000.0")
        self.log.configure(state="disabled")

    def _set_status(self, key):
        text, color = STATUS[key]
        self.lbl_status.config(text=text, fg=color)

    def _refresh_models(self):
        self.models = list_checkpoints()
        values = self.models or [""]
        if self.var_model.get() not in values:
            self.var_model.set(values[-1])
        self.combo.config(values=values)
        self._append_log(f"已刷新：发现 {len(self.models)} 个模型。")

    def _on_fps(self, value):
        self.fps_value = float(value)
        self.lbl_fps.config(text=f"{int(self.fps_value)} FPS")

    # ---------- start / stop ----------

    def _start(self):
        if self.worker_thread and self.worker_thread.is_alive():
            return
        model = self.var_model.get()
        if not model or not os.path.exists(model):
            messagebox.showerror("模型无效", f"找不到模型文件：\n{model}")
            return
        self.stop_event = threading.Event()
        worker = EvalWorker(model_path=model,
                            show=self.var_show.get(),
                            fps_getter=lambda: self.fps_value,
                            msgs=self.msgs,
                            stop_event=self.stop_event)
        self.worker_thread = threading.Thread(target=worker.run, daemon=True,
                                              name="eval-worker")
        self.worker_thread.start()
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self._set_status("running")
        self._append_log("▶ 验证已开始。")

    def _stop(self):
        if self.stop_event:
            self.stop_event.set()
            self.btn_stop.config(state="disabled")
            self._set_status("stopping")
            self._append_log("■ 停止中：当前步完成后立即退出…")

    # ---------- message pump ----------

    def _poll(self):
        try:
            while True:
                msg = self.msgs.get_nowait()
                kind = msg[0]
                if kind == "stats":
                    _, ep, score, best, avg, deaths = msg
                    self.values["ep"].config(text=str(ep))
                    self.values["score"].config(text=str(score))
                    self.values["best"].config(text=str(best))
                    self.values["avg"].config(text=f"{avg:.2f}")
                    self.values["death"].config(
                        text=f"{deaths['self']}/{deaths['wall']}/{deaths['timeout']}")
                elif kind == "log":
                    self._append_log(msg[1])
                elif kind == "done":
                    self.btn_start.config(state="normal")
                    self.btn_stop.config(state="disabled")
                    self._set_status("finished")
        except queue.Empty:
            pass
        self.root.after(150, self._poll)

    # ---------- shutdown ----------

    def _on_close(self):
        if self.worker_thread and self.worker_thread.is_alive():
            if not messagebox.askokcancel("退出", "验证正在进行中。\n停止并退出吗？"):
                return
            self.stop_event.set()
            self._append_log("■ 等待验证线程退出…")
            self._wait_worker_close()
        else:
            self.root.destroy()

    def _wait_worker_close(self):
        if self.worker_thread and self.worker_thread.is_alive():
            self.root.after(300, self._wait_worker_close)
        else:
            self.root.destroy()


def main():
    import sys
    root = tk.Tk()
    app = TrainerGUI(root)
    if "--auto-start" in sys.argv:
        root.after(500, app._start)  # begin verification with current settings
    if "--selftest" in sys.argv:
        root.after(2000, root.destroy)  # build UI, pump events briefly, exit
    root.mainloop()


if __name__ == "__main__":
    main()
