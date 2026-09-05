"""
D3QN Snake Training Script
Main training loop for D3QN agent to play Snake game
"""

import os
import re
import glob
import json
import time
import sys
import argparse
from pathlib import Path

# Add src directory to path for imports
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt
from envs import SnakeEnv
from envs.snake_env import STEP_GUIDANCE_COEFF
from agents import D3QNAgent
from ckpt_utils import infer_model_arch

# Anchor all project paths here so the script works from any cwd
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def find_latest_checkpoint(checkpoint_dir=None):
    """Find the newest d3qn_snake_episode_NNNN.pth in checkpoint_dir.
    Returns (episode_number, path) or None if no checkpoint exists."""
    checkpoint_dir = checkpoint_dir or str(PROJECT_ROOT / 'models')
    if not os.path.isdir(checkpoint_dir):
        return None
    candidates = []
    for f in glob.glob(os.path.join(checkpoint_dir, 'd3qn_snake_episode_*.pth')):
        m = re.search(r'episode_(\d+)\.pth$', f)
        if m:
            candidates.append((int(m.group(1)), f))
    return max(candidates) if candidates else None


class Trainer:
    """Trainer class for managing the complete training process"""
    
    def __init__(self, n_episodes=10000, max_steps_per_episode=500,
                 log_interval=10, save_interval=50, render_training=False,
                 resume=False, model_path=None,
                 on_episode=None, stop_event=None,
                 obs_type='vision', n_step=1, grid_size=30,
                 epsilon_decay=None,
                 reward_shaping='scaled',
                 self_death_factor=1.5,
                 preview_interval=100):
        self.n_episodes = n_episodes
        self.max_steps = max_steps_per_episode
        self.render_training = render_training  # Enable visualization during training

        # GUI integration hooks (both optional, unused by plain console runs)
        self.on_episode = on_episode    # callable(episode, score, reward, epsilon)
        self.stop_event = stop_event    # threading.Event; .set() requests graceful stop

        # Create environment and agent
        self.env = SnakeEnv(grid_size=grid_size, observation_type=obs_type,
                            reward_shaping=reward_shaping,
                            self_death_factor=self_death_factor)
        if obs_type == 'grid':
            # Full-board CNN: bigger batches help, image replays need more RAM;
            # much slower epsilon decay — pixel CNNs learn late, and if
            # exploration dies before food-seeking emerges the greedy policy
            # freezes into wall-avoiding circles (0.999 hit the 0.10 floor at
            # ~ep 2300 on 30×30 with avg score still ≈ 0.1 — never recovered).
            # Buffer must hold enough EPISODES, not just transitions: a 20×20
            # episode runs up to 400 steps, so 50k transitions was a ~150-episode
            # window of highly correlated experience (vs ~1000 episodes at
            # 10×10, which trains fine) — 200k restores a healthy mix.
            self.agent = D3QNAgent(obs_type='grid', n_step=n_step, grid_size=grid_size,
                                   buffer_size=200000, batch_size=128,
                                   epsilon_decay=epsilon_decay if epsilon_decay is not None else 0.9997)
        else:
            self.agent = D3QNAgent(n_step=n_step)
        
        # Logging
        self.log_interval = log_interval
        self.save_interval = save_interval
        # Every N episodes play ONE rendered episode in a visible window so
        # headless training stays observable. 0 disables the preview entirely.
        self.preview_interval = preview_interval
        
        # Tracking metrics
        self.rewards = []
        self.scores = []
        self.losses = []
        self.epsilon_values = []
        # Death-cause tally for this run: late game is dominated by
        # self-collisions, and this makes that visible in logs/params
        self.death_counts = {'wall': 0, 'self': 0, 'timeout': 0}
        
        # Start step
        self.total_steps = 0
        
        # Best-model tracking: the "best" checkpoint is the moment the
        # trailing-average score peaked (more robust than a single lucky episode)
        self.run_id = time.strftime('%Y%m%d_%H%M%S')
        self.best_avg_window = 100    # trailing window for the best criterion
        self.best_min_episodes = 20   # episodes before a "best" is trustworthy
        self.best_avg = None          # best trailing-average score this run
        self.best_episode = None      # global episode number where it happened
        self.resumed_from = None
        
        # Resume support: load latest checkpoint (or an explicit model_path)
        self.start_episode = 1
        if model_path is None and resume:
            found = find_latest_checkpoint()
            if found:
                model_path = found[1]
        if model_path:
            if os.path.exists(model_path):
                self.agent.load(model_path)
                m = re.search(r'episode_(\d+)\.pth$', model_path)
                if m:
                    self.start_episode = int(m.group(1)) + 1
                self.resumed_from = model_path
                print(f"📂 Resumed from: {model_path}")
                print(f"   Continuing at episode {self.start_episode}, "
                      f"ε={self.agent.epsilon:.4f}, agent steps={self.agent.steps}")
            else:
                print(f"⚠️ Checkpoint not found: {model_path} — starting fresh")
    
    def train_one_episode(self, episode_num, render=False):
        """Train for one episode"""
        
        # Show the episode number and exploration rate on the pygame window
        self.env.current_episode = episode_num
        self.env.current_epsilon = self.agent.epsilon
        state, _ = self.env.reset()
        total_reward = 0
        total_score = 0
        done = False
        step = 0
        
        while not done and step < self.max_steps:
            # Select action
            action = self.agent.select_action(state)
            
            # Take step
            next_state, reward, terminated, truncated, info = self.env.step(action)
            
            # Store transition
            self.agent.store_transition(state, action, reward, next_state, int(terminated or truncated))
            
            # Update
            state = next_state
            total_reward += reward
            
            # Render (either requested or from training mode)
            should_render = render or self.render_training
            if should_render:
                self.env.render()
            
            # Optimize model
            loss = self.agent.optimize_model()
            if loss is not None:
                self.losses.append(loss)
            
            # Step counter
            self.agent.step()
            self.total_steps += 1
            
            done = terminated or truncated
            step += 1
        
        self.agent.end_episode()
        return total_reward, self.env.score
    
    def _maybe_update_best(self, episode):
        """Record a new best when the trailing-average score improves.
        Averages over the last `best_avg_window` episodes (all episodes so
        far if fewer); only eligible once `best_min_episodes` were played —
        earlier averages are too noisy to call a best."""
        if len(self.scores) < self.best_min_episodes:
            return False
        avg = float(np.mean(self.scores[-self.best_avg_window:]))
        if self.best_avg is None or avg > self.best_avg:
            self.best_avg, self.best_episode = avg, episode
            return True
        return False
    
    def evaluate_performance(self, window=50):
        """Calculate moving average metrics"""
        if len(self.rewards) < window:
            return []
        
        avg_rewards = []
        for i in range(len(self.rewards)):
            end = i + 1
            start = max(0, end - window)
            avg_rewards.append(np.mean(self.rewards[start:end]))
        
        return avg_rewards
    
    def plot_training_metrics(self, save_path=None, best_episode=None):
        """Plot and save training curves.
        save_path: where to write the PNG (default output/training_curves.png).
        best_episode: 0-based index into this run's metrics marking the
        trailing-average peak — drawn as an orange dashed line."""
        window = 50
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Episode rewards
        axes[0, 0].plot(range(len(self.rewards)), self.rewards, alpha=0.3, label='Episode Reward')
        avg_rew = self.evaluate_performance()
        if avg_rew:
            axes[0, 0].plot(range(len(avg_rew)), avg_rew, 'r-', linewidth=2, label=f'{window}-episode Avg')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title('Rewards Over Time')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Scores per episode
        axes[0, 1].plot(range(len(self.scores)), self.scores, alpha=0.3, label='Score')
        avg_score = self.evaluate_performance()
        if avg_score:
            axes[0, 1].plot(range(len(avg_score)), avg_score, 'g-', linewidth=2, label=f'{window}-episode Avg')
        if best_episode is not None and 0 <= best_episode < len(self.scores):
            line = axes[0, 1].axvline(x=best_episode, color='orange',
                                      linestyle='--', linewidth=1.5, alpha=0.9)
            line.set_label('Best (trailing avg)')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Food Eaten')
        axes[0, 1].set_title('Score (Food Eaten)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Loss curve (downsampled when huge — per-step losses can reach millions,
        # and this plot is regenerated on every auto-save)
        if len(self.losses) > 20000:
            loss_step = max(1, len(self.losses) // 10000)
            losses_plot = self.losses[::loss_step]
            loss_label = f'Training Loss (1 in {loss_step})'
        else:
            losses_plot = self.losses
            loss_label = 'Training Loss'
        axes[1, 0].plot(range(len(losses_plot)), losses_plot, alpha=0.3, color='blue', label=loss_label)
        if len(self.losses) > 0:
            axes[1, 0].axhline(y=np.mean(self.losses), color='red', linestyle='--', label='Mean Loss')
        axes[1, 0].set_xlabel('Training Step')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Training Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Epsilon decay
        axes[1, 1].plot(range(len(self.epsilon_values)), self.epsilon_values, 'purple', linewidth=2)
        axes[1, 1].fill_between(range(len(self.epsilon_values)), self.agent.epsilon_end, 
                               self.epsilon_values, alpha=0.3)
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Epsilon')
        axes[1, 1].set_title('Exploration Rate Decay')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path is None:
            out_dir = PROJECT_ROOT / 'output'
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / 'training_curves.png'
        else:
            out_path = Path(save_path)
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Training curves saved to '{out_path}'")
    
    def save_trained_model(self, episode):
        """Save the trained model"""
        models_dir = PROJECT_ROOT / 'models'
        models_dir.mkdir(parents=True, exist_ok=True)
        filepath = models_dir / f'd3qn_snake_episode_{episode}.pth'
        self.agent.save(str(filepath))
        print(f"Model saved to {filepath}")
    
    def _run_dir(self):
        """Folder for this training session: models/run_{run_id}/
        Holds BOTH models (best + last) plus params and training curves."""
        return PROJECT_ROOT / 'models' / f'run_{self.run_id}'

    def build_params_dict(self, stop_reason=None, snapshot=False):
        """All parameters and run statistics for this training session's folder."""
        a = self.agent
        return {
            'run_id': self.run_id,
            'exported_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'export_status': 'auto_save' if snapshot else 'final',
            'stop_reason': stop_reason,
            'network': type(a.policy_net).__name__,
            'network_params': sum(p.numel() for p in a.policy_net.parameters()),
            'hyperparams': {
                'obs_type': a.obs_type,
                'grid_size': getattr(a, 'grid_size', None),
                'input_dim': getattr(a, 'input_dim', None),
                'reward_shaping': getattr(self.env, 'reward_shaping', 'flat'),
                'self_death_factor': getattr(self.env, 'self_death_factor', 1.0),
                'step_guidance_coeff': STEP_GUIDANCE_COEFF,
                'num_actions': a.policy_net.num_actions,
                'n_step': a.n_step,
                'gamma': a.gamma,
                'lr': a.optimizer.param_groups[0]['lr'],
                'optimizer': type(a.optimizer).__name__,
                'loss': type(a.criterion).__name__,
                'grad_clip': 10,
                'batch_size': a.batch_size,
                'buffer_size': a.buffer_size,
                'target_update_steps': a.target_update,
                'epsilon_start': a.epsilon_start,
                'epsilon_end': a.epsilon_end,
                'epsilon_decay': a.epsilon_decay,
                'epsilon_at_export': a.epsilon,
                'agent_steps': a.steps,
            },
            'run': {
                'episodes_this_run': len(self.scores),
                'start_episode': self.start_episode,
                'end_episode': self.last_episode,
                'resumed_from': self.resumed_from,
                'total_env_steps': self.total_steps,
            },
            'metrics': {
                'max_score': max(self.scores) if self.scores else None,
                'avg_score_last_100': round(float(np.mean(self.scores[-100:])), 2) if self.scores else None,
                'avg_reward_last_100': round(float(np.mean(self.rewards[-100:])), 2) if self.rewards else None,
                'final_epsilon': self.epsilon_values[-1] if self.epsilon_values else None,
                'losses_recorded': len(self.losses),
                'deaths': dict(self.death_counts),
            },
            'best': {
                'criterion': f'avg_score_last_{self.best_avg_window}',
                'scope': 'this training run',
                'min_episodes_required': self.best_min_episodes,
                'episode': self.best_episode,
                'avg_score': round(self.best_avg, 2) if self.best_avg is not None else None,
                'fallback_to_last': self.best_episode is None,
            },
            'last': {
                'episode': self.last_episode,
                'saved_by': 'auto-save every save_interval episodes + exit-save',
            },
            'files': {
                'best_model': 'best_model.pth',
                'last_model': 'last_model.pth',
                'params': 'params.json',
                'plot': 'training_metrics.png',
                'history': 'history.jsonl',
            },
        }

    def _write_params(self, run_dir, stop_reason=None, snapshot=False):
        (run_dir / 'params.json').write_text(
            json.dumps(self.build_params_dict(stop_reason, snapshot),
                       indent=2, ensure_ascii=False), encoding='utf-8')

    def _append_history(self, episode, score, reward, death=None):
        """Append one line of per-episode history to the run folder —
        the source data for the GUI's chart view. A locked file costs one
        chart point, never the training run."""
        d = self._run_dir()
        d.mkdir(parents=True, exist_ok=True)
        try:
            with open(d / 'history.jsonl', 'a', encoding='utf-8') as f:
                f.write(json.dumps({'episode': episode, 'score': score,
                                    'reward': round(reward, 2),
                                    'epsilon': round(self.agent.epsilon, 4),
                                    'death': death}) + '\n')
            self._history_io_warned = False
        except OSError as e:
            if not getattr(self, '_history_io_warned', False):
                print(f"⚠️ history.jsonl 写入失败（图表会缺数据点，训练继续）：{e}")
                self._history_io_warned = True

    def _write_plot(self, run_dir):
        best_idx = (self.best_episode - self.start_episode) \
            if self.best_episode is not None else None
        self.plot_training_metrics(save_path=run_dir / 'training_metrics.png',
                                   best_episode=best_idx)

    def _save_best_to_run_dir(self):
        """Auto-save on a new best: refresh best_model.pth + params.json.
        Fires the moment the trailing-average score peaks, so the best
        weights survive even an ungraceful crash."""
        d = self._run_dir()
        d.mkdir(parents=True, exist_ok=True)
        self.agent.save(str(d / 'best_model.pth'))
        self._write_params(d, snapshot=True)

    def _auto_save_run(self):
        """Auto-save every save_interval episodes: last_model.pth + params + curves."""
        d = self._run_dir()
        d.mkdir(parents=True, exist_ok=True)
        self.agent.save(str(d / 'last_model.pth'))
        self._write_params(d, snapshot=True)
        # Curves are cosmetic: a locked target (e.g. the png open in a
        # viewer) must never kill the run — it once crashed a 20000-episode
        # session at ep 4000 via OSError(22).
        try:
            self._write_plot(d)
        except Exception as e:
            print(f"⚠️ 曲线图保存失败，已跳过（训练继续）：{e}")
        print(f"⏳ Auto-save → {d}")

    def _finalize_run(self, stop_reason):
        """Exit-save: final last_model.pth + params.json + curves.
        The best snapshot taken during the run is NOT overwritten; if no
        best was ever recorded, best_model.pth falls back to final weights."""
        d = self._run_dir()
        d.mkdir(parents=True, exist_ok=True)
        self.agent.save(str(d / 'last_model.pth'))
        if self.best_episode is None:
            self.agent.save(str(d / 'best_model.pth'))
        self._write_params(d, stop_reason=stop_reason)
        try:
            self._write_plot(d)
        except Exception as e:
            print(f"⚠️ 曲线图保存失败，已跳过：{e}")
        return d

    def print_training_status(self, episode, reward, score, steps):
        """Print training status every log_interval episodes"""
        avg_reward = np.mean(self.rewards[-10:]) if len(self.rewards) >= 10 else reward
        print(f"\n{'='*60}")
        print(f"Episode: {episode:5d} | Epsilon: {self.agent.epsilon:.4f} | "
              f"Avg Reward: {avg_reward:8.2f} | Score: {score:4d} | "
              f"Steps: {steps:5d} | Current ε: {self.agent.epsilon:.4f}")
        total = sum(self.death_counts.values())
        if total:
            c = self.death_counts
            print(f"Deaths this run: wall {c['wall']} ({c['wall']/total*100:.0f}%) | "
                  f"self {c['self']} ({c['self']/total*100:.0f}%) | "
                  f"timeout {c['timeout']} ({c['timeout']/total*100:.0f}%)")
        print(f"{'='*60}\n")
    
    def train(self):
        """Run the full training loop"""
        
        print("\n" + "="*60)
        print("🐍 D3QN Snake Training Started!")
        print("="*60)
        print(f"Target episodes: {self.n_episodes}")
        print(f"Device: {self.agent.device}")
        print(f"Reward shaping: {self.env.reward_shaping}"
              + ("（按蛇长缩放：长蛇吃果奖励更高、死亡惩罚更轻）"
                 if self.env.reward_shaping == 'scaled' else "（固定 +10 / -10）"))
        print(f"Self-collision penalty: ×"
              f"{getattr(self.env, 'self_death_factor', 1.0):g} "
              f"（撞自己死得比撞墙更亏，让蛇学会区分两种死法）")
        print(f"Step guidance: ±{STEP_GUIDANCE_COEFF:g}/len"
              f"（每步靠近食物加分、远离扣分；蛇越长信号越弱，避免长蛇无脑直冲）")
        print(f"Epsilon decay: {self.agent.epsilon_decay:g} "
              f"(floor {self.agent.epsilon_end:g})")
        print(f"Save interval: every {self.save_interval} episodes")
        if self.start_episode > self.n_episodes:
            print(f"⚠️ 起始局数 {self.start_episode} 已超过总局数 {self.n_episodes}"
                  f" — 本轮将立即结束；如需继续训练请在设置中调大总局数")
        print("Stop anytime: close the game window, press Ctrl+C,")
        print("              or run 停止训练.bat — the model is auto-saved")
        print("="*60)
        
        self.last_episode = self.start_episode - 1
        stop_reason = None
        
        try:
            for episode in range(self.start_episode, self.n_episodes + 1):
                self.last_episode = episode
                
                # Train one episode
                reward, score = self.train_one_episode(episode)

                # Attribute the death cause before the next reset clears it
                cause = getattr(self.env, 'last_death_cause', None)
                if cause in self.death_counts:
                    self.death_counts[cause] += 1

                # Per-episode progress line
                print(f"[Episode {episode:4d}/{self.n_episodes}] Score: {score:3d} | "
                      f"Reward: {reward:8.2f} | ε: {self.agent.epsilon:.3f}")

                # Record metrics
                self.rewards.append(reward)
                self.scores.append(score)
                self.epsilon_values.append(self.agent.epsilon)
                self._append_history(episode, score, reward, death=cause)
                
                # Snapshot the best model whenever the trailing-average
                # score peaks (quietly — the folder is summarized at exit)
                if self._maybe_update_best(episode):
                    self._save_best_to_run_dir()
                
                # Report progress to a GUI (if attached)
                if self.on_episode:
                    self.on_episode(episode, score, reward, self.agent.epsilon)

                # Graceful stop requested externally (e.g. GUI stop button)
                if self.stop_event is not None and self.stop_event.is_set():
                    stop_reason = "stop requested"
                    break

                # Print status
                if episode % self.log_interval == 0:
                    self.print_training_status(episode, reward, score, self.total_steps)
                
                # Save model periodically: root checkpoint for resume
                # + auto-save into this run's folder (last model + params + curves).
                # I/O hiccups (locked files, full disk) must not kill the run.
                if episode % self.save_interval == 0:
                    try:
                        self.save_trained_model(episode)
                        self._auto_save_run()
                    except Exception as e:
                        print(f"⚠️ 第 {episode} 局自动保存失败，已跳过（训练继续）：{e}")
                
                # Graceful stop: user closed the game window
                if getattr(self.env, 'close_requested', False):
                    stop_reason = "game window closed"
                    break
                
                # Graceful stop: STOP file found in project root
                stop_file = PROJECT_ROOT / 'STOP'
                if stop_file.exists():
                    stop_reason = "STOP file detected"
                    try:
                        stop_file.unlink()
                    except OSError:
                        pass
                    break
                
                # Check for convergence criteria
                if len(self.rewards) >= 100:
                    recent_avg = np.mean(self.rewards[-100:])
                    if recent_avg > 50.0:
                        print(f"\n✅ Converged! Recent 100-episode avg reward: {recent_avg:.2f}")
                        break
                
                # Live preview: play one rendered episode every N episodes in a
                # visible pygame window (headless-friendly UI). Set
                # preview_interval=0 to disable, or render_training=True to
                # render EVERY step (much slower: 20 FPS cap).
                if self.preview_interval and episode % self.preview_interval == 0:
                    print(f"\n🎮 Live preview — episode {episode} (rendering)...")
                    reward_demo, score_demo = self.train_one_episode(episode, render=True)
        
        except KeyboardInterrupt:
            stop_reason = "Ctrl+C"
        
        # Final summary + auto-save — runs on normal completion AND any manual stop
        self._training_summary()

        print("\n📊 Generating training plots...")
        try:
            self.plot_training_metrics()
        except Exception as e:
            print(f"⚠️ 训练曲线生成失败，已跳过：{e}")

        if self.last_episode >= self.start_episode:
            try:
                self.save_trained_model(self.last_episode)

                # Exit-save: finalize this run's folder (best + last + params + curves)
                print("\n📦 Exporting training run package...")
                run_dir = self._finalize_run(stop_reason)
            except Exception as e:
                run_dir = None
                print(f"⚠️ 结束导出失败（根目录检查点可能已保存）：{e}")
            if run_dir is not None:
                if self.best_episode is not None:
                    print(f"🏆 Best model → {run_dir / 'best_model.pth'}")
                    print(f"            (avg score {self.best_avg:.2f} over last "
                          f"{self.best_avg_window} eps, peaked at episode {self.best_episode})")
                else:
                    print(f"🏆 Best model → {run_dir / 'best_model.pth'} "
                          f"(no reliable best this run — identical to final weights)")
                print(f"📍 Last model → {run_dir / 'last_model.pth'} (episode {self.last_episode})")
                print(f"📄 Params     → {run_dir / 'params.json'}")
                print(f"📈 Curves     → {run_dir / 'training_metrics.png'}")
        else:
            print("No new episodes completed this session — keeping existing checkpoints.")
        
        # Close environment
        self.env.close()
        
        if stop_reason:
            print(f"\n🛑 Training stopped early ({stop_reason}).")
        print(f"\n✨ Model auto-saved at episode {self.last_episode} — "
              f"restart with resume=True to continue from here.")
    
    def _training_summary(self):
        """Print training summary statistics"""
        print(f"\n{'='*60}")
        print("📊 TRAINING SUMMARY")
        print(f"{'='*60}")
        if len(self.rewards) > 0:
            print(f"Total Episodes:      {len(self.rewards)}")
            print(f"Max Score:           {max(self.scores)}")
            print(f"Min Score:           {min(self.scores)}")
            print(f"Avg Score (last 100): {np.mean(self.scores[-100:]):.2f}")
            if self.best_episode is not None:
                print(f"Best Avg Score:      {self.best_avg:.2f} (at episode {self.best_episode})")
            print(f"Best Reward:         {max(self.rewards):.2f}")
            print(f"Avg Reward (last 100): {np.mean(self.rewards[-100:]):.2f}")
        if len(self.epsilon_values) > 0:
            print(f"Final Epsilon:       {self.epsilon_values[-1]:.4f}")
        if len(self.losses) > 0:
            print(f"Final Loss:          {self.losses[-1]:.4f}")
        print(f"Total Training Steps:{self.total_steps:,}")
        print(f"{'='*60}\n")


def resolve_training_config(resume_from=None, fresh=False, grid_size=30):
    """Map start options (CLI / GUI) to Trainer kwargs. The agent architecture
    must match the checkpoint being loaded, so whenever weights are loaded
    (explicit --resume-from, or auto-resume finding a checkpoint) the grid
    size comes from the weights themselves; the requested grid size only
    applies to fresh starts and checkpoint-less auto runs."""
    if fresh:
        return {'resume': False, 'model_path': None,
                'obs_type': 'grid', 'grid_size': grid_size}
    if resume_from:
        obs_type, ckpt_grid = infer_model_arch(resume_from)
        return {'resume': False, 'model_path': resume_from,
                'obs_type': obs_type, 'grid_size': ckpt_grid}
    found = find_latest_checkpoint()
    if found:
        obs_type, ckpt_grid = infer_model_arch(found[1])
        return {'resume': True, 'model_path': None,
                'obs_type': obs_type, 'grid_size': ckpt_grid}
    return {'resume': True, 'model_path': None,
            'obs_type': 'grid', 'grid_size': grid_size}


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='D3QN Snake training')
    parser.add_argument('--resume-from', dest='resume_from', default=None,
                        help='continue from a specific checkpoint '
                             '(default: newest d3qn_snake_episode_*.pth in models/)')
    parser.add_argument('--fresh', action='store_true',
                        help='start from randomly initialized weights (no resume)')
    parser.add_argument('--episodes', type=int, default=20000,
                        help='total target episodes (default 20000)')
    parser.add_argument('--grid-size', dest='grid_size', type=int, default=30,
                        help='board size in cells for fresh/checkpoint-less runs '
                             '(default 30; ignored when resuming — weights decide)')
    parser.add_argument('--n-step', dest='n_step', type=int, default=3,
                        help='n-step returns (default 3)')
    parser.add_argument('--epsilon-decay', dest='epsilon_decay', type=float,
                        default=0.9997, help='per-episode epsilon decay (default 0.9997)')
    parser.add_argument('--reward-shaping', dest='reward_shaping',
                        choices=['flat', 'scaled'], default='scaled',
                        help="reward mode: 'scaled' = size-dependent food/death "
                             "rewards (default), 'flat' = fixed +10/-10")
    parser.add_argument('--self-death-factor', dest='self_death_factor',
                        type=float, default=1.5,
                        help='self-collision death penalty multiplier vs wall '
                             '(default 1.5; 1.0 = identical penalties)')
    parser.add_argument('--save-interval', dest='save_interval', type=int,
                        default=100, help='save checkpoint every N episodes (default 100)')
    args = parser.parse_args()
    cfg = resolve_training_config(args.resume_from, args.fresh, args.grid_size)
    if args.resume_from:
        print(f"🎯 Starting from checkpoint: {args.resume_from} "
              f"({cfg['obs_type']} {cfg['grid_size']}×{cfg['grid_size']})")
    elif args.fresh:
        print("🌱 Fresh start: random initialization, no resume")
    if cfg['grid_size'] != args.grid_size:
        print(f"ℹ️ 地图大小已按检查点调整为 {cfg['grid_size']}×{cfg['grid_size']}"
              f"（请求 {args.grid_size}）— 网络结构必须与已训练权重匹配")
    trainer = Trainer(
        n_episodes=args.episodes,
        max_steps_per_episode=1000,
        log_interval=20,           # Print status every N episodes
        save_interval=args.save_interval,
        render_training=False,     # Headless for speed; watch via scripts/demo.py
        resume=cfg['resume'],
        model_path=cfg['model_path'],
        obs_type=cfg['obs_type'],  # matched to the resume source automatically
        n_step=args.n_step,
        epsilon_decay=args.epsilon_decay,
        reward_shaping=args.reward_shaping,
        self_death_factor=args.self_death_factor,
        grid_size=cfg['grid_size'],
        preview_interval=0         # 0 = fully headless, no live window
    )
    trainer.train()


if __name__ == '__main__':
    main()
