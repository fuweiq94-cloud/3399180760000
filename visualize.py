"""
Visual report for the D3QN Snake training run.

Combines the full training history (parsed from a training log) with a fresh
greedy evaluation of the trained model into one figure:
  - learning curve (score + proper score moving-average + epsilon overlay)
  - evaluation scores per episode
  - evaluation score distribution
  - death-cause breakdown

Usage:
    python visualize.py                       # default: training_log.txt + latest model, 50 eval eps
    python visualize.py --log mylog.txt -n 100
    python visualize.py --no-eval             # training curves only
"""

import argparse
import os
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')  # save to file, no window needed
import matplotlib.pyplot as plt

from snake_env import SnakeEnv
from d3qn_agent import D3QNAgent
from eval_model import find_latest_checkpoint, evaluate

EP_LINE = re.compile(r'\[Episode\s+(\d+)/\d+\] Score:\s+(\d+).*ε:\s+([\d.]+)')


def moving_avg(x, window=50):
    """Trailing moving average, same length as x (leading values are partial)."""
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return x
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        lo = max(0, i - window + 1)
        out[i] = x[lo:i + 1].mean()
    return out


def parse_training_log(path):
    """Extract (episodes, scores, epsilons) from a train.py stdout log."""
    episodes, scores, epsilons = [], [], []
    with open(path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = EP_LINE.search(line)
            if m:
                episodes.append(int(m.group(1)))
                scores.append(int(m.group(2)))
                epsilons.append(float(m.group(3)))
    return np.array(episodes), np.array(scores), np.array(epsilons)


def main():
    parser = argparse.ArgumentParser(description='Visual training/eval report')
    parser.add_argument('--log', type=str, default='training_log.txt')
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('-n', '--episodes', type=int, default=50)
    parser.add_argument('--out', type=str, default='training_report.png')
    parser.add_argument('--no-eval', action='store_true')
    args = parser.parse_args()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ---- (0,0) full learning curve from log ----
    ax = axes[0, 0]
    if os.path.exists(args.log):
        eps, scores, epsilons = parse_training_log(args.log)
        if len(eps):
            ax.scatter(eps, scores, s=6, alpha=0.25, color='steelblue',
                       label='Score per episode')
            ax.plot(eps, moving_avg(scores, 50), color='red', linewidth=2,
                    label='50-episode avg (scores)')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Score (food eaten)')
            ax2 = ax.twinx()
            ax2.plot(eps, epsilons, color='purple', linewidth=1.2, linestyle='--',
                     label='epsilon (right axis)')
            ax2.set_ylabel('Epsilon')
            ax2.set_ylim(0, 1.05)
            lines = ax.get_lines() + ax2.get_lines()
            ax.legend(lines, [l.get_label() for l in lines], loc='upper left', fontsize=9)
            ax.set_title(f'Learning Curve ({len(eps)} episodes, log: {os.path.basename(args.log)})')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No episode lines found in log', ha='center', va='center')
    else:
        ax.text(0.5, 0.5, f'Log not found: {args.log}', ha='center', va='center')
        ax.set_title('Learning Curve')

    if not args.no_eval:
        # ---- run greedy evaluation ----
        path = args.model
        if path is None:
            found = find_latest_checkpoint()
            path = found[1] if found else None
        if path is None:
            raise SystemExit('No checkpoint in models/ — pass --model or use --no-eval')
        agent = D3QNAgent()
        agent.load(path)
        agent.policy_net.eval()
        print(f'Evaluating {path} for {args.episodes} greedy episodes...')

        env = SnakeEnv(grid_size=20)
        scores, rewards, steps_list, deaths = evaluate(
            agent, env, args.episodes, verbose=False)
        env.close()

        # ---- (0,1) eval scores per episode ----
        ax = axes[0, 1]
        ax.plot(range(1, len(scores) + 1), scores, 'o-', color='seagreen',
                markersize=4, alpha=0.6, label='Score')
        ax.plot(range(1, len(scores) + 1), moving_avg(scores, 10), color='darkgreen',
                linewidth=2, label='10-ep avg')
        ax.axhline(np.mean(scores), color='gray', linestyle=':',
                   label=f'mean = {np.mean(scores):.1f}')
        ax.set_xlabel('Eval episode')
        ax.set_ylabel('Score')
        ax.set_title(f'Greedy Evaluation ({args.episodes} episodes, no exploration/dropout)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # ---- (1,0) score distribution ----
        ax = axes[1, 0]
        bins = np.arange(-0.5, max(scores) + 1.5, 1)
        ax.hist(scores, bins=bins, color='steelblue', edgecolor='black', alpha=0.75)
        ax.axvline(np.mean(scores), color='red', linestyle='--',
                   label=f'mean = {np.mean(scores):.1f}')
        ax.axvline(np.median(scores), color='orange', linestyle='--',
                   label=f'median = {np.median(scores):.1f}')
        ax.set_xlabel('Score')
        ax.set_ylabel('Episodes')
        ax.set_title('Eval Score Distribution')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # ---- (1,1) death causes ----
        ax = axes[1, 1]
        labels = ['wall', 'self', 'timeout']
        counts = [deaths[k] for k in labels]
        if sum(counts) > 0:
            ax.pie(counts, labels=[f'{k}\n{v} ({v / sum(counts) * 100:.0f}%)' for k, v in zip(labels, counts)],
                   colors=['#ff9999', '#66b3ff', '#ffcc99'],
                   explode=(0.03, 0.03, 0.03), startangle=90)
        else:
            ax.text(0.5, 0.5, 'no deaths recorded', ha='center', va='center')
        ax.set_title('Eval Death Causes')

    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f'Report saved to {args.out}')


if __name__ == '__main__':
    main()
