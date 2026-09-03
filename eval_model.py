"""
Evaluate a trained D3QN Snake model.

Runs the trained agent in pure exploitation mode (no exploration, dropout off)
for N episodes, reports score statistics / death causes, and compares against
a random-action baseline of the same size.

Usage:
    python eval_model.py                     # 100 episodes, latest checkpoint
    python eval_model.py -n 50               # fewer episodes
    python eval_model.py --model models/d3qn_snake_episode_5000.pth
    python eval_model.py --render            # watch the games
"""

import argparse
import glob
import os
import random
import re

import numpy as np

from snake_env import SnakeEnv
from d3qn_agent import D3QNAgent


def find_latest_checkpoint(checkpoint_dir='models'):
    """Find the newest d3qn_snake_episode_NNNN.pth (same logic as train.py)."""
    if not os.path.isdir(checkpoint_dir):
        return None
    candidates = []
    for f in glob.glob(os.path.join(checkpoint_dir, 'd3qn_snake_episode_*.pth')):
        m = re.search(r'episode_(\d+)\.pth$', f)
        if m:
            candidates.append((int(m.group(1)), f))
    return max(candidates) if candidates else None


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


def evaluate(agent, env, n_episodes, render=False, verbose=True):
    """Run n_episodes with the agent's greedy policy, collect stats."""
    scores, rewards, steps_list = [], [], []
    deaths = {'wall': 0, 'self': 0, 'timeout': 0}

    for ep in range(1, n_episodes + 1):
        state, _ = env.reset()
        done = False
        total_reward, steps = 0.0, 0

        while not done:
            action = agent.select_action(state, train=False)
            cause = classify_death(env, action)
            state, r, terminated, truncated, _ = env.step(action)
            total_reward += r
            steps += 1
            done = terminated or truncated
            if render:
                env.render()

        scores.append(env.score)
        rewards.append(total_reward)
        steps_list.append(steps)
        if r == -1.0:  # timeout penalty is the only -1.0
            deaths['timeout'] += 1
        elif cause:
            deaths[cause] += 1
        if verbose:
            print(f"  ep {ep:3d}/{n_episodes}: score={env.score:2d}  reward={total_reward:7.1f}  steps={steps}")

    return scores, rewards, steps_list, deaths


def evaluate_random(env, n_episodes):
    """Random-action baseline of the same size."""
    scores, deaths = [], {'wall': 0, 'self': 0, 'timeout': 0}
    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = random.randint(0, 3)
            cause = classify_death(env, action)
            state, r, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        scores.append(env.score)
        if r == -1.0:
            deaths['timeout'] += 1
        elif cause:
            deaths[cause] += 1
    return scores, deaths


def print_block(title, scores, rewards, deaths):
    s = np.array(scores)
    print(f"\n{'=' * 60}")
    print(f"{title}")
    print(f"{'=' * 60}")
    print(f"  Avg score     : {s.mean():6.2f}")
    print(f"  Median        : {np.median(s):6.1f}")
    print(f"  Min / Max     : {s.min():d} / {s.max():d}")
    print(f"  Std           : {s.std():6.2f}")
    print(f"  Avg reward    : {np.mean(rewards):6.2f}")
    print(f"  Score >= 3    : {(s >= 3).sum()}/{len(s)} episodes ({(s >= 3).mean() * 100:.0f}%)")
    print(f"  Score == 0    : {(s == 0).sum()}/{len(s)} episodes ({(s == 0).mean() * 100:.0f}%)")
    n = max(1, sum(deaths.values()))
    print(f"  Death causes  : wall {deaths['wall']} ({deaths['wall'] / n * 100:.0f}%)"
          f" | self {deaths['self']} ({deaths['self'] / n * 100:.0f}%)"
          f" | timeout {deaths['timeout']} ({deaths['timeout'] / n * 100:.0f}%)")


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained D3QN snake model')
    parser.add_argument('-n', '--episodes', type=int, default=100)
    parser.add_argument('--model', type=str, default=None,
                        help='checkpoint path (default: latest in models/)')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--no-baseline', action='store_true',
                        help='skip the random-action comparison')
    args = parser.parse_args()

    path = args.model
    if path is None:
        found = find_latest_checkpoint()
        path = found[1] if found else None
    if path is None or not os.path.exists(path):
        raise SystemExit(f"No checkpoint found ({path}). Train first or pass --model.")

    agent = D3QNAgent()
    agent.load(path)
    agent.policy_net.eval()  # disable dropout: deterministic greedy policy
    print(f"Model: {path}")
    print(f"Device: {agent.device} | greedy policy (no exploration, dropout off)")

    env = SnakeEnv(grid_size=20)
    scores, rewards, steps_list, deaths = evaluate(agent, env, args.episodes, args.render)
    print_block(f"TRAINED AGENT ({args.episodes} greedy episodes)", scores, rewards, deaths)

    if not args.no_baseline:
        random_scores, random_deaths = evaluate_random(env, args.episodes)
        print_block(f"RANDOM BASELINE ({args.episodes} random episodes)", random_scores, [0] * len(random_scores), random_deaths)
        ratio = np.mean(scores) / max(1e-6, np.mean(random_scores))
        print(f"\nTrained/random avg-score ratio: {ratio:.1f}x")
        if ratio >= 3:
            print("Verdict: model clearly learned (well above random).")
        elif ratio >= 1.5:
            print("Verdict: model learned somewhat, but weakly.")
        else:
            print("Verdict: model barely better than random — likely NOT learned.")

    env.close()


if __name__ == '__main__':
    main()
