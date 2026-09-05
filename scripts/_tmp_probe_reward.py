# Throwaway: validate the no-living-cost reward — fresh CNN, watch for the
# mid-run slide-to-wall-collapse that killed the previous 20x20 run.
# Usage: python _tmp_probe_reward.py <grid> <episodes> <tag>
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import train as train_mod

grid, n_eps, tag = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
tmp = Path(tempfile.mkdtemp(prefix=f'probe_{tag}_'))
train_mod.PROJECT_ROOT = tmp

t = train_mod.Trainer(n_episodes=n_eps, log_interval=100, save_interval=100000,
                      resume=False, obs_type='grid', grid_size=grid,
                      n_step=3, preview_interval=0)

print(f"[{tag}] grid={grid} decay={t.agent.epsilon_decay} "
      f"floor={t.agent.epsilon_end}", flush=True)
scores = []
for ep in range(1, n_eps + 1):
    _, s = t.train_one_episode(ep)
    scores.append(s)
    if ep % 100 == 0:
        c = scores[-100:]
        print(f"[{tag}] ep {ep:4d}: avg100 {sum(c)/100:5.2f}  best100 "
              f"{max(c):2d}  eps {t.agent.epsilon:.3f}  "
              f"deaths(last100) wall {t.death_counts.get('wall',0)} "
              f"self {t.death_counts.get('self',0)} "
              f"timeout {t.death_counts.get('timeout',0)}", flush=True)
        t.death_counts = {'wall': 0, 'self': 0, 'timeout': 0}
