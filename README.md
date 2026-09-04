# D3QN Snake AI Training System 🐍🤖

A complete implementation of **D3QN (Dueling Double Deep Q-Network)** trained to play the classic Snake game, with a well-organized modular project structure.

## 📁 Project Structure

```
D3QN/
├── src/                          # Source code - Modular package structure
│   ├── envs/                     # Environment definitions
│   │   ├── __init__.py
│   │   └── snake_env.py          # Snake game environment (Gymnasium)
│   │
│   ├── models/                   # Neural network architectures
│   │   ├── __init__.py
│   │   └── d3qn_network.py       # D3QN neural network
│   │
│   ├── agents/                   # Reinforcement learning agents
│   │   ├── __init__.py
│   │   └── d3qn_agent.py         # D3QN agent with replay buffer
│   │
│   ├── utils/                    # Utilities and helper tools
│   │   ├── __init__.py
│   │   ├── gui.py                # Tkinter GUI interface
│   │   └── visualize.py          # Training visualization tools
│   │
│   ├── data/                     # Data processing modules (future use)
│   │   └── __init__.py
│   │
│   └── __init__.py               # Package initialization
│
├── scripts/                      # Launch and training scripts
│   ├── start.ps1                 # PowerShell launcher ⭐ Recommended
│   ├── start_training.bat        # Batch file launcher
│   ├── train.py                  # Main training script
│   ├── demo.py                   # Demo and testing script
│   ├── eval_model.py             # Model evaluation script
│   └── 停止训练.bat               # Stop training script
│
├── models/                       # Saved model checkpoints
│   └── *.pth                     # 101 PyTorch model files
│
├── docs/                         # Documentation
│   ├── README.md                 # Main documentation
│   └── 启动指南.md                # Startup guide
│
├── output/                       # Generated outputs
│   ├── training_curves.png       # Training metrics visualization
│   └── training_report.png       # Performance report charts
│
├── tests/                        # Test files (reserved)
├── skills/                       # Agent skills
│   └── src-hunter/
│
├── .venv/                        # Python virtual environment
├── requirements.txt              # Python dependencies
├── .gitignore
└── README.md                     # This file
```

## ✨ Key Features

- ✅ **Modular Architecture**: Clean separation of concerns (envs, models, agents, utils)
- ✅ **D3QN Algorithm**: Advanced deep RL combining Dueling DQN + Double DQN
- ✅ **Visualization**: Real-time pygame rendering during training
- ✅ **Comprehensive Logging**: Track rewards, scores, losses, and exploration rate
- ✅ **Model Persistence**: Save/load trained models for later use
- ✅ **GUI Support**: Tkinter-based control panel

## 🚀 Quick Start

### Prerequisites

- Python 3.14+
- [uv](https://github.com/astral-sh/uv) package manager
- Windows OS (for batch scripts)

### Setup

1. **Create Virtual Environment** (if not exists):
   ```powershell
   if (-not (Test-Path ".venv")) {
       uv venv --python 3.14
   }
   ```

2. **Install Dependencies**:
   ```powershell
   uv pip install torch numpy gymnasium matplotlib pygame
   ```

## Running the Project

### Method 1: PowerShell Script (Recommended) ⭐

```powershell
cd scripts
.\start.ps1 -Train      # Start training
.\start.ps1 -Demo       # Run demo mode  
.\start.ps1 -Install    # Install dependencies only
.\start.ps1             # Default: starts training
```

### Method 2: Batch File

```batch
cd scripts
.\start_training.bat    # Start training
```

### Method 3: Direct Execution

```powershell
# From project root
cd scripts
python train.py         # Train the agent
python demo.py          # Test the trained agent
python eval_model.py    # Evaluate performance
```

## 🎯 What is D3QN?

**D3QN (Dueling Double DQN)** combines two advanced techniques:

1. **Double DQN**: Decouples action selection from evaluation to reduce overestimation bias
2. **Dueling Networks**: Separates value estimation V(s) and advantage estimation A(s,a)

**Network Architecture:**
```
Input (10 dims) → Conv Layers → Shared FC Layer
                                   ├→ Value Stream → V(s)
                                   └→ Advantage Stream → A(s,a)
Output: Q(s,a) = V(s) + (A(s,a) - mean(A))
```

## 🐍 How It Works

The agent learns to play Snake by:

1. **Observation**: Vision-based input (10 dimensions)
   - 8 directions: proximity to walls/snake body (negative values)
   - 2 dimensions: food direction relative to snake head

2. **Actions**: [Up, Down, Left, Right]

3. **Rewards** (default `scaled` mode, 按蛇长缩放):
   - Eat food: `+1 → +10`，蛇越长奖励越高
   - Collision: `-10 → -1`，蛇越长惩罚越轻；撞自己再乘 1.5 倍（区分两种死法）
   - Normal move: **± `0.3/蛇长`**（靠近食物加分、远离扣分，蛇越长引导越弱；无固定步惩罚——步惩罚累计会超过死亡惩罚，诱发“撞墙自杀”）
   - -1.0: Timeout penalty

After ~5000 episodes, the agent typically achieves:
- ✅ Average score: 8+ food per game
- ✅ Intelligent obstacle avoidance
- ✅ Efficient pathfinding

## 📊 Training Output

During training you'll see real-time progress including episode number, epsilon (exploration rate), average reward, score, and steps. Training curves are automatically saved to `output/training_curves.png`.

## 💡 Usage Examples

### Train from scratch

```bash
cd scripts
python train.py --n_episodes 5000
```

### Resume training

```bash
cd scripts
python train.py --resume  # Auto-loads latest checkpoint
```

### Test trained agent

```bash
cd scripts
python demo.py            # Interactive menu
```

### Evaluate performance

```bash
cd scripts
python eval_model.py -n 100 --render   # 100 episodes with visualization
```

## 🔧 Configuration

### Training Parameters (in `scripts/train.py`)

```python
trainer = Trainer(
    n_episodes=5000,           # Number of training episodes
    max_steps_per_episode=500, # Maximum steps per episode
    log_interval=10,           # Print status every N episodes
    save_interval=50,          # Save model every N episodes
    render_training=True,      # Enable visualization during training
    resume=True                # Auto-resume from checkpoint
)
```

### Agent Hyperparameters (in `src/agents/d3qn_agent.py`)

```python
self.gamma = 0.99            # Discount factor
self.epsilon_start = 1.0     # Initial exploration rate
self.epsilon_end = 0.05      # Final exploration rate  
self.epsilon_decay = 0.995   # Decay rate per episode
self.batch_size = 64         # Mini-batch size
self.buffer_size = 100000    # Experience replay capacity
self.target_update = 1000    # Target network update frequency
```

## 🛠️ Advanced Tips

### Change Observation Type
```python
# In src/envs/snake_env.py
env = SnakeEnv(grid_size=20, observation_type='state')  # Full grid vs vision-based
```

### Use GPU Acceleration
PyTorch auto-detects CUDA. Check device in training output: `Device: cuda`

### Add TensorBoard Logging
```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('logs/')
writer.add_scalar('Loss', loss, global_step)
```

## 📈 Performance Benchmarks

| Episodes | Avg Score | Epsilon | Loss |
|----------|-----------|---------|------|
| 100      | 0.5       | 0.60    | 0.25 |
| 500      | 2.3       | 0.08    | 0.12 |
| 1000     | 5.8       | 0.05    | 0.08 |
| 5000     | 8.2       | 0.05    | 0.05 |

## 📚 References

- **[Dueling DQN](https://arxiv.org/abs/1511.06581)** - Wang et al., ICML 2016
- **[Double DQN](https://arxiv.org/abs/1512.06750)** - Van Hasselt et al., AAAI 2015
- **[Prioritized Replay](https://arxiv.org/abs/1511.05952)** - Schaul et al., ICLR 2016

## 🙏 Acknowledgments

Inspired by:
- Original OpenAI Snake implementations
- Alexey Mordvintsev's RL repository
- Stable Baselines3 library

---

**Happy Training!** 🎮🤖

For detailed startup instructions, see [`docs/启动指南.md`](docs/启动指南.md).
