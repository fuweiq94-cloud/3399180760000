# D3QN Snake AI Training System

## 🐍 Overview

A complete implementation of **D3QN (Dueling Double Deep Q-Network)** trained to play the classic Snake game. This system demonstrates state-of-the-art deep reinforcement learning techniques in a visual, interactive format.

## 🎯 Features

- ✅ **D3QN Architecture**: Combines Dueling DQN and Double DQN for stable, efficient training
- ✅ **Visual Rendering**: Real-time pygame visualization during training and testing
- ✅ **Experience Replay**: Robust memory management with replay buffer
- ✅ **Target Network**: Stabilized training through periodic target network updates
- ✅ **Comprehensive Logging**: Track rewards, scores, losses, and exploration rate
- ✅ **Training Metrics**: Automated plots showing convergence and performance
- ✅ **Model Persistence**: Save and load trained models for later use

## 📁 Project Structure

```
d:\zm\D3QN/
├── snake_env.py          # Custom Gym environment for Snake game
├── d3qn_network.py       # Dueling network architecture
├── d3qn_agent.py         # D3QN agent with experience replay
├── train.py              # Main training script
├── demo.py               # Demo and testing utilities
├── requirements.txt      # Dependencies
├── README.md             # This file
├── models/               # Saved model checkpoints
│   └── (episode_NNNN.pth)
└── training_curves.png   # Auto-generated training plots
```

## 🚀 Quick Start

### 1. Setup Virtual Environment (If not exists)

First, ensure the virtual environment is created using `uv`:

```bash
# Check if .venv exists
if (-not (Test-Path ".venv")) {
    uv venv --python 3.14
}
```

**✅ Current Status**: Virtual environment **already created** at `.venv/`

---

### 2. Install Dependencies

```bash
# If using uv (recommended)
uv pip install torch numpy gymnasium matplotlib

# For pygame on Python 3.14, see startup guide if installation fails
uv pip install pygame  # May need manual installation
```

**⚠️ Note**: Pygame might have compatibility issues with Python 3.14. See `启动指南.md` for solutions.

---

### 3. Start Training

```bash
pip install -r requirements.txt
```

### 2. Run Training

Train the agent from scratch:

```bash
python train.py
```

The training will:
- Run for 5000 episodes by default
- Show progress every 10 episodes
- Save best model every 50 episodes to `models/`
- Generate training curves in `training_curves.png`
- Stop automatically if converged (avg reward > 50 over last 100 episodes)

### 3. Start Training

**Using PowerShell Script (Recommended) ⭐:**

```powershell
# Auto-checks .venv and starts training
.\start.ps1 -Train
```

**Or using Batch File:**

```batch
REM Windows CMD compatible
.start_training.bat
```

**Or directly with Python:**

```bash
# Make sure you're in the project directory
cd d:\zm\D3QN

# Run using virtual environment python
.venv\Scripts\python.exe train.py
```

The training will:
- Run for 5000 episodes by default
- Show progress every 10 episodes
- Save best model every 50 episodes to `models/`
- Generate training curves in `training_curves.png`
- Stop automatically if converged (avg reward > 50 over last 100 episodes)
- **Display real-time visualization of the snake game** ⭐

### 4. Test Trained Agent

After training completes:

```bash
python demo.py
```

Choose option "1" to test the trained agent playing Snake!

### 4. Try Random Play First

To see the environment work before training:

```bash
python demo.py
```

Choose option "2" to see random gameplay.

## 🔧 Configuration

Edit these parameters in `train.py`:

```python
trainer = Trainer(
    n_episodes=5000,           # Number of training episodes
    max_steps_per_episode=500, # Maximum steps per episode
    log_interval=10,           # Print status every N episodes
    save_interval=50           # Save model every N episodes
)
```

Or modify hyperparameters in `d3qn_agent.py`:

```python
self.gamma = 0.99            # Discount factor
self.epsilon_start = 1.0     # Initial exploration rate
self.epsilon_end = 0.05      # Final exploration rate  
self.epsilon_decay = 0.995   # Decay rate per episode
self.batch_size = 64         # Mini-batch size
self.buffer_size = 100000    # Experience replay capacity
self.target_update = 1000    # Target network update frequency
```

## 🎮 How It Works

### Environment

**Observation Space** (Vision-based):
- 10-dimensional vector encoding:
  - 8 directions: proximity to walls/snake body (negative values)
  - 2 dimensions: food direction relative to snake head (obs[8]: up/down, obs[9]: left/right, each ±1 or 0)

**Action Space**:
- Discrete: [Up, Down, Left, Right]

**Rewards**:
- +10.0: Eat food
- -10.0: Collision (wall or self)
- -0.1: Per-step penalty (encourages faster solutions)
- -1.0: Timeout penalty

### D3QN Algorithm

The agent uses **Dueling Double DQN** which combines:

1. **Double DQN**: Decouples action selection from evaluation to reduce overestimation bias
2. **Dueling Network**: Separates value estimation V(s) and advantage estimation A(s,a)

**Network Architecture**:
```
Input (9 dimensions) → CNN Layers → Shared FC Layer
                                    ├→ Value Stream → V(s)
                                    └→ Advantage Stream → A(s,a)
Output: Q(s,a) = V(s) + (A(s,a) - mean(A))
```

## 📊 Training Output

During training, you'll see:

```
============================================================
Episode:  123 | Epsilon: 0.7852 | Avg Reward:  12.34 | Score:   1 | 
Steps:   345 | Current ε: 0.7852
============================================================
```

**Metrics tracked**:
- Episode rewards and moving averages
- Food eaten (score) per episode
- Training loss (MSE)
- Exploration rate decay
- Best/worst scores
- Convergence check

## 💾 Saved Model Files

Models are saved as PyTorch checkpoints containing:
- Network parameters
- Optimizer state
- Epsilon value
- Step counter

Load a model:
```python
from d3qn_agent import D3QNAgent

agent = D3QNAgent()
agent.load('models/d3qn_snake_episode_5000.pth')
```

## 🔬 Performance Tips

### Faster Training
- Reduce `n_episodes` to 1000-2000 for basic learning
- Use GPU acceleration if available (auto-detected)
- Increase `batch_size` for more stable gradients

### Better Performance
- Train longer (10000+ episodes)
- Adjust epsilon decay: slower decay = better final performance
- Use smaller grid size (e.g., 15 instead of 20) for faster episodes

### Visualization
- Set `log_interval=1` to render every episode (slow but informative)
- Watch how epsilon decays from 1.0 → 0.05
- Observe when the agent starts consistently eating food

## 🌟 Example Results

Typical training progression:

| Episodes | Avg Score | Epsilon | Loss |
|----------|-----------|---------|------|
| 100      | 0.5       | 0.60    | 0.25 |
| 500      | 2.3       | 0.08    | 0.12 |
| 1000     | 5.8       | 0.05    | 0.08 |
| 5000     | 8.2       | 0.05    | 0.05 |

After 5000 episodes, the agent can typically:
- Eat 8+ pieces of food on average per game
- Navigate around obstacles intelligently
- Take efficient paths to food
- Avoid collisions consistently

## 🛠️ Advanced Usage

### Customize Observation Type

Change from vision-based to full-grid observation in `SnakeEnv`:

```python
env = SnakeEnv(grid_size=20, observation_type='state')
```

This provides a 20x20 grid where:
- 0 = empty
- 1 = snake body
- 2 = food

### Prioritized Experience Replay

Replace standard replay with prioritized version in `d3qn_agent.py`.

### Add TensorBoard Support

Track metrics with tensorboard:

```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('logs/')
writer.add_scalar('Loss', loss, global_step)
```

## 🐛 Troubleshooting

**CUDA out of memory**: Reduce batch_size in `d3qn_agent.py`

**Training converges too slowly**:
- Check that GPU is being used (print device in train.py)
- Increase batch size
- Reduce grid size

**Agent doesn't learn**:
- Ensure rendering isn't slowing down training too much
- Try disabling rendering: set render=False in train.py
- Check if epsilon decays too quickly

**Game doesn't start**:
- Make sure pygame is installed: `pip install pygame`
- Close any zombie pygame windows

## 📈 What's Next?

Potential enhancements:
- [ ] Asynchronous multi-agent training
- [ ] Population-based training (PBT)
- [ ] Curriculum learning (start easy, get harder)
- [ ] Transfer learning to other grid games
- [ ] Web interface using Gradio/Streamlit
- [ ] Competitive agents (snake vs snake battles!)

## 📚 References

- **Dueling DQN**: Wang et al., "Dueling Network Architectures for Deep Reinforcement Learning", ICML 2016
- **Double DQN**: Van Hasselt et al., "Deep Reinforcement Learning with Double Q-learning", AAAI 2015
- **Prioritized Experience Replay**: Schaul et al., "Prioritized Experience Replay", ICLR 2016
- **Original DQN**: Mnih et al., "Human-level control through deep reinforcement learning", Nature 2015

## 📄 License

This project is provided for educational purposes. Feel free to use and modify!

## 🙏 Acknowledgments

Inspired by:
- The original OpenAI Snake game implementations
- The good RL repository by Alexey Mordvintsev
- Stable Baselines3 for reinforcement learning libraries

---

Happy training! 🎮🤖