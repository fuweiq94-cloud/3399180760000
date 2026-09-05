"""
D3QN Agent - Dueling Double Deep Q-Network with Experience Replay
Main training agent that implements the D3QN algorithm
"""

import torch
import torch.nn as nn
import numpy as np
from collections import deque, namedtuple
from models import D3QN, D3QNCNN
import random


Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])


class D3QNAgent:
    """
    D3QN Agent with Double Q-learning and Dueling Architecture
    
    Key components:
    - Main network (online): selects actions and computes Q-values
    - Target network: provides stable target values for training
    - Experience replay: stores past experiences for training
    - Optional n-step returns: bootstrap targets look n steps ahead, which
      propagates delayed death-by-self-trap credit back to the fatal decision
    """
    
    def __init__(self, input_dim=10, num_actions=4, obs_type='vision',
                 n_step=1, buffer_size=100000, batch_size=64, grid_size=20,
                 epsilon_decay=0.995, epsilon_end=None,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.obs_type = obs_type
        self.input_dim = input_dim
        self.grid_size = grid_size

        # Hyperparameters
        self.gamma = 0.99          # Discount factor
        self.epsilon_start = 1.0   # Initial exploration rate
        # Pixel CNNs infer food geometry from raw boards — far slower than
        # the 10-dim vision features — so they keep a higher exploration
        # floor: at 0.05 the greedy policy froze into wall-avoiding circles
        # long before food-seeking emerged (observed on 30×30: floor hit at
        # ~ep 3000 with avg score still ≈ 0.1, then 17k episodes of
        # self-reinforcement).
        if epsilon_end is None:
            epsilon_end = 0.10 if obs_type == 'grid' else 0.05
        self.epsilon_end = epsilon_end  # Final exploration rate
        # 0.995/episode fits the fast-learning feature model; pixel CNNs
        # need a longer exploration tail (e.g. 0.9997 keeps ε>0.3 until
        # ~ep 4000 and reaches the 0.10 floor near ep 7700)
        self.epsilon_decay = epsilon_decay
        
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.target_update = 1000  # legacy field (params.json); sync is now soft
        self.tau = 0.005           # per-step Polyak coefficient for the target net
        
        # n-step returns (1 = classic one-step TD). Plain deque: the oldest
        # full window is emitted on the (n+1)-th push, so episode-end flush
        # never re-emits an already composed transition.
        self.n_step = n_step
        self._nstep_buf = deque()
        
        # Networks
        if obs_type == 'grid':
            net_fn = lambda: D3QNCNN(num_actions=num_actions, grid_size=grid_size)
        else:
            net_fn = lambda: D3QN(input_dim=input_dim, num_actions=num_actions)
        self.policy_net = net_fn().to(device)
        self.target_net = net_fn().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.criterion = nn.MSELoss()
        
        # Experience replay
        self.memory = deque(maxlen=self.buffer_size)
        self.steps = 0
        
        # Epsilon scheduling
        self.epsilon = self.epsilon_start
    
    def select_action(self, state, train=True):
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current observation (numpy array or tensor)
            train: Whether to use exploration or exploitation
            
        Returns:
            Action index
        """
        if train and np.random.rand() < self.epsilon:
            # Exploration: random action
            return np.random.randint(0, self.policy_net.num_actions)
        else:
            # Exploitation: best action according to policy
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store experience in replay buffer (composed into n-step returns)"""
        if self.n_step <= 1:
            self.memory.append(Transition(state, action, reward, next_state, done))
            return
        self._nstep_buf.append((state, action, reward, next_state, done))
        if len(self._nstep_buf) > self.n_step:
            self.memory.append(self._compose_nstep(self.n_step))
            self._nstep_buf.popleft()

    def _compose_nstep(self, k):
        """k-step return from the oldest k buffered transitions:
        R = r_t + γ r_{t+1} + ... + γ^{k-1} r_{t+k-1}, bootstrap state s_{t+k}"""
        R = sum((self.gamma ** i) * self._nstep_buf[i][2] for i in range(k))
        s0, a0 = self._nstep_buf[0][0], self._nstep_buf[0][1]
        next_state, done = self._nstep_buf[k - 1][3], self._nstep_buf[k - 1][4]
        return Transition(s0, a0, R, next_state, done)

    def _flush_nstep(self):
        """At episode end, emit the remaining k<n partial transitions"""
        while self._nstep_buf:
            self.memory.append(self._compose_nstep(len(self._nstep_buf)))
            self._nstep_buf.popleft()
    
    def step(self):
        """Per-step bookkeeping: step counter + soft (Polyak) target sync.

        The old hard copy every `target_update` steps gave the bootstrap no
        correction between swaps: once exploration thinned (ε < ~0.25) the
        narrowed replay data couldn't anchor Q-values against each 1000-step
        jump, and performance collapsed (20×20 peaked at avg 0.92 near
        ε 0.35, fell to 0.05 by ε 0.10, never recovered). A per-step soft
        update (τ=0.005, effective lag ~200 steps) keeps the target
        continuously honest."""
        self.steps += 1
        with torch.no_grad():
            for tp, pp in zip(self.target_net.parameters(),
                              self.policy_net.parameters()):
                tp.mul_(1.0 - self.tau).add_(pp, alpha=self.tau)
    
    def end_episode(self):
        """Decay epsilon once per episode and flush partial n-step transitions"""
        self._flush_nstep()
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def optimize_model(self):
        """
        Train the policy network using sampled transitions
        
        Implements Double DQN update:
        - Use policy_net to select best action
        - Use target_net to evaluate the selected action
        """
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch of transitions
        transitions = random.sample(self.memory, self.batch_size)
        batch = Transition(*zip(*transitions))
        
        # Prepare tensors (rewards/dones as [B,1] columns to avoid
        # broadcasting with next_q's [B,1] into a wrong [B,B] target)
        states = torch.FloatTensor(np.array(batch.state)).to(self.device)
        actions = torch.LongTensor(batch.action).to(self.device)
        rewards = torch.FloatTensor(np.array(batch.reward)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        dones = torch.FloatTensor(np.array(batch.done)).unsqueeze(1).to(self.device)
        
        # Current Q-values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Compute targets using Double DQN formula
        # Step 1: Use policy_net to select best action at next state
        best_actions = self.policy_net(next_states).argmax(dim=1)
        
        # Step 2: Use target_net to evaluate selected actions
        with torch.no_grad():
            next_q = self.target_net(next_states).gather(1, best_actions.unsqueeze(1))
            
        # Calculate targets. Stored rewards are already n-step discounted sums,
        # so the bootstrap uses gamma^n (gamma^1 = classic one-step TD)
        targets = rewards + (1 - dones) * (self.gamma ** self.n_step) * next_q
        
        # Compute loss
        loss = self.criterion(current_q, targets)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        
        self.optimizer.step()
        
        return loss.item()
    
    def save(self, path):
        """Save model checkpoint"""
        torch.save({
            'epsilon': self.epsilon,
            'steps': self.steps,
            'model_dict': self.policy_net.state_dict(),
            'optimizer_dict': self.optimizer.state_dict()
        }, path)
    
    def load(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        self.policy_net.load_state_dict(checkpoint['model_dict'])
        self.target_net.load_state_dict(checkpoint['model_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_dict'])


class MemoryReplayBuffer:
    """Enhanced experience replay buffer with prioritized sampling (optional)"""
    
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
    
    def add(self, priority, *args):
        """Add new experience"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = args
        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, n_samples):
        """Sample random batch"""
        indices = np.random.choice(len(self.buffer), n_samples)
        samples = [self.buffer[i] for i in indices]
        return zip(*samples)
    
    def update_priorities(self, indices, priorities):
        """Update priorities for sampled transitions"""
        for idx, pri in zip(indices, priorities):
            self.priorities[idx] = pri


def test_agent():
    """Test the D3QN agent"""
    print("Testing D3QN Agent...")
    
    # Create agent
    agent = D3QNAgent(input_dim=10, num_actions=4)
    print(f"Agent created on device: {agent.device}")
    print(f"Policy net parameters: {sum(p.numel() for p in agent.policy_net.parameters()):,}")
    
    # Test action selection
    state = np.random.randn(10).astype(np.float32)
    action = agent.select_action(state, train=True)
    print(f"\nSelected action: {action}")
    
    # Test storing transitions
    next_state = np.random.randn(10).astype(np.float32)
    agent.store_transition(state, action, 1.0, next_state, False)
    print(f"\nStored transition. Buffer size: {len(agent.memory)}")
    
    # Test optimization step
    for _ in range(10):
        loss = agent.optimize_model()
        if loss is not None:
            print(f"Loss: {loss:.4f}")
    
    print("\n✓ All tests passed!")


if __name__ == '__main__':
    test_agent()
