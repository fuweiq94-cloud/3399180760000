"""
D3QN Snake Training Script
Main training loop for D3QN agent to play Snake game
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from snake_env import SnakeEnv
from d3qn_agent import D3QNAgent


class Trainer:
    """Trainer class for managing the complete training process"""
    
    def __init__(self, n_episodes=10000, max_steps_per_episode=500, 
                 log_interval=10, save_interval=50):
        self.n_episodes = n_episodes
        self.max_steps = max_steps_per_episode
        
        # Create environment and agent
        self.env = SnakeEnv(grid_size=20)
        self.agent = D3QNAgent()
        
        # Logging
        self.log_interval = log_interval
        self.save_interval = save_interval
        
        # Tracking metrics
        self.rewards = []
        self.scores = []
        self.losses = []
        self.epsilon_values = []
        
        # Start step
        self.total_steps = 0
    
    def train_one_episode(self, episode_num, render=False):
        """Train for one episode"""
        
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
            total_score += 1 if terminated or truncated else 0
            
            if render:
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
        
        return total_reward, total_score
    
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
    
    def plot_training_metrics(self):
        """Plot and save training curves"""
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
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Food Eaten')
        axes[0, 1].set_title('Score (Food Eaten)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Loss curve
        axes[1, 0].plot(range(len(self.losses)), self.losses, alpha=0.3, color='blue', label='Training Loss')
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
        plt.savefig('training_curves.png', dpi=150)
        print("Training curves saved to 'training_curves.png'")
    
    def save_trained_model(self, episode):
        """Save the trained model"""
        os.makedirs('models', exist_ok=True)
        filepath = f'models/d3qn_snake_episode_{episode}.pth'
        self.agent.save(filepath)
        print(f"Model saved to {filepath}")
    
    def print_training_status(self, episode, reward, score, steps):
        """Print training status every log_interval episodes"""
        avg_reward = np.mean(self.rewards[-10:]) if len(self.rewards) >= 10 else reward
        print(f"\n{'='*60}")
        print(f"Episode: {episode:5d} | Epsilon: {self.agent.epsilon:.4f} | "
              f"Avg Reward: {avg_reward:8.2f} | Score: {score:4d} | "
              f"Steps: {steps:5d} | Current ε: {self.agent.epsilon:.4f}")
        print(f"{'='*60}\n")
    
    def train(self):
        """Run the full training loop"""
        
        print("\n" + "="*60)
        print("🐍 D3QN Snake Training Started!")
        print("="*60)
        print(f"Target episodes: {self.n_episodes}")
        print(f"Device: {self.agent.device}")
        print(f"Epsilon decay: {self.agent.epsilon_decay}")
        print(f"Save interval: every {self.save_interval} episodes")
        print("="*60)
        
        for episode in range(1, self.n_episodes + 1):
            # Start tracking
            start_time = time.time()
            
            # Train one episode
            reward, score = self.train_one_episode(episode)
            
            # Record metrics
            self.rewards.append(reward)
            self.scores.append(score)
            self.epsilon_values.append(self.agent.epsilon)
            
            # Print status
            if episode % self.log_interval == 0:
                self.print_training_status(episode, reward, score, self.total_steps)
            
            # Save model periodically
            if episode % self.save_interval == 0:
                self.save_trained_model(episode)
            
            # Check for convergence criteria
            if len(self.rewards) >= 100:
                recent_avg = np.mean(self.rewards[-100:])
                if recent_avg > 50.0:
                    print(f"\n✅ Converged! Recent 100-episode avg reward: {recent_avg:.2f}")
                    break
            
            # Optional rendering for demonstration
            if episode <= 10 or (episode % 100 == 0 and episode > 0):
                print(f"\n🎮 Playing episode {episode} (rendering)...")
                reward_demo, score_demo = self.train_one_episode(episode, render=True)
        
        # Final summary
        self._training_summary()
        
        # Plot and save metrics
        print("\n📊 Generating training plots...")
        self.plot_training_metrics()
        
        # Save final model
        self.save_trained_model(self.n_episodes)
        
        # Close environment
        self.env.close()
        
        print("\n✨ Training completed successfully!")
    
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
            print(f"Best Reward:         {max(self.rewards):.2f}")
            print(f"Avg Reward (last 100): {np.mean(self.rewards[-100:]):.2f}")
        if len(self.epsilon_values) > 0:
            print(f"Final Epsilon:       {self.epsilon_values[-1]:.4f}")
        if len(self.losses) > 0:
            print(f"Final Loss:          {self.losses[-1]:.4f}")
        print(f"Total Training Steps:{self.total_steps:,}")
        print(f"{'='*60}\n")


def main():
    """Main entry point"""
    trainer = Trainer(
        n_episodes=5000,     # Number of training episodes
        max_steps_per_episode=500,
        log_interval=10,
        save_interval=50
    )
    trainer.train()


if __name__ == '__main__':
    main()
