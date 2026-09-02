"""
D3QN Snake AI - Complete Demo and Test Script
Combines all components for testing the trained agent
"""

import numpy as np
from snake_env import SnakeEnv
from d3qn_agent import D3QNAgent


def test_trained_agent():
    """Test a trained agent playing Snake"""
    
    print("\n" + "="*60)
    print("🐍 Testing Trained D3QN Agent")
    print("="*60)
    
    # Load or create agent
    try:
        agent = D3QNAgent()
        checkpoint_path = 'models/d3qn_snake_episode_5000.pth'
        agent.load(checkpoint_path)
        print(f"✓ Loaded trained model from {checkpoint_path}")
    except Exception as e:
        print(f"⚠ Could not load trained model: {e}")
        print("  Starting with untrained agent...")
        agent = D3QNAgent()
    
    # Create environment
    env = SnakeEnv(grid_size=20)
    
    # Play multiple episodes
    n_episodes = 10
    
    print(f"\nTesting {n_episodes} episodes...")
    print(f"Epsilon: {agent.epsilon:.4f} (exploitation mode)\n")
    
    for episode in range(1, n_episodes + 1):
        state, _ = env.reset()
        total_score = 0
        done = False
        
        while not done:
            # Render game
            env.render()
            
            # Select action (exploit only)
            action = agent.select_action(state, train=False)
            
            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)
            
            state = next_state
            done = terminated or truncated
            
            if done:
                env.close()
                print(f"Episode {episode:2d}: Score={total_score}, Reward={reward:.1f}")
                
                # Start new episode
                state, _ = env.reset()
                env.screen.fill((0, 0, 0))
                pygame.display.update()
                total_score = 0


def play_with_random_agent():
    """Play with random actions to test environment"""
    
    print("\n" + "="*60)
    print("🎮 Random Play Demo (to test environment)")
    print("="*60)
    print("\nThis demonstrates basic environment functionality.")
    print("\nPress Ctrl+C to stop...\n")
    
    env = SnakeEnv(grid_size=20)
    state, _ = env.reset()
    done = False
    
    try:
        while True:
            env.render()
            
            # Random action
            action = np.random.randint(0, 4)
            
            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)
            
            state = next_state
            done = terminated or truncated
            
            if done:
                print(f"Game Over! Score: {state.shape[0]}")
                input("Press Enter to start new episode...")
                state, _ = env.reset()
    except KeyboardInterrupt:
        print("\nStopped by user")
        env.close()


def compare_architectures():
    """Compare different network architectures"""
    
    print("\n" + "="*60)
    print("🔬 Network Architecture Comparison")
    print("="*60)
    
    # Import needed modules
    from d3qn_network import D3QN
    import torch
    
    architectures = [
        ("Simple MLP", {'input_dim': 9, 'num_actions': 4}),
        ("Dueling CNN", {'input_dim': 9, 'num_actions': 4}),
        ("Larger Dueling", {'input_dim': 9, 'num_actions': 4, 'fc1_dim': 256, 'fc2_dim': 256})
    ]
    
    for name, kwargs in architectures:
        net = D3QN(**kwargs)
        params = sum(p.numel() for p in net.parameters())
        
        # Forward pass test
        x = torch.randn(32, 9)
        with torch.no_grad():
            out = net(x)
        
        print(f"\n{name}:")
        print(f"  Parameters: {params:,}")
        print(f"  Output shape: {out.shape}")
    
    print("\n✅ Architecture comparison complete!")


if __name__ == '__main__':
    import pygame
    pygame.init()
    
    # Choose which demo to run
    choice = input("""
Choose demonstration:
1. Test trained agent (requires saved model)
2. Test with random agent
3. Compare architectures
Enter your choice [1/2/3]: """)
    
    if choice == '1':
        test_trained_agent()
    elif choice == '2':
        play_with_random_agent()
    elif choice == '3':
        compare_architectures()
    else:
        print("Invalid choice, running architecture comparison...")
        compare_architectures()
    
    pygame.quit()
