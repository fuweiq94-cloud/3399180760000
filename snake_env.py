"""
D3QN Snake Game Environment
A simple environment for training D3QN agent to play Snake
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame


class SnakeEnv(gym.Env):
    """Simple Snake game environment"""
    
    def __init__(self, grid_size=20, observation_type='vision'):
        super(SnakeEnv, self).__init__()
        
        self.grid_size = grid_size
        self.close_requested = False  # set True when user closes the game window
        
        # Observation space: vision-based (local view around snake head)
        if observation_type == 'vision':
            self.observation_space = spaces.Box(
                low=-1, high=2, shape=(10,), dtype=np.float32
            )
        elif observation_type == 'state':
            self.observation_space = spaces.Box(
                low=0, high=grid_size, shape=(grid_size, grid_size), dtype=np.int32
            )
        
        # Action space: Up, Down, Left, Right
        self.action_space = spaces.Discrete(4)
        
        # Direction mappings
        self.directions = {
            0: np.array([-1, 0]),   # Up
            1: [1, 0],              # Down
            2: [0, -1],             # Left
            3: [0, 1]               # Right
        }
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize snake in the middle of the grid
        self.snake = [(self.grid_size // 2, self.grid_size // 2)]
        self.food = self._place_food()
        self.score = 0
        self.steps_without_food = 0
        self.max_steps = self.grid_size * self.grid_size
        
        # Get initial observation
        obs = self._get_observation()
        
        return obs, {}
    
    def step(self, action):
        # Move snake
        head = self.snake[0]
        direction = self.directions[action]
        new_head = (head[0] + direction[0], head[1] + direction[1])
        
        # Update steps without food counter
        if new_head != self.food:
            self.steps_without_food += 1
        
        done = False
        
        # Check collision with walls
        if (not 0 <= new_head[0] < self.grid_size or 
            not 0 <= new_head[1] < self.grid_size):
            done = True
            reward = -10.0
        
        # Check collision with self
        elif new_head in self.snake[:-1]:
            done = True
            reward = -10.0
        
        # Check timeout
        elif self.steps_without_food >= self.max_steps:
            done = True
            reward = -1.0
        
        # Check if ate food
        elif new_head == self.food:
            self.snake.insert(0, new_head)
            self.score += 1
            self.food = self._place_food()
            self.steps_without_food = 0
            reward = 10.0
        else:
            # Normal move
            self.snake.insert(0, new_head)
            self.snake.pop()
            reward = -0.1  # Small penalty to encourage finding food faster
        
        obs = self._get_observation()
        truncated = False
        return obs, reward, done, truncated, {}
    
    def _place_food(self):
        """Place food at random position not occupied by snake"""
        while True:
            pos = (np.random.randint(0, self.grid_size), 
                   np.random.randint(0, self.grid_size))
            if pos not in self.snake:
                return pos
    
    def _get_observation(self):
        """Get vision-based observation (8 directions + 2-axis food direction)"""
        head = self.snake[0]
        obs = np.zeros(10, dtype=np.float32)
        
        # Check 8 directions
        directions_8 = [
            (-1, 0), (1, 0), (0, -1), (0, 1),           # N, S, W, E
            (-1, -1), (-1, 1), (1, -1), (1, 1)          # NW, NE, SW, SE
        ]
        
        for i, (dx, dy) in enumerate(directions_8):
            # Check if there's a wall/obstacle in this direction
            obstacle_found = False
            dist = 0
            
            for step in range(1, self.grid_size):
                check_pos = (head[0] + dx * step, head[1] + dy * step)
                
                # Wall detection
                if not (0 <= check_pos[0] < self.grid_size and 
                        0 <= check_pos[1] < self.grid_size):
                    obstacle_found = True
                    dist = step
                    break
                
                # Snake body detection
                if check_pos in self.snake:
                    obstacle_found = True
                    dist = step
                    break
            
            # Set observation value
            if obstacle_found:
                obs[i] = -dist  # Negative value indicates obstacle
            else:
                obs[i] = 0      # No obstacle in this direction
        
        # Food direction encoding
        food_x, food_y = self.food
        head_x, head_y = head
        
        if food_x < head_x:
            obs[8] -= 1  # Food is above
        elif food_x > head_x:
            obs[8] += 1  # Food is below

        if food_y < head_y:
            obs[9] -= 1  # Food is to the left
        elif food_y > head_y:
            obs[9] += 1  # Food is to the right

        return obs
    
    def render(self, mode='human'):
        """Render the game state using pygame"""
        if mode == 'human':
            if not hasattr(self, 'screen'):
                pygame.init()
                self.screen = pygame.display.set_mode((400, 400))
                pygame.display.set_caption('D3QN Snake AI')
                self.clock = pygame.time.Clock()
                self.font = pygame.font.Font(None, 36)
            
            # Process OS events; closing the window requests a graceful training stop
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close_requested = True
            self.screen.fill((0, 0, 0))
            
            # Draw grid
            cell_size = 400 // self.grid_size
            for x in range(self.grid_size + 1):
                pygame.draw.line(self.screen, (50, 50, 50), 
                                (x * cell_size, 0), 
                                (x * cell_size, 400))
                pygame.draw.line(self.screen, (50, 50, 50), 
                                (0, x * cell_size), 
                                (400, x * cell_size))
            
            # Draw snake
            for i, (x, y) in enumerate(self.snake):
                color = (0, 255, 0) if i == 0 else (0, 200, 0)
                rect = pygame.Rect(x * cell_size, y * cell_size, cell_size-1, cell_size-1)
                pygame.draw.rect(self.screen, color, rect)
            
            # Draw food
            fx, fy = self.food
            food_rect = pygame.Rect(fx * cell_size, fy * cell_size, cell_size-1, cell_size-1)
            pygame.draw.rect(self.screen, (255, 0, 0), food_rect)
            
            # Draw score
            score_text = self.font.render(f'Score: {self.score}', True, (255, 255, 255))
            self.screen.blit(score_text, (10, 10))
            
            # Draw steps
            steps_text = self.font.render(f'Steps: {self.steps_without_food}', True, (255, 255, 255))
            self.screen.blit(steps_text, (10, 50))
            
            # Draw current episode number (kept up to date by the trainer)
            episode_text = self.font.render(f'Episode: {getattr(self, "current_episode", 1)}', True, (255, 255, 255))
            self.screen.blit(episode_text, (10, 90))

            # Draw current exploration rate (kept up to date by the trainer)
            epsilon_text = self.font.render(f'Epsilon: {getattr(self, "current_epsilon", 1.0):.3f}', True, (255, 255, 0))
            self.screen.blit(epsilon_text, (10, 130))
            
            pygame.display.flip()
            # FPS can be adjusted at runtime (e.g. by the GUI speed slider)
            self.clock.tick(getattr(self, 'render_fps', 20))
    
    def close(self):
        """Close pygame window"""
        if hasattr(self, 'screen'):
            pygame.quit()


if __name__ == '__main__':
    # Test the environment
    env = SnakeEnv(grid_size=20)
    
    print("Testing Snake Environment...")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    obs, _ = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial observation: {obs}")
    
    # Random play to test rendering
    env.reset()
    print("\nPress Ctrl+C to stop...")
    
    try:
        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            
            if terminated or truncated:
                print(f"\nGame Over! Score: {env.score}")
                obs, _ = env.reset()
    except KeyboardInterrupt:
        print("\nStopped by user")
        env.close()
