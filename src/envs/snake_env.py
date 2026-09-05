"""
D3QN Snake Game Environment
A simple environment for training D3QN agent to play Snake
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame

# Per-step directional guidance: a normal move that shortens the Manhattan
# distance to the food earns +COEFF/len, one that lengthens it loses the same.
# Dividing by snake length fades the signal as the snake grows: it bootstraps
# food-seeking early, but stops pushing long snakes to beeline through their
# own body (late game the scaled food/death rewards dominate instead).
# COEFF=0.3 puts the signal at the ±0.1 living-cost magnitude once the snake
# reaches length 3, and at ±0.01 by length 30.
STEP_GUIDANCE_COEFF = 0.3


class SnakeEnv(gym.Env):
    """Simple Snake game environment"""
    
    def __init__(self, grid_size=20, observation_type='vision',
                 reward_shaping='scaled', self_death_factor=1.5):
        super(SnakeEnv, self).__init__()

        self.grid_size = grid_size
        self.observation_type = observation_type
        # 'scaled': size-dependent shaping — the longer the snake, the bigger
        #            the food reward and the milder the death penalty, so the
        #            learning signal strengthens as the game progresses
        # 'flat':   legacy fixed +10 food / -10 death (only via explicit opt-in)
        self.reward_shaping = reward_shaping
        # Self-collision deaths are penalized harder than wall deaths by this
        # factor: in late game nearly every death is self-collision, and the
        # distinct penalty forces Q(s,a) to represent WHICH fatal obstacle an
        # action runs into (walls never move; the body does).
        self.self_death_factor = float(self_death_factor)
        # 'wall' | 'self' | 'timeout' of the last step; None if it survived.
        # Lets trainers/loggers attribute deaths without re-deriving them.
        self.last_death_cause = None
        self.close_requested = False  # set True when user closes the game window
        
        # Observation space: vision-based (local view around snake head)
        if observation_type == 'vision':
            self.observation_space = spaces.Box(
                low=-1, high=2, shape=(10,), dtype=np.float32
            )
        elif observation_type == 'grid':
            # Full-board 3-channel image for CNN: body / food / head
            self.observation_space = spaces.Box(
                low=0, high=1, shape=(3, grid_size, grid_size), dtype=np.float32
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
        self.last_death_cause = None
        
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
            self.last_death_cause = 'wall'
            reward = self._death_reward('wall')

        # Check collision with self
        elif new_head in self.snake[:-1]:
            done = True
            self.last_death_cause = 'self'
            reward = self._death_reward('self')

        # Check timeout
        elif self.steps_without_food >= self.max_steps:
            done = True
            self.last_death_cause = 'timeout'
            reward = -1.0

        # Check if ate food
        elif new_head == self.food:
            self.snake.insert(0, new_head)
            self.score += 1
            self.food = self._place_food()
            self.steps_without_food = 0
            self.last_death_cause = None
            reward = self._food_reward()
        else:
            # Normal move
            self.snake.insert(0, new_head)
            self.snake.pop()
            self.last_death_cause = None
            # No flat living cost: with γ=0.99 a -0.1/step cost accumulates
            # to ≈ -10 across a full grid×grid episode — worse than the
            # ≈ -9.5 early-death penalty, which made "charge the nearest
            # wall" the reward-optimal policy for a food-blind snake
            # (observed: 20×20 run collapsed to 100% wall deaths at
            # ep 14500+). Directional guidance alone provides the per-step
            # pressure, matching the reference implementation.
            reward = self._step_guidance(head, new_head)
        
        obs = self._get_observation()
        truncated = False
        return obs, reward, done, truncated, {}

    def _death_reward(self, cause):
        """Collision penalty, differentiated by cause. 'scaled': shrinks with
        snake length — an early death costs almost -10, dying once the snake
        spans the board edge costs only -1. Self-collision is further scaled
        by self_death_factor so the two failure modes get distinct values."""
        if self.reward_shaping != 'scaled':
            base = -10.0
        else:
            frac = min(1.0, len(self.snake) / self.grid_size)
            base = -(1.0 + 9.0 * (1.0 - frac))
        if cause == 'self' and self.self_death_factor != 1.0:
            return base * self.self_death_factor
        return base

    def _food_reward(self):
        """Food reward. 'scaled': grows with snake length (~+1 early → +10
        once the snake spans the board edge) so late-game food justifies
        risky maneuvers. Called after the new head is inserted."""
        if self.reward_shaping != 'scaled':
            return 10.0
        frac = min(1.0, len(self.snake) / self.grid_size)
        return 1.0 + 9.0 * frac

    def _step_guidance(self, old_head, new_head):
        """Direction bonus for a non-eating move: positive when the move
        shortened the Manhattan distance to the food, negative otherwise.
        Magnitude is STEP_GUIDANCE_COEFF / snake_length, so the guidance
        fades as the snake grows. The food never moves on a normal step,
        so comparing distances to the same food cell is exact. With
        4-directional moves the Manhattan distance always changes by ±1,
        so every normal move gets a decisive nonzero signal."""
        before = (abs(old_head[0] - self.food[0])
                  + abs(old_head[1] - self.food[1]))
        after = (abs(new_head[0] - self.food[0])
                 + abs(new_head[1] - self.food[1]))
        amount = STEP_GUIDANCE_COEFF / max(1, len(self.snake))
        return amount if after < before else -amount

    def _place_food(self):
        """Place food at random position not occupied by snake"""
        while True:
            pos = (np.random.randint(0, self.grid_size), 
                   np.random.randint(0, self.grid_size))
            if pos not in self.snake:
                return pos
    
    def _get_observation(self):
        """Dispatch to the configured observation encoder"""
        if self.observation_type == 'grid':
            return self._get_grid_observation()
        return self._get_vision_observation()

    def _get_grid_observation(self):
        """Full-board 3-channel image: ch0 snake body (head excluded),
        ch1 food, ch2 head. Food and head are painted as 3x3 blobs: a
        single cell gets aliased away by the stride-2 conv stack — with
        point food a supervised probe could not learn the food's
        row-direction at all (26% accuracy, chance level) vs 97% with
        blobs, while large structures (walls of body) learned fine."""
        obs = np.zeros((3, self.grid_size, self.grid_size), dtype=np.float32)
        head = self.snake[0]
        for cell in self.snake:
            obs[0, cell[0], cell[1]] = 1.0
        obs[0, head[0], head[1]] = 0.0  # head lives in its own channel
        self._paint_blob(obs, 1, self.food)
        self._paint_blob(obs, 2, head)
        return obs

    def _paint_blob(self, obs, channel, pos):
        """Paint pos as a 3x3 blob, clipped at the walls."""
        r, c = pos
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < self.grid_size and 0 <= cc < self.grid_size:
                    obs[channel, rr, cc] = 1.0

    def _get_vision_observation(self):
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
            
            # +1: a wall can sit exactly grid_size steps away (head on the
            # opposite border) and must still be detected
            for step in range(1, self.grid_size + 1):
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
