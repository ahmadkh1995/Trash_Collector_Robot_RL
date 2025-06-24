import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import os

class GarbageCollectorEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(GarbageCollectorEnv, self).__init__()
        self.grid_size = 6
        self.window_size = 600
        self.cell_size = self.window_size // self.grid_size
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=self.grid_size-1, shape=(2,), dtype=np.int32)

        self.score = 0  # Initialize score

        if self.render_mode == 'human':
            pygame.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Garbage Collector RL")
            self.clock = pygame.time.Clock()

            asset_path = lambda name: os.path.join("assets", "icons", f"{name}.png")
            self.agent_icon = pygame.transform.scale(pygame.image.load(asset_path("agent")), (self.cell_size, self.cell_size))
            self.trash_icon = pygame.transform.scale(pygame.image.load(asset_path("trash")), (self.cell_size, self.cell_size))
            self.obstacle_icon = pygame.transform.scale(pygame.image.load(asset_path("obstacle")), (self.cell_size, self.cell_size))

        self.reset()


    def reset(self):
        self.agent_pos = np.array([0, 0])
        self.score = 0  # Reset score

        # Generate trash positions
        num_trashes = 6  # Increase number of trashes
        all_positions = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)]
        all_positions.remove((self.agent_pos[0], self.agent_pos[1]))
        trash_positions = np.random.choice(len(all_positions), num_trashes, replace=False)
        self.trash_positions = [np.array(all_positions[i]) for i in trash_positions]

        # Generate obstacle positions
        num_obstacles = 3  # Reduce number of obstacles
        remaining_positions = [
            pos for pos in all_positions
            if not any(np.array_equal(pos, trash_pos) for trash_pos in self.trash_positions)
        ]
        obstacle_positions = np.random.choice(len(remaining_positions), num_obstacles, replace=False)
        self.obstacle_positions = [np.array(remaining_positions[i]) for i in obstacle_positions]

        return self.agent_pos

    def step(self, action):
        if action == 0 and self.agent_pos[1] > 0:
            self.agent_pos[1] -= 1
        elif action == 1 and self.agent_pos[1] < self.grid_size - 1:
            self.agent_pos[1] += 1
        elif action == 2 and self.agent_pos[0] > 0:
            self.agent_pos[0] -= 1
        elif action == 3 and self.agent_pos[0] < self.grid_size - 1:
            self.agent_pos[0] += 1

        reward = -0.01
        done = False

        # Check if agent collects any trash
        for trash_pos in self.trash_positions:
            if np.array_equal(self.agent_pos, trash_pos):
                reward = 1.0
                self.trash_positions = [pos for pos in self.trash_positions if not np.array_equal(pos, trash_pos)]
                if not self.trash_positions:  # All trashes collected
                    done = True
                break

        # Check if agent hits an obstacle
        for obstacle_pos in self.obstacle_positions:
            if np.array_equal(self.agent_pos, obstacle_pos):
                reward = -1.0
                done = True
                break

        self.score += reward  # Update score
        return self.agent_pos, reward, done, {}

    # Python
    def render(self, mode="human"):
        if self.render_mode != 'human':
            return

        self.window.fill((255, 255, 255))

        for x in range(0, self.window_size, self.cell_size):
            pygame.draw.line(self.window, (200, 200, 200), (x, 0), (x, self.window_size))
        for y in range(0, self.window_size, self.cell_size):
            pygame.draw.line(self.window, (200, 200, 200), (0, y), (self.window_size, y))

        # Smooth movement
        start_pos = self.agent_icon.get_rect(
            topleft=(self.agent_pos[0] * self.cell_size, self.agent_pos[1] * self.cell_size))
        target_pos = (self.agent_pos[0] * self.cell_size, self.agent_pos[1] * self.cell_size)
        steps = 10  # Number of interpolation steps

        for i in range(steps):
            interpolated_pos = (
                start_pos.x + (target_pos[0] - start_pos.x) * (i / steps),
                start_pos.y + (target_pos[1] - start_pos.y) * (i / steps)
            )
            self.window.fill((255, 255, 255))  # Clear the screen
            for x in range(0, self.window_size, self.cell_size):
                pygame.draw.line(self.window, (200, 200, 200), (x, 0), (x, self.window_size))
            for y in range(0, self.window_size, self.cell_size):
                pygame.draw.line(self.window, (200, 200, 200), (0, y), (self.window_size, y))

            self.window.blit(self.agent_icon, interpolated_pos)
            for trash_pos in self.trash_positions:
                self.window.blit(self.trash_icon, (trash_pos[0] * self.cell_size, trash_pos[1] * self.cell_size))
            for obstacle_pos in self.obstacle_positions:
                self.window.blit(self.obstacle_icon,
                                 (obstacle_pos[0] * self.cell_size, obstacle_pos[1] * self.cell_size))

            # Render score
            font = pygame.font.Font(None, 36)
            score_text = font.render(f"Score: {self.score:.2f}", True, (0, 0, 0))
            self.window.blit(score_text, (10, 10))
            pygame.display.flip()
            self.clock.tick(30)  # Control animation speed


    def close(self):
        if self.render_mode == 'human':
            pygame.quit()
