import numpy as np

class QLearningAgent:
    def __init__(self, env, epsilon=0.1, learning_rate=0.1, discount_factor=0.99):
        self.env = env
        self.epsilon = epsilon  # Exploration rate
        self.learning_rate = learning_rate  # Learning rate
        self.discount_factor = discount_factor  # Discount factor
        self.q_table = {}  # Initialize Q-table

    def learn(self, state, action, reward, next_state, done):
        # Check if the state contains the expected value
        agent_positions = np.argwhere(state == 3)
        if agent_positions.size == 0:  # Handle empty result
            print("Warning: Agent position not found in the state.")
            return  # Skip learning for this step

        agent_pos = tuple(agent_positions[0])  # Extract position
        # Perform Q-learning updates (example logic)
        if agent_pos not in self.q_table:
            self.q_table[agent_pos] = np.zeros(self.env.action_space.n)

        # Update Q-value (example logic)
        max_future_q = np.max(self.q_table.get(tuple(np.argwhere(next_state == 3)[0]), np.zeros(self.env.action_space.n)))
        current_q = self.q_table[agent_pos][action]
        self.q_table[agent_pos][action] = current_q + 0.1 * (reward + 0.99 * max_future_q - current_q)

    def choose_action(self, state):
        # Exploration vs. exploitation
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()  # Explore: random action
        else:
            agent_positions = np.argwhere(state == 3)
            if agent_positions.size == 0:
                return self.env.action_space.sample()  # Default to random action if state is invalid
            agent_pos = tuple(agent_positions[0])
            if agent_pos not in self.q_table:
                self.q_table[agent_pos] = np.zeros(self.env.action_space.n)
            return np.argmax(self.q_table[agent_pos])  # Exploit: best action