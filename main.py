import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QRadioButton, QPushButton, QWidget, QMessageBox
from environment.garbage_collector_env import GarbageCollectorEnv
from agent.Q_learning_agent import QLearningAgent
from agent.DQN_learning_agent import DQNAgent
import pygame

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Garbage Collector Robot")

        # Central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        # Radio buttons for algorithm selection
        self.q_learning_radio = QRadioButton("Q-Learning")
        self.q_learning_radio.setChecked(True)  # Set Q-Learning as default
        self.dqn_radio = QRadioButton("DQN")
        layout.addWidget(self.q_learning_radio)
        layout.addWidget(self.dqn_radio)

        # Start button
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_program)
        layout.addWidget(self.start_button)

    def start_program(self):
        if self.q_learning_radio.isChecked():
            selected_algorithm = "Q-Learning"
        elif self.dqn_radio.isChecked():
            selected_algorithm = "DQN"
        else:
            QMessageBox.critical(self, "Error", "Please select a learning algorithm!")
            return

        self.close()  # Close the first UI

        env = GarbageCollectorEnv(render_mode='human')

        if selected_algorithm == "Q-Learning":
            agent = QLearningAgent(env)
        elif selected_algorithm == "DQN":
            state_size = env.observation_space.shape[0]
            action_size = env.action_space.n
            agent = DQNAgent(env, state_size, action_size)

        total_score = 0  # Initialize cumulative score

        for episode in range(100):  # Number of episodes for training/testing
            done = False
            state = env.reset()

            while not done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

                action = agent.choose_action(state)  # Use the selected agent to choose action
                next_state, reward, done, info = env.step(action)

                if selected_algorithm == "Q-Learning":
                    agent.learn(state, action, reward, next_state, done)  # Update Q-table
                elif selected_algorithm == "DQN":
                    agent.store_experience(state, action, reward, next_state, done)
                    agent.train()  # Train the neural network

                state = next_state
                total_score += reward  # Update cumulative score
                env.render()

                if done:
                    # Display message
                    font = pygame.font.Font(None, 36)
                    if reward == 1.0:
                        message = "Goal Reached!"
                    elif reward == -1.0:
                        message = "Collision with Obstacle!"
                    else:
                        message = "Episode Ended!"

                    text = font.render(f"{message} | Score: {total_score:.2f}", True, (255, 0, 0))
                    env.window.blit(text, (env.window_size // 2 - text.get_width() // 2, env.window_size // 2 - text.get_height() // 2))
                    pygame.display.flip()
                    pygame.time.wait(2000)  # Wait for 2 seconds

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())