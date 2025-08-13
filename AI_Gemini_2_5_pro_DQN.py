import numpy as np
import random
import os
from collections import deque
from Game import Game  # Assuming Game.py is in the same directory

# --- HYPERPARAMETERS ---
# Game and Board settings
BOARD_SIZE = 4
STATE_SIZE = BOARD_SIZE * BOARD_SIZE
ACTION_SIZE = BOARD_SIZE * BOARD_SIZE - 1 # Tiles are numbered 1-15

# DQN Agent settings
LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 0.95  # gamma

# Exploration/Exploitation settings
EPSILON_START = 1.0       # Initial exploration rate
EPSILON_DECAY = 0.9995    # Rate at which epsilon decreases
EPSILON_MIN = 0.01        # Minimum exploration rate

# Training settings
BATCH_SIZE = 64
REPLAY_MEMORY_SIZE = 10000
MIN_REPLAY_MEMORY_SIZE = 1000 # Minimum memory size before training starts
UPDATE_TARGET_EVERY = 5 # Terminal states (episodes)

# Reward Shaping settings
REWARD_WINDOW_SIZE = 10   # N, the moving window for entropy calculation
REWARD_SCALING_K = 0.1    # k, scaling factor for entropy reward
SOLVE_REWARD = 1000
MOVE_PENALTY = -1

# File settings
MODEL_NAME = "FifteenPuzzleDQN"
# --- END HYPERPARAMETERS ---


# Attempt to import TensorFlow, provide guidance if it fails
try:
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.optimizers import Adam
except ImportError:
    print("TensorFlow not found. Please install it using: pip install tensorflow")
    exit()


class DQNAgent:
    """
    A Deep Q-Network Agent to learn to solve the Fifteen Puzzle.
    """
    def __init__(self):
        # Main model, gets trained every step
        self.model = self._build_model()

        # Target model, this is what we predict against every step
        self.target_model = self._build_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.epsilon = EPSILON_START

    def _build_model(self):
        """Builds the Neural Network for the DQN agent."""
        model = Sequential()
        # Input layer: The flattened game board
        model.add(Dense(128, input_dim=STATE_SIZE, activation='relu'))
        model.add(Dense(128, activation='relu'))
        # Output layer: Q-values for each possible tile move (1-15)
        model.add(Dense(ACTION_SIZE, activation='linear'))
        model.compile(loss="mse", optimizer=Adam(learning_rate=LEARNING_RATE), metrics=['accuracy'])
        return model

    def update_target_model(self):
        """Copies weights from the main model to the target model."""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """Stores an experience tuple in the replay memory."""
        # The action is the tile label (1-15), so we subtract 1 for the index
        self.replay_memory.append((state, action - 1, reward, next_state, done))

    def act(self, state, valid_moves):
        """
        Chooses an action based on the current state and valid moves.
        Uses epsilon-greedy policy for exploration/exploitation.
        """
        if np.random.random() > self.epsilon:
            # Exploit: Get Q-values from the model
            q_values = self.model.predict(state.reshape(1, -1), verbose=0)[0]

            # Apply action masking: set Q-values of invalid moves to -infinity
            masked_q_values = np.full(ACTION_SIZE, -np.inf)
            for move in valid_moves:
                masked_q_values[move - 1] = q_values[move - 1] # move is 1-15

            # Choose the best valid move
            action_index = np.argmax(masked_q_values)
            return action_index + 1 # Return tile label
        else:
            # Explore: Choose a random valid move
            return random.choice(valid_moves)

    def replay(self):
        """
        Trains the main network using a batch of experiences from replay memory.
        """
        # Start training only if we have enough samples
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a random batch of samples from replay memory
        minibatch = random.sample(self.replay_memory, BATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states, verbose=0)

        # Get future states from minibatch, then query target model for Q values
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states, verbose=0)

        X = []
        y = []

        # Enumerate transitions and build training data
        for index, (current_state, action_index, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                # If not a terminal state, calculate the new Q-value
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT_FACTOR * max_future_q
            else:
                # If it is a terminal state, the new Q-value is just the reward
                new_q = reward

            # Update the Q value for the action taken
            current_qs = current_qs_list[index]
            current_qs[action_index] = new_q

            X.append(current_state)
            y.append(current_qs)

        # Fit the model on the training data
        self.model.fit(np.array(X), np.array(y), batch_size=BATCH_SIZE, verbose=0, shuffle=False)

        # Decay epsilon
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY
            self.epsilon = max(EPSILON_MIN, self.epsilon)

    def load(self, name):
        """Loads the model weights from a file."""
        if os.path.exists(name):
            print(f"Loading model from {name}...")
            self.model = load_model(name)
            self.update_target_model()
        else:
            print(f"No model found at {name}. Starting with a new model.")

    def save(self, name):
        """Saves the model weights to a file."""
        print(f"Saving model to {name}...")
        self.model.save(name)


def get_entropy(game):
    """Calculates the sum of Manhattan distances for all tiles."""
    distances = game.get_distance_set()
    return sum(distances.values())

def run_training(agent, episodes, verbosity=1):
    """Main training loop."""
    target_update_counter = 0

    for episode in range(1, episodes + 1):
        game = Game(BOARD_SIZE)
        game.shuffle() # Start with a shuffled board

        current_state = game.get_labels_as_matrix().flatten()
        entropy_window = deque(maxlen=REWARD_WINDOW_SIZE)
        
        done = False
        step = 0
        max_steps = BOARD_SIZE * 100 # Set a max step limit to avoid infinite loops
        
        while not done and step < max_steps:
            step += 1
            valid_moves = game.get_valid_moves()
            action_tile = agent.act(current_state, valid_moves)
            
            # Perform the move
            game.move_tile(action_tile)
            
            # Check if solved
            done = game.is_solved()
            
            # Calculate reward
            reward = 0
            if done:
                reward = SOLVE_REWARD
            else:
                current_entropy = get_entropy(game)
                
                # Calculate entropy-based reward
                reward_entropy = 0
                if len(entropy_window) == REWARD_WINDOW_SIZE:
                    avg_entropy = sum(entropy_window) / REWARD_WINDOW_SIZE
                    reward_entropy = REWARD_SCALING_K * (avg_entropy - current_entropy)

                reward = MOVE_PENALTY + reward_entropy
                entropy_window.append(current_entropy)

            new_state = game.get_labels_as_matrix().flatten()
            
            # Remember the experience
            agent.remember(current_state, action_tile, reward, new_state, done)
            
            current_state = new_state

            # Train the agent
            agent.replay()

        # Post-episode logic
        if done:
            target_update_counter += 1
            if verbosity >= 1:
                print(f"EPISODE: {episode}/{episodes} | SOLVED in {step} steps | Epsilon: {agent.epsilon:.5f}")

        if target_update_counter > UPDATE_TARGET_EVERY:
            agent.update_target_model()
            target_update_counter = 0
            if verbosity >= 1:
                print("Updated target model.")
        
        if verbosity == 2 and not done:
             print(f"EPISODE: {episode}/{episodes} | FAILED after {max_steps} steps.")


if __name__ == "__main__":
    agent = DQNAgent()
    
    print("Fifteen Puzzle ML Controller")
    print("Commands: train <episodes>, load, save, verbosity <0-2>, exit")

    while True:
        cmd_input = input("> ").strip().lower().split()
        command = cmd_input[0]

        if command == "train":
            try:
                num_episodes = int(cmd_input[1])
                run_training(agent, num_episodes)
            except (IndexError, ValueError):
                print("Usage: train <number_of_episodes>")
        
        elif command == "load":
            agent.load(f"{MODEL_NAME}.h5")

        elif command == "save":
            agent.save(f"{MODEL_NAME}.h5")

        elif command == "verbosity":
            # This is a placeholder for now. The training function would need to be updated
            # to accept and use this value.
            print("Verbosity setting is not yet fully implemented.")
            
        elif command == "exit":
            break
            
        else:
            print("Unknown command.")
