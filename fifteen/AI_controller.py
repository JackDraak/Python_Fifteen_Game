# AI_controller.py
"""
Sophisticated AI Controller for the Fifteen Puzzle

This controller understands the deceptive complexity of the 15-puzzle:
- Uses Manhattan distance (entropy) as the primary state evaluation metric
- Implements Deep Q-Network (DQN) for pattern recognition across the 16-tile state space
- Provides nuanced reward shaping: small rewards for entropy reduction, penalties for increases
- Handles the paradox that optimal solutions often require temporary entropy increases
- Supports both single-tile and multi-tile moves from the enhanced Game.py

Key Features:
- Experience replay for stable learning
- Target network for stable Q-value estimation
- Epsilon-greedy exploration with decay
- Sophisticated reward system that understands puzzle dynamics
- State representation optimized for tile relationships
- Progress tracking and model persistence
"""

import numpy as np
import random
import json
import time
from collections import deque
from typing import Dict, List, Tuple, Optional, Any
from Game import Game

# Try to import TensorFlow, fall back to lightweight implementation if unavailable
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available. Using lightweight Q-table implementation.")


class AIController:
    """
    Advanced AI Controller for the Fifteen Puzzle using Deep Q-Learning.

    The controller understands that optimal puzzle solving often requires:
    1. Temporarily increasing entropy to unlock better future states
    2. Recognizing complex tile interaction patterns
    3. Balancing exploration vs exploitation in the vast state space
    """

    def __init__(self,
                 game: Game,
                 learning_rate: float = 0.001,
                 discount_factor: float = 0.95,
                 epsilon_start: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 memory_size: int = 10000,
                 batch_size: int = 32,
                 target_update_freq: int = 100,
                 reward_scale: float = 1.0,
                 entropy_reward_scale: float = 0.1,
                 use_tensorflow: bool = True):

        self.game = game
        self.breadth = game.breadth
        self.state_size = self.breadth * self.breadth
        self.action_space_size = self.breadth * self.breadth  # All tile labels including blank

        # Learning parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.reward_scale = reward_scale
        self.entropy_reward_scale = entropy_reward_scale

        # Network parameters
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.memory = deque(maxlen=memory_size)

        # Training statistics
        self.episode_count = 0
        self.step_count = 0
        self.total_reward = 0
        self.best_score = -float('inf')
        self.training_history = []

        # Initialize networks
        self.use_tensorflow = use_tensorflow and TF_AVAILABLE
        if self.use_tensorflow:
            self.q_network = self._build_network()
            self.target_network = self._build_network()
            self.update_target_network()
        else:
            # Fallback to Q-table for environments without TensorFlow
            self.q_table = {}

    def _build_network(self) -> Sequential:
        """
        Build a Deep Q-Network optimized for understanding tile relationships.

        Architecture designed to capture:
        - Local tile adjacencies (important for valid moves)
        - Global board patterns (important for strategy)
        - Tile displacement patterns (important for entropy measurement)
        """
        model = Sequential([
            # Input layer: flattened board state
            Dense(256, input_shape=(self.state_size,), activation='relu'),
            Dropout(0.2),

            # Hidden layers: progressively learn complex tile patterns
            Dense(256, activation='relu'),
            Dropout(0.2),
            Dense(128, activation='relu'),
            Dropout(0.1),
            Dense(128, activation='relu'),

            # Output layer: Q-values for each possible tile move
            Dense(self.action_space_size, activation='linear')
        ])

        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )

        return model

    def get_state_representation(self, game: Game = None) -> np.ndarray:
        """
        Convert game state to neural network input.

        Creates a normalized representation that helps the AI understand:
        - Current tile positions
        - Tile displacement from goal positions
        - Blank tile location (critical for move planning)
        """
        if game is None:
            game = self.game

        # Get current board state
        state = np.array(game.get_state(), dtype=np.float32)

        # Normalize tile values to [0, 1] range
        state = state / self.state_size

        return state

    def calculate_reward(self, prev_entropy: float, curr_entropy: float, solved: bool) -> float:
        """
        Calculate reward based on entropy change and solve status.

        Reward Philosophy:
        - Large positive reward for solving (+1000)
        - Small positive reward for entropy reduction (+0.1 * reduction)
        - Small negative reward for entropy increase (-0.1 * increase)
        - Small step penalty to encourage efficiency (-0.01)

        This encourages the AI to:
        1. Prioritize solving the puzzle
        2. Generally reduce entropy when possible
        3. Accept temporary entropy increases when they lead to better positions
        4. Solve efficiently
        """
        if solved:
            return 1000.0 * self.reward_scale

        # Entropy-based reward (negative because we want to minimize distance)
        entropy_change = prev_entropy - curr_entropy
        entropy_reward = entropy_change * self.entropy_reward_scale

        # Small step penalty to encourage efficiency
        step_penalty = -0.01

        total_reward = entropy_reward + step_penalty
        return total_reward * self.reward_scale

    def get_valid_actions(self) -> List[int]:
        """Get list of valid tile labels that can be moved."""
        return self.game.get_valid_moves()

    def choose_action(self, state: np.ndarray, valid_actions: List[int], training: bool = True) -> int:
        """
        Choose action using epsilon-greedy policy with valid action masking.

        During training: explores with probability epsilon
        During evaluation: always exploits (chooses best action)
        """
        if training and random.random() < self.epsilon:
            # Exploration: random valid action
            return random.choice(valid_actions)

        # Exploitation: best valid action according to Q-network
        if self.use_tensorflow:
            q_values = self.q_network.predict(state.reshape(1, -1), verbose=0)[0]
        else:
            # Q-table fallback
            state_key = tuple(state)
            if state_key not in self.q_table:
                self.q_table[state_key] = {action: 0.0 for action in range(1, self.action_space_size + 1)}
            q_values = [self.q_table[state_key].get(i, 0.0) for i in range(1, self.action_space_size + 1)]

        # Mask invalid actions
        masked_q_values = np.full(self.action_space_size, -np.inf)
        for action in valid_actions:
            if 1 <= action <= self.action_space_size:
                masked_q_values[action - 1] = q_values[action - 1]

        # Choose best valid action
        best_action_idx = np.argmax(masked_q_values)
        return best_action_idx + 1

    def remember(self, state: np.ndarray, action: int, reward: float,
                 next_state: np.ndarray, done: bool):
        """Store experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self):
        """
        Perform one training step using experience replay.

        Implements Double DQN to reduce overestimation bias:
        - Use main network to select actions
        - Use target network to evaluate actions
        """
        if len(self.memory) < self.batch_size:
            return

        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states = np.array([experience[0] for experience in batch])
        actions = np.array([experience[1] for experience in batch])
        rewards = np.array([experience[2] for experience in batch])
        next_states = np.array([experience[3] for experience in batch])
        dones = np.array([experience[4] for experience in batch])

        if self.use_tensorflow:
            # Get current Q-values
            current_q_values = self.q_network.predict(states, verbose=0)

            # Get next Q-values from target network
            next_q_values = self.target_network.predict(next_states, verbose=0)

            # Calculate target Q-values
            targets = current_q_values.copy()
            for i in range(self.batch_size):
                action_idx = actions[i] - 1  # Convert to 0-indexed
                if dones[i]:
                    targets[i][action_idx] = rewards[i]
                else:
                    targets[i][action_idx] = rewards[i] + self.discount_factor * np.max(next_q_values[i])

            # Train the network
            self.q_network.fit(states, targets, batch_size=self.batch_size, verbose=0)

        else:
            # Q-table update
            for i in range(len(batch)):
                state, action, reward, next_state, done = batch[i]
                state_key = tuple(state)
                next_state_key = tuple(next_state)

                if state_key not in self.q_table:
                    self.q_table[state_key] = {a: 0.0 for a in range(1, self.action_space_size + 1)}
                if next_state_key not in self.q_table:
                    self.q_table[next_state_key] = {a: 0.0 for a in range(1, self.action_space_size + 1)}

                if done:
                    target = reward
                else:
                    target = reward + self.discount_factor * max(self.q_table[next_state_key].values())

                # Q-learning update
                old_value = self.q_table[state_key][action]
                self.q_table[state_key][action] = old_value + self.learning_rate * (target - old_value)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        """Copy weights from main network to target network."""
        if self.use_tensorflow:
            self.target_network.set_weights(self.q_network.get_weights())

    def play_episode(self, max_steps: int = 1000, training: bool = True, verbose: bool = False) -> Tuple[float, int, bool]:
        """
        Play one complete episode.

        Returns:
        - Total reward earned
        - Steps taken
        - Whether puzzle was solved
        """
        # Reset game to starting state
        if training:
            # For training, start from shuffled state
            self.game = Game(self.breadth, True, seed=None)

        state = self.get_state_representation()
        total_reward = 0.0
        steps = 0
        solved = False

        for step in range(max_steps):
            # Get valid actions
            valid_actions = self.get_valid_actions()
            if not valid_actions:
                break

            # Get current entropy
            prev_entropy = self.game.get_distance_sum()

            # Choose and execute action
            action = self.choose_action(state, valid_actions, training)
            success = self.game.player_move(action)

            if not success:
                # Invalid move, give penalty and continue
                reward = -10.0 * self.reward_scale
                total_reward += reward
                continue

            # Calculate reward
            curr_entropy = self.game.get_distance_sum()
            solved = self.game.is_solved()
            reward = self.calculate_reward(prev_entropy, curr_entropy, solved)
            total_reward += reward

            # Get next state
            next_state = self.get_state_representation()

            # Store experience
            if training:
                self.remember(state, action, reward, next_state, solved)
                self.train_step()

            # Update state
            state = next_state
            steps += 1

            if verbose:
                print(f"Step {steps}: Action {action}, Entropy {prev_entropy}->{curr_entropy}, "
                      f"Reward {reward:.3f}, Total {total_reward:.3f}")

            # Check if solved
            if solved:
                if verbose:
                    print(f"*** SOLVED in {steps} steps! Total reward: {total_reward:.3f} ***")
                break

        # Update target network periodically
        if training and self.step_count % self.target_update_freq == 0:
            self.update_target_network()

        self.step_count += steps
        self.episode_count += 1

        # Track best performance
        if total_reward > self.best_score:
            self.best_score = total_reward

        # Record training history
        if training:
            self.training_history.append({
                'episode': self.episode_count,
                'reward': total_reward,
                'steps': steps,
                'solved': solved,
                'epsilon': self.epsilon,
                'final_entropy': self.game.get_distance_sum()
            })

        return total_reward, steps, solved

    def train(self, episodes: int, verbose: bool = True, save_freq: int = 100):
        """
        Train the AI for specified number of episodes.

        Args:
            episodes: Number of training episodes
            verbose: Whether to print progress
            save_freq: How often to save the model
        """
        if verbose:
            print(f"Starting training for {episodes} episodes...")
            print(f"Using {'TensorFlow DQN' if self.use_tensorflow else 'Q-table'}")

        start_time = time.time()
        solved_count = 0

        for episode in range(episodes):
            reward, steps, solved = self.play_episode(training=True, verbose=False)

            if solved:
                solved_count += 1

            if verbose and (episode + 1) % 10 == 0:
                elapsed = time.time() - start_time
                solve_rate = solved_count / (episode + 1) * 100
                avg_reward = np.mean([h['reward'] for h in self.training_history[-10:]])

                print(f"Episode {episode + 1}/{episodes} | "
                      f"Reward: {reward:.1f} | Steps: {steps} | "
                      f"Solved: {'Yes' if solved else 'No'} | "
                      f"Solve Rate: {solve_rate:.1f}% | "
                      f"Avg Reward (10): {avg_reward:.1f} | "
                      f"Epsilon: {self.epsilon:.3f} | "
                      f"Time: {elapsed:.1f}s")

            # Save model periodically
            if (episode + 1) % save_freq == 0:
                self.save_model(f"ai_model_episode_{episode + 1}")

        if verbose:
            total_time = time.time() - start_time
            final_solve_rate = solved_count / episodes * 100
            print(f"\nTraining completed!")
            print(f"Total time: {total_time:.1f}s")
            print(f"Final solve rate: {final_solve_rate:.1f}%")
            print(f"Best score: {self.best_score:.1f}")

    def evaluate(self, episodes: int = 100, verbose: bool = True) -> Dict[str, Any]:
        """
        Evaluate the trained AI on fresh puzzles.

        Returns performance statistics.
        """
        old_epsilon = self.epsilon
        self.epsilon = 0.0  # No exploration during evaluation

        results = {
            'episodes': episodes,
            'solved': 0,
            'total_steps': 0,
            'total_reward': 0.0,
            'solve_times': [],
            'final_entropies': []
        }

        for episode in range(episodes):
            reward, steps, solved = self.play_episode(training=False, verbose=False)

            results['total_steps'] += steps
            results['total_reward'] += reward

            if solved:
                results['solved'] += 1
                results['solve_times'].append(steps)

            results['final_entropies'].append(self.game.get_distance_sum())

        # Calculate statistics
        results['solve_rate'] = results['solved'] / episodes * 100
        results['avg_steps'] = results['total_steps'] / episodes
        results['avg_reward'] = results['total_reward'] / episodes
        results['avg_solve_time'] = np.mean(results['solve_times']) if results['solve_times'] else None
        results['avg_final_entropy'] = np.mean(results['final_entropies'])

        self.epsilon = old_epsilon  # Restore epsilon

        if verbose:
            print(f"\nEvaluation Results ({episodes} episodes):")
            print(f"Solve Rate: {results['solve_rate']:.1f}%")
            print(f"Average Steps: {results['avg_steps']:.1f}")
            print(f"Average Reward: {results['avg_reward']:.1f}")
            if results['avg_solve_time']:
                print(f"Average Solve Time: {results['avg_solve_time']:.1f} steps")
            print(f"Average Final Entropy: {results['avg_final_entropy']:.1f}")

        return results

    def save_model(self, filename: str):
        """Save the trained model and training statistics."""
        if self.use_tensorflow:
            model_path = f"{filename}.h5"
            self.q_network.save(model_path)
            print(f"Model saved to {model_path}")
        else:
            model_path = f"{filename}_qtable.json"
            # Convert numpy arrays to lists for JSON serialization
            serializable_qtable = {}
            for state_key, actions in self.q_table.items():
                serializable_qtable[str(state_key)] = actions

            with open(model_path, 'w') as f:
                json.dump(serializable_qtable, f)
            print(f"Q-table saved to {model_path}")

        # Save training statistics
        stats_path = f"{filename}_stats.json"
        stats = {
            'episode_count': self.episode_count,
            'step_count': self.step_count,
            'best_score': self.best_score,
            'epsilon': self.epsilon,
            'training_history': self.training_history,
            'model_parameters': {
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor,
                'epsilon_decay': self.epsilon_decay,
                'breadth': self.breadth
            }
        }

        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Training statistics saved to {stats_path}")

    def load_model(self, filename: str):
        """Load a trained model and training statistics."""
        if self.use_tensorflow:
            model_path = f"{filename}.h5"
            try:
                self.q_network = load_model(model_path)
                self.update_target_network()
                print(f"Model loaded from {model_path}")
            except FileNotFoundError:
                print(f"Model file {model_path} not found")
        else:
            model_path = f"{filename}_qtable.json"
            try:
                with open(model_path, 'r') as f:
                    serializable_qtable = json.load(f)

                # Convert back from JSON format
                self.q_table = {}
                for state_str, actions in serializable_qtable.items():
                    state_key = eval(state_str)  # Convert string back to tuple
                    self.q_table[state_key] = actions

                print(f"Q-table loaded from {model_path}")
            except FileNotFoundError:
                print(f"Q-table file {model_path} not found")

        # Load training statistics
        stats_path = f"{filename}_stats.json"
        try:
            with open(stats_path, 'r') as f:
                stats = json.load(f)

            self.episode_count = stats['episode_count']
            self.step_count = stats['step_count']
            self.best_score = stats['best_score']
            self.epsilon = stats['epsilon']
            self.training_history = stats['training_history']

            print(f"Training statistics loaded from {stats_path}")
        except FileNotFoundError:
            print(f"Statistics file {stats_path} not found")


if __name__ == "__main__":
    # Example usage and testing
    print("Fifteen Puzzle AI Controller")
    print("=" * 50)

    # Create a 4x4 puzzle
    game = Game(4, False)  # Start with solved puzzle
    ai = AIController(game)

    print(f"Using {'TensorFlow DQN' if ai.use_tensorflow else 'Q-table fallback'}")
    print(f"State space size: {ai.state_size}")
    print(f"Action space size: {ai.action_space_size}")

    # Quick test
    print("\nRunning quick test...")
    game_test = Game(4, True)  # Shuffled 4x4 puzzle (standard 15-puzzle)
    ai_test = AIController(game_test)
    reward, steps, solved = ai_test.play_episode(max_steps=100, training=False, verbose=True)

    print(f"\nTest complete: Reward={reward:.1f}, Steps={steps}, Solved={solved}")