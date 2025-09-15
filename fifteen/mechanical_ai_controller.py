"""
Enhanced AI Controller with Mechanical Awareness for Fifteen Puzzle

This controller implements deep understanding of fifteen-puzzle mechanical constraints:
- Blank position analysis (corner/edge/interior)
- Progressive vs regressive move classification
- Position-dependent action utilities
- Mechanical constraint-aware training

Educational Focus: Demonstrates how domain knowledge can enhance RL performance
"""

import numpy as np
import random
from typing import List, Tuple, Dict, Optional
from collections import deque
from Game import Game

# Optional TensorFlow import with fallback
try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. Using Q-table fallback.")


class MechanicalAIController:
    """AI Controller with deep mechanical understanding of fifteen-puzzle constraints."""

    def __init__(self, game: Game, epsilon_start: float = 0.9, epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995, learning_rate: float = 0.001,
                 memory_size: int = 10000, batch_size: int = 32):
        """
        Initialize mechanical-aware AI controller.

        Args:
            game: Game instance to control
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Exploration decay rate
            learning_rate: Neural network learning rate
            memory_size: Experience replay buffer size
            batch_size: Training batch size
        """
        self.game = game
        self.breadth = game.breadth
        self.state_size = game.breadth * game.breadth
        self.action_space_size = self.state_size

        # Enhanced parameters for mechanical awareness
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Experience replay and model training
        self.memory = deque(maxlen=memory_size)
        self.target_update_frequency = 10
        self.training_steps = 0

        # Mechanical awareness tracking
        self.previous_move = None
        self.previous_blank_pos = None
        self.move_history = deque(maxlen=10)  # Track move sequence

        # Enhanced reward parameters
        self.progressive_move_bonus = 2.0
        self.regressive_move_penalty = -1.0
        self.mechanical_awareness_bonus = 0.5

        # Neural network setup
        self.use_tensorflow = TENSORFLOW_AVAILABLE
        if self.use_tensorflow:
            self._build_enhanced_network()
            self._build_target_network()
        else:
            self.q_table = {}

        # Statistics tracking
        self.statistics_tracker = None

    def _build_enhanced_network(self):
        """Build enhanced neural network with mechanical awareness features."""
        if not self.use_tensorflow:
            return

        # Enhanced input: state + blank position features + move history features
        input_size = self.state_size + 6  # +6 for blank position encoding and move features

        self.q_network = keras.Sequential([
            keras.layers.Dense(256, activation='relu', input_shape=(input_size,)),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(self.action_space_size, activation='linear')
        ])

        self.q_network.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )

    def _build_target_network(self):
        """Build target network for stable training."""
        if not self.use_tensorflow:
            return

        input_size = self.state_size + 6

        self.target_network = keras.Sequential([
            keras.layers.Dense(256, activation='relu', input_shape=(input_size,)),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(self.action_space_size, activation='linear')
        ])

        self.target_network.set_weights(self.q_network.get_weights())

    def get_blank_position_category(self, state: List[int]) -> str:
        """
        Determine blank position category for mechanical analysis.

        Returns:
            'corner', 'edge', or 'interior'
        """
        # Find blank position (represented as None in state)
        try:
            blank_pos = state.index(None)
        except ValueError:
            # Fallback: look for blank_label value
            blank_pos = state.index(self.breadth * self.breadth)

        row = blank_pos // self.breadth
        col = blank_pos % self.breadth

        # Corner positions
        if (row == 0 or row == self.breadth - 1) and (col == 0 or col == self.breadth - 1):
            return 'corner'

        # Edge positions (not corner)
        if row == 0 or row == self.breadth - 1 or col == 0 or col == self.breadth - 1:
            return 'edge'

        # Interior positions
        return 'interior'

    def classify_move_utility(self, action: int, state: List[int]) -> str:
        """
        Classify move based on fifteen-puzzle mechanical truth table.

        TRUTH TABLE (4x4 puzzle always has exactly 6 moves):
        - Corner blank: 1 regressive, 1 progressive, 0 exploratory, 4 chain
        - Edge blank: 1 regressive, 1 progressive, 1 exploratory, 3 chain
        - Surrounded blank: 1 regressive, 1 progressive, 2 exploratory, 2 chain

        Args:
            action: Tile to move (1-16)
            state: Current puzzle state

        Returns:
            'progressive', 'regressive', 'exploratory', or 'chain'
        """
        # Regressive move: undoes the previous move
        if self.previous_move is not None and action == self.previous_move:
            return 'regressive'

        # For now, classify non-regressive moves as exploratory
        # A full implementation would need:
        # 1. Progressive move detection (requires solution path analysis)
        # 2. Chain move detection (requires neighbor-swap sequence analysis)
        # 3. Position-specific move counting per truth table

        blank_category = self.get_blank_position_category(state)

        # Simplified classification - in real implementation, would need:
        # - Progressive move detection algorithm
        # - Chain move sequence analysis
        # - Position-specific move ratio enforcement

        return 'exploratory'  # Placeholder for non-regressive moves

    def get_enhanced_state_representation(self, game: Game = None) -> np.ndarray:
        """
        Create enhanced state representation with mechanical awareness features.

        Features:
        - Normalized tile positions (16 values)
        - Blank position category encoding (3 values: corner/edge/interior)
        - Previous move encoding (1 value)
        - Move history pattern (2 values)

        Returns:
            Enhanced state vector for neural network
        """
        if game is None:
            game = self.game

        # Base state representation
        state = np.array(game.get_state(), dtype=np.float32)
        state = state / self.state_size  # Normalize

        # Blank position category features
        blank_category = self.get_blank_position_category(game.get_state())
        blank_features = np.zeros(3, dtype=np.float32)
        if blank_category == 'corner':
            blank_features[0] = 1.0
        elif blank_category == 'edge':
            blank_features[1] = 1.0
        else:  # interior
            blank_features[2] = 1.0

        # Previous move feature
        prev_move_feature = np.array([
            self.previous_move / self.action_space_size if self.previous_move else 0.0
        ], dtype=np.float32)

        # Move history pattern features
        history_features = np.zeros(2, dtype=np.float32)
        if len(self.move_history) >= 2:
            # Recent move diversity
            recent_unique = len(set(list(self.move_history)[-4:]))
            history_features[0] = recent_unique / 4.0

            # Repetition pattern
            if len(self.move_history) >= 4:
                recent_4 = list(self.move_history)[-4:]
                if recent_4[0] == recent_4[2] and recent_4[1] == recent_4[3]:
                    history_features[1] = 1.0  # Back-and-forth pattern detected

        # Combine all features
        enhanced_state = np.concatenate([
            state,
            blank_features,
            prev_move_feature,
            history_features
        ])

        return enhanced_state

    def calculate_mechanical_reward(self, prev_state: List[int], action: int,
                                  curr_state: List[int], solved: bool) -> float:
        """
        Calculate reward incorporating mechanical understanding.

        Args:
            prev_state: Previous puzzle state
            action: Action taken
            curr_state: Resulting puzzle state
            solved: Whether puzzle is solved

        Returns:
            Enhanced reward value
        """
        # Base entropy reward (existing system)
        prev_entropy = self._calculate_entropy(prev_state)
        curr_entropy = self._calculate_entropy(curr_state)
        entropy_reward = (prev_entropy - curr_entropy) * 10.0

        # Solve bonus
        if solved:
            return entropy_reward + 100.0

        # Mechanical understanding rewards
        move_utility = self.classify_move_utility(action, prev_state)

        mechanical_bonus = 0.0
        if move_utility == 'progressive':
            mechanical_bonus = self.progressive_move_bonus
        elif move_utility == 'regressive':
            mechanical_bonus = self.regressive_move_penalty
        elif move_utility == 'exploratory':
            mechanical_bonus = self.mechanical_awareness_bonus

        # Blank position awareness bonus
        blank_category = self.get_blank_position_category(prev_state)
        position_bonus = 0.0

        if blank_category == 'corner' and move_utility == 'progressive':
            position_bonus = 1.0  # Reward finding the one good move from corner
        elif blank_category == 'interior' and move_utility != 'regressive':
            position_bonus = 0.3  # Moderate reward for good interior moves

        total_reward = entropy_reward + mechanical_bonus + position_bonus
        return total_reward

    def _calculate_entropy(self, state: List[int]) -> float:
        """Calculate puzzle entropy (disorder measure)."""
        entropy = 0.0
        for i, tile in enumerate(state):
            if tile != self.breadth * self.breadth:  # Skip blank
                target_pos = tile - 1  # 0-indexed target position
                current_pos = i

                # Manhattan distance
                target_row, target_col = target_pos // self.breadth, target_pos % self.breadth
                current_row, current_col = current_pos // self.breadth, current_pos % self.breadth

                entropy += abs(target_row - current_row) + abs(target_col - current_col)

        return entropy

    def choose_action(self, state: np.ndarray, valid_actions: List[int], training: bool = True) -> int:
        """
        Choose action using mechanical-aware policy.

        Args:
            state: Enhanced state representation
            valid_actions: List of valid actions
            training: Whether in training mode

        Returns:
            Selected action
        """
        # Epsilon-greedy exploration
        if training and random.random() < self.epsilon:
            # Mechanical-aware exploration: prefer non-regressive moves
            game_state = self.game.get_state()
            non_regressive_actions = []

            for action in valid_actions:
                move_utility = self.classify_move_utility(action, game_state)
                if move_utility != 'regressive':
                    non_regressive_actions.append(action)

            if non_regressive_actions:
                return random.choice(non_regressive_actions)
            else:
                return random.choice(valid_actions)

        # Exploitation: use enhanced Q-values
        if self.use_tensorflow:
            q_values = self.q_network.predict(state.reshape(1, -1), verbose=0)[0]
        else:
            # Q-table fallback
            state_key = tuple(state)
            if state_key not in self.q_table:
                self.q_table[state_key] = {action: 0.0 for action in range(1, self.action_space_size + 1)}
            q_values = [self.q_table[state_key].get(i, 0.0) for i in range(1, self.action_space_size + 1)]

        # Mask invalid actions and apply mechanical bias
        masked_q_values = np.full(self.action_space_size, -np.inf)
        game_state = self.game.get_state()

        for action in valid_actions:
            if 1 <= action <= self.action_space_size:
                base_q = q_values[action - 1]

                # Apply mechanical bias
                move_utility = self.classify_move_utility(action, game_state)
                mechanical_bias = 0.0

                if move_utility == 'progressive':
                    mechanical_bias = 0.5
                elif move_utility == 'regressive':
                    mechanical_bias = -0.5

                masked_q_values[action - 1] = base_q + mechanical_bias

        # Choose best action
        if np.all(masked_q_values == -np.inf):
            return random.choice(valid_actions)

        best_action_idx = np.argmax(masked_q_values)
        return best_action_idx + 1

    def remember(self, state: np.ndarray, action: int, reward: float,
                 next_state: np.ndarray, done: bool):
        """Store experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self):
        """Perform one training step with enhanced features."""
        if not self.use_tensorflow or len(self.memory) < self.batch_size:
            return

        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])

        # Current Q-values
        current_q_values = self.q_network.predict(states, verbose=0)

        # Next Q-values from target network
        next_q_values = self.target_network.predict(next_states, verbose=0)

        # Calculate targets
        targets = current_q_values.copy()
        for i in range(self.batch_size):
            if dones[i]:
                targets[i][actions[i] - 1] = rewards[i]
            else:
                targets[i][actions[i] - 1] = rewards[i] + 0.95 * np.max(next_q_values[i])

        # Train network
        self.q_network.fit(states, targets, epochs=1, verbose=0)

        # Update target network periodically
        self.training_steps += 1
        if self.training_steps % self.target_update_frequency == 0:
            self.target_network.set_weights(self.q_network.get_weights())

        # Decay epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

    def play_episode(self, max_steps: int = 300, training: bool = True, verbose: bool = False) -> Tuple[float, int, bool]:
        """
        Play one episode with mechanical awareness.

        Returns:
            (total_reward, steps_taken, solved)
        """
        self.game.restart_game()
        state = self.get_enhanced_state_representation()
        total_reward = 0.0
        steps = 0
        solved = False

        # Reset mechanical tracking
        self.previous_move = None
        self.previous_blank_pos = None
        self.move_history.clear()

        for step in range(max_steps):
            valid_actions = self.game.get_valid_moves()
            if not valid_actions:
                break

            # Choose action with mechanical awareness
            action = self.choose_action(state, valid_actions, training)

            # Store current state for reward calculation
            prev_state = self.game.get_state().copy()

            # Execute action
            success = self.game.player_move(action)
            if not success:
                break

            # Update mechanical tracking
            self.move_history.append(action)
            curr_state = self.game.get_state()
            self.previous_move = action

            # Calculate enhanced reward
            reward = self.calculate_mechanical_reward(prev_state, action, curr_state, self.game.is_solved())

            # Get next state representation
            next_state = self.get_enhanced_state_representation()
            solved = self.game.is_solved()

            if verbose:
                blank_cat = self.get_blank_position_category(prev_state)
                move_util = self.classify_move_utility(action, prev_state)
                print(f"Step {step + 1}: Move {action}, Blank: {blank_cat}, Utility: {move_util}, Reward: {reward:.2f}")

            # Store experience and train
            if training:
                self.remember(state, action, reward, next_state, solved)
                self.train_step()

            total_reward += reward
            state = next_state
            steps += 1

            if solved:
                if verbose:
                    print(f"Solved in {steps} steps with mechanical awareness!")
                break

        return total_reward, steps, solved

    def attach_statistics_tracker(self, tracker):
        """Attach statistics tracker for learning analysis."""
        self.statistics_tracker = tracker