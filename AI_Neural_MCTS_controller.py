# AI_Neural_MCTS_controller.py
"""
Neural Network enhanced MCTS controller for the sliding puzzle game.
Uses Game.py helper functions intelligently for state evaluation and learning.
"""

import random
import math
import time
import numpy as np
import pickle
import os
from copy import deepcopy
from typing import List, Tuple, Dict
from Game import Game

# Simple neural network for position evaluation
class PuzzleNet:
    def __init__(self, input_size: int, hidden_size: int = 128):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Initialize weights randomly
        self.w1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = np.random.randn(hidden_size, hidden_size // 2) * 0.1
        self.b2 = np.zeros((1, hidden_size // 2))
        self.w3 = np.random.randn(hidden_size // 2, 1) * 0.1
        self.b3 = np.zeros((1, 1))
        
        self.learning_rate = 0.001
        
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow
    
    def forward(self, x):
        self.z1 = np.dot(x, self.w1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = self.relu(self.z2)
        self.z3 = np.dot(self.a2, self.w3) + self.b3
        self.output = self.sigmoid(self.z3)
        return self.output
    
    def backward(self, x, y, output):
        m = x.shape[0]
        
        # Output layer
        dz3 = output - y
        dw3 = np.dot(self.a2.T, dz3) / m
        db3 = np.sum(dz3, axis=0, keepdims=True) / m
        
        # Hidden layer 2
        da2 = np.dot(dz3, self.w3.T)
        dz2 = da2 * self.relu_derivative(self.z2)
        dw2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        # Hidden layer 1
        da1 = np.dot(dz2, self.w2.T)
        dz1 = da1 * self.relu_derivative(self.z1)
        dw1 = np.dot(x.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        # Update weights
        self.w3 -= self.learning_rate * dw3
        self.b3 -= self.learning_rate * db3
        self.w2 -= self.learning_rate * dw2
        self.b2 -= self.learning_rate * db2
        self.w1 -= self.learning_rate * dw1
        self.b1 -= self.learning_rate * db1
    
    def train(self, x, y):
        output = self.forward(x)
        self.backward(x, y, output)
        return output
    
    def predict(self, x):
        return self.forward(x)


class NeuralMCTSNode:
    def __init__(self, game_state: Game, parent=None, move_made=None):
        self.game_state = game_state
        self.parent = parent
        self.move_made = move_made
        self.children = {}
        self.visits = 0
        self.total_reward = 0.0
        self.neural_value = 0.0  # Neural network evaluation
        self.untried_moves = game_state.get_valid_moves().copy()

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def best_child(self, exploration_weight=1.4):
        """Enhanced UCB1 using neural network evaluation."""
        if not self.children:
            return None
        
        best_score = float('-inf')
        best_child = None
        
        for child in self.children.values():
            if child.visits == 0:
                # Use neural network prior for unvisited nodes
                ucb_score = child.neural_value + exploration_weight * math.sqrt(math.log(self.visits + 1))
            else:
                # Standard UCB1 enhanced with neural prior
                exploitation = child.total_reward / child.visits
                exploration = exploration_weight * math.sqrt(math.log(self.visits) / child.visits)
                neural_bonus = 0.1 * child.neural_value  # Small neural network bias
                ucb_score = exploitation + exploration + neural_bonus
            
            if ucb_score > best_score:
                best_score = ucb_score
                best_child = child
                
        return best_child


class NeuralMCTSController:
    def __init__(self, game: Game, iterations=500, show_thinking=False, model_file="puzzle_model.pkl"):
        self.game = game
        self.iterations = iterations
        self.show_thinking = show_thinking
        self.move_count = 0
        self.model_file = model_file
        
        # Calculate feature size based on actual features generated
        # We'll determine this dynamically by extracting features from the initial game state
        temp_features = self.extract_intelligent_features(game)
        feature_size = temp_features.shape[1]
        
        if self.show_thinking:
            print(f"Detected feature size: {feature_size}")
        
        # Initialize or load neural network
        self.neural_net = PuzzleNet(feature_size)
        self.load_model()
        
        # Training data collection
        self.training_positions = []
        self.training_outcomes = []
        self.games_played = 0

    def extract_intelligent_features(self, game_state: Game) -> np.ndarray:
        """Extract comprehensive features using Game.py helper functions."""
        features = []
        
        # 1. Normalized tile positions (0-1 range)
        labels_list = game_state.get_labels_as_list()
        normalized_positions = [label / (game_state.breadth * game_state.breadth) for label in labels_list]
        features.extend(normalized_positions)
        
        # 2. Distance features for each tile position
        distance_features = []
        for row in range(game_state.breadth):
            for col in range(game_state.breadth):
                label = game_state.get_label(row, col)
                if label != game_state.blank_label:
                    distance = game_state.get_distance_by_label(label)
                    # Normalize distance (max possible distance in a breadth x breadth grid)
                    max_distance = 2 * (game_state.breadth - 1)
                    normalized_distance = distance / max_distance if max_distance > 0 else 0
                else:
                    normalized_distance = 0  # Blank tile
                distance_features.append(normalized_distance)
        features.extend(distance_features)
        
        # 3. Move and game state statistics
        valid_moves = game_state.get_valid_moves()
        features.append(len(valid_moves) / (game_state.breadth * game_state.breadth))  # Mobility
        features.append(game_state.get_distance_sum() / (game_state.breadth ** 4))      # Total disorder
        
        # Blank position features (normalized)
        blank_row, blank_col = game_state.blank_position
        features.append(blank_row / (game_state.breadth - 1))
        features.append(blank_col / (game_state.breadth - 1))
        
        # 4. Pattern recognition features
        labels_matrix = game_state.get_labels_as_matrix()
        
        # Count correctly placed tiles in each row/column
        correctly_placed_rows = 0
        correctly_placed_cols = 0
        
        for row in range(game_state.breadth):
            row_correct = True
            for col in range(game_state.breadth):
                expected_label = row * game_state.breadth + col + 1
                actual_label = labels_matrix[row][col]
                if actual_label != expected_label and expected_label != game_state.blank_label:
                    row_correct = False
                    break
            if row_correct:
                correctly_placed_rows += 1
        
        for col in range(game_state.breadth):
            col_correct = True
            for row in range(game_state.breadth):
                expected_label = row * game_state.breadth + col + 1
                actual_label = labels_matrix[row][col]
                if actual_label != expected_label and expected_label != game_state.blank_label:
                    col_correct = False
                    break
            if col_correct:
                correctly_placed_cols += 1
        
        features.append(correctly_placed_rows / game_state.breadth)
        features.append(correctly_placed_cols / game_state.breadth)
        
        # Adjacent correct pairs (horizontal and vertical)
        correct_h_pairs = 0
        correct_v_pairs = 0
        total_h_pairs = game_state.breadth * (game_state.breadth - 1)
        total_v_pairs = (game_state.breadth - 1) * game_state.breadth
        
        for row in range(game_state.breadth):
            for col in range(game_state.breadth - 1):
                label1 = labels_matrix[row][col]
                label2 = labels_matrix[row][col + 1]
                if label1 != game_state.blank_label and label2 != game_state.blank_label:
                    if label2 == label1 + 1:
                        correct_h_pairs += 1
        
        for row in range(game_state.breadth - 1):
            for col in range(game_state.breadth):
                label1 = labels_matrix[row][col]
                label2 = labels_matrix[row + 1][col]
                if label1 != game_state.blank_label and label2 != game_state.blank_label:
                    if label2 == label1 + game_state.breadth:
                        correct_v_pairs += 1
        
        features.append(correct_h_pairs / total_h_pairs if total_h_pairs > 0 else 0)
        features.append(correct_v_pairs / total_v_pairs if total_v_pairs > 0 else 0)
        
        # Corner and edge placement accuracy
        corners_correct = 0
        
        # Check corners
        corner_positions = [(0, 0), (0, game_state.breadth-1), (game_state.breadth-1, 0), (game_state.breadth-1, game_state.breadth-1)]
        for row, col in corner_positions:
            expected_label = row * game_state.breadth + col + 1
            actual_label = labels_matrix[row][col]
            if actual_label == expected_label:
                corners_correct += 1
        
        features.append(corners_correct / 4)
        
        # Linear conflict detection (tiles in correct row/col but wrong order)
        linear_conflicts = 0
        for row in range(game_state.breadth):
            row_tiles = []
            for col in range(game_state.breadth):
                label = labels_matrix[row][col]
                if label != game_state.blank_label:
                    # Check if this tile belongs in this row
                    correct_row = (label - 1) // game_state.breadth
                    if correct_row == row:
                        row_tiles.append((label, col))
            
            # Count inversions in this row
            for i in range(len(row_tiles)):
                for j in range(i + 1, len(row_tiles)):
                    if row_tiles[i][0] > row_tiles[j][0]:  # Wrong order
                        linear_conflicts += 1
        
        features.append(linear_conflicts / (game_state.breadth ** 2))
        
        return np.array(features).reshape(1, -1)

    def evaluate_position(self, game_state: Game) -> float:
        """Use neural network to evaluate game position."""
        if game_state.is_solved():
            return 1.0
        
        features = self.extract_intelligent_features(game_state)
        neural_value = self.neural_net.predict(features)[0, 0]
        return float(neural_value)

    def select(self, node: NeuralMCTSNode) -> NeuralMCTSNode:
        """Selection with neural network guidance."""
        current = node
        while not current.game_state.is_solved() and current.is_fully_expanded():
            current = current.best_child()
            if current is None:
                break
        return current

    def expand(self, node: NeuralMCTSNode) -> NeuralMCTSNode:
        """Expansion with neural network evaluation of new nodes."""
        if not node.untried_moves or node.game_state.is_solved():
            return node
        
        # Prioritize moves using heuristics
        move_scores = []
        for move in node.untried_moves:
            # Quick heuristic: prefer moves that reduce total distance
            temp_game = self.copy_game_state(node.game_state)
            if temp_game.player_move(move):
                distance_reduction = node.game_state.get_distance_sum() - temp_game.get_distance_sum()
                move_scores.append((move, distance_reduction))
            else:
                move_scores.append((move, -1))  # Invalid move penalty
        
        # Sort by heuristic score and pick from top moves
        move_scores.sort(key=lambda x: x[1], reverse=True)
        top_moves = [move for move, _ in move_scores[:min(3, len(move_scores))]]
        move = random.choice(top_moves)
        
        node.untried_moves.remove(move)
        
        # Create new game state
        new_game_state = self.copy_game_state(node.game_state)
        success = new_game_state.player_move(move)
        
        if not success:
            return node
        
        # Create child node with neural evaluation
        child = NeuralMCTSNode(new_game_state, parent=node, move_made=move)
        child.neural_value = self.evaluate_position(new_game_state)
        node.children[move] = child
        
        return child

    def simulate(self, node: NeuralMCTSNode) -> float:
        """Neural network guided simulation."""
        simulation_game = self.copy_game_state(node.game_state)
        moves_made = 0
        max_moves = 50  # Shorter simulations with neural guidance
        
        while not simulation_game.is_solved() and moves_made < max_moves:
            valid_moves = simulation_game.get_valid_moves()
            if not valid_moves:
                break
            
            # Use neural network to bias move selection
            if moves_made < 10:  # Use neural guidance for early moves
                move_evaluations = []
                for move in valid_moves:
                    temp_game = self.copy_game_state(simulation_game)
                    if temp_game.player_move(move):
                        eval_score = self.evaluate_position(temp_game)
                        move_evaluations.append((move, eval_score))
                
                if move_evaluations:
                    # Weighted random selection based on evaluations
                    moves, scores = zip(*move_evaluations)
                    scores = np.array(scores)
                    # Softmax selection
                    exp_scores = np.exp(scores * 5)  # Temperature = 5
                    probabilities = exp_scores / np.sum(exp_scores)
                    move = np.random.choice(moves, p=probabilities)
                else:
                    move = random.choice(valid_moves)
            else:
                # Random selection for later moves
                move = random.choice(valid_moves)
            
            simulation_game.player_move(move)
            moves_made += 1
        
        # Enhanced reward function
        if simulation_game.is_solved():
            return max(0.8, 2.0 - moves_made * 0.02)
        else:
            # Use neural network evaluation for terminal non-solved positions
            neural_eval = self.evaluate_position(simulation_game)
            return neural_eval * 0.5  # Scale down non-winning positions

    def backpropagate(self, node: NeuralMCTSNode, reward: float):
        """Standard backpropagation with training data collection."""
        current = node
        while current is not None:
            current.visits += 1
            current.total_reward += reward
            
            # Collect training data
            if len(self.training_positions) < 10000:  # Limit training data size
                features = self.extract_intelligent_features(current.game_state)
                self.training_positions.append(features)
                self.training_outcomes.append(reward)
            
            current = current.parent

    def copy_game_state(self, game: Game) -> Game:
        """Create a deep copy of the game state."""
        new_game = Game(game.breadth, False)
        
        for i, tile in enumerate(game.tiles):
            new_game.tiles[i].label = tile.label
            new_game.tiles[i].row = tile.row
            new_game.tiles[i].column = tile.column
            new_game.tiles[i].ordinal = tile.ordinal
        
        new_game.blank_position = game.blank_position
        return new_game

    def train_neural_network(self):
        """Train the neural network on collected game data."""
        if len(self.training_positions) < 100:
            return
        
        # Prepare training data
        X = np.vstack(self.training_positions)
        y = np.array(self.training_outcomes).reshape(-1, 1)
        
        # Normalize outcomes
        y = (y - np.min(y)) / (np.max(y) - np.min(y) + 1e-8)
        
        # Train in batches
        batch_size = 32
        for epoch in range(10):
            indices = np.random.permutation(len(X))
            for i in range(0, len(X), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_X = X[batch_indices]
                batch_y = y[batch_indices]
                self.neural_net.train(batch_X, batch_y)
        
        if self.show_thinking:
            print(f"Trained neural network on {len(X)} positions")

    def save_model(self):
        """Save the trained model."""
        model_data = {
            'w1': self.neural_net.w1,
            'b1': self.neural_net.b1,
            'w2': self.neural_net.w2,
            'b2': self.neural_net.b2,
            'w3': self.neural_net.w3,
            'b3': self.neural_net.b3,
            'input_size': self.neural_net.input_size,
            'games_played': self.games_played
        }
        with open(self.model_file, 'wb') as f:
            pickle.dump(model_data, f)

    def load_model(self):
        """Load a pre-trained model if it exists."""
        if os.path.exists(self.model_file):
            try:
                with open(self.model_file, 'rb') as f:
                    model_data = pickle.load(f)
                
                # Check if the saved model has the same input size
                if model_data.get('input_size') == self.neural_net.input_size:
                    self.neural_net.w1 = model_data['w1']
                    self.neural_net.b1 = model_data['b1']
                    self.neural_net.w2 = model_data['w2']
                    self.neural_net.b2 = model_data['b2']
                    self.neural_net.w3 = model_data['w3']
                    self.neural_net.b3 = model_data['b3']
                    self.games_played = model_data.get('games_played', 0)
                    
                    if self.show_thinking:
                        print(f"Loaded model with {self.games_played} games of experience")
                else:
                    if self.show_thinking:
                        print(f"Model input size mismatch. Expected {self.neural_net.input_size}, got {model_data.get('input_size', 'unknown')}")
                        print("Starting with fresh neural network")
                        
            except Exception as e:
                if self.show_thinking:
                    print(f"Could not load model: {e}")

    def get_best_move(self) -> int:
        """MCTS with neural network enhancement."""
        if self.game.is_solved():
            return None
            
        root = NeuralMCTSNode(self.copy_game_state(self.game))
        root.neural_value = self.evaluate_position(root.game_state)
        
        for iteration in range(self.iterations):
            leaf = self.select(root)
            if not leaf.game_state.is_solved():
                leaf = self.expand(leaf)
            reward = self.simulate(leaf)
            self.backpropagate(leaf, reward)
            
            if self.show_thinking and iteration % 100 == 0:
                best_child = max(root.children.values(), key=lambda c: c.visits) if root.children else None
                if best_child:
                    print(f"Iteration {iteration}: Move {best_child.move_made} "
                          f"(visits: {best_child.visits}, neural: {best_child.neural_value:.3f})")
        
        if not root.children:
            return None
            
        best_child = max(root.children.values(), key=lambda c: c.visits)
        return best_child.move_made

    def play_game(self, max_moves=100, delay=0.5, train_after=True):
        """Play a complete game with learning."""
        print(f"Starting game {self.games_played + 1}...")
        initial_distance = self.game.get_distance_sum()
        
        while self.move_count < max_moves and not self.game.is_solved():
            print(f"\n--- Move {self.move_count + 1} ---")
            print(self.game)
            
            # Get neural network evaluation of current position
            current_eval = self.evaluate_position(self.game)
            print(f"Neural evaluation: {current_eval:.3f}")
            print(f"Total distance: {self.game.get_distance_sum()}")
            
            best_move = self.get_best_move()
            if best_move is None:
                break
                
            success = self.game.player_move(best_move)
            if success:
                self.move_count += 1
                sequence = self.game.get_move_sequence(best_move)
                move_type = "single" if len(sequence) == 1 else f"multi ({len(sequence)})"
                print(f"AI move: {best_move} ({move_type})")
            else:
                break
                
            time.sleep(delay)
        
        # Game finished
        self.games_played += 1
        final_distance = self.game.get_distance_sum()
        improvement = initial_distance - final_distance
        
        print(f"\n--- Game {self.games_played} Complete ---")
        print(self.game)
        
        if self.game.is_solved():
            print(f"*** SOLVED! *** Moves: {self.move_count}")
        else:
            print(f"Game ended after {self.move_count} moves")
            print(f"Distance improvement: {improvement}")
        
        # Train and save model
        if train_after:
            self.train_neural_network()
            self.save_model()
            # Clear training data for next game
            self.training_positions = []
            self.training_outcomes = []


if __name__ == '__main__':
    print("Neural MCTS Puzzle Solver")
    print("=" * 30)
    
    # Create game
    size = int(input("Game size (3-5 recommended): ") or "4")
    game = Game(size, True)
    
    # Create AI controller
    ai = NeuralMCTSController(game, iterations=300, show_thinking=True)
    
    # Play multiple games for learning
    num_games = int(input("Number of games to play (1-10): ") or "1")
    
    for game_num in range(num_games):
        if game_num > 0:
            # Create new game for next iteration
            ai.game = Game(size, True)
            ai.move_count = 0
        
        ai.play_game(max_moves=200, delay=1.0)
        
        if game_num < num_games - 1:
            print(f"\nStarting next game in 3 seconds...")
            time.sleep(3)
