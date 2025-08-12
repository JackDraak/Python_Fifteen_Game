# AI_Neural_MCTS_controller.py
"""
Enhanced Neural Network MCTS controller with better exploration and test modes.
"""

import random
import math
import time
import numpy as np
import pickle
import os
from collections import defaultdict, deque
from copy import deepcopy
from typing import List, Tuple, Dict
from Game import Game

class PuzzleNet:
    def __init__(self, input_size: int, hidden_size: int = 128):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Initialize weights with better initialization
        std1 = math.sqrt(2.0 / input_size)
        std2 = math.sqrt(2.0 / hidden_size)
        std3 = math.sqrt(2.0 / (hidden_size // 2))
        
        self.w1 = np.random.normal(0, std1, (input_size, hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = np.random.normal(0, std2, (hidden_size, hidden_size // 2))
        self.b2 = np.zeros((1, hidden_size // 2))
        self.w3 = np.random.normal(0, std3, (hidden_size // 2, 1))
        self.b3 = np.zeros((1, 1))
        
        self.learning_rate = 0.001
        self.momentum = 0.9
        
        # Add momentum terms
        self.vw1 = np.zeros_like(self.w1)
        self.vb1 = np.zeros_like(self.b1)
        self.vw2 = np.zeros_like(self.w2)
        self.vb2 = np.zeros_like(self.b2)
        self.vw3 = np.zeros_like(self.w3)
        self.vb3 = np.zeros_like(self.b3)
        
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
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
        
        # Update weights with momentum
        self.vw3 = self.momentum * self.vw3 - self.learning_rate * dw3
        self.vb3 = self.momentum * self.vb3 - self.learning_rate * db3
        self.vw2 = self.momentum * self.vw2 - self.learning_rate * dw2
        self.vb2 = self.momentum * self.vb2 - self.learning_rate * db2
        self.vw1 = self.momentum * self.vw1 - self.learning_rate * dw1
        self.vb1 = self.momentum * self.vb1 - self.learning_rate * db1
        
        self.w3 += self.vw3
        self.b3 += self.vb3
        self.w2 += self.vw2
        self.b2 += self.vb2
        self.w1 += self.vw1
        self.b1 += self.vb1
    
    def train(self, x, y):
        output = self.forward(x)
        self.backward(x, y, output)
        loss = np.mean((output - y) ** 2)
        return output, loss
    
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
        self.neural_value = 0.0
        self.untried_moves = game_state.get_valid_moves().copy()

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def best_child(self, exploration_weight=1.4):
        """Enhanced UCB1 with progressive widening and neural guidance."""
        if not self.children:
            return None
        
        best_score = float('-inf')
        best_child = None
        
        for child in self.children.values():
            if child.visits == 0:
                ucb_score = child.neural_value + exploration_weight * math.sqrt(math.log(self.visits + 1))
            else:
                exploitation = child.total_reward / child.visits
                exploration = exploration_weight * math.sqrt(math.log(self.visits) / child.visits)
                neural_bonus = 0.2 * child.neural_value
                ucb_score = exploitation + exploration + neural_bonus
            
            if ucb_score > best_score:
                best_score = ucb_score
                best_child = child
                
        return best_child


class EnhancedNeuralMCTSController:
    def __init__(self, game: Game, iterations=500, show_thinking=False, 
                 model_file="puzzle_model.pkl", test_mode=False, rapid_test=False):
        self.game = game
        self.iterations = iterations
        self.show_thinking = show_thinking
        self.test_mode = test_mode
        self.rapid_test = rapid_test
        self.move_count = 0
        self.model_file = model_file
        
        # Enhanced exploration tracking
        self.move_history = deque(maxlen=20)  # Track recent moves for repetition detection
        self.state_visit_count = defaultdict(int)  # Track state visitations
        self.exploration_bonus = 0.1  # Bonus for exploring new states
        self.repetition_penalty = 0.3  # Penalty for repeating recent moves
        
        # Performance tracking
        self.position_evaluations = []
        self.distance_history = []
        self.unique_positions_visited = set()
        self.training_losses = []
        
        # Dynamic parameters
        self.exploration_weight = 2.0  # Start with higher exploration
        self.min_exploration = 0.5
        self.exploration_decay = 0.95
        
        # Calculate feature size dynamically
        temp_features = self.extract_intelligent_features(game)
        feature_size = temp_features.shape[1]
        
        if self.show_thinking:
            print(f"Detected feature size: {feature_size}")
        
        # Initialize neural network
        self.neural_net = PuzzleNet(feature_size)
        self.load_model()
        
        # Training data collection
        self.training_positions = []
        self.training_outcomes = []
        self.games_played = 0
        self.training_frequency = 50  # Train every 50 moves instead of only at game end

    def get_state_hash(self, game_state: Game) -> str:
        """Create a hash of the game state for tracking unique positions."""
        return str(game_state.get_labels_as_list())

    def extract_intelligent_features(self, game_state: Game) -> np.ndarray:
        """Enhanced feature extraction with better normalization."""
        features = []
        
        # 1. Normalized tile positions
        labels_list = game_state.get_labels_as_list()
        max_label = game_state.breadth * game_state.breadth
        normalized_positions = [label / max_label for label in labels_list]
        features.extend(normalized_positions)
        
        # 2. Manhattan distances (normalized)
        max_distance = 2 * (game_state.breadth - 1)
        for row in range(game_state.breadth):
            for col in range(game_state.breadth):
                label = game_state.get_label(row, col)
                if label != game_state.blank_label:
                    distance = game_state.get_distance_by_label(label)
                    normalized_distance = distance / max_distance if max_distance > 0 else 0
                else:
                    normalized_distance = 0
                features.append(normalized_distance)
        
        # 3. Game state metrics
        valid_moves = game_state.get_valid_moves()
        features.append(len(valid_moves) / max_label)  # Mobility
        features.append(game_state.get_distance_sum() / (game_state.breadth ** 4))  # Total disorder
        
        # Blank position features
        blank_row, blank_col = game_state.blank_position
        features.append(blank_row / (game_state.breadth - 1) if game_state.breadth > 1 else 0)
        features.append(blank_col / (game_state.breadth - 1) if game_state.breadth > 1 else 0)
        
        # 4. Pattern recognition features
        labels_matrix = game_state.get_labels_as_matrix()
        
        # Correctly placed tiles
        correctly_placed = 0
        total_tiles = game_state.breadth * game_state.breadth - 1  # Exclude blank
        
        for row in range(game_state.breadth):
            for col in range(game_state.breadth):
                expected_label = row * game_state.breadth + col + 1
                actual_label = labels_matrix[row][col]
                if expected_label != game_state.blank_label and actual_label == expected_label:
                    correctly_placed += 1
        
        features.append(correctly_placed / total_tiles if total_tiles > 0 else 0)
        
        # Adjacent correct pairs
        correct_pairs = 0
        total_pairs = 0
        
        # Horizontal pairs
        for row in range(game_state.breadth):
            for col in range(game_state.breadth - 1):
                label1 = labels_matrix[row][col]
                label2 = labels_matrix[row][col + 1]
                if label1 != game_state.blank_label and label2 != game_state.blank_label:
                    total_pairs += 1
                    if label2 == label1 + 1:
                        correct_pairs += 1
        
        # Vertical pairs
        for row in range(game_state.breadth - 1):
            for col in range(game_state.breadth):
                label1 = labels_matrix[row][col]
                label2 = labels_matrix[row + 1][col]
                if label1 != game_state.blank_label and label2 != game_state.blank_label:
                    total_pairs += 1
                    if label2 == label1 + game_state.breadth:
                        correct_pairs += 1
        
        features.append(correct_pairs / total_pairs if total_pairs > 0 else 0)
        
        # Linear conflicts (tiles in correct row/col but wrong relative order)
        linear_conflicts = 0
        max_conflicts = game_state.breadth ** 2
        
        # Row conflicts
        for row in range(game_state.breadth):
            row_tiles = []
            for col in range(game_state.breadth):
                label = labels_matrix[row][col]
                if label != game_state.blank_label:
                    correct_row = (label - 1) // game_state.breadth
                    if correct_row == row:
                        row_tiles.append(label)
            
            # Count inversions
            for i in range(len(row_tiles)):
                for j in range(i + 1, len(row_tiles)):
                    if row_tiles[i] > row_tiles[j]:
                        linear_conflicts += 1
        
        features.append(linear_conflicts / max_conflicts if max_conflicts > 0 else 0)
        
        return np.array(features).reshape(1, -1)

    def evaluate_position(self, game_state: Game) -> float:
        """Enhanced position evaluation with exploration bonus."""
        if game_state.is_solved():
            return 1.0
        
        # Get neural network evaluation
        features = self.extract_intelligent_features(game_state)
        neural_value = self.neural_net.predict(features)[0, 0]
        
        # Add exploration bonus for unvisited states
        state_hash = self.get_state_hash(game_state)
        exploration_bonus = 0.0
        if state_hash not in self.unique_positions_visited:
            exploration_bonus = self.exploration_bonus
        
        # Track position for analysis
        if self.test_mode:
            self.position_evaluations.append(float(neural_value))
            self.unique_positions_visited.add(state_hash)
        
        return float(neural_value) + exploration_bonus

    def select(self, node: NeuralMCTSNode) -> NeuralMCTSNode:
        """Selection with adaptive exploration."""
        current = node
        while not current.game_state.is_solved() and current.is_fully_expanded():
            current = current.best_child(self.exploration_weight)
            if current is None:
                break
        return current

    def expand(self, node: NeuralMCTSNode) -> NeuralMCTSNode:
        """Enhanced expansion with move prioritization."""
        if not node.untried_moves or node.game_state.is_solved():
            return node
        
        # Evaluate all untried moves and select based on potential
        move_scores = []
        for move in node.untried_moves:
            temp_game = self.copy_game_state(node.game_state)
            if temp_game.player_move(move):
                # Multiple scoring criteria
                distance_reduction = node.game_state.get_distance_sum() - temp_game.get_distance_sum()
                neural_eval = self.evaluate_position(temp_game)
                
                # Penalize recent moves to encourage exploration
                repetition_penalty = 0
                if move in self.move_history:
                    repetition_penalty = self.repetition_penalty * (self.move_history.count(move) / len(self.move_history))
                
                combined_score = distance_reduction + neural_eval - repetition_penalty
                move_scores.append((move, combined_score))
            else:
                move_scores.append((move, -10))  # Invalid move penalty
        
        # Use weighted random selection favoring better moves
        if move_scores:
            moves, scores = zip(*move_scores)
            scores = np.array(scores)
            
            # Shift scores to be positive and add small epsilon
            min_score = np.min(scores)
            adjusted_scores = scores - min_score + 0.1
            
            # Apply softmax with temperature for exploration
            temperature = 2.0
            exp_scores = np.exp(adjusted_scores / temperature)
            probabilities = exp_scores / np.sum(exp_scores)
            
            move = np.random.choice(moves, p=probabilities)
        else:
            move = random.choice(node.untried_moves)
        
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
        """Enhanced simulation with better guidance."""
        simulation_game = self.copy_game_state(node.game_state)
        moves_made = 0
        max_moves = 30  # Shorter simulations
        initial_distance = simulation_game.get_distance_sum()
        
        while not simulation_game.is_solved() and moves_made < max_moves:
            valid_moves = simulation_game.get_valid_moves()
            if not valid_moves:
                break
            
            # Use neural guidance for early moves
            if moves_made < 5:
                move_evaluations = []
                for move in valid_moves[:min(6, len(valid_moves))]:  # Limit evaluation to top 6 moves
                    temp_game = self.copy_game_state(simulation_game)
                    if temp_game.player_move(move):
                        eval_score = self.evaluate_position(temp_game)
                        move_evaluations.append((move, eval_score))
                
                if move_evaluations:
                    moves, scores = zip(*move_evaluations)
                    scores = np.array(scores)
                    
                    # Softmax selection with temperature
                    exp_scores = np.exp(scores * 3)
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
            return max(0.9, 1.5 - moves_made * 0.01)
        else:
            # Reward based on distance improvement and neural evaluation
            final_distance = simulation_game.get_distance_sum()
            distance_improvement = (initial_distance - final_distance) / initial_distance if initial_distance > 0 else 0
            neural_eval = self.evaluate_position(simulation_game)
            
            return max(0.1, (distance_improvement + neural_eval) * 0.3)

    def backpropagate(self, node: NeuralMCTSNode, reward: float):
        """Enhanced backpropagation with continuous learning."""
        current = node
        while current is not None:
            current.visits += 1
            current.total_reward += reward
            
            # Collect training data more frequently
            if len(self.training_positions) < 5000:
                features = self.extract_intelligent_features(current.game_state)
                self.training_positions.append(features)
                self.training_outcomes.append(reward)
            
            current = current.parent
        
        # Train neural network periodically during search
        if len(self.training_positions) >= self.training_frequency:
            self.train_neural_network_incremental()

    def train_neural_network_incremental(self):
        """Incremental training during search."""
        if len(self.training_positions) < 20:
            return
        
        # Use recent data for training
        recent_count = min(100, len(self.training_positions))
        X = np.vstack(self.training_positions[-recent_count:])
        y = np.array(self.training_outcomes[-recent_count:]).reshape(-1, 1)
        
        # Normalize outcomes
        if np.max(y) > np.min(y):
            y = (y - np.min(y)) / (np.max(y) - np.min(y))
        
        # Single training epoch
        _, loss = self.neural_net.train(X, y)
        self.training_losses.append(loss)

    def copy_game_state(self, game: Game) -> Game:
        """Efficient game state copying."""
        new_game = Game(game.breadth, False)
        
        for i, tile in enumerate(game.tiles):
            new_game.tiles[i].label = tile.label
            new_game.tiles[i].row = tile.row
            new_game.tiles[i].column = tile.column
            new_game.tiles[i].ordinal = tile.ordinal
        
        new_game.blank_position = game.blank_position
        return new_game

    def train_neural_network(self):
        """Full training at game end."""
        if len(self.training_positions) < 50:
            return
        
        X = np.vstack(self.training_positions)
        y = np.array(self.training_outcomes).reshape(-1, 1)
        
        # Normalize outcomes
        if np.max(y) > np.min(y):
            y = (y - np.min(y)) / (np.max(y) - np.min(y))
        
        # Train in batches
        batch_size = 32
        total_loss = 0
        batch_count = 0
        
        for epoch in range(5):  # Fewer epochs for faster training
            indices = np.random.permutation(len(X))
            for i in range(0, len(X), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_X = X[batch_indices]
                batch_y = y[batch_indices]
                _, loss = self.neural_net.train(batch_X, batch_y)
                total_loss += loss
                batch_count += 1
        
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        self.training_losses.append(avg_loss)
        
        if self.show_thinking:
            print(f"Trained NN on {len(X)} positions, avg loss: {avg_loss:.4f}")

    def save_model(self):
        """Save the trained model with enhanced metadata."""
        model_data = {
            'w1': self.neural_net.w1, 'b1': self.neural_net.b1,
            'w2': self.neural_net.w2, 'b2': self.neural_net.b2,
            'w3': self.neural_net.w3, 'b3': self.neural_net.b3,
            'vw1': self.neural_net.vw1, 'vb1': self.neural_net.vb1,
            'vw2': self.neural_net.vw2, 'vb2': self.neural_net.vb2,
            'vw3': self.neural_net.vw3, 'vb3': self.neural_net.vb3,
            'input_size': self.neural_net.input_size,
            'games_played': self.games_played,
            'training_losses': self.training_losses[-100:],  # Keep recent losses
            'exploration_weight': self.exploration_weight
        }
        with open(self.model_file, 'wb') as f:
            pickle.dump(model_data, f)

    def load_model(self):
        """Load model with momentum terms."""
        if os.path.exists(self.model_file):
            try:
                with open(self.model_file, 'rb') as f:
                    model_data = pickle.load(f)
                
                if model_data.get('input_size') == self.neural_net.input_size:
                    self.neural_net.w1 = model_data['w1']
                    self.neural_net.b1 = model_data['b1']
                    self.neural_net.w2 = model_data['w2']
                    self.neural_net.b2 = model_data['b2']
                    self.neural_net.w3 = model_data['w3']
                    self.neural_net.b3 = model_data['b3']
                    
                    # Load momentum terms if available
                    if 'vw1' in model_data:
                        self.neural_net.vw1 = model_data['vw1']
                        self.neural_net.vb1 = model_data['vb1']
                        self.neural_net.vw2 = model_data['vw2']
                        self.neural_net.vb2 = model_data['vb2']
                        self.neural_net.vw3 = model_data['vw3']
                        self.neural_net.vb3 = model_data['vb3']
                    
                    self.games_played = model_data.get('games_played', 0)
                    self.training_losses = model_data.get('training_losses', [])
                    self.exploration_weight = model_data.get('exploration_weight', 2.0)
                    
                    if self.show_thinking:
                        recent_loss = self.training_losses[-1] if self.training_losses else 0
                        print(f"Loaded model: {self.games_played} games, recent loss: {recent_loss:.4f}")
                else:
                    if self.show_thinking:
                        print("Model size mismatch - starting fresh")
            except Exception as e:
                if self.show_thinking:
                    print(f"Could not load model: {e}")

    def get_best_move(self) -> int:
        """Enhanced MCTS with better exploration."""
        if self.game.is_solved():
            return None
            
        root = NeuralMCTSNode(self.copy_game_state(self.game))
        root.neural_value = self.evaluate_position(root.game_state)
        
        # Track search progress
        best_moves = []
        
        for iteration in range(self.iterations):
            leaf = self.select(root)
            if not leaf.game_state.is_solved():
                leaf = self.expand(leaf)
            reward = self.simulate(leaf)
            self.backpropagate(leaf, reward)
            
            # Track progress for test mode
            if self.test_mode and iteration % 50 == 0:
                if root.children:
                    current_best = max(root.children.values(), key=lambda c: c.visits)
                    best_moves.append(current_best.move_made)
                    
            if self.show_thinking and iteration % 100 == 0:
                if root.children:
                    best_child = max(root.children.values(), key=lambda c: c.visits)
                    print(f"Iter {iteration}: Move {best_child.move_made} "
                          f"(visits: {best_child.visits}, value: {best_child.total_reward/max(1, best_child.visits):.3f})")
        
        if not root.children:
            return None
            
        # Select move with highest visit count (most robust)
        best_child = max(root.children.values(), key=lambda c: c.visits)
        
        # Update exploration weight (decay over time)
        self.exploration_weight = max(self.min_exploration, 
                                    self.exploration_weight * self.exploration_decay)
        
        return best_child.move_made

    def rapid_assessment(self, max_moves=50) -> Dict:
        """Quick assessment of model capability."""
        print("Running rapid assessment...")
        
        initial_distance = self.game.get_distance_sum()
        moves_made = 0
        unique_moves = set()
        distance_improvements = []
        move_sequence = []
        
        while moves_made < max_moves and not self.game.is_solved():
            current_distance = self.game.get_distance_sum()
            best_move = self.get_best_move()
            
            if best_move is None:
                break
                
            success = self.game.player_move(best_move)
            if success:
                moves_made += 1
                unique_moves.add(best_move)
                move_sequence.append(best_move)
                
                new_distance = self.game.get_distance_sum()
                improvement = current_distance - new_distance
                distance_improvements.append(improvement)
                
                if self.rapid_test:
                    print(f"Move {moves_made}: {best_move} (dist: {new_distance}, change: {improvement:+d})")
        
        # Analysis
        total_improvement = initial_distance - self.game.get_distance_sum()
        avg_improvement = np.mean(distance_improvements) if distance_improvements else 0
        exploration_ratio = len(unique_moves) / moves_made if moves_made > 0 else 0
        
        results = {
            'moves_made': moves_made,
            'total_distance_improvement': total_improvement,
            'average_improvement_per_move': avg_improvement,
            'unique_moves_tried': len(unique_moves),
            'exploration_ratio': exploration_ratio,
            'final_distance': self.game.get_distance_sum(),
            'solved': self.game.is_solved(),
            'move_sequence': move_sequence
        }
        
        return results

    def play_game(self, max_moves=100, delay=0.5, train_after=True):
        """Enhanced game play with better monitoring."""
        print(f"Starting game {self.games_played + 1}...")
        
        if self.rapid_test:
            return self.rapid_assessment(max_moves)
        
        initial_distance = self.game.get_distance_sum()
        self.distance_history = [initial_distance]
        recent_moves = deque(maxlen=10)
        
        while self.move_count < max_moves and not self.game.is_solved():
            if not self.test_mode:
                print(f"\n--- Move {self.move_count + 1} ---")
                print(self.game)
            
            # Get current state info
            current_distance = self.game.get_distance_sum()
            current_eval = self.evaluate_position(self.game)
            
            if not self.test_mode:
                print(f"Neural evaluation: {current_eval:.3f}")
                print(f"Total distance: {current_distance}")
                print(f"Exploration weight: {self.exploration_weight:.3f}")
            
            best_move = self.get_best_move()
            if best_move is None:
                print("No valid moves available!")
                break
                
            success = self.game.player_move(best_move)
            if success:
                self.move_count += 1
                recent_moves.append(best_move)
                self.move_history.append(best_move)
                
                new_distance = self.game.get_distance_sum()
                self.distance_history.append(new_distance)
                improvement = current_distance - new_distance
                
                # Track state visitation
                state_hash = self.get_state_hash(self.game)
                self.state_visit_count[state_hash] += 1
                
                if self.test_mode:
                    # Test mode: show move sequence
                    print(f"{best_move}", end=", " if self.move_count < max_moves else "\n")
                    
                    # Check for excessive repetition
                    if len(recent_moves) >= 6:
                        unique_recent = len(set(recent_moves))
                        if unique_recent <= 3:
                            print(f"\n[Warning: Low exploration - only {unique_recent} unique moves in last {len(recent_moves)}]")
                else:
                    sequence = self.game.get_move_sequence(best_move)
                    move_type = "single" if len(sequence) == 1 else f"multi ({len(sequence)})"
                    print(f"AI move: {best_move} ({move_type}) -> distance: {new_distance} (change: {improvement:+d})")
                
                # Check for progress stagnation
                if len(self.distance_history) >= 10:
                    recent_progress = self.distance_history[-10] - self.distance_history[-1]
                    if recent_progress <= 0 and not self.test_mode:
                        print(f"[No progress in last 10 moves - increasing exploration]")
                        self.exploration_weight = min(3.0, self.exploration_weight * 1.2)
            else:
                print(f"Move {best_move} failed!")
                break
                
            if not self.test_mode and delay > 0:
                time.sleep(delay)
        
        # Game finished
        self.games_played += 1
        final_distance = self.game.get_distance_sum()
        total_improvement = initial_distance - final_distance
        
        if not self.test_mode:
            print(f"\n--- Game {self.games_played} Complete ---")
            print(self.game)
        
        if self.game.is_solved():
            print(f"*** SOLVED! *** Moves: {self.move_count}")
        else:
            print(f"Game ended after {self.move_count} moves")
            print(f"Distance: {initial_distance} -> {final_distance} (improvement: {total_improvement})")
            print(f"Unique positions visited: {len(self.unique_positions_visited)}")
            
            # Show repetition analysis
            if len(self.move_history) > 0:
                move_counts = {}
                for move in self.move_history:
                    move_counts[move] = move_counts.get(move, 0) + 1
                most_common = sorted(move_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                print(f"Most frequent moves: {most_common}")
        
        # Performance analysis
        if self.test_mode and len(self.distance_history) > 1:
            improvements = [self.distance_history[i] - self.distance_history[i+1] 
                          for i in range(len(self.distance_history)-1)]
            positive_moves = sum(1 for imp in improvements if imp > 0)
            print(f"Positive moves: {positive_moves}/{len(improvements)} ({positive_moves/len(improvements)*100:.1f}%)")
            
            if self.training_losses:
                print(f"Recent training loss: {self.training_losses[-1]:.4f}")
        
        # Train and save model
        if train_after:
            self.train_neural_network()
            self.save_model()
            # Clear some training data to prevent memory issues
            if len(self.training_positions) > 2000:
                # Keep most recent data
                self.training_positions = self.training_positions[-1000:]
                self.training_outcomes = self.training_outcomes[-1000:]


def run_comparative_test(size=4, iterations_per_test=200):
    """Run a comparative test to see learning progress."""
    print("Running Comparative Learning Test")
    print("=" * 40)
    
    # Test 1: Fresh model
    print("Test 1: Fresh model (no prior learning)")
    fresh_game = Game(size, True)
    fresh_ai = EnhancedNeuralMCTSController(fresh_game, iterations=iterations_per_test, 
                                          test_mode=True, rapid_test=True, model_file="temp_fresh.pkl")
    # Remove any existing temp model
    if os.path.exists("temp_fresh.pkl"):
        os.remove("temp_fresh.pkl")
    
    fresh_results = fresh_ai.play_game(max_moves=100)
    
    # Test 2: Pre-trained model
    print("\nTest 2: Pre-trained model")
    trained_game = Game(size, True)
    # Copy the same initial state for fair comparison
    for i, tile in enumerate(fresh_game.tiles):
        trained_game.tiles[i].label = tile.label
        trained_game.tiles[i].row = tile.row
        trained_game.tiles[i].column = tile.column
    trained_game.blank_position = fresh_game.blank_position
    
    trained_ai = EnhancedNeuralMCTSController(trained_game, iterations=iterations_per_test,
                                            test_mode=True, rapid_test=True)
    trained_results = trained_ai.play_game(max_moves=100)
    
    # Compare results
    print("\n" + "="*40)
    print("COMPARISON RESULTS:")
    print("="*40)
    
    metrics = ['moves_made', 'total_distance_improvement', 'average_improvement_per_move', 
               'exploration_ratio', 'final_distance']
    
    for metric in metrics:
        fresh_val = fresh_results.get(metric, 0)
        trained_val = trained_results.get(metric, 0)
        print(f"{metric:30}: Fresh={fresh_val:8.3f} | Trained={trained_val:8.3f}")
    
    # Clean up temp file
    if os.path.exists("temp_fresh.pkl"):
        os.remove("temp_fresh.pkl")
    
    return fresh_results, trained_results


if __name__ == '__main__':
    print("Enhanced Neural MCTS Puzzle Solver")
    print("=" * 40)
    
    # Get user preferences
    size = int(input("Game size (3-5 recommended): ") or "4")
    mode = input("Mode (1=Normal, 2=Test, 3=Rapid, 4=Comparative): ") or "1"
    
    if mode == "4":
        run_comparative_test(size)
    else:
        # Create game
        game = Game(size, True)
        
        # Configure based on mode
        test_mode = mode in ["2", "3"]
        rapid_test = mode == "3"
        show_thinking = mode == "1"
        iterations = 300 if mode == "1" else 200
        
        # Create AI controller
        ai = EnhancedNeuralMCTSController(game, iterations=iterations, 
                                        show_thinking=show_thinking,
                                        test_mode=test_mode, 
                                        rapid_test=rapid_test)
        
        # Play games
        if mode == "1":
            # Normal mode - multiple games for learning
            num_games = int(input("Number of games to play (1-10): ") or "1")
            
            for game_num in range(num_games):
                if game_num > 0:
                    ai.game = Game(size, True)
                    ai.move_count = 0
                    ai.move_history.clear()
                    ai.unique_positions_visited.clear()
                
                ai.play_game(max_moves=200, delay=1.0)
                
                if game_num < num_games - 1:
                    print(f"\nStarting next game in 3 seconds...")
                    time.sleep(3)
        else:
            # Test modes - single game analysis
            ai.play_game(max_moves=150, delay=0.1)
            
