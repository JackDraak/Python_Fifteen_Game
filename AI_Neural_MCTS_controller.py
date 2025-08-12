# AI_Intelligent_MCTS_controller.py
"""
Simple, intelligent MCTS controller that actually understands the sliding puzzle.
No neural network complexity - just smart heuristics and proper game understanding.
"""

import random
import math
import time
import numpy as np
from collections import defaultdict, deque
from copy import deepcopy
from typing import List, Tuple, Dict
from Game import Game


class IntelligentMCTSNode:
    def __init__(self, game_state: Game, parent=None, move_made=None):
        self.game_state = game_state
        self.parent = parent
        self.move_made = move_made
        self.children = {}
        self.visits = 0
        self.total_reward = 0.0
        self.untried_moves = game_state.get_valid_moves().copy()

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def best_child(self, exploration_weight=1.4):
        if not self.children:
            return None
        
        best_score = float('-inf')
        best_child = None
        
        for child in self.children.values():
            if child.visits == 0:
                ucb_score = float('inf')  # Prioritize unvisited
            else:
                exploitation = child.total_reward / child.visits
                exploration = exploration_weight * math.sqrt(math.log(self.visits) / child.visits)
                ucb_score = exploitation + exploration
            
            if ucb_score > best_score:
                best_score = ucb_score
                best_child = child
                
        return best_child


class IntelligentMCTSController:
    def __init__(self, game: Game, iterations=300, show_thinking=False, test_mode=False):
        self.game = game
        self.iterations = iterations
        self.show_thinking = show_thinking
        self.test_mode = test_mode
        self.move_count = 0
        
        # Track recent moves to avoid repetition
        self.recent_moves = deque(maxlen=8)
        self.move_history = []
        
    def evaluate_position(self, game_state: Game) -> float:
        """Crystal clear position evaluation based on actual puzzle progress."""
        if game_state.is_solved():
            return 100.0  # Maximum reward for solved state
        
        matrix = game_state.get_labels_as_matrix()
        breadth = game_state.breadth
        score = 0.0
        
        # 1. TILES IN CORRECT POSITIONS (most important)
        correct_positions = 0
        for row in range(breadth):
            for col in range(breadth):
                expected_label = row * breadth + col + 1
                actual_label = matrix[row][col]
                
                if actual_label == expected_label:
                    correct_positions += 1
                    score += 10.0  # Big reward for correct placement
        
        # Percentage of correctly placed tiles
        correct_percentage = correct_positions / (breadth * breadth)
        
        # 2. MANHATTAN DISTANCE (how far each tile is from its home)
        total_manhattan = 0
        for row in range(breadth):
            for col in range(breadth):
                actual_label = matrix[row][col]
                if actual_label != game_state.blank_label:
                    # Where should this tile be?
                    target_row = (actual_label - 1) // breadth
                    target_col = (actual_label - 1) % breadth
                    
                    # How far is it?
                    distance = abs(row - target_row) + abs(col - target_col)
                    total_manhattan += distance
        
        # Convert to score (less distance = higher score)
        max_possible_distance = breadth * breadth * (breadth - 1)  # Worst case
        if max_possible_distance > 0:
            distance_score = (max_possible_distance - total_manhattan) / max_possible_distance * 20.0
        else:
            distance_score = 20.0
        
        score += distance_score
        
        # 3. BONUS for correct rows/columns
        for row in range(breadth):
            row_correct = True
            for col in range(breadth):
                expected = row * breadth + col + 1
                if matrix[row][col] != expected and expected != game_state.blank_label:
                    row_correct = False
                    break
            if row_correct:
                score += 5.0  # Bonus for complete row
        
        for col in range(breadth):
            col_correct = True
            for row in range(breadth):
                expected = row * breadth + col + 1
                if matrix[row][col] != expected and expected != game_state.blank_label:
                    col_correct = False
                    break
            if col_correct:
                score += 5.0  # Bonus for complete column
        
        # 4. LINEAR CONFLICTS (tiles in right row/col but wrong order)
        conflicts = 0
        for row in range(breadth):
            row_tiles = []
            for col in range(breadth):
                label = matrix[row][col]
                if label != game_state.blank_label:
                    correct_row = (label - 1) // breadth
                    if correct_row == row:
                        row_tiles.append((label, col))
            
            # Count conflicts in this row
            for i in range(len(row_tiles)):
                for j in range(i + 1, len(row_tiles)):
                    if row_tiles[i][0] > row_tiles[j][0] and row_tiles[i][1] < row_tiles[j][1]:
                        conflicts += 1
        
        score -= conflicts * 2.0  # Penalty for conflicts
        
        # 5. MOBILITY (having options is good)
        valid_moves = len(game_state.get_valid_moves())
        score += valid_moves * 0.1
        
        return max(0, score)

    def get_move_value(self, game_state: Game, move: int) -> float:
        """Evaluate how good a specific move is."""
        # Create temporary game state
        temp_game = self.copy_game_state(game_state)
        
        if not temp_game.player_move(move):
            return -100.0  # Invalid move
        
        current_score = self.evaluate_position(game_state)
        new_score = self.evaluate_position(temp_game)
        improvement = new_score - current_score
        
        # Penalty for repeating recent moves
        repetition_penalty = 0
        if move in self.recent_moves:
            count = list(self.recent_moves).count(move)
            repetition_penalty = count * 5.0  # Increasing penalty
        
        return improvement - repetition_penalty

    def select(self, node: IntelligentMCTSNode) -> IntelligentMCTSNode:
        current = node
        while not current.game_state.is_solved() and current.is_fully_expanded():
            current = current.best_child()
            if current is None:
                break
        return current

    def expand(self, node: IntelligentMCTSNode) -> IntelligentMCTSNode:
        if not node.untried_moves or node.game_state.is_solved():
            return node
        
        # Evaluate all untried moves and pick the best one
        move_values = []
        for move in node.untried_moves:
            value = self.get_move_value(node.game_state, move)
            move_values.append((move, value))
        
        # Sort by value and pick from top moves
        move_values.sort(key=lambda x: x[1], reverse=True)
        
        # Use some randomness but bias toward better moves
        if len(move_values) >= 3:
            # Pick randomly from top 3 moves
            top_moves = move_values[:3]
            weights = [3, 2, 1]  # Bias toward best move
            move = random.choices([m[0] for m in top_moves], weights=weights)[0]
        else:
            move = move_values[0][0]  # Just pick the best
        
        node.untried_moves.remove(move)
        
        # Create new game state
        new_game_state = self.copy_game_state(node.game_state)
        success = new_game_state.player_move(move)
        
        if not success:
            return node
        
        child = IntelligentMCTSNode(new_game_state, parent=node, move_made=move)
        node.children[move] = child
        
        return child

    def simulate(self, node: IntelligentMCTSNode) -> float:
        """Simulate game with intelligent move selection."""
        simulation_game = self.copy_game_state(node.game_state)
        moves_made = 0
        max_moves = 30
        initial_score = self.evaluate_position(simulation_game)
        
        while not simulation_game.is_solved() and moves_made < max_moves:
            valid_moves = simulation_game.get_valid_moves()
            if not valid_moves:
                break
            
            if moves_made < 10:  # Use intelligent selection for first 10 moves
                move_scores = []
                for move in valid_moves:
                    score = self.get_move_value(simulation_game, move)
                    move_scores.append((move, score))
                
                # Sort and pick from top half
                move_scores.sort(key=lambda x: x[1], reverse=True)
                top_half = max(1, len(move_scores) // 2)
                move = random.choice(move_scores[:top_half])[0]
            else:
                # Random for later moves
                move = random.choice(valid_moves)
            
            simulation_game.player_move(move)
            moves_made += 1
        
        # Calculate reward based on improvement
        if simulation_game.is_solved():
            return 100.0 - moves_made  # Bonus for solving quickly
        else:
            final_score = self.evaluate_position(simulation_game)
            improvement = final_score - initial_score
            return max(0, improvement)

    def backpropagate(self, node: IntelligentMCTSNode, reward: float):
        current = node
        while current is not None:
            current.visits += 1
            current.total_reward += reward
            current = current.parent

    def copy_game_state(self, game: Game) -> Game:
        new_game = Game(game.breadth, False)
        
        for i, tile in enumerate(game.tiles):
            new_game.tiles[i].label = tile.label
            new_game.tiles[i].row = tile.row
            new_game.tiles[i].column = tile.column
            new_game.tiles[i].ordinal = tile.ordinal
        
        new_game.blank_position = game.blank_position
        return new_game

    def get_best_move(self) -> int:
        """MCTS search for best move."""
        if self.game.is_solved():
            return None
            
        root = IntelligentMCTSNode(self.copy_game_state(self.game))
        
        for iteration in range(self.iterations):
            leaf = self.select(root)
            if not leaf.game_state.is_solved():
                leaf = self.expand(leaf)
            reward = self.simulate(leaf)
            self.backpropagate(leaf, reward)
        
        if not root.children:
            return None
            
        # Pick move with highest average reward
        best_child = max(root.children.values(), 
                        key=lambda c: c.total_reward / max(1, c.visits))
        
        if self.show_thinking:
            print(f"Move analysis:")
            for move, child in root.children.items():
                avg_reward = child.total_reward / max(1, child.visits)
                print(f"  Move {move}: {child.visits} visits, avg reward: {avg_reward:.2f}")
            print(f"Selected: {best_child.move_made}")
        
        return best_child.move_made

    def analyze_position(self):
        """Show detailed analysis of current position."""
        print("\n--- Position Analysis ---")
        matrix = self.game.get_labels_as_matrix()
        breadth = self.game.breadth
        
        print("Current state:")
        for row in matrix:
            print("  ", row)
        
        print(f"Overall score: {self.evaluate_position(self.game):.2f}")
        
        # Show what's wrong
        misplaced = []
        for row in range(breadth):
            for col in range(breadth):
                expected = row * breadth + col + 1
                actual = matrix[row][col]
                if actual != expected and expected != self.game.blank_label:
                    target_row = (actual - 1) // breadth
                    target_col = (actual - 1) % breadth
                    distance = abs(row - target_row) + abs(col - target_col)
                    misplaced.append((actual, (row, col), (target_row, target_col), distance))
        
        print("Misplaced tiles:")
        for tile, current, target, dist in misplaced[:5]:  # Show worst 5
            print(f"  Tile {tile}: at {current}, should be {target} (distance: {dist})")
        
        # Show move evaluations
        valid_moves = self.game.get_valid_moves()
        print(f"Move evaluations:")
        move_scores = []
        for move in valid_moves:
            score = self.get_move_value(self.game, move)
            move_scores.append((move, score))
        
        move_scores.sort(key=lambda x: x[1], reverse=True)
        for move, score in move_scores[:5]:
            print(f"  Move {move}: {score:.2f}")

    def play_game(self, max_moves=100, delay=0.5):
        """Play a complete game with detailed analysis."""
        print(f"Starting intelligent MCTS game...")
        initial_score = self.evaluate_position(self.game)
        
        while self.move_count < max_moves and not self.game.is_solved():
            if not self.test_mode:
                print(f"\n--- Move {self.move_count + 1} ---")
                print(self.game)
                self.analyze_position()
            
            best_move = self.get_best_move()
            if best_move is None:
                break
                
            success = self.game.player_move(best_move)
            if success:
                self.move_count += 1
                self.recent_moves.append(best_move)
                self.move_history.append(best_move)
                
                new_score = self.evaluate_position(self.game)
                improvement = new_score - initial_score
                
                if self.test_mode:
                    print(f"{best_move}", end=", ")
                else:
                    sequence = self.game.get_move_sequence(best_move)
                    move_type = "single" if len(sequence) == 1 else f"multi ({len(sequence)})"
                    print(f"AI move: {best_move} ({move_type})")
                    print(f"Score improvement: {improvement:+.2f} (total: {new_score:.2f})")
            else:
                break
                
            if not self.test_mode and delay > 0:
                time.sleep(delay)
        
        # Game finished
        final_score = self.evaluate_position(self.game)
        total_improvement = final_score - initial_score
        
        print(f"\n--- Game Complete ---")
        if not self.test_mode:
            print(self.game)
        
        if self.game.is_solved():
            print(f"*** SOLVED! *** in {self.move_count} moves")
        else:
            print(f"Game ended after {self.move_count} moves")
            print(f"Score improvement: {total_improvement:.2f}")
            
        # Show move frequency analysis
        if len(self.move_history) > 0:
            move_counts = {}
            for move in self.move_history:
                move_counts[move] = move_counts.get(move, 0) + 1
            most_common = sorted(move_counts.items(), key=lambda x: x[1], reverse=True)
            print(f"Move frequency: {most_common}")


if __name__ == '__main__':
    print("Intelligent MCTS Puzzle Solver")
    print("=" * 30)
    
    # Create game
    size = int(input("Game size (3-4 recommended): ") or "4")
    mode = input("Mode (1=Normal with analysis, 2=Test mode): ") or "1"
    
    game = Game(size, True)
    
    # Create AI controller
    test_mode = mode == "2"
    ai = IntelligentMCTSController(game, iterations=300, show_thinking=True, test_mode=test_mode)
    
    # Show initial analysis
    if not test_mode:
        print("\nInitial position:")
        print(game)
        ai.analyze_position()
    
    # Play game
    ai.play_game(max_moves=150, delay=1.0 if not test_mode else 0.1)
    
