# AI_MCTS_controller.py
"""
Clean MCTS (Monte Carlo Tree Search) AI controller for the sliding puzzle game.
Uses the enhanced Game.py with multi-tile movement support.
"""

import random
import math
import time
from copy import deepcopy
from Game import Game


class MCTSNode:
    def __init__(self, game_state: Game, parent=None, move_made=None):
        self.game_state = game_state
        self.parent = parent
        self.move_made = move_made  # The move that led to this state
        self.children = {}  # Dict of {move: child_node}
        self.visits = 0
        self.total_reward = 0.0
        self.untried_moves = game_state.get_valid_moves().copy()

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def best_child(self, exploration_weight=1.4):
        """Select best child using UCB1 formula."""
        if not self.children:
            return None
        
        best_score = float('-inf')
        best_child = None
        
        for child in self.children.values():
            if child.visits == 0:
                return child  # Prioritize unvisited children
            
            # UCB1 formula
            exploitation = child.total_reward / child.visits
            exploration = exploration_weight * math.sqrt(math.log(self.visits) / child.visits)
            ucb_score = exploitation + exploration
            
            if ucb_score > best_score:
                best_score = ucb_score
                best_child = child
                
        return best_child

    def most_visited_child(self):
        """Return the child with the most visits (for final move selection)."""
        if not self.children:
            return None
        return max(self.children.values(), key=lambda child: child.visits)


class MCTSController:
    def __init__(self, game: Game, iterations=1000, show_thinking=False):
        self.game = game
        self.iterations = iterations
        self.show_thinking = show_thinking
        self.move_count = 0

    def select(self, node: MCTSNode) -> MCTSNode:
        """MCTS Selection: traverse tree using UCB1 until we reach a leaf."""
        current = node
        while not current.game_state.is_solved() and current.is_fully_expanded():
            current = current.best_child()
            if current is None:
                break
        return current

    def expand(self, node: MCTSNode) -> MCTSNode:
        """MCTS Expansion: add a new child node for an untried move."""
        if not node.untried_moves or node.game_state.is_solved():
            return node
        
        # Pick a random untried move
        move = random.choice(node.untried_moves)
        node.untried_moves.remove(move)
        
        # Create new game state by making the move
        new_game_state = self.copy_game_state(node.game_state)
        success = new_game_state.player_move(move)
        
        if not success:
            # If move failed, try another
            return node
        
        # Create child node
        child = MCTSNode(new_game_state, parent=node, move_made=move)
        node.children[move] = child
        
        return child

    def simulate(self, node: MCTSNode) -> float:
        """MCTS Simulation: random playout from current state."""
        simulation_game = self.copy_game_state(node.game_state)
        moves_made = 0
        max_moves = 200  # Prevent infinite loops
        
        while not simulation_game.is_solved() and moves_made < max_moves:
            valid_moves = simulation_game.get_valid_moves()
            if not valid_moves:
                break
                
            move = random.choice(valid_moves)
            simulation_game.player_move(move)
            moves_made += 1
        
        # Reward function: higher reward for solving in fewer moves
        if simulation_game.is_solved():
            return max(1.0, 50.0 - moves_made * 0.1)  # Reward decreases with more moves
        else:
            # Partial reward based on how close we are to solution
            total_distance = simulation_game.get_distance_sum()
            return max(0.1, 1.0 / (1.0 + total_distance * 0.1))

    def backpropagate(self, node: MCTSNode, reward: float):
        """MCTS Backpropagation: update all nodes in the path."""
        current = node
        while current is not None:
            current.visits += 1
            current.total_reward += reward
            current = current.parent

    def copy_game_state(self, game: Game) -> Game:
        """Create a deep copy of the game state."""
        new_game = Game(game.breadth, False)  # Create unsolved game
        
        # Copy the tile positions
        for i, tile in enumerate(game.tiles):
            new_game.tiles[i].label = tile.label
            new_game.tiles[i].row = tile.row
            new_game.tiles[i].column = tile.column
            new_game.tiles[i].ordinal = tile.ordinal
        
        new_game.blank_position = game.blank_position
        return new_game

    def get_best_move(self) -> int:
        """Use MCTS to find the best move from current game state."""
        if self.game.is_solved():
            return None
            
        root = MCTSNode(self.copy_game_state(self.game))
        
        # MCTS iterations
        for iteration in range(self.iterations):
            # Selection
            leaf = self.select(root)
            
            # Expansion
            if not leaf.game_state.is_solved():
                leaf = self.expand(leaf)
            
            # Simulation
            reward = self.simulate(leaf)
            
            # Backpropagation
            self.backpropagate(leaf, reward)
            
            if self.show_thinking and iteration % 100 == 0:
                best_child = root.most_visited_child()
                if best_child:
                    print(f"Iteration {iteration}: Best move so far: {best_child.move_made} "
                          f"(visits: {best_child.visits}, avg reward: {best_child.total_reward/best_child.visits:.3f})")
        
        # Select move with most visits
        best_child = root.most_visited_child()
        return best_child.move_made if best_child else None

    def play_one_move(self) -> bool:
        """Make one AI move. Returns True if move was made, False if game is solved."""
        if self.game.is_solved():
            return False
            
        print(f"\n--- Move {self.move_count + 1} ---")
        print(self.game)
        print("AI is thinking...")
        
        start_time = time.time()
        best_move = self.get_best_move()
        think_time = time.time() - start_time
        
        if best_move is None:
            print("No valid moves found!")
            return False
            
        # Make the move
        success = self.game.player_move(best_move)
        if success:
            self.move_count += 1
            sequence = self.game.get_move_sequence(best_move)
            move_type = "single tile" if len(sequence) == 1 else f"multi-tile ({len(sequence)} tiles)"
            print(f"AI chose move: {best_move} ({move_type}) - Think time: {think_time:.2f}s")
        else:
            print(f"Move {best_move} failed!")
            
        return success

    def play_auto(self, max_moves=100, delay=1.0):
        """Play automatically until solved or max moves reached."""
        print("Starting AI automatic play...")
        print(f"Initial state:")
        print(self.game)
        
        while self.move_count < max_moves and not self.game.is_solved():
            if not self.play_one_move():
                break
            time.sleep(delay)
        
        print(f"\n--- Final Result ---")
        print(self.game)
        if self.game.is_solved():
            print(f"*** AI SOLVED THE PUZZLE! ***")
            print(f"Total moves: {self.move_count}")
        else:
            print(f"AI stopped after {self.move_count} moves (max: {max_moves})")

    def play_interactive(self):
        """Interactive mode - press Enter for each move."""
        print("Interactive AI mode - Press Enter for each move, 'q' to quit")
        print(f"Initial state:")
        print(self.game)
        
        while not self.game.is_solved():
            user_input = input("\nPress Enter for AI move (or 'q' to quit): ").strip().lower()
            if user_input == 'q':
                break
                
            if not self.play_one_move():
                break
        
        if self.game.is_solved():
            print(f"*** AI SOLVED THE PUZZLE! ***")
            print(f"Total moves: {self.move_count}")


if __name__ == '__main__':
    # Test the AI
    print("Creating 4x4 puzzle...")
    game = Game(4, True)  # 4x4 shuffled puzzle
    
    # Create AI controller
    ai = MCTSController(game, iterations=500, show_thinking=True)
    
    # Choose play mode
    mode = input("Choose mode: (1) Auto play, (2) Interactive, (3) Single move: ").strip()
    
    if mode == "1":
        ai.play_auto(max_moves=50, delay=2.0)
    elif mode == "2":
        ai.play_interactive()
    else:
        ai.play_one_move()
        print(f"\nFinal state:")
        print(ai.game)
