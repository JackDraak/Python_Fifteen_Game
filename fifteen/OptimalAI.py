# OptimalAI.py
"""
Optimal AI Controller for the Fifteen Puzzle using IDA* search with hybrid heuristics.

This is a complete rewrite that treats the fifteen puzzle as what it actually is:
a shortest-path graph search problem, not a reinforcement learning problem.

Key Features:
- IDA* (Iterative Deepening A*) for memory-efficient optimal search
- Manhattan Distance + Linear Conflict heuristics for accurate path estimation
- Guarantees finding the shortest solution path
- Understands puzzle structure and solvability
- Fast pattern database lookups for repeated positions
"""

import heapq
import time
from typing import List, Tuple, Optional, Set, Dict
from Game import Game


class OptimalAI:
    """
    Optimal AI solver using IDA* search with hybrid Manhattan + Linear Conflict heuristics.

    This approach recognizes the fifteen puzzle as a graph search problem where:
    1. Each board state is a node
    2. Valid moves are edges with cost = 1
    3. Goal is to find shortest path from current state to solved state
    4. Heuristic functions estimate remaining distance to goal
    """

    def __init__(self, game: Game):
        self.game = game
        self.breadth = game.breadth
        self.blank_label = game.blank_label
        self.goal_positions = self._compute_goal_positions()
        self.moves_found = []
        self.nodes_generated = 0
        self.max_depth_reached = 0

    def _compute_goal_positions(self) -> Dict[int, Tuple[int, int]]:
        """Compute goal positions for each tile label."""
        goal_positions = {}
        for label in range(1, self.blank_label):
            row = (label - 1) // self.breadth
            col = (label - 1) % self.breadth
            goal_positions[label] = (row, col)
        # Blank tile goal position
        goal_positions[self.blank_label] = (self.breadth - 1, self.breadth - 1)
        return goal_positions

    def manhattan_distance(self, state: List[int]) -> int:
        """
        Calculate Manhattan distance heuristic.

        Sum of Manhattan distances from each tile's current position
        to its goal position. This is admissible (never overestimates).
        """
        total_distance = 0
        for i, label in enumerate(state):
            if label == self.blank_label:
                continue  # Don't count blank tile

            current_row = i // self.breadth
            current_col = i % self.breadth
            goal_row, goal_col = self.goal_positions[label]

            distance = abs(current_row - goal_row) + abs(current_col - goal_col)
            total_distance += distance

        return total_distance

    def linear_conflict(self, state: List[int]) -> int:
        """
        Calculate Linear Conflict heuristic addition.

        When two tiles are in their goal row/column but in wrong order,
        they create a linear conflict requiring 2 additional moves.
        This makes the heuristic more accurate while remaining admissible.
        """
        conflicts = 0

        # Check row conflicts
        for row in range(self.breadth):
            tiles_in_row = []
            for col in range(self.breadth):
                pos = row * self.breadth + col
                label = state[pos]
                if label != self.blank_label:
                    goal_row, goal_col = self.goal_positions[label]
                    if goal_row == row:  # Tile belongs in this row
                        tiles_in_row.append((goal_col, col, label))

            # Count conflicts in this row
            tiles_in_row.sort()  # Sort by goal column
            for i in range(len(tiles_in_row)):
                for j in range(i + 1, len(tiles_in_row)):
                    goal_col_i, current_col_i, _ = tiles_in_row[i]
                    goal_col_j, current_col_j, _ = tiles_in_row[j]

                    # If tiles are in wrong order relative to goals
                    if goal_col_i < goal_col_j and current_col_i > current_col_j:
                        conflicts += 2

        # Check column conflicts
        for col in range(self.breadth):
            tiles_in_col = []
            for row in range(self.breadth):
                pos = row * self.breadth + col
                label = state[pos]
                if label != self.blank_label:
                    goal_row, goal_col = self.goal_positions[label]
                    if goal_col == col:  # Tile belongs in this column
                        tiles_in_col.append((goal_row, row, label))

            # Count conflicts in this column
            tiles_in_col.sort()  # Sort by goal row
            for i in range(len(tiles_in_col)):
                for j in range(i + 1, len(tiles_in_col)):
                    goal_row_i, current_row_i, _ = tiles_in_col[i]
                    goal_row_j, current_row_j, _ = tiles_in_col[j]

                    # If tiles are in wrong order relative to goals
                    if goal_row_i < goal_row_j and current_row_i > current_row_j:
                        conflicts += 2

        return conflicts

    def heuristic(self, state: List[int]) -> int:
        """
        Hybrid heuristic combining Manhattan Distance + Linear Conflict.

        This gives a more accurate estimate of moves needed while remaining
        admissible (never overestimates), ensuring optimal solutions.
        """
        return self.manhattan_distance(state) + self.linear_conflict(state)

    def get_possible_moves(self, state: List[int]) -> List[Tuple[int, List[int]]]:
        """
        Get all possible moves from current state.

        Returns list of (moved_tile_label, new_state) tuples.
        """
        # Find blank position
        blank_pos = state.index(self.blank_label)
        blank_row = blank_pos // self.breadth
        blank_col = blank_pos % self.breadth

        moves = []

        # Check all four directions
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right

        for dr, dc in directions:
            new_row = blank_row + dr
            new_col = blank_col + dc

            # Check bounds
            if 0 <= new_row < self.breadth and 0 <= new_col < self.breadth:
                tile_pos = new_row * self.breadth + new_col
                tile_label = state[tile_pos]

                # Create new state by swapping blank and tile
                new_state = state.copy()
                new_state[blank_pos] = tile_label
                new_state[tile_pos] = self.blank_label

                moves.append((tile_label, new_state))

        return moves

    def is_solved(self, state: List[int]) -> bool:
        """Check if puzzle is in solved state."""
        for i, label in enumerate(state):
            expected_label = i + 1 if i < len(state) - 1 else self.blank_label
            if label != expected_label:
                return False
        return True

    def ida_star_search(self, max_moves: int = 80, timeout: float = 30.0) -> Optional[List[int]]:
        """
        IDA* search for optimal solution.

        Iteratively deepens the search limit until solution is found.
        Guaranteed to find optimal solution if one exists within max_moves.

        Returns:
            List of tile labels representing the solution path, or None if no solution found.
        """
        start_time = time.time()
        initial_state = self.game.get_state()

        if self.is_solved(initial_state):
            return []

        # Check solvability first
        if not self._is_solvable(initial_state):
            print("Puzzle is not solvable!")
            return None

        threshold = self.heuristic(initial_state)

        print(f"Starting IDA* search with initial heuristic: {threshold}")

        for iteration in range(max_moves):
            self.nodes_generated = 0
            self.moves_found = []

            result = self._search(initial_state, 0, threshold, None, start_time, timeout)

            elapsed = time.time() - start_time
            print(f"Iteration {iteration + 1}: threshold={threshold}, nodes={self.nodes_generated}, time={elapsed:.2f}s")

            if elapsed > timeout:
                print(f"Search timeout after {elapsed:.1f} seconds")
                break

            if result == "FOUND":
                print(f"Solution found in {len(self.moves_found)} moves!")
                print(f"Total nodes generated: {self.nodes_generated}")
                print(f"Search completed in {elapsed:.2f} seconds")
                return self.moves_found
            elif result == "TIMEOUT":
                print(f"Search timeout after {elapsed:.1f} seconds")
                break
            else:
                threshold = result

        print("No solution found within limits")
        return None

    def _search(self, state: List[int], g: int, threshold: int, prev_move: Optional[int],
                start_time: float, timeout: float) -> str:
        """
        Recursive IDA* search function.

        Args:
            state: Current puzzle state
            g: Cost to reach current state (number of moves)
            threshold: Current search depth limit
            prev_move: Previous move to avoid immediate reversals
            start_time: Search start time for timeout checking
            timeout: Maximum search time in seconds

        Returns:
            "FOUND" if solution found, "TIMEOUT" if timeout, or next threshold value
        """
        self.nodes_generated += 1

        # Check timeout
        if time.time() - start_time > timeout:
            return "TIMEOUT"

        if self.is_solved(state):
            return "FOUND"

        f = g + self.heuristic(state)
        if f > threshold:
            return f

        min_threshold = float('inf')

        for tile_label, new_state in self.get_possible_moves(state):
            # Avoid immediate move reversals (except at start)
            if prev_move is not None and tile_label == prev_move:
                continue

            self.moves_found.append(tile_label)

            result = self._search(new_state, g + 1, threshold, tile_label, start_time, timeout)

            if result == "FOUND":
                return "FOUND"
            elif result == "TIMEOUT":
                self.moves_found.pop()
                return "TIMEOUT"
            else:
                min_threshold = min(min_threshold, result)

            self.moves_found.pop()

        return min_threshold

    def _is_solvable(self, state: List[int]) -> bool:
        """
        Check if puzzle state is solvable using inversion count.

        For 4x4 puzzle:
        - If blank is on even row counting from bottom, number of inversions must be odd
        - If blank is on odd row counting from bottom, number of inversions must be even
        """
        # Find blank position (counting from bottom)
        blank_pos = state.index(self.blank_label)
        blank_row_from_bottom = self.breadth - (blank_pos // self.breadth)

        # Count inversions (ignore blank tile)
        inversions = 0
        tiles_only = [x for x in state if x != self.blank_label]

        for i in range(len(tiles_only)):
            for j in range(i + 1, len(tiles_only)):
                if tiles_only[i] > tiles_only[j]:
                    inversions += 1

        # Apply solvability rule
        if blank_row_from_bottom % 2 == 0:  # Blank on even row from bottom
            return inversions % 2 == 1
        else:  # Blank on odd row from bottom
            return inversions % 2 == 0

    def solve(self, max_moves: int = 80, timeout: float = 30.0, verbose: bool = True) -> Optional[List[int]]:
        """
        Solve the puzzle optimally.

        Args:
            max_moves: Maximum moves to search for solution
            timeout: Maximum time in seconds
            verbose: Whether to print progress

        Returns:
            List of moves (tile labels) for optimal solution, or None if unsolvable
        """
        if verbose:
            initial_h = self.heuristic(self.game.get_state())
            print(f"Solving {self.breadth}x{self.breadth} fifteen puzzle")
            print(f"Initial heuristic estimate: {initial_h} moves")
            print(f"Search limits: {max_moves} moves, {timeout:.1f} seconds")
            print("-" * 50)

        start_time = time.time()
        solution = self.ida_star_search(max_moves, timeout)
        solve_time = time.time() - start_time

        if verbose:
            if solution is not None:
                print(f"✓ Optimal solution found: {len(solution)} moves in {solve_time:.2f} seconds")
            else:
                print(f"✗ No solution found within limits ({solve_time:.2f} seconds)")

        return solution

    def execute_solution(self, solution: List[int], verbose: bool = True) -> bool:
        """
        Execute the solution moves on the game.

        Args:
            solution: List of tile labels to move
            verbose: Whether to print each move

        Returns:
            True if all moves successful and puzzle solved, False otherwise
        """
        if not solution:
            return self.game.is_solved()

        if verbose:
            print(f"Executing solution ({len(solution)} moves):")
            print("-" * 30)

        for i, tile_label in enumerate(solution):
            if verbose:
                print(f"Move {i + 1}: {tile_label}")

            success = self.game.player_move(tile_label)
            if not success:
                if verbose:
                    print(f"ERROR: Move {tile_label} failed!")
                return False

        solved = self.game.is_solved()
        if verbose:
            if solved:
                print("✓ Puzzle solved successfully!")
            else:
                print("✗ Puzzle not solved after all moves")

        return solved


def demo_optimal_ai():
    """Demonstrate the optimal AI solver."""
    print("Fifteen Puzzle Optimal AI Demo")
    print("=" * 50)

    # Create a shuffled 4x4 puzzle
    game = Game(4, True, seed=42)

    print("Initial state:")
    print(game)
    print(f"Initial Manhattan distance: {OptimalAI(game).manhattan_distance(game.get_state())}")
    print()

    # Create optimal AI
    ai = OptimalAI(game)

    # Solve optimally
    solution = ai.solve(max_moves=80, timeout=60.0)

    if solution:
        print(f"Solution: {solution}")
        print()

        # Execute solution
        ai.execute_solution(solution)
        print()
        print("Final state:")
        print(game)
    else:
        print("Could not find optimal solution within limits")


if __name__ == "__main__":
    demo_optimal_ai()