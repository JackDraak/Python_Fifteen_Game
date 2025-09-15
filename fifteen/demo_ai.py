#!/usr/bin/env python3
# demo_ai.py
"""
Quick demo of the AI Controller for the Fifteen Puzzle

This script demonstrates the AI's ability to understand the puzzle's complexity
and make intelligent moves based on entropy reduction and pattern recognition.
"""

import time
from Game import Game
from AI_controller import AIController


def simple_demo():
    """Run a simple demo showing AI solving the standard 15-puzzle."""
    print("Fifteen Puzzle AI Controller Demo")
    print("=" * 50)

    # Start with the standard 4x4 fifteen puzzle
    print("Creating a 4x4 puzzle (standard 15-puzzle)...")
    game = Game(4, False)  # Start solved
    print("Solved state:")
    print(game)

    # Shuffle it
    print("Shuffling puzzle...")
    game.shuffle(20)
    print("Shuffled state:")
    print(game)
    print(f"Initial entropy (distance sum): {game.get_distance_sum()}")

    # Create AI
    ai = AIController(game)

    print(f"\nAI Configuration:")
    print(f"  Implementation: Optimal IDA* Search")
    print(f"  Heuristic: Manhattan Distance + Linear Conflict")
    print(f"  Guarantees: Optimal solutions (shortest path)")

    # Use the optimal AI to solve
    print("\nSolving puzzle optimally...")
    print("(This AI finds the shortest possible solution)")

    solution = ai.solve_puzzle(verbose=False)

    if solution:
        print(f"✓ Optimal solution found: {len(solution)} moves")
        print(f"  Solve time: {ai.solve_time:.3f} seconds")
        print(f"  Nodes generated: {ai.nodes_generated}")
        print(f"  Initial heuristic: {ai.optimal_ai.heuristic(game.get_state())} moves")

        print(f"\nSolution sequence: {solution}")

        print("\nExecuting solution step by step...")
        success = ai.execute_solution(verbose=True)

        if success:
            print(f"\n✓ Puzzle solved successfully!")
            print("Final state:")
            print(game)
        else:
            print("✗ Solution execution failed")
    else:
        print("✗ Could not find solution within limits")
        return

    # Show statistics
    stats = ai.get_statistics()
    print(f"\nFinal Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")



def entropy_analysis_demo():
    """Demonstrate the entropy concept central to the puzzle."""
    print("\n" + "=" * 50)
    print("ENTROPY ANALYSIS DEMO")
    print("=" * 50)

    print("Understanding puzzle entropy (disorder)...")

    # Show solved state
    game = Game(4, False)
    print("\nSOLVED STATE (minimum entropy):")
    print(game)
    print(f"Entropy (distance sum): {game.get_distance_sum()}")

    # Show progressively more shuffled states
    entropies = []

    for shuffles in [5, 20, 50, 100]:
        game = Game(4, False)
        game.shuffle(shuffles)
        entropy = game.get_distance_sum()
        entropies.append(entropy)

        print(f"\nAfter {shuffles} shuffle moves:")
        print(game)
        print(f"Entropy: {entropy}")

    print(f"\nEntropy progression: {entropies}")
    print("Notice how entropy generally increases with more shuffling")
    print("The AI must learn to reduce this entropy back to 0")


def optimal_solver_demo():
    """Show how the optimal solver works."""
    print("\n" + "=" * 50)
    print("OPTIMAL SOLVER ANALYSIS")
    print("=" * 50)

    game = Game(4, True)  # Start with shuffled puzzle
    ai = AIController(game)

    print("This is how the optimal solver works:")
    print(f"1. Analyze current puzzle state (entropy: {game.get_distance_sum()})")
    print(f"2. Calculate heuristic estimate: {ai.get_heuristic_value()} moves")
    print(f"3. Check solvability: {ai.is_solvable()}")
    print("4. Use IDA* search to find shortest path")
    print("5. Manhattan Distance + Linear Conflict heuristics")
    print("6. Guarantee optimal (shortest) solution")

    print(f"\nHeuristic Analysis:")
    state = game.get_state()
    manhattan_dist = ai.optimal_ai.manhattan_distance(state)
    linear_conflict = ai.optimal_ai.linear_conflict(state)
    total_heuristic = ai.get_heuristic_value()

    print(f"  Manhattan Distance: {manhattan_dist}")
    print(f"  Linear Conflict: {linear_conflict}")
    print(f"  Total Heuristic: {total_heuristic}")
    print(f"  (Lower bound estimate of moves needed)")

    print(f"\nKey Differences from Learning Approaches:")
    print(f"  ✓ No training required - always optimal")
    print(f"  ✓ Deterministic - same input gives same output")
    print(f"  ✓ Mathematically sound - uses graph theory")
    print(f"  ✓ Fast for practical puzzles")
    print(f"  ✓ Understands puzzle structure, not just patterns")


if __name__ == "__main__":
    try:
        simple_demo()

        input("\nPress Enter to continue to entropy analysis...")
        entropy_analysis_demo()

        input("\nPress Enter to continue to optimal solver analysis...")
        optimal_solver_demo()

        print("\n" + "=" * 50)
        print("Demo completed!")
        print("The optimal AI now guarantees shortest solutions!")
        print("No training required - it uses mathematical graph search.")
        print("For evaluation on multiple puzzles, run:")
        print("  python AI_controller.py")
        print("  python OptimalAI.py")

    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"Demo error: {e}")
        import traceback
        traceback.print_exc()