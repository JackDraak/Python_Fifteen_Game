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
    print(f"  Implementation: {'Deep Q-Network' if ai.use_tensorflow else 'Q-Table'}")
    print(f"  State size: {ai.state_size}")
    print(f"  Action space: {ai.action_space_size}")

    # Let AI attempt to solve (without training)
    print("\nLetting untrained AI attempt solution...")
    print("(Note: Untrained AI will make mostly random moves)")

    steps = 0
    max_steps = 20
    last_entropy = game.get_distance_sum()

    while not game.is_solved() and steps < max_steps:
        steps += 1

        valid_moves = game.get_valid_moves()
        action = ai.choose_action(ai.get_state_representation(), valid_moves, training=False)

        print(f"\nStep {steps}:")
        print(f"  Valid moves: {valid_moves}")
        print(f"  AI chooses: {action}")

        success = game.player_move(action)
        if success:
            current_entropy = game.get_distance_sum()
            entropy_change = last_entropy - current_entropy

            print(f"  Entropy: {last_entropy} -> {current_entropy} ({entropy_change:+d})")
            print(game)

            if entropy_change > 0:
                print("  ✓ Good move - entropy reduced!")
            elif entropy_change < 0:
                print("  ↑ Entropy increased - might be strategic?")
            else:
                print("  → Entropy unchanged")

            last_entropy = current_entropy
        else:
            print("  ✗ Move failed!")

        time.sleep(1)  # Pause for readability

    if game.is_solved():
        print(f"\n🎉 Puzzle solved in {steps} steps!")
    else:
        print(f"\nPuzzle not solved in {max_steps} steps")
        print(f"Final entropy: {game.get_distance_sum()}")

    print("\nDemo Notes:")
    print("- This AI is untrained, so moves are mostly random")
    print("- With proper training, the AI learns complex patterns")
    print("- Training teaches the AI when to increase entropy strategically")
    print("- Use train_ai.py to train the AI properly")


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


def training_preview():
    """Preview what training would look like."""
    print("\n" + "=" * 50)
    print("TRAINING PREVIEW")
    print("=" * 50)

    game = Game(4, True)  # Start with shuffled puzzle
    ai = AIController(game)

    print("This is how the AI learns:")
    print(f"1. Start with shuffled puzzle (entropy: {game.get_distance_sum()})")
    print("2. Try random moves initially (high exploration)")
    print("3. Get rewards for reducing entropy")
    print("4. Get penalties for increasing entropy")
    print("5. Get big reward for solving puzzle")
    print("6. Over many episodes, learn which moves lead to solutions")

    print(f"\nReward System Example:")
    initial_entropy = game.get_distance_sum()

    # Simulate a move
    valid_moves = game.get_valid_moves()
    action = valid_moves[0]  # Take first valid move

    game.player_move(action)
    new_entropy = game.get_distance_sum()

    # Calculate what reward would be
    entropy_change = initial_entropy - new_entropy
    reward = ai.calculate_reward(initial_entropy, new_entropy, game.is_solved())

    print(f"  Move: Tile {action}")
    print(f"  Entropy change: {initial_entropy} -> {new_entropy} ({entropy_change:+d})")
    print(f"  Reward: {reward:.3f}")

    if entropy_change > 0:
        print(f"  → Positive reward for reducing entropy!")
    elif entropy_change < 0:
        print(f"  → Negative reward for increasing entropy")
        print(f"  → But sometimes necessary for better future positions!")
    else:
        print(f"  → Neutral entropy change")

    print(f"\n  If puzzle was solved: Reward would be {1000 * ai.reward_scale:.0f}!")


if __name__ == "__main__":
    try:
        simple_demo()

        input("\nPress Enter to continue to entropy analysis...")
        entropy_analysis_demo()

        input("\nPress Enter to continue to training preview...")
        training_preview()

        print("\n" + "=" * 50)
        print("Demo completed!")
        print("To train the AI properly, run:")
        print("  python train_ai.py --episodes 500 --verbose")
        print("To see interactive demos with trained AI:")
        print("  python train_ai.py --load model_name --interactive")

    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"Demo error: {e}")
        import traceback
        traceback.print_exc()