#!/usr/bin/env python3
# train_ai.py
"""
Training script for the Fifteen Puzzle AI Controller

This script provides a comprehensive training and evaluation pipeline for the AI.
It demonstrates the controller's ability to learn the complex patterns required
for optimal puzzle solving, including the paradoxical need to sometimes increase
entropy to achieve better long-term positions.

Usage:
    python train_ai.py --episodes 1000 --eval-freq 100 --save-freq 50
    python train_ai.py --load model_name --evaluate-only
    python train_ai.py --interactive  # Watch AI solve puzzles step by step
"""

import argparse
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from Game import Game
from AI_controller import AIController


def plot_training_progress(ai: AIController, save_path: str = None):
    """Plot training progress graphs."""
    try:
        import matplotlib.pyplot as plt

        history = ai.training_history
        if not history:
            print("No training history to plot")
            return

        episodes = [h['episode'] for h in history]
        rewards = [h['reward'] for h in history]
        steps = [h['steps'] for h in history]
        solved = [h['solved'] for h in history]
        epsilons = [h['epsilon'] for h in history]
        entropies = [h['final_entropy'] for h in history]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Reward over time
        ax1.plot(episodes, rewards, alpha=0.7, label='Episode Reward')
        # Moving average
        window = 50
        if len(rewards) >= window:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax1.plot(episodes[window-1:], moving_avg, 'r-', linewidth=2, label=f'{window}-Episode Average')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Training Reward Progress')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Steps to completion (for solved puzzles only)
        solved_episodes = [e for e, s in zip(episodes, solved) if s]
        solved_steps = [st for st, s in zip(steps, solved) if s]
        if solved_steps:
            ax2.scatter(solved_episodes, solved_steps, alpha=0.6, s=20)
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Steps to Solution')
            ax2.set_title('Solution Efficiency (Solved Puzzles Only)')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No solved puzzles yet', transform=ax2.transAxes, ha='center')
            ax2.set_title('Solution Efficiency (No Solutions Yet)')

        # Exploration rate (epsilon)
        ax3.plot(episodes, epsilons, 'g-', alpha=0.8)
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Epsilon (Exploration Rate)')
        ax3.set_title('Exploration vs Exploitation Balance')
        ax3.grid(True, alpha=0.3)

        # Final entropy distribution
        ax4.hist(entropies, bins=30, alpha=0.7, edgecolor='black')
        ax4.axvline(0, color='red', linestyle='--', linewidth=2, label='Perfect Solution')
        ax4.set_xlabel('Final Entropy (Manhattan Distance Sum)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Final State Quality Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training plots saved to {save_path}")

        plt.show()

    except ImportError:
        print("Matplotlib not available. Install with: pip install matplotlib")


def interactive_demo(ai: AIController, max_episodes: int = 5):
    """Run interactive demo showing AI solving puzzles step by step."""
    print("\n" + "="*60)
    print("INTERACTIVE AI DEMO")
    print("="*60)
    print("Watch the AI solve puzzles step by step!")
    print("The AI will show its thinking process and move decisions.")
    print()

    for episode in range(max_episodes):
        print(f"\n--- Demo Episode {episode + 1}/{max_episodes} ---")

        # Create a new shuffled puzzle
        demo_game = Game(4, True, seed=episode + 42)  # Fixed seed for reproducibility
        ai.game = demo_game

        print("Initial puzzle state:")
        print(demo_game)
        print(f"Initial entropy: {demo_game.get_distance_sum()}")

        input("Press Enter to start solving...")

        state = ai.get_state_representation()
        steps = 0
        max_steps = 100

        while not demo_game.is_solved() and steps < max_steps:
            steps += 1

            # Show current state
            print(f"\n--- Step {steps} ---")
            valid_actions = ai.get_valid_actions()
            current_entropy = demo_game.get_distance_sum()

            # Get AI's action choice
            action = ai.choose_action(state, valid_actions, training=False)

            print(f"Current entropy: {current_entropy}")
            print(f"Valid moves: {valid_actions}")
            print(f"AI chooses to move tile: {action}")

            # Execute the move
            success = demo_game.player_move(action)
            if not success:
                print("Move failed! (This shouldn't happen with a trained AI)")
                continue

            # Show result
            new_entropy = demo_game.get_distance_sum()
            entropy_change = current_entropy - new_entropy

            print(demo_game)
            print(f"New entropy: {new_entropy} (change: {entropy_change:+.1f})")

            if entropy_change > 0:
                print("👍 Entropy reduced - good move!")
            elif entropy_change < 0:
                print("🤔 Entropy increased - strategic move for future benefit?")
            else:
                print("➡️ Entropy unchanged")

            if demo_game.is_solved():
                print("🎉 PUZZLE SOLVED! 🎉")
                break

            # Update state for next iteration
            state = ai.get_state_representation()

            # Pause for readability
            time.sleep(0.5)

        if not demo_game.is_solved():
            print(f"Puzzle not solved within {max_steps} steps")

        print(f"\nDemo episode completed in {steps} steps")

        if episode < max_episodes - 1:
            input("\nPress Enter for next demo episode (or Ctrl+C to exit)...")


def main():
    parser = argparse.ArgumentParser(description='Train and evaluate Fifteen Puzzle AI')
    parser.add_argument('--episodes', type=int, default=500,
                        help='Number of training episodes (default: 500)')
    parser.add_argument('--eval-freq', type=int, default=50,
                        help='Evaluate every N episodes (default: 50)')
    parser.add_argument('--save-freq', type=int, default=100,
                        help='Save model every N episodes (default: 100)')
    parser.add_argument('--board-size', type=int, default=4,
                        help='Board size (default: 4 for classic 15-puzzle)')
    parser.add_argument('--load', type=str, default=None,
                        help='Load pretrained model')
    parser.add_argument('--evaluate-only', action='store_true',
                        help='Only evaluate, do not train')
    parser.add_argument('--interactive', action='store_true',
                        help='Run interactive demo')
    parser.add_argument('--plot', action='store_true',
                        help='Plot training progress')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output during training')

    args = parser.parse_args()

    print("Fifteen Puzzle AI Training System")
    print("=" * 50)
    print(f"Board size: {args.board_size}x{args.board_size}")

    # Create game and AI
    game = Game(args.board_size, False)
    ai = AIController(game)

    print(f"AI Implementation: {'Deep Q-Network (TensorFlow)' if ai.use_tensorflow else 'Q-Table Fallback'}")
    print(f"State space size: {ai.state_size}")
    print(f"Action space size: {ai.action_space_size}")

    # Load pretrained model if specified
    if args.load:
        ai.load_model(args.load)

    # Interactive demo mode
    if args.interactive:
        if not args.load:
            print("\nWARNING: Running interactive demo with untrained AI!")
            print("The AI will make mostly random moves. Consider loading a trained model with --load.")
            if input("Continue anyway? (y/N): ").lower() != 'y':
                return

        interactive_demo(ai)
        return

    # Evaluation only mode
    if args.evaluate_only:
        if not args.load:
            print("ERROR: --evaluate-only requires --load to specify a trained model")
            return

        print(f"\nEvaluating loaded model...")
        results = ai.evaluate(episodes=100, verbose=True)

        if args.plot and ai.training_history:
            plot_training_progress(ai, f"evaluation_plots_{int(time.time())}.png")

        return

    # Training mode
    print(f"\nStarting training for {args.episodes} episodes...")
    print(f"Evaluation frequency: every {args.eval_freq} episodes")
    print(f"Save frequency: every {args.save_freq} episodes")

    start_time = time.time()

    try:
        # Training loop with periodic evaluation
        for checkpoint in range(0, args.episodes, args.eval_freq):
            episodes_to_run = min(args.eval_freq, args.episodes - checkpoint)

            print(f"\nTraining episodes {checkpoint + 1}-{checkpoint + episodes_to_run}...")
            ai.train(episodes_to_run, verbose=args.verbose, save_freq=args.save_freq)

            # Evaluation
            print(f"\nEvaluating at episode {ai.episode_count}...")
            results = ai.evaluate(episodes=20, verbose=False)

            print(f"Evaluation Results:")
            print(f"  Solve Rate: {results['solve_rate']:.1f}%")
            print(f"  Avg Steps: {results['avg_steps']:.1f}")
            print(f"  Avg Reward: {results['avg_reward']:.1f}")
            if results['avg_solve_time']:
                print(f"  Avg Solve Time: {results['avg_solve_time']:.1f} steps")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    # Final evaluation and save
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f} seconds")

    # Final evaluation
    print("\nFinal evaluation (100 episodes)...")
    final_results = ai.evaluate(episodes=100, verbose=True)

    # Save final model
    final_model_name = f"final_model_{args.board_size}x{args.board_size}_{int(time.time())}"
    ai.save_model(final_model_name)

    # Generate plots
    if args.plot:
        plot_path = f"training_progress_{int(time.time())}.png"
        plot_training_progress(ai, plot_path)

    print(f"\nTraining Summary:")
    print(f"  Total Episodes: {ai.episode_count}")
    print(f"  Total Steps: {ai.step_count}")
    print(f"  Final Solve Rate: {final_results['solve_rate']:.1f}%")
    print(f"  Best Training Score: {ai.best_score:.1f}")
    print(f"  Model saved as: {final_model_name}")


if __name__ == "__main__":
    main()