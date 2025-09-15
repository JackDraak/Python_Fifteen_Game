#!/usr/bin/env python3
# monitor_ai.py
"""
Comprehensive AI Monitoring and Visualization Tool

This script provides real-time and post-training analysis of the AI's learning progress,
including detailed statistics, visualizations, and learning state inspection.
"""

import json
import numpy as np
import argparse
import os
from typing import Dict, List, Any
import time
from datetime import datetime

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.patches import Rectangle
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Matplotlib not available. Install with: pip install matplotlib")

from Game import Game
from AI_controller import AIController


class AIMonitor:
    """Comprehensive monitoring system for AI training progress."""

    def __init__(self, ai: AIController = None):
        self.ai = ai
        self.live_data = {
            'episodes': [],
            'rewards': [],
            'steps': [],
            'solve_rates': [],
            'entropies': [],
            'epsilons': [],
            'timestamps': []
        }

    def load_training_stats(self, stats_file: str) -> Dict[str, Any]:
        """Load training statistics from saved file."""
        try:
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            print(f"Loaded training statistics from {stats_file}")
            return stats
        except FileNotFoundError:
            print(f"Statistics file {stats_file} not found")
            return {}

    def print_training_summary(self, stats: Dict[str, Any]):
        """Print comprehensive training summary."""
        if not stats:
            print("No statistics to display")
            return

        print("=" * 60)
        print("AI TRAINING SUMMARY")
        print("=" * 60)

        # Basic info
        print(f"Total Episodes: {stats.get('episode_count', 'Unknown')}")
        print(f"Total Steps: {stats.get('step_count', 'Unknown')}")
        print(f"Best Score: {stats.get('best_score', 'Unknown'):.1f}")
        print(f"Current Epsilon: {stats.get('epsilon', 'Unknown'):.4f}")

        # Model parameters
        params = stats.get('model_parameters', {})
        if params:
            print(f"\nModel Parameters:")
            print(f"  Learning Rate: {params.get('learning_rate', 'Unknown')}")
            print(f"  Discount Factor: {params.get('discount_factor', 'Unknown')}")
            print(f"  Epsilon Decay: {params.get('epsilon_decay', 'Unknown')}")
            print(f"  Board Size: {params.get('breadth', 'Unknown')}x{params.get('breadth', 'Unknown')}")

        # Training history analysis
        history = stats.get('training_history', [])
        if history:
            print(f"\nTraining History Analysis:")
            print(f"  Episodes Recorded: {len(history)}")

            # Calculate solve rate over time
            solved_episodes = [h for h in history if h.get('solved', False)]
            total_solve_rate = len(solved_episodes) / len(history) * 100

            # Recent performance (last 100 episodes)
            recent_history = history[-100:]
            recent_solved = [h for h in recent_history if h.get('solved', False)]
            recent_solve_rate = len(recent_solved) / len(recent_history) * 100

            print(f"  Overall Solve Rate: {total_solve_rate:.1f}%")
            print(f"  Recent Solve Rate (last 100): {recent_solve_rate:.1f}%")

            # Average rewards
            rewards = [h.get('reward', 0) for h in history]
            recent_rewards = [h.get('reward', 0) for h in recent_history]

            print(f"  Average Reward: {np.mean(rewards):.1f}")
            print(f"  Recent Average Reward: {np.mean(recent_rewards):.1f}")

            # Steps analysis (for solved puzzles only)
            if solved_episodes:
                solve_steps = [h.get('steps', 0) for h in solved_episodes]
                recent_solve_steps = [h.get('steps', 0) for h in recent_solved]

                print(f"  Average Steps to Solve: {np.mean(solve_steps):.1f}")
                if recent_solve_steps:
                    print(f"  Recent Average Steps to Solve: {np.mean(recent_solve_steps):.1f}")

            # Learning progress indicators
            first_100 = history[:100]
            last_100 = history[-100:]

            first_solve_rate = len([h for h in first_100 if h.get('solved', False)]) / len(first_100) * 100
            last_solve_rate = len([h for h in last_100 if h.get('solved', False)]) / len(last_100) * 100

            print(f"\nLearning Progress:")
            print(f"  First 100 Episodes Solve Rate: {first_solve_rate:.1f}%")
            print(f"  Last 100 Episodes Solve Rate: {last_solve_rate:.1f}%")
            print(f"  Improvement: {last_solve_rate - first_solve_rate:+.1f}%")

    def plot_training_progress(self, stats: Dict[str, Any], save_path: str = None):
        """Create comprehensive training progress plots."""
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib required for plotting")
            return

        history = stats.get('training_history', [])
        if not history:
            print("No training history to plot")
            return

        # Extract data
        episodes = [h.get('episode', i+1) for i, h in enumerate(history)]
        rewards = [h.get('reward', 0) for h in history]
        steps = [h.get('steps', 0) for h in history]
        solved = [h.get('solved', False) for h in history]
        epsilons = [h.get('epsilon', 0) for h in history]
        entropies = [h.get('final_entropy', 0) for h in history]

        # Create comprehensive plot
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(16, 12))

        # 1. Reward progression with moving average
        ax1.plot(episodes, rewards, alpha=0.6, linewidth=0.5, color='blue', label='Episode Reward')
        if len(rewards) >= 50:
            window = min(50, len(rewards)//4)
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax1.plot(episodes[window-1:], moving_avg, color='red', linewidth=2, label=f'{window}-Episode Moving Average')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Training Reward Progression')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Solve rate over time (rolling window)
        window_size = 50
        solve_rates = []
        episode_windows = []
        for i in range(window_size-1, len(solved)):
            window_solved = solved[i-window_size+1:i+1]
            solve_rate = sum(window_solved) / len(window_solved) * 100
            solve_rates.append(solve_rate)
            episode_windows.append(episodes[i])

        ax2.plot(episode_windows, solve_rates, color='green', linewidth=2)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Solve Rate (%)')
        ax2.set_title(f'Solve Rate Over Time (Rolling {window_size}-Episode Window)')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)

        # 3. Steps to solution (solved puzzles only)
        solved_episodes = [ep for ep, s in zip(episodes, solved) if s]
        solved_steps = [st for st, s in zip(steps, solved) if s]
        if solved_steps:
            ax3.scatter(solved_episodes, solved_steps, alpha=0.6, s=15, color='orange')
            if len(solved_steps) >= 10:
                # Add trend line
                z = np.polyfit(solved_episodes, solved_steps, 1)
                p = np.poly1d(z)
                ax3.plot(solved_episodes, p(solved_episodes), "r--", alpha=0.8, linewidth=2, label='Trend')
                ax3.legend()
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Steps to Solution')
        ax3.set_title('Solution Efficiency (Solved Puzzles Only)')
        ax3.grid(True, alpha=0.3)

        # 4. Exploration rate (epsilon) decay
        ax4.plot(episodes, epsilons, color='purple', linewidth=2)
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Epsilon (Exploration Rate)')
        ax4.set_title('Exploration vs Exploitation Balance')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1)

        # 5. Final entropy distribution
        ax5.hist(entropies, bins=30, alpha=0.7, color='brown', edgecolor='black')
        ax5.axvline(0, color='red', linestyle='--', linewidth=2, label='Perfect Solution (Entropy = 0)')
        ax5.set_xlabel('Final Entropy (Manhattan Distance Sum)')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Final State Quality Distribution')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 6. Learning velocity (reward improvement rate)
        if len(rewards) >= 100:
            velocity_window = 50
            velocities = []
            velocity_episodes = []
            for i in range(velocity_window, len(rewards) - velocity_window):
                before = np.mean(rewards[i-velocity_window:i])
                after = np.mean(rewards[i:i+velocity_window])
                velocity = after - before
                velocities.append(velocity)
                velocity_episodes.append(episodes[i])

            ax6.plot(velocity_episodes, velocities, color='teal', linewidth=2)
            ax6.axhline(0, color='red', linestyle='--', alpha=0.5)
            ax6.set_xlabel('Episode')
            ax6.set_ylabel('Learning Velocity (Reward Change)')
            ax6.set_title('Learning Velocity Over Time')
            ax6.grid(True, alpha=0.3)
        else:
            ax6.text(0.5, 0.5, 'Insufficient data for\nvelocity calculation\n(need 100+ episodes)',
                    transform=ax6.transAxes, ha='center', va='center')
            ax6.set_title('Learning Velocity (Insufficient Data)')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training plots saved to {save_path}")

        plt.show()

    def live_monitor(self, ai: AIController, refresh_interval: int = 10):
        """Monitor AI training in real-time."""
        print("Starting live AI monitoring...")
        print("Press Ctrl+C to stop")

        try:
            while True:
                # Get current stats
                current_time = datetime.now().strftime("%H:%M:%S")

                if hasattr(ai, 'training_history') and ai.training_history:
                    latest = ai.training_history[-1]

                    print(f"\n[{current_time}] AI Status:")
                    print(f"  Episode: {latest.get('episode', '?')}")
                    print(f"  Last Reward: {latest.get('reward', 0):.1f}")
                    print(f"  Last Steps: {latest.get('steps', 0)}")
                    print(f"  Solved: {'Yes' if latest.get('solved', False) else 'No'}")
                    print(f"  Epsilon: {latest.get('epsilon', 0):.4f}")
                    print(f"  Final Entropy: {latest.get('final_entropy', 0)}")

                    # Recent performance
                    if len(ai.training_history) >= 10:
                        recent = ai.training_history[-10:]
                        recent_solve_rate = sum(1 for h in recent if h.get('solved', False)) / len(recent) * 100
                        recent_avg_reward = np.mean([h.get('reward', 0) for h in recent])

                        print(f"  Recent Solve Rate (10): {recent_solve_rate:.1f}%")
                        print(f"  Recent Avg Reward (10): {recent_avg_reward:.1f}")

                else:
                    print(f"[{current_time}] Waiting for training to start...")

                time.sleep(refresh_interval)

        except KeyboardInterrupt:
            print("\nLive monitoring stopped")

    def analyze_model_state(self, ai: AIController):
        """Analyze the current state of the AI model."""
        print("=" * 60)
        print("AI MODEL STATE ANALYSIS")
        print("=" * 60)

        if ai.use_tensorflow:
            print("Model Type: Deep Q-Network (TensorFlow)")
            print(f"Network Architecture:")
            ai.q_network.summary()

            # Get layer weights statistics
            print(f"\nWeight Statistics:")
            for i, layer in enumerate(ai.q_network.layers):
                if hasattr(layer, 'get_weights') and layer.get_weights():
                    weights = layer.get_weights()[0]  # Get weight matrix
                    print(f"  Layer {i+1}: Mean={np.mean(weights):.6f}, "
                          f"Std={np.std(weights):.6f}, "
                          f"Shape={weights.shape}")
        else:
            print("Model Type: Q-Table")
            print(f"Q-Table Size: {len(ai.q_table)} states")

            if ai.q_table:
                # Analyze Q-values
                all_q_values = []
                for state_actions in ai.q_table.values():
                    all_q_values.extend(state_actions.values())

                print(f"Q-Value Statistics:")
                print(f"  Total Q-Values: {len(all_q_values)}")
                print(f"  Mean Q-Value: {np.mean(all_q_values):.3f}")
                print(f"  Std Q-Value: {np.std(all_q_values):.3f}")
                print(f"  Min Q-Value: {np.min(all_q_values):.3f}")
                print(f"  Max Q-Value: {np.max(all_q_values):.3f}")

        print(f"\nTraining Parameters:")
        print(f"  Learning Rate: {ai.learning_rate}")
        print(f"  Discount Factor: {ai.discount_factor}")
        print(f"  Current Epsilon: {ai.epsilon:.6f}")
        print(f"  Memory Size: {len(ai.memory)}/{ai.memory.maxlen}")

    def interactive_dashboard(self, stats_file: str):
        """Create an interactive dashboard for monitoring."""
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib required for interactive dashboard")
            return

        stats = self.load_training_stats(stats_file)
        if not stats:
            return

        # Create the dashboard
        self.plot_training_progress(stats)


def main():
    parser = argparse.ArgumentParser(description='Monitor AI training progress')
    parser.add_argument('--stats', type=str, help='Path to training statistics JSON file')
    parser.add_argument('--live', action='store_true', help='Monitor live training')
    parser.add_argument('--model', type=str, help='Path to saved model for analysis')
    parser.add_argument('--interval', type=int, default=10, help='Live monitoring refresh interval (seconds)')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--save-plot', type=str, help='Save plots to file')

    args = parser.parse_args()

    monitor = AIMonitor()

    if args.stats:
        # Load and analyze saved statistics
        stats = monitor.load_training_stats(args.stats)
        if stats:
            monitor.print_training_summary(stats)

            if args.plot or args.save_plot:
                save_path = args.save_plot if args.save_plot else None
                monitor.plot_training_progress(stats, save_path)

    elif args.live:
        # Live monitoring mode
        print("Live monitoring requires an active AI training session")
        print("This would typically be called from within a training script")
        # Example of how to use:
        # game = Game(4, False)
        # ai = AIController(game)
        # monitor = AIMonitor(ai)
        # monitor.live_monitor(ai, args.interval)

    elif args.model:
        # Analyze saved model
        game = Game(4, False)
        ai = AIController(game)
        ai.load_model(args.model)
        monitor.analyze_model_state(ai)

    else:
        # Show usage
        print("AI Monitoring Tool")
        print("================")
        print("Usage examples:")
        print("  python monitor_ai.py --stats model_stats.json")
        print("  python monitor_ai.py --stats model_stats.json --plot")
        print("  python monitor_ai.py --model saved_model --analyze")
        print("  python monitor_ai.py --live --interval 5")


if __name__ == "__main__":
    main()