#!/usr/bin/env python3
# integrated_demo.py
"""
Integrated Demo: Reinforcement Learning vs Optimal Search
Educational comparison of two AI approaches for the Fifteen Puzzle

This demo shows students:
1. How RL explores and learns through trial-and-error
2. How Optimal Search uses mathematical guarantees
3. When to use each approach
4. Performance trade-offs and characteristics
5. Real-time statistics and model updates
"""

import time
import json
from typing import Dict, Any, List
from Game import Game
from AI_controller import AIController
from OptimalAI import OptimalAI


class IntegratedDemo:
    """
    Educational demo comparing RL and Optimal AI approaches.

    Provides side-by-side analysis with statistics collection,
    performance metrics, and educational insights.
    """

    def __init__(self):
        self.results_history = []
        self.session_stats = {
            'total_puzzles': 0,
            'rl_successes': 0,
            'optimal_successes': 0,
            'rl_total_time': 0.0,
            'optimal_total_time': 0.0,
            'rl_total_steps': 0,
            'optimal_total_steps': 0
        }

    def run_single_comparison(self, shuffle_count: int = 20, max_steps: int = 100,
                            show_moves: bool = True, seed: int = None) -> Dict[str, Any]:
        """
        Run a single comparison between RL and Optimal AI on the same puzzle.

        Args:
            shuffle_count: Number of shuffle moves to create puzzle
            max_steps: Maximum steps for RL attempt
            show_moves: Whether to show move-by-move progress
            seed: Random seed for reproducible puzzles

        Returns:
            Dictionary with detailed comparison results
        """
        print(f"\n{'='*80}")
        print(f"PUZZLE COMPARISON: {shuffle_count} shuffles (seed: {seed})")
        print(f"{'='*80}")

        # Create identical puzzle for both AIs
        game_rl = Game(4, False, seed=seed)
        game_rl.shuffle(shuffle_count)

        game_optimal = Game(4, False, seed=seed)
        game_optimal.shuffle(shuffle_count)

        print(f"\nINITIAL PUZZLE STATE:")
        print(game_rl)
        print(f"Initial Entropy: {game_rl.get_distance_sum()}")
        print(f"Solvable: {OptimalAI(game_rl)._is_solvable(game_rl.get_state())}")

        results = {
            'shuffle_count': shuffle_count,
            'seed': seed,
            'initial_entropy': game_rl.get_distance_sum(),
            'solvable': True  # We'll verify this
        }

        # Test with Reinforcement Learning AI
        print(f"\n{'-'*40} REINFORCEMENT LEARNING AI {'-'*40}")
        rl_ai = AIController(game_rl, epsilon_start=0.1)  # Reduced exploration for demo

        print(f"Network Type: {'TensorFlow DQN' if rl_ai.use_tensorflow else 'Q-table'}")
        print(f"Exploration Rate (ε): {rl_ai.epsilon:.3f}")
        print(f"Anti-loop measures: Active (max {rl_ai.max_consecutive_moves} consecutive moves)")
        print("\nStarting RL attempt...")

        start_time = time.time()
        rl_reward, rl_steps, rl_solved = rl_ai.play_episode(
            max_steps=max_steps,
            training=False,
            verbose=show_moves
        )
        rl_time = time.time() - start_time

        results['rl'] = {
            'solved': rl_solved,
            'steps': rl_steps,
            'time': rl_time,
            'reward': rl_reward,
            'final_entropy': game_rl.get_distance_sum(),
            'stagnation_episodes': rl_ai.stagnation_counter,
            'exploration_forced': rl_ai.stagnation_counter >= rl_ai.force_exploration_threshold
        }

        # Test with Optimal AI
        print(f"\n{'-'*40} OPTIMAL SEARCH AI {'-'*40}")
        optimal_ai = OptimalAI(game_optimal)

        heuristic_estimate = optimal_ai.heuristic(game_optimal.get_state())
        manhattan_dist = optimal_ai.manhattan_distance(game_optimal.get_state())
        linear_conflict = optimal_ai.linear_conflict(game_optimal.get_state())

        print(f"Algorithm: IDA* (Iterative Deepening A*)")
        print(f"Heuristic: Manhattan Distance + Linear Conflict")
        print(f"Manhattan Distance: {manhattan_dist}")
        print(f"Linear Conflict: {linear_conflict}")
        print(f"Total Heuristic: {heuristic_estimate} moves (lower bound)")
        print("\nStarting optimal search...")

        start_time = time.time()
        optimal_solution = optimal_ai.solve(max_moves=80, timeout=30.0, verbose=show_moves)
        optimal_time = time.time() - start_time

        if optimal_solution:
            optimal_steps = len(optimal_solution)
            optimal_solved = True

            if show_moves:
                print(f"\nExecuting optimal solution...")
                optimal_ai.execute_solution(optimal_solution, verbose=False)
        else:
            optimal_steps = 0
            optimal_solved = False

        results['optimal'] = {
            'solved': optimal_solved,
            'steps': optimal_steps,
            'time': optimal_time,
            'nodes_generated': optimal_ai.nodes_generated,
            'heuristic_estimate': heuristic_estimate,
            'manhattan_distance': manhattan_dist,
            'linear_conflict': linear_conflict,
            'solution': optimal_solution if optimal_solution else None
        }

        # Update session statistics
        self._update_session_stats(results)

        # Print comparison analysis
        self._print_comparison_analysis(results)

        return results

    def _update_session_stats(self, results: Dict[str, Any]):
        """Update running session statistics."""
        self.session_stats['total_puzzles'] += 1

        if results['rl']['solved']:
            self.session_stats['rl_successes'] += 1
            self.session_stats['rl_total_steps'] += results['rl']['steps']
        self.session_stats['rl_total_time'] += results['rl']['time']

        if results['optimal']['solved']:
            self.session_stats['optimal_successes'] += 1
            self.session_stats['optimal_total_steps'] += results['optimal']['steps']
        self.session_stats['optimal_total_time'] += results['optimal']['time']

        self.results_history.append(results)

    def _print_comparison_analysis(self, results: Dict[str, Any]):
        """Print detailed comparison analysis for educational insight."""
        print(f"\n{'='*80}")
        print(f"COMPARISON ANALYSIS")
        print(f"{'='*80}")

        rl = results['rl']
        opt = results['optimal']

        # Success comparison
        print(f"RESULTS:")
        print(f"  RL AI:      {'✓ SOLVED' if rl['solved'] else '✗ FAILED'} in {rl['steps']} steps, {rl['time']:.3f}s")
        print(f"  Optimal AI: {'✓ SOLVED' if opt['solved'] else '✗ FAILED'} in {opt['steps']} steps, {opt['time']:.3f}s")

        if rl['solved'] and opt['solved']:
            efficiency_diff = ((rl['steps'] - opt['steps']) / opt['steps']) * 100
            speed_ratio = rl['time'] / opt['time'] if opt['time'] > 0 else float('inf')

            print(f"\nEFFICIENCY ANALYSIS:")
            print(f"  RL used {efficiency_diff:+.1f}% more moves than optimal")
            print(f"  RL took {speed_ratio:.1f}x longer than optimal search")
            print(f"  Optimal guaranteed shortest path: {opt['steps']} moves")
            print(f"  RL found sub-optimal path: {rl['steps']} moves")

        # Approach comparison
        print(f"\nMETHODOLOGY COMPARISON:")
        print(f"  RL Approach:")
        print(f"    - Trial-and-error learning")
        print(f"    - Epsilon-greedy exploration (ε={rl.get('epsilon', 'N/A')})")
        print(f"    - Anti-loop measures: {'Active' if rl.get('exploration_forced') else 'Inactive'}")
        print(f"    - Stagnation counter: {rl.get('stagnation_episodes', 0)}")
        print(f"    - Final entropy: {rl['final_entropy']}")

        print(f"  Optimal Approach:")
        print(f"    - Mathematical graph search (IDA*)")
        print(f"    - Heuristic-guided: {opt['heuristic_estimate']} move estimate")
        print(f"    - Nodes searched: {opt['nodes_generated']:,}")
        print(f"    - Guaranteed optimal solution")

        # Educational insights
        print(f"\nEDUCATIONAL INSIGHTS:")
        if rl['solved'] and opt['solved']:
            if rl['time'] < opt['time']:
                print(f"  • RL was faster this time, but solution was sub-optimal")
            else:
                print(f"  • Optimal search was faster AND found shortest path")

        if rl.get('exploration_forced'):
            print(f"  • RL activated forced exploration due to stagnation")

        if opt['heuristic_estimate'] == opt['steps']:
            print(f"  • Optimal heuristic was perfect (estimated = actual)")
        else:
            print(f"  • Optimal heuristic underestimated by {opt['steps'] - opt['heuristic_estimate']} moves")

    def run_performance_suite(self, difficulties: List[int] = None, trials_per_difficulty: int = 3):
        """
        Run comprehensive performance comparison across different difficulties.

        Args:
            difficulties: List of shuffle counts to test
            trials_per_difficulty: Number of trials per difficulty level
        """
        if difficulties is None:
            difficulties = [5, 10, 20, 30]

        print(f"\n{'='*80}")
        print(f"PERFORMANCE SUITE: {len(difficulties)} difficulties × {trials_per_difficulty} trials each")
        print(f"{'='*80}")

        suite_results = {}

        for difficulty in difficulties:
            print(f"\n{'#'*60}")
            print(f"DIFFICULTY LEVEL: {difficulty} shuffles")
            print(f"{'#'*60}")

            difficulty_results = []

            for trial in range(trials_per_difficulty):
                print(f"\n--- Trial {trial + 1}/{trials_per_difficulty} ---")

                result = self.run_single_comparison(
                    shuffle_count=difficulty,
                    max_steps=min(100, difficulty * 3),  # Scale steps with difficulty
                    show_moves=False,  # Less verbose for suite
                    seed=trial * 42  # Reproducible seeds
                )

                difficulty_results.append(result)

            suite_results[difficulty] = difficulty_results

        # Print suite summary
        self._print_suite_summary(suite_results)
        return suite_results

    def _print_suite_summary(self, suite_results: Dict[int, List[Dict]]):
        """Print comprehensive summary of performance suite."""
        print(f"\n{'='*80}")
        print(f"PERFORMANCE SUITE SUMMARY")
        print(f"{'='*80}")

        print(f"{'Difficulty':<10} | {'RL Success':<10} | {'Opt Success':<11} | {'Avg RL Steps':<12} | {'Avg Opt Steps':<13} | {'Time Ratio':<10}")
        print(f"{'-'*10} | {'-'*10} | {'-'*11} | {'-'*12} | {'-'*13} | {'-'*10}")

        for difficulty, results in suite_results.items():
            rl_success_rate = sum(1 for r in results if r['rl']['solved']) / len(results) * 100
            opt_success_rate = sum(1 for r in results if r['optimal']['solved']) / len(results) * 100

            rl_avg_steps = sum(r['rl']['steps'] for r in results if r['rl']['solved']) / max(1, sum(1 for r in results if r['rl']['solved']))
            opt_avg_steps = sum(r['optimal']['steps'] for r in results if r['optimal']['solved']) / max(1, sum(1 for r in results if r['optimal']['solved']))

            rl_avg_time = sum(r['rl']['time'] for r in results) / len(results)
            opt_avg_time = sum(r['optimal']['time'] for r in results) / len(results)
            time_ratio = rl_avg_time / max(opt_avg_time, 0.001)

            print(f"{difficulty:<10} | {rl_success_rate:>8.1f}% | {opt_success_rate:>9.1f}% | {rl_avg_steps:>10.1f} | {opt_avg_steps:>11.1f} | {time_ratio:>8.1f}x")

        print(f"\n{'='*80}")
        print(f"SESSION TOTALS:")
        print(f"  Total puzzles tested: {self.session_stats['total_puzzles']}")
        print(f"  RL success rate: {self.session_stats['rl_successes']/max(1,self.session_stats['total_puzzles'])*100:.1f}%")
        print(f"  Optimal success rate: {self.session_stats['optimal_successes']/max(1,self.session_stats['total_puzzles'])*100:.1f}%")
        print(f"  Total RL time: {self.session_stats['rl_total_time']:.2f}s")
        print(f"  Total Optimal time: {self.session_stats['optimal_total_time']:.2f}s")

    def _convert_to_json_serializable(self, obj):
        """Convert numpy types and other non-serializable objects to JSON-compatible types."""
        import numpy as np

        if isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def save_session_data(self, filename: str = None):
        """Save all session data for analysis."""
        if filename is None:
            filename = f"demo_session_{int(time.time())}.json"

        data = {
            'session_stats': self.session_stats,
            'results_history': self.results_history,
            'timestamp': time.time(),
            'summary': 'Integrated demo comparing RL vs Optimal AI approaches'
        }

        # Convert numpy types to JSON-serializable types
        serializable_data = self._convert_to_json_serializable(data)

        with open(filename, 'w') as f:
            json.dump(serializable_data, f, indent=2)

        print(f"\nSession data saved to: {filename}")
        return filename


def main():
    """Main demo runner with interactive menu."""
    print("Fifteen Puzzle: Reinforcement Learning vs Optimal Search")
    print("=" * 60)
    print("Educational AI Comparison Demo")
    print("=" * 60)

    demo = IntegratedDemo()

    while True:
        print(f"\nDEMO MENU:")
        print(f"1. Single puzzle comparison (detailed)")
        print(f"2. Quick comparison (same puzzle, less verbose)")
        print(f"3. Performance suite (multiple difficulties)")
        print(f"4. View session statistics")
        print(f"5. Save session data")
        print(f"6. Exit")

        choice = input(f"\nSelect option (1-6): ").strip()

        if choice == '1':
            shuffle_count = int(input("Enter shuffle count (5-50): ") or "20")
            seed = int(input("Enter seed for reproducibility (or press Enter for random): ") or str(int(time.time())))
            demo.run_single_comparison(shuffle_count=shuffle_count, show_moves=True, seed=seed)

        elif choice == '2':
            demo.run_single_comparison(shuffle_count=15, show_moves=False, seed=42)

        elif choice == '3':
            difficulties = [5, 10, 20, 30, 40]
            trials = int(input("Trials per difficulty (1-5): ") or "2")
            demo.run_performance_suite(difficulties, trials)

        elif choice == '4':
            stats = demo.session_stats
            print(f"\nSESSION STATISTICS:")
            for key, value in stats.items():
                print(f"  {key}: {value}")

        elif choice == '5':
            filename = demo.save_session_data()

        elif choice == '6':
            print("Demo completed. Thank you!")
            break

        else:
            print("Invalid choice. Please select 1-6.")

        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()