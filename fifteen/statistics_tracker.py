# statistics_tracker.py
"""
Comprehensive Statistics and Model Update Tracking System
For Educational Analysis of AI Performance

This module provides detailed tracking of:
- RL model training progress and updates
- Performance metrics over time
- Behavioral analysis and learning patterns
- Comparative analysis between approaches
- Educational insights for students
"""

import json
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass, asdict


@dataclass
class EpisodeStats:
    """Statistics for a single episode."""
    episode_num: int
    solved: bool
    steps: int
    reward: float
    final_entropy: int
    epsilon: float
    training_time: float
    stagnation_count: int
    forced_exploration: bool
    action_diversity: float  # Number of unique actions / total actions
    entropy_reduction: int  # Initial - final entropy


@dataclass
class ModelUpdate:
    """Information about a model weight update."""
    episode: int
    timestamp: float
    loss: float
    learning_rate: float
    weights_changed: bool
    gradient_norm: Optional[float] = None


@dataclass
class ComparisonResult:
    """Results comparing RL vs Optimal AI on same puzzle."""
    puzzle_id: str
    shuffle_count: int
    initial_entropy: int
    rl_result: Dict[str, Any]
    optimal_result: Dict[str, Any]
    timestamp: float


class StatisticsTracker:
    """
    Comprehensive tracking system for AI performance analysis.

    Provides educational insights into learning progress, model updates,
    and behavioral patterns for both RL and Optimal approaches.
    """

    def __init__(self, save_directory: str = "ai_statistics"):
        self.save_directory = Path(save_directory)
        self.save_directory.mkdir(exist_ok=True)

        # Episode tracking
        self.episode_history: List[EpisodeStats] = []
        self.model_updates: List[ModelUpdate] = []
        self.comparison_results: List[ComparisonResult] = []

        # Real-time metrics
        self.current_session = {
            'start_time': time.time(),
            'episodes_trained': 0,
            'total_steps': 0,
            'solve_count': 0,
            'model_updates': 0
        }

        # Learning curves data
        self.learning_curves = {
            'rewards': deque(maxlen=1000),
            'solve_rates': deque(maxlen=100),  # Rolling solve rate per 10 episodes
            'efficiency': deque(maxlen=1000),  # Steps to solve when successful
            'exploration_rates': deque(maxlen=1000)
        }

        # Behavioral patterns
        self.action_preferences = defaultdict(int)
        self.state_visit_counts = defaultdict(int)
        self.problem_states = []  # States that cause repeated failures

    def record_episode(self, ai_controller, episode_num: int, reward: float,
                      steps: int, solved: bool, training_time: float = 0.0) -> EpisodeStats:
        """
        Record detailed statistics for a single episode.

        Args:
            ai_controller: The AI controller used
            episode_num: Episode number
            reward: Total reward earned
            steps: Steps taken
            solved: Whether puzzle was solved
            training_time: Time spent training

        Returns:
            EpisodeStats object with all collected data
        """
        # Calculate action diversity
        recent_moves = list(ai_controller.recent_moves)
        if recent_moves:
            unique_actions = len(set(recent_moves))
            action_diversity = unique_actions / len(recent_moves)
        else:
            action_diversity = 0.0

        # Calculate entropy reduction
        if hasattr(ai_controller, 'initial_entropy'):
            entropy_reduction = ai_controller.initial_entropy - ai_controller.game.get_distance_sum()
        else:
            entropy_reduction = 0

        # Create episode stats
        episode_stats = EpisodeStats(
            episode_num=episode_num,
            solved=solved,
            steps=steps,
            reward=reward,
            final_entropy=ai_controller.game.get_distance_sum(),
            epsilon=ai_controller.epsilon,
            training_time=training_time,
            stagnation_count=ai_controller.stagnation_counter,
            forced_exploration=ai_controller.stagnation_counter >= ai_controller.force_exploration_threshold,
            action_diversity=action_diversity,
            entropy_reduction=entropy_reduction
        )

        self.episode_history.append(episode_stats)

        # Update real-time metrics
        self.current_session['episodes_trained'] += 1
        self.current_session['total_steps'] += steps
        if solved:
            self.current_session['solve_count'] += 1

        # Update learning curves
        self.learning_curves['rewards'].append(reward)
        self.learning_curves['exploration_rates'].append(ai_controller.epsilon)
        if solved:
            self.learning_curves['efficiency'].append(steps)

        # Update rolling solve rate every 10 episodes
        if len(self.episode_history) % 10 == 0:
            recent_solves = sum(1 for ep in self.episode_history[-10:] if ep.solved)
            self.learning_curves['solve_rates'].append(recent_solves / 10.0)

        # Track action preferences
        for action in recent_moves:
            self.action_preferences[action] += 1

        return episode_stats

    def record_model_update(self, episode: int, loss: float, learning_rate: float,
                          weights_changed: bool = True, gradient_norm: float = None) -> ModelUpdate:
        """
        Record information about model weight updates.

        Args:
            episode: Current episode number
            loss: Training loss
            learning_rate: Current learning rate
            weights_changed: Whether weights were actually updated
            gradient_norm: L2 norm of gradients (if available)

        Returns:
            ModelUpdate object
        """
        update = ModelUpdate(
            episode=episode,
            timestamp=time.time(),
            loss=loss,
            learning_rate=learning_rate,
            weights_changed=weights_changed,
            gradient_norm=gradient_norm
        )

        self.model_updates.append(update)
        self.current_session['model_updates'] += 1

        return update

    def record_comparison(self, puzzle_id: str, shuffle_count: int, initial_entropy: int,
                         rl_result: Dict[str, Any], optimal_result: Dict[str, Any]) -> ComparisonResult:
        """
        Record results from comparing RL vs Optimal AI on same puzzle.

        Args:
            puzzle_id: Unique identifier for the puzzle
            shuffle_count: Number of shuffles used to create puzzle
            initial_entropy: Starting entropy/difficulty
            rl_result: Results from RL AI attempt
            optimal_result: Results from Optimal AI attempt

        Returns:
            ComparisonResult object
        """
        comparison = ComparisonResult(
            puzzle_id=puzzle_id,
            shuffle_count=shuffle_count,
            initial_entropy=initial_entropy,
            rl_result=rl_result,
            optimal_result=optimal_result,
            timestamp=time.time()
        )

        self.comparison_results.append(comparison)
        return comparison

    def generate_learning_report(self, last_n_episodes: int = 100) -> Dict[str, Any]:
        """
        Generate comprehensive learning progress report.

        Args:
            last_n_episodes: Number of recent episodes to analyze

        Returns:
            Dictionary with detailed learning analysis
        """
        if not self.episode_history:
            return {"error": "No episode data available"}

        recent_episodes = self.episode_history[-last_n_episodes:]

        # Basic statistics
        solve_rate = sum(1 for ep in recent_episodes if ep.solved) / len(recent_episodes)
        avg_steps = np.mean([ep.steps for ep in recent_episodes])
        avg_reward = np.mean([ep.reward for ep in recent_episodes])
        avg_entropy_reduction = np.mean([ep.entropy_reduction for ep in recent_episodes])

        # Solved episodes only
        solved_episodes = [ep for ep in recent_episodes if ep.solved]
        if solved_episodes:
            avg_solve_steps = np.mean([ep.steps for ep in solved_episodes])
            avg_solve_reward = np.mean([ep.reward for ep in solved_episodes])
        else:
            avg_solve_steps = 0
            avg_solve_reward = 0

        # Learning trends
        if len(recent_episodes) >= 20:
            first_half = recent_episodes[:len(recent_episodes)//2]
            second_half = recent_episodes[len(recent_episodes)//2:]

            solve_rate_trend = (
                sum(1 for ep in second_half if ep.solved) / len(second_half) -
                sum(1 for ep in first_half if ep.solved) / len(first_half)
            )

            reward_trend = (
                np.mean([ep.reward for ep in second_half]) -
                np.mean([ep.reward for ep in first_half])
            )
        else:
            solve_rate_trend = 0
            reward_trend = 0

        # Behavioral analysis
        forced_exploration_rate = sum(1 for ep in recent_episodes if ep.forced_exploration) / len(recent_episodes)
        avg_action_diversity = np.mean([ep.action_diversity for ep in recent_episodes])
        avg_stagnation = np.mean([ep.stagnation_count for ep in recent_episodes])

        # Most common actions
        action_counts = defaultdict(int)
        for ep in recent_episodes:
            # Would need access to actual moves - simplified for now
            pass

        report = {
            'analysis_period': f"Last {len(recent_episodes)} episodes",
            'basic_metrics': {
                'solve_rate': f"{solve_rate:.1%}",
                'avg_steps_per_episode': f"{avg_steps:.1f}",
                'avg_reward_per_episode': f"{avg_reward:.2f}",
                'avg_entropy_reduction': f"{avg_entropy_reduction:.1f}",
                'avg_steps_when_solved': f"{avg_solve_steps:.1f}",
                'avg_reward_when_solved': f"{avg_solve_reward:.2f}"
            },
            'learning_trends': {
                'solve_rate_improvement': f"{solve_rate_trend:+.1%}",
                'reward_trend': f"{reward_trend:+.2f}",
                'trend_direction': 'improving' if solve_rate_trend > 0.05 else 'stable' if abs(solve_rate_trend) <= 0.05 else 'declining'
            },
            'behavioral_patterns': {
                'forced_exploration_rate': f"{forced_exploration_rate:.1%}",
                'avg_action_diversity': f"{avg_action_diversity:.2f}",
                'avg_stagnation_episodes': f"{avg_stagnation:.1f}",
                'exploration_health': 'good' if forced_exploration_rate < 0.3 else 'concerning'
            },
            'session_totals': {
                'total_episodes': len(self.episode_history),
                'session_duration': f"{(time.time() - self.current_session['start_time'])/60:.1f} minutes",
                'total_model_updates': len(self.model_updates)
            }
        }

        return report

    def generate_comparison_report(self) -> Dict[str, Any]:
        """
        Generate report comparing RL vs Optimal AI performance.

        Returns:
            Dictionary with comparative analysis
        """
        if not self.comparison_results:
            return {"error": "No comparison data available"}

        comparisons = self.comparison_results

        # Success rates
        rl_successes = sum(1 for comp in comparisons if comp.rl_result.get('solved', False))
        optimal_successes = sum(1 for comp in comparisons if comp.optimal_result.get('solved', False))

        rl_success_rate = rl_successes / len(comparisons)
        optimal_success_rate = optimal_successes / len(comparisons)

        # Efficiency when both succeed
        both_solved = [comp for comp in comparisons
                      if comp.rl_result.get('solved', False) and comp.optimal_result.get('solved', False)]

        if both_solved:
            rl_avg_steps = np.mean([comp.rl_result['steps'] for comp in both_solved])
            optimal_avg_steps = np.mean([comp.optimal_result['steps'] for comp in both_solved])
            efficiency_ratio = rl_avg_steps / optimal_avg_steps if optimal_avg_steps > 0 else float('inf')

            rl_avg_time = np.mean([comp.rl_result.get('time', 0) for comp in both_solved])
            optimal_avg_time = np.mean([comp.optimal_result.get('time', 0) for comp in both_solved])
            time_ratio = rl_avg_time / optimal_avg_time if optimal_avg_time > 0 else float('inf')
        else:
            efficiency_ratio = float('inf')
            time_ratio = float('inf')

        # Difficulty analysis
        difficulty_breakdown = defaultdict(lambda: {'rl_solved': 0, 'optimal_solved': 0, 'total': 0})
        for comp in comparisons:
            difficulty = comp.shuffle_count
            difficulty_breakdown[difficulty]['total'] += 1
            if comp.rl_result.get('solved', False):
                difficulty_breakdown[difficulty]['rl_solved'] += 1
            if comp.optimal_result.get('solved', False):
                difficulty_breakdown[difficulty]['optimal_solved'] += 1

        report = {
            'total_comparisons': len(comparisons),
            'success_rates': {
                'rl_success_rate': f"{rl_success_rate:.1%}",
                'optimal_success_rate': f"{optimal_success_rate:.1%}",
                'success_gap': f"{optimal_success_rate - rl_success_rate:+.1%}"
            },
            'efficiency_analysis': {
                'both_solved_cases': len(both_solved),
                'rl_vs_optimal_steps': f"{efficiency_ratio:.1f}x longer" if efficiency_ratio != float('inf') else "N/A",
                'rl_vs_optimal_time': f"{time_ratio:.1f}x slower" if time_ratio != float('inf') else "N/A"
            },
            'difficulty_breakdown': {
                str(difficulty): {
                    'rl_rate': f"{data['rl_solved']/data['total']:.1%}",
                    'optimal_rate': f"{data['optimal_solved']/data['total']:.1%}",
                    'cases': data['total']
                }
                for difficulty, data in difficulty_breakdown.items()
            },
            'educational_insights': self._generate_educational_insights(rl_success_rate, optimal_success_rate, efficiency_ratio)
        }

        return report

    def _generate_educational_insights(self, rl_success_rate: float, optimal_success_rate: float,
                                     efficiency_ratio: float) -> List[str]:
        """Generate educational insights from comparison data."""
        insights = []

        if optimal_success_rate > rl_success_rate + 0.2:
            insights.append("Optimal AI significantly outperforms RL in success rate - demonstrates value of mathematical guarantees")

        if efficiency_ratio > 2.0:
            insights.append("RL solutions are much longer than optimal - shows exploration vs exploitation trade-off")
        elif efficiency_ratio < 1.5:
            insights.append("RL solutions are surprisingly close to optimal - indicates good learning")

        if rl_success_rate < 0.3:
            insights.append("RL struggling with current puzzles - may need more training or parameter tuning")

        if optimal_success_rate < 0.8:
            insights.append("Some puzzles too hard even for optimal solver - hitting complexity limits")

        return insights

    def save_statistics(self, filename: str = None) -> str:
        """
        Save all tracked statistics to file.

        Args:
            filename: Optional custom filename

        Returns:
            Path to saved file
        """
        if filename is None:
            filename = f"statistics_{int(time.time())}.json"

        filepath = self.save_directory / filename

        # Prepare data for JSON serialization
        data = {
            'episode_history': [asdict(ep) for ep in self.episode_history],
            'model_updates': [asdict(update) for update in self.model_updates],
            'comparison_results': [asdict(comp) for comp in self.comparison_results],
            'current_session': self.current_session,
            'learning_curves': {key: list(values) for key, values in self.learning_curves.items()},
            'action_preferences': dict(self.action_preferences),
            'metadata': {
                'created_at': time.time(),
                'total_episodes': len(self.episode_history),
                'total_comparisons': len(self.comparison_results)
            }
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Statistics saved to: {filepath}")
        return str(filepath)

    def load_statistics(self, filename: str) -> bool:
        """
        Load previously saved statistics.

        Args:
            filename: Path to statistics file

        Returns:
            True if successful, False otherwise
        """
        try:
            filepath = Path(filename)
            if not filepath.exists():
                filepath = self.save_directory / filename

            with open(filepath, 'r') as f:
                data = json.load(f)

            # Restore episode history
            self.episode_history = [EpisodeStats(**ep) for ep in data.get('episode_history', [])]

            # Restore model updates
            self.model_updates = [ModelUpdate(**update) for update in data.get('model_updates', [])]

            # Restore comparison results
            self.comparison_results = [ComparisonResult(**comp) for comp in data.get('comparison_results', [])]

            # Restore other data
            self.current_session = data.get('current_session', self.current_session)

            learning_curves_data = data.get('learning_curves', {})
            for key, values in learning_curves_data.items():
                self.learning_curves[key] = deque(values, maxlen=self.learning_curves[key].maxlen)

            self.action_preferences = defaultdict(int, data.get('action_preferences', {}))

            print(f"Statistics loaded from: {filepath}")
            print(f"Loaded {len(self.episode_history)} episodes, {len(self.comparison_results)} comparisons")

            return True

        except Exception as e:
            print(f"Error loading statistics: {e}")
            return False

    def create_learning_plots(self, save_plots: bool = True) -> List[str]:
        """
        Create visualization plots of learning progress.

        Args:
            save_plots: Whether to save plots to files

        Returns:
            List of plot filenames created
        """
        if not self.episode_history:
            print("No data available for plotting")
            return []

        plot_files = []

        try:
            # Plot 1: Solve rate over time
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 3, 1)
            solve_rates = list(self.learning_curves['solve_rates'])
            if solve_rates:
                episodes = range(10, len(solve_rates) * 10 + 1, 10)
                plt.plot(episodes, solve_rates, 'b-', linewidth=2)
                plt.title('Solve Rate Over Time')
                plt.xlabel('Episode')
                plt.ylabel('Solve Rate (per 10 episodes)')
                plt.grid(True, alpha=0.3)

            # Plot 2: Reward progression
            plt.subplot(1, 3, 2)
            rewards = list(self.learning_curves['rewards'])
            if rewards:
                # Moving average for smoother curve
                window_size = min(50, len(rewards) // 10)
                if window_size > 1:
                    moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                    plt.plot(range(window_size//2, len(rewards) - window_size//2), moving_avg, 'r-', linewidth=2, label=f'Moving Avg ({window_size})')
                plt.plot(rewards, 'r-', alpha=0.3, label='Raw Rewards')
                plt.title('Reward Progression')
                plt.xlabel('Episode')
                plt.ylabel('Reward')
                plt.legend()
                plt.grid(True, alpha=0.3)

            # Plot 3: Exploration rate
            plt.subplot(1, 3, 3)
            exploration_rates = list(self.learning_curves['exploration_rates'])
            if exploration_rates:
                plt.plot(exploration_rates, 'g-', linewidth=2)
                plt.title('Exploration Rate (ε)')
                plt.xlabel('Episode')
                plt.ylabel('Epsilon')
                plt.grid(True, alpha=0.3)

            plt.tight_layout()

            if save_plots:
                plot_file = self.save_directory / "learning_progress.png"
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plot_files.append(str(plot_file))

            plt.show()

        except ImportError:
            print("Matplotlib not available - cannot create plots")
        except Exception as e:
            print(f"Error creating plots: {e}")

        return plot_files


def create_demo_tracker() -> StatisticsTracker:
    """Create a pre-configured tracker for demo use."""
    tracker = StatisticsTracker()

    # Add some sample data for demonstration
    print("Statistics Tracker initialized")
    print(f"Save directory: {tracker.save_directory}")

    return tracker


if __name__ == "__main__":
    # Demo of statistics tracker
    tracker = create_demo_tracker()

    print("\nStatistics Tracker Demo")
    print("=" * 50)

    # Simulate some episode data
    print("Simulating training data...")

    # This would normally be integrated with actual training
    # For demo, we'll create synthetic data

    report = tracker.generate_learning_report(10)
    print("\nSample Learning Report:")
    print(json.dumps(report, indent=2))