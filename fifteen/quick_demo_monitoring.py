#!/usr/bin/env python3
# quick_demo_monitoring.py
"""
Quick demo showing what AI monitoring looks like during training
"""

from Game import Game
from AI_controller import AIController
import json
import time

def demo_monitoring():
    """Demonstrate AI monitoring capabilities with a short training session."""
    print("AI Monitoring Demonstration")
    print("=" * 50)

    # Create AI
    game = Game(4, False)
    ai = AIController(
        game,
        learning_rate=0.01,  # Higher learning rate for faster demo
        epsilon_decay=0.99,  # Faster exploration decay
        memory_size=1000     # Smaller memory for demo
    )

    print("Training AI for 20 episodes to demonstrate monitoring...")
    print("Watch the statistics evolve:")

    # Train for just 20 episodes to show monitoring
    for episode in range(20):
        reward, steps, solved = ai.play_episode(max_steps=100, training=True, verbose=False)

        # Show detailed progress every 5 episodes
        if (episode + 1) % 5 == 0:
            print(f"\n--- Episode {episode + 1} Update ---")
            print(f"Latest Episode:")
            print(f"  Reward: {reward:.1f}")
            print(f"  Steps: {steps}")
            print(f"  Solved: {'✓' if solved else '✗'}")
            print(f"  Final Entropy: {ai.game.get_distance_sum()}")

            # Recent performance
            recent_history = ai.training_history[-5:]
            recent_rewards = [h['reward'] for h in recent_history]
            recent_solved = [h['solved'] for h in recent_history]

            print(f"Recent 5 Episodes:")
            print(f"  Avg Reward: {sum(recent_rewards)/len(recent_rewards):.1f}")
            print(f"  Solve Rate: {sum(recent_solved)/len(recent_solved)*100:.1f}%")
            print(f"  Current Epsilon: {ai.epsilon:.4f}")

    # Save example statistics
    demo_stats_file = "demo_training_stats.json"
    ai.save_model("demo_model")

    print(f"\n--- Final Training Summary ---")
    print(f"Episodes Completed: {ai.episode_count}")
    print(f"Total Steps: {ai.step_count}")
    print(f"Best Score: {ai.best_score:.1f}")
    print(f"Final Epsilon: {ai.epsilon:.4f}")

    # Calculate final solve rate
    solved_count = sum(1 for h in ai.training_history if h['solved'])
    solve_rate = solved_count / len(ai.training_history) * 100
    print(f"Overall Solve Rate: {solve_rate:.1f}%")

    # Show what gets saved
    print(f"\nSaved Files:")
    print(f"  Model: demo_model.h5 (or demo_model_qtable.json)")
    print(f"  Statistics: demo_model_stats.json")

    print(f"\nYou can now analyze this training with:")
    print(f"  python monitor_ai.py --stats demo_model_stats.json")

    # Show sample of what's in the statistics file
    print(f"\nSample Training History (first 3 episodes):")
    for i, episode_data in enumerate(ai.training_history[:3]):
        print(f"  Episode {i+1}: Reward={episode_data['reward']:.1f}, "
              f"Steps={episode_data['steps']}, "
              f"Solved={'Yes' if episode_data['solved'] else 'No'}, "
              f"Epsilon={episode_data['epsilon']:.3f}")

    return ai

if __name__ == "__main__":
    try:
        ai = demo_monitoring()

        print(f"\n" + "=" * 50)
        print("MONITORING CAPABILITIES SUMMARY")
        print("=" * 50)
        print("✓ Real-time episode statistics")
        print("✓ Learning progress tracking")
        print("✓ Model state persistence")
        print("✓ Performance trend analysis")
        print("✓ Solve rate monitoring")
        print("✓ Exploration/exploitation balance")
        print("✓ Reward progression tracking")
        print("✓ Solution efficiency metrics")

        print(f"\nFor longer training sessions, you get:")
        print(f"• Comprehensive plots and visualizations")
        print(f"• Live monitoring during training")
        print(f"• Model architecture analysis")
        print(f"• Comparative performance studies")
        print(f"• Interactive puzzle-solving demos")

    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"Demo error: {e}")
        import traceback
        traceback.print_exc()