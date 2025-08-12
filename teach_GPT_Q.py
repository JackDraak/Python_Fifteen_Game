#teach_GPT_Q.py
from ai_controller import AIController
from game import Game
import time
import os


def run_learning_session(
    breadth=3,
    episodes=100,
    max_steps=200,
    epsilon_start=0.2,
    epsilon_decay=0.995,
    epsilon_min=0.01,
    model_path="q_table.json",
    verbose_interval=10,
):
    """
    Runs multiple episodes of AI-controlled play to train the model.

    breadth          : puzzle size (3 => 3x3)
    episodes         : how many episodes to run
    max_steps        : max moves per episode
    epsilon_start    : initial exploration rate
    epsilon_decay    : multiplicative decay after each episode
    epsilon_min      : lower bound for epsilon
    model_path       : file to save Q-table after each episode
    verbose_interval : print status every N episodes
    """
    game = Game(breadth, shuffled=True)
    ai = AIController(game)
    
    # Load prior learning if available
    if os.path.exists(model_path):
        ai.load_learning(model_path)
        if verbose_interval:
            print(f"Loaded existing Q-table from {model_path}.")

    ai.epsilon = epsilon_start

    for ep in range(1, episodes + 1):
        # Reset game each episode
        game = Game(breadth, shuffled=True)
        ai.game = game

        total_reward, steps = ai.play_episode(max_steps=max_steps, verbose=False)

        # Decay epsilon for less exploration over time
        ai.epsilon = max(epsilon_min, ai.epsilon * epsilon_decay)

        # Save learning progress
        ai.save_learning(model_path)

        if verbose_interval and ep % verbose_interval == 0:
            print(
                f"Episode {ep}/{episodes}: steps={steps}, reward={total_reward:.2f}, epsilon={ai.epsilon:.3f}"
            )
            time.sleep(0.05)

    print("Training complete.")
    print(f"Q-table saved to {model_path}.")


if __name__ == "__main__":
    run_learning_session()

