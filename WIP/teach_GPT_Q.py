#teach_GPT_Q.py
from AI_GPT5_Q import AIController
from Game import Game
import time
import os
import matplotlib.pyplot as plt

num_episodes = 500
stats = {
    "rewards": [],
    "steps": [],
    "win_rates": [],
    "distances": []
}

ai = AIController(Game(breadth=3, shuffled=True))

window = 50  # moving average window
wins_in_window = 0

for episode in range(num_episodes):
    total_reward, steps, solved, distance = ai.play_episode(max_steps=200, verbose=False)

    stats["rewards"].append(total_reward)
    stats["steps"].append(steps)
    stats["distances"].append(distance)

    if solved:
        wins_in_window += 1

    # Print stats every N episodes
    if (episode + 1) % window == 0:
        win_rate = wins_in_window / window
        stats["win_rates"].append(win_rate)
        avg_reward = sum(stats["rewards"][-window:]) / window
        avg_steps = sum(stats["steps"][-window:]) / window
        avg_distance = sum(stats["distances"][-window:]) / window
        print(f"Episode {episode+1}: "
              f"Win rate={win_rate:.2%}, "
              f"Avg reward={avg_reward:.2f}, "
              f"Avg steps={avg_steps:.1f}, "
              f"Avg distance={avg_distance:.2f}")
        wins_in_window = 0

# Plot learning curves
plt.figure(figsize=(10,6))
plt.subplot(2, 1, 1)
plt.plot(stats["rewards"], label="Reward per Episode", alpha=0.7)
plt.plot(stats["steps"], label="Steps per Episode", alpha=0.7)
plt.legend()
plt.title("Learning Performance")

plt.subplot(2, 1, 2)
plt.plot(stats["distances"], label="Final Distance", alpha=0.7)
plt.xlabel("Episode")
plt.ylabel("Distance")
plt.legend()

plt.tight_layout()
plt.show()
