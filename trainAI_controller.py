# deprecated 
#
#trainAI_controller.py
import numpy as np
from tile_game_env import TileGameEnv, state_to_index

# Parameters
game_size = 4
num_episodes = 1000
alpha = 0.1  # learning rate
gamma = 0.99  # discount factor
epsilon = 0.1  # exploration rate

# Initialize the environment
env = TileGameEnv(game_size)

# Initialize the Q-table
q_table = np.zeros((game_size ** 2, env.action_space.n))

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    done = False

    while not done:
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(q_table[state_to_index(np.array(state))])  # Exploit
        next_state, reward, done, _ = env.step(action)

        state_index = state_to_index(np.array(state))
        next_state_index = state_to_index(np.array(next_state))
        
        # Update the Q-table (using vectorized operations)
        q_table[state_index, action] += alpha * (reward + gamma * np.max(q_table[next_state_index]) - q_table[state_index, action])

        state = next_state

    # Update epsilon (decay exploration rate)
    epsilon *= 0.99

    if episode % 5 == 0:
        print(f"Episode {episode} of {num_episodes}")

# Save the Q-table for later use
np.save("q_table.npy", q_table)
