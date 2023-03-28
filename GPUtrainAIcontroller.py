import torch
from tile_game_env import TileGameEnv, state_to_index

# Parameters
game_size = 4
num_episodes = 1000
alpha = 0.1  # learning rate
gamma = 0.99  # discount factor
epsilon = 0.1  # exploration rate

# Check for GPU availability and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device) # TODO consider removing 

# Initialize the environment
env = TileGameEnv(game_size)

# Initialize the Q-table
q_table = torch.zeros(game_size ** 2, env.action_space.n, device=device)

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    done = False

    while not done:
        if torch.rand(1).item() < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = torch.argmax(q_table[state_to_index(torch.tensor(state, device=device))]).item()  # Exploit
        next_state, reward, done, _ = env.step(action)

        state_index = state_to_index(torch.tensor(state, device=device))
        next_state_index = state_to_index(torch.tensor(next_state, device=device))

        # Update the Q-table (using vectorized operations)
        q_table[state_index, action] += alpha * (reward + gamma * torch.max(q_table[next_state_index]) - q_table[state_index, action])

        state = next_state

    # Update epsilon (decay exploration rate)
    epsilon *= 0.99

    if episode % 5 == 0:
        print(f"Episode {episode} of {num_episodes}")

# Save the Q-table for later use
torch.save(q_table, "q_table.pt")