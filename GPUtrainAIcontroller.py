import torch
from tile_game_env import TileGameEnv, state_to_index

# Parameters
game_size = 4
num_episodes = 1000
alpha = 0.1  # learning rate
gamma = 0.99  # discount factor
epsilon = 0.1  # exploration rate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Initialize the environment
env = TileGameEnv(game_size)

# Initialize the Q-table as a PyTorch tensor on the GPU
q_table = torch.zeros((game_size ** 2, env.action_space.n), device=device)

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    state_torch = torch.tensor(state, dtype=torch.uint8, device=device)  # Convert the state to a torch tensor on the GPU device
    done = False

    while not done:
        if torch.rand(1).item() < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = torch.argmax(q_table[state_to_index(state_torch)]).item()  # Exploit
        next_state, reward, done, _ = env.step(action)
        next_state_torch = torch.tensor(next_state, dtype=torch.uint8, device=device)

        state_index = state_to_index(state_torch)
        next_state_index = state_to_index(next_state_torch)

        # Update the Q-table (using vectorized operations)
        q_table[state_index, action] += alpha * (reward + gamma * torch.max(q_table[next_state_index]) - q_table[state_index, action])

        state_torch = next_state_torch

    # Update epsilon (decay exploration rate)
    epsilon *= 0.99

    if episode % 1 == 0:
        print(f"Episode {episode} of {num_episodes}")
        if device.type == 'cuda':
            print(f"Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f"Memory Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

# Save the Q-table for later use
torch.save(q_table, "q_table.pt")