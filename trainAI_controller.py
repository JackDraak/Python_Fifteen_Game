# trainAI_controller.py
import numpy as np
import gym
from gym import spaces
from Game import Game

# Create custom environment      
class TileGameEnv(gym.Env):
    def __init__(self, game_size):
        super(TileGameEnv, self).__init__()

        self.game_size = game_size
        self.game = Game(game_size, True)  # Pass 'True' to shuffle the game
        
        # Define action space
        self.action_space = spaces.Discrete(4)  # Four possible actions: up, down, left, right
        
        # Define observation space
        self.observation_space = spaces.Box(low=0, high=game_size**2-1, shape=(game_size, game_size), dtype=np.uint8)

    def step(self, action):
        # Apply the action to the game, update the state, and calculate the reward
        prev_state = self.game.get_state()
        self.game.slide_tile(action)
        new_state = self.game.get_state()

        if new_state == prev_state:  # invalid move
            reward = -1
            done = False
        elif self.game.is_solved():  # puzzle solved
            reward = 100
            done = True
        else:
            reward = -1
            done = False

        return new_state, reward, done, {}

    def reset(self):
        # Reset the game and return the initial state
        self.game = Game(self.game_size, True)
        return self.game.get_state()

    def render(self, mode='human'):
        # Render the current game state
        if mode == 'human':
            print(self.game)
        else:
            raise NotImplementedError

    def close(self):
        pass

def cantor_pair(a, b):
    ab = a + b
    ab1 = a + b + 1
    factors = []
    i = 2
    while ab > 1 or ab1 > 1:
        if ab % i == 0 or ab1 % i == 0:
            factors.append(i)
            if ab % i == 0:
                ab //= i
            if ab1 % i == 0:
                ab1 //= i
        else:
            i += 1
    result = 1
    for factor in set(factors):
        count1 = factors.count(factor)
        count2 = 0
        for j in range(count1):
            if b % factor == 0:
                b //= factor
                count2 += 1
        result *= factor ** (count1 - count2)
    return result

def state_to_index(state):
    index = 0
    for i in range(1, len(state)):
        index = cantor_pair(index, state[i])
    return index

# Implement Q-Learning RLA
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
            action = np.argmax(q_table[state_to_index(state)])  # Exploit
        next_state, reward, done, _ = env.step(action)

        state_index = state_to_index(state)
        next_state_index = state_to_index(next_state)
        # Update the Q-table
        q_table[state_index, action] += alpha * (reward + gamma * np.max(q_table[next_state_index]) - q_table[state_index, action])

        state = next_state

    # Update epsilon (decay exploration rate)
    epsilon *= 0.99

# Save the Q-table for later use
np.save("q_table.npy", q_table)
