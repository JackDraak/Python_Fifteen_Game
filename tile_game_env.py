import gym
import numpy as np
from gym import spaces
from Game import Game
from numba import jit

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

        return np.array(new_state, dtype=np.uint8), reward, done, {}

    def reset(self):
        # Reset the game and return the initial state
        self.game = Game(self.game_size, True)
        return np.array(self.game.get_state(), dtype=np.uint8)

    def render(self, mode='human'):
        # Render the current game state
        if mode == 'human':
            print(self.game)
        else:
            raise NotImplementedError

    def close(self):
        pass

@jit(nopython=True)
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

@jit(nopython=True)
def state_to_index(state):
    index = 0
    for i in range(1, len(state)):
        index = cantor_pair(index, state[i])
    return index
