# Python: Fifteen Puzzle Game
_The classic 2-dimensional tile-game "Fifteen", modeled in Python 3_

This GitHub repository contains a Python implementation of the 15-puzzle game with an AI trainer. Here's a brief analysis of the 3 key files:

1. AI_trainer_controller.py:
This file contains the AITrainerController class which uses Python decouple for specific configurations. It handles the integration of the AI in training the model using the game environment provided.

2. Game.py:
This file defines the 'Game' class, which handles the game's core logic. It includes functions for initializing, shuffling, moving, and checking the game's state. It also works on a 4x4 grid and provides a simple console interface for interaction.

3. Tile.py:
Tile.py contains the 'Tile' class, which represents the numbered tiles of the 15-puzzle game. The class includes basic information about each tile (like its value and position) and the functions to check for equality and obtain a string representation.

TLDR:
The 15-puzzle game project, written in Python, includes an AI trainer and essential classes for the game's core mechanics. Key classes are the AITrainerController, Game, and Tile, which handle training the model, the gameplay, and tile representation, respectively.

FUTURE:
This was built as an Hello-ML-World project. Begun in 2021, set a side a couple years, and now refurbished and extended with the assistance of ChatGPT4. 

UPDATE:
28 Mar 2023
--
AI_trainer_controller.py
_(written with the aid of ChatGPT4)_

One of my objectives a couple years ago when I wrote this code, was to toy with MLAs. As a neophyte coder I didn't not find much success at the time.

In summary, the code defines a Q-Network and an AI trainer controller for training an AI to play a sliding puzzle game. The AI trainer controller uses a deep Q-learning algorithm to train the Q-Network based on the game states and valid moves. The model is GPU-optimized and uses PyTorch.

Here's the high-level training process:

Initialize the AI_trainer_controller with game dimensions, learning rate, discount factor (gamma), and exploration rate (epsilon).
Train the model for the specified number of episodes.
Save the trained model to disk using the save_model() method.
Load the trained model using the load_model() method before playing the game.
Play the game using the play() method, which uses the trained model to make moves.
The code creates a Q-Network for approximating the Q-function, which is used to estimate the expected future rewards for taking actions in a given state. The AI trainer controller trains the Q-Network by interacting with the game environment, storing transitions in memory, and learning from the stored transitions using a mean squared error loss function and an Adam optimizer.

During training, the AI selects actions using an epsilon-greedy exploration strategy, where it either chooses a random action with probability epsilon or selects the action with the highest Q-value according to the Q-Network.

After training, the AI plays the game using the learned Q-Network to choose the best actions based on the current game state. The trained model is saved and loaded from disk, making it suitable for research, re-training, or other purposes.