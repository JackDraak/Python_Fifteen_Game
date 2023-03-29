# Python: Fifteen Puzzle Game

Welcome to the Python implementation of the classic 2-dimensional tile game "Fifteen," available on GitHub. This project includes an AI trainer and essential classes for the game's core mechanics. The primary purpose of this repo is to get some introduction to machine learning. It will be done poorly, I guarantee, so do not use it as an example. If that's why you're here, I highly recommend that you don't even look at the AI trainer code. All you need to take away from here for some plug-and-play 15 puzzle are:

- Game.py # Game class, core of the game
- Tile.py # Tile class required by Game class

One or both of the manual controllers, if you need the patterns:

- console_controller.py # play from console
- GUI_controller.py # play with a mouse using PyGame

## Key RL Files

Here's a brief analysis of the key files, if you are looking for a playground to start in:

### AI_trainer_controller.py

This file contains the AITrainerController class, which uses Python decouple for specific configurations. It handles the integration of the AI in training the model using the game environment provided.

### Game.py

This file defines the 'Game' class, which handles the game's core logic. It includes functions for initializing, shuffling, moving, and checking the game's state. It also works on a 4x4 grid and provides a simple console interface for interaction.

### Tile.py

Tile.py contains the 'Tile' class, which represents the numbered tiles of the 15-puzzle game. The class includes basic information about each tile (like its value and position) and the functions to check for equality and obtain a string representation.

## TLDR

The 15-puzzle game project, written in Python, includes an AI trainer and essential classes for the game's core mechanics. Key classes are the AITrainerController, Game, and Tile, which handle training the model, the gameplay, and tile representation, respectively.

## Future

This project was initially built as a "Hello-ML-World" project and started in 2021. It was set aside for a couple of years, and now it has been refurbished and extended with the assistance of ChatGPT4.

## Update

Date: 28 Mar 2023

### AI Trainer Controller

The AI trainer controller is the key component of this project. In summary, the code defines a Q-Network and an AI trainer controller for training an AI to play a sliding puzzle game. The AI trainer controller uses a deep Q-learning algorithm to train the Q-Network based on the game states and valid moves. The model is GPU-optimized and uses PyTorch.

### Training Process

The high-level training process includes the following steps:

1. Initialize the AI_trainer_controller with game dimensions, learning rate, discount factor (gamma), and exploration rate (epsilon).
2. Train the model for the specified number of episodes.
3. Save the trained model to disk using the save_model() method.
4. Load the trained model using the load_model() method before playing the game.
5. Play the game using the play() method, which uses the trained model to make moves.

During training, the AI selects actions using an epsilon-greedy exploration strategy, where it either chooses a random action with probability epsilon or selects the action with the highest Q-value according to the Q-Network.

After training, the AI plays the game using the learned Q-Network to choose the best actions based on the current game state. The trained model is saved and loaded from disk, making it suitable for research, re-training, or other purposes.
