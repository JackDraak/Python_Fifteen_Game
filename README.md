# Python: Fifteen Puzzle Game
_The classic 2-dimensional tile-game "Fifteen", modeled in Python 3_

Thank you for your interest in my project. I've been using this repository as a vehicle to get some experience with the Python syntax. Fundamentally, this means it's in flux. It began as a console game, so that I could hit the ground running (i.e. get some solid results, rapidly, without the need to dive into multiple layers of new framework). If you start to dig through the history, your sanity is in your hands -- I take no responsibility. 

At this point, the _console version_ is fairly polished, and offers not only the game "15", but also any matrix dimension from 3-31 (and now, as a bonus, it offers a (q)uit option at every prompt, as well as dual-play modes: WASD or numerical.)

The code is not perfect, but I was also itching to get a _GUI mode_ going, and so now if you run from __GUI_controller.py__ a crude PyGame interface allows the game to be played with a pointer device.


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