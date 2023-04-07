# AI_trainer_controller.py

# Straight-up: Q-model is almost certainly a terrible choice for this application, and the
# splatter of data tossed at it is not managed well. Again, this is a starting-point, a
# proof of concept, and a vehicle for explosure to ML. Try to not see it as anything more 
# than that, unless you're actually enjoying playing the game -- in that case, live your
# best life!

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from copy import deepcopy
from Game import Game
import multiprocessing as mp


class QNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class AI_trainer_controller:
    def __init__(self, game_breadth: int, learning_rate: float, gamma: float, epsilon: float, buffer_size: int, min_epsilon: float, decay_factor: float, batch_size: int):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device: ", self.device) # INFO
        self.game_breadth = game_breadth
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.buffer_size = buffer_size
        self.min_epsilon = min_epsilon
        self.decay_factor = decay_factor
        self.batch_size = batch_size
        self.q_network = QNetwork(37, 64, game_breadth ** 2).to(self.device) # updated for more parameters
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_function = nn.MSELoss()
        self.memory = []

    def _game_state_to_tensor(self, game: Game) -> torch.Tensor:
        state_1d = game.get_labels_as_list()    # Length 16
        state_2d = game.get_labels_as_matrix()  # Length 16
        distance_set = game.get_distance_set()  # Length 32
        distance_sum = game.get_distance_sum()  # Length 1

        # Flatten the state_2d
        state_2d_flattened = [item for sublist in state_2d for item in sublist]

        # Flatten the distance_set
        distance_set_flattened = [item for sublist in distance_set for inner_list in sublist for item in inner_list]

        # Combine all the features into a single list
        state = state_1d + state_2d_flattened + distance_set_flattened + [distance_sum]

        # # Print lengths of the features, for fun and profit
        # print(f"Length of state_1d: {len(state_1d)}")
        # print(f"Length of state_2d_flattened: {len(state_2d_flattened)}")
        # print(f"Length of distance_set_flattened: {len(distance_set_flattened)}")
        # print(f"Length of distance_sum: 1")

        state = np.array(state, dtype=np.float32)  # Convert the list to a NumPy array
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        return state
    
    def save_model(self, file_path: str) -> None:
        torch.save(self.q_network.state_dict(), file_path)

    def load_model(self, file_path: str) -> None:
        self.q_network.load_state_dict(torch.load(file_path))
        self.q_network.eval()

    def _choose_action(self, game: Game) -> int:
        valid_moves = game.get_valid_moves()
        if np.random.rand() < self.epsilon:
            return random.choice(valid_moves)
        else:
            state = self._game_state_to_tensor(game)
            with torch.no_grad():
                q_values = self.q_network(state)
            valid_q_values = [q_values[0, move - 1] for move in valid_moves]
            return valid_moves[np.argmax(valid_q_values)]

    def _store_transition(self, state: Game, action: int, reward: float, next_state: Game, done: bool) -> None:
        transition = (self._game_state_to_tensor(state), action, reward, self._game_state_to_tensor(next_state), done)
        if len(self.memory) < self.buffer_size:
            self.memory.append(transition)
        else:
            self.memory.pop(0)  # Remove the oldest transition
            self.memory.append(transition)  # Add the new transition

    def _learn_from_memory(self) -> None:
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.cat(states)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.cat(next_states)

    def train(self, episodes: int) -> None:
        for episode in range(episodes):
            game = Game(self.game_breadth, True)
            done = False

            while not done:
                state = deepcopy(game)
                action = self._choose_action(game)
                game.slide_tile(action)
                next_state = deepcopy(game)
                done = game.is_solved()

                if done:
                    reward = 1
                else:
                    reward = -1

                self._store_transition(state, action, reward, next_state, done)
                self._learn_from_memory()

            # Update epsilon using the decay schedule
            self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_factor)

            if episode % 2 == 0:  # TODO Adjust according to preference
                print(f"Episode {episode}: Epsilon {self.epsilon}")              

    def play(self) -> None:
        game = Game(self.game_breadth, True)
        print("Initial state:")
        print(game)

        while not game.is_solved():
            action = self._choose_action(game)
            game.slide_tile(action)
            print(f"\nMoved tile {action}:")
            print(game)

            print("Solved!")

def worker_train(trainer: AI_trainer_controller, episodes: int, result_queue: mp.Queue):
    trainer.train(episodes)
    result_queue.put(trainer.q_network.state_dict())


if __name__ == "__main__":
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    game_breadth = 4 ## TODO set to 3 for low-power systems
    learning_rate = 0.001
    gamma = 0.99
    epsilon = 1.0
    buffer_size = 1000
    min_epsilon = 0.01
    decay_factor = 0.995
    batch_size = 32
    episodes = 1000  # TODO Define the number of episodes here
    model_file_path = "Q_model.pth"
    num_workers = mp.cpu_count()

    workers = []
    result_queue = mp.Queue()
    for i in range(num_workers):
        ai_trainer = AI_trainer_controller(game_breadth, learning_rate, gamma, epsilon, buffer_size, min_epsilon, decay_factor, batch_size)
        worker = mp.Process(target=worker_train, args=(ai_trainer, episodes, result_queue))
        workers.append(worker)

    for worker in workers:
        worker.start()

    for worker in workers:
        worker.join()

    # Retrieve the model states from the result_queue and save the first one as an example
    state_dicts = [result_queue.get() for _ in range(num_workers)]
    torch.save(state_dicts[0], model_file_path)

    # Load the model before playing
    ai_trainer.load_model(model_file_path)
    ai_trainer.play()
