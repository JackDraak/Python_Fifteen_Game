import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from copy import deepcopy
from Game import Game


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
    def __init__(self, game_dimension: int, learning_rate: float, gamma: float, epsilon: float):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device: ", self.device) # INFO
        self.game_dimension = game_dimension
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_network = QNetwork(37, 64, game_dimension ** 2).to(self.device) # updated for more parameters
        # self.q_network = QNetwork((game_dimension ** 2) * 2 + game_dimension + 1, 64, game_dimension ** 2).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_function = nn.MSELoss()
        self.memory = []

    def _game_state_to_tensor(self, game: Game) -> torch.Tensor:
        state_1d = game.get_labels_as_list()    # Length 9
        state_2d = game.get_labels_as_matrix()  # Length 9
        distance_set = game.get_distance_set()  # Length 18
        distance_sum = game.get_distance_sum()  # Length 1

        # Flatten the state_2d
        state_2d_flattened = [item for sublist in state_2d for item in sublist]

        # Flatten the distance_set
        distance_set_flattened = [item for sublist in distance_set for inner_list in sublist for item in inner_list]

        # Combine all the features into a single list
        state = state_1d + state_2d_flattened + distance_set_flattened + [distance_sum]

        # # Print lengths of the features
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
        self.memory.append(transition)

    def _learn_from_memory(self) -> None:
        if not self.memory:
            return

        state, action, reward, next_state, done = random.choice(self.memory)
        q_values = self.q_network(state)
        next_q_values = self.q_network(next_state)

        target_q_values = q_values.clone()
        if done:
            target_q_values[0, action - 1] = reward
        else:
            target_q_values[0, action - 1] = reward + self.gamma * torch.max(next_q_values)

        loss = self.loss_function(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, episodes: int) -> None:
        for episode in range(episodes):
            game = Game(self.game_dimension, True)
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

        if episode % 2 == 0: # TODO Adjust according to preference
            print(f"Episode {episode}: Epsilon {self.epsilon}")                        

    def play(self) -> None:
        game = Game(self.game_dimension, True)
        print("Initial state:")
        print(game)

        while not game.is_solved():
            action = self._choose_action(game)
            game.slide_tile(action)
            print(f"\nMoved tile {action}:")
            print(game)

            print("Solved!")


if __name__ == "__main__":
    game_dimension = 3
    learning_rate = 0.001
    gamma = 0.95
    epsilon = 1.0
    episodes = 2
    model_file_path = "Q_model.pth"

    ai_trainer = AI_trainer_controller(game_dimension, learning_rate, gamma, epsilon)
    ai_trainer.train(episodes)
    ai_trainer.save_model(model_file_path)

    # Load the model before playing
    ai_trainer.load_model(model_file_path)
    ai_trainer.play()

