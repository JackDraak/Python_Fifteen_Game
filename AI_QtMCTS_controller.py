'''
    This module contains the AI_QtMCTS Controller class, which is responsible for handling AI input and updating the console (for now).
'''
# AI_QtMCTS_controller.py
from time import sleep
from console_controller import Controller as cc
from Game import Game
import random
from typing import Union, Tuple
import numpy as np
    
def uct(node):
    return (node.total_reward / node.visits) + np.sqrt(2 * np.log(node.parent.visits) / node.visits)

class Node:
    def __init__(self, game_state, parent=None):
        self.game_state = game_state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.total_reward = 0

class Controller:
    def __init__(self, game: Game):
        self.game = game
        self.console_controller = cc(game)
    
    def command_check(self, command: str) -> Union[str, Tuple[int, int]]:
        return self.console_controller.command_check(command)
    
    def input_shuffle(self, game: Game) -> None:
        self.console_controller.game.shuffle(50) # TODO: Until further notice, this method will always shuffle 50(?) times, for simplicity.   

    def input_turn(self, game: Game) -> None:
        move_set = self.console_controller.game.get_valid_moves()
        self.console_controller.process_turn(self.game, random.choice(move_set))
            
    # metadata required for ML algorithms includes the following:
    # - game state, represented by a 2D array of integers, the tile labels
    def get_game_state(self) -> list:
        game_labels_as_matrix = game.get_labels_as_matrix()
        return game_labels_as_matrix
        
    # - distance pairings, represented by a 2D array of paired integers, the label & distance from each tile to its goal position
    def get_distance_scores(self) -> list:
        game_distance_scores = game.get_distance_scores()
        return game_distance_scores

    def select(self, node):
        # MCTS Selection step
        while len(node.children) > 0:
            node = max(node.children, key=uct)
        return node

    def expand(self, node):
        # MCTS Expansion step
        if not node.game_state.is_solved():
            moves = node.game_state.get_valid_moves()
            for move in moves:
                new_game_state = node.game_state.copy()
                new_game_state.slide_tile(move)
                new_node = Node(new_game_state, parent=node)
                node.children.append(new_node)
        return node

    def simulate(self, node):
        # MCTS Simulation step
        simulation_game_state = node.game_state.copy()
        last_move = None
        while not simulation_game_state.is_solved():
            moves = simulation_game_state.get_valid_moves()
            # Exclude the last move (inverse move) from the list of valid moves
            if last_move is not None:
                moves = [move for move in moves if move != last_move]
            random_move = random.choice(moves)
            simulation_game_state.move_tile(random_move)
            last_move = (random_move[1], random_move[0])  # Invert the move (row, col) -> (col, row)
            print(game)
            sleep(0.15)
        return 10  # Assuming a reward of 1 for reaching the goal

    def backpropagate(self, node, reward):
        # MCTS Backpropagation step
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    def mcts_search(self, root, iterations):
        for _ in range(iterations):
            selected_node = self.select(root)
            expanded_node = self.expand(selected_node)
            reward = self.simulate(expanded_node)
            self.backpropagate(expanded_node, reward)
        best_child = max(root.children, key=lambda child: child.visits)
        return best_child.game_state
    
    def play(self) -> None:
        moves = list()
        moves.append(0)
        while not self.game.is_solved():
            print(game)
            move_set = self.game.get_valid_moves()
            no_move = True
            while no_move:
                root = Node(self.game)
                best_move = self.mcts_search(root, iterations=100)  # You can adjust the number of iterations
                if not best_move == moves[-1]:
                    if moves[-1] == 0:
                        moves.pop()  # remove leading placeholder 0 from moves list
                    self.console_controller.process_turn(self.game, str(best_move))  # Convert best_move to string
                    moves.append(best_move)
                    no_move = False
                    sleep(0.05)
        print(game)
        print("*** Congratulations, you solved the puzzle! ***\n")
        print(f"Total moves: {len(moves)}")
         
    
if __name__ == '__main__':
    game_size = 4 # TODO extend this so AIs can play any size game
    game = Game(game_size, True)
    controller = Controller(game)
    controller.play()