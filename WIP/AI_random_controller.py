'''
    This module contains the AI_random Controller class, which is responsible for handling AI input and updating the console (for now).
'''
# AI_random_controller.py
from time import sleep
from console_controller import Controller as cc
from Game import Game
import random
from typing import Union, Tuple

class Controller:
    def __init__(self, game: Game):
        self.game = game
        self.console_controller = cc(game)
    
    # wrapper for the command_check method in console_controller.py
    def command_check(self, command: str) -> Union[str, Tuple[int, int]]:
        return self.console_controller.command_check(command)
    
    # wrapper for the input_shuffle method in console_controller.py
    def input_shuffle(self, game: Game) -> None:
        self.console_controller.game.shuffle(50) # TODO: Until further notice, this method will always shuffle 50(?) times, for simplicity.
        
    # wrapper for the input_turn method in console_controller.py
    def input_turn(self, game: Game) -> None:
        # use random choice from move_set for process_turn
        move_set = self.console_controller.game.get_valid_moves()
        self.console_controller.process_turn(self.game, random.choice(move_set))
    
    # Modelled after console_controller.py, this method is a wrapper for the play method in console_controller.py
    def play(self) -> None:
        moves = list()
        moves.append(0)
        while not self.game.is_solved():
            print(game)
            move_set = self.game.get_valid_moves()
            no_move = True
            while no_move:
                random_move = random.choice(move_set)
                # if random move is last in moves list, pick another random move
                if not random_move == moves[-1]:
                    if moves[-1] == 0:
                        moves.pop() # remove leading placeholder 0 from moves list
                    self.console_controller.process_turn(self.game, str(random_move))  # Convert random_move to string
                    moves.append(random_move)
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