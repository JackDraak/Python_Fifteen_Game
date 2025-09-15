'''
    This module contains the console Controller class, which is responsible for handling user input and updating the console.
'''
# console_controller.py 
from Game import Game
from typing import Union, Tuple

class Controller:
    def __init__(self, game: Game):
        self.game = game
    
    @staticmethod
    def command_check(command: str) -> Union[str, Tuple[int, int]]:
        if command == "":                   #  Default behaviour selected.
            return command
        else:
            direction = (0, 0)
            check_command = command.lower()[0]
            if check_command == "q":        #  Quit.
                Controller.quit_game()
            elif check_command == "d":      #  Swap: move target right, move blank left.  
                direction = (0, -1)
            elif check_command == "a":      #  Swap: move target left, move blank right.
                direction = (0, 1)
            elif check_command == "s":      #  Swap: move target down, move blank up.   
                direction = (-1, 0)
            elif check_command == "w":      #  Swap: move target up, move blank down.  
                direction = (1, 0)
            if direction != (0, 0):
                return direction
            return command

    @staticmethod
    def input_game_size() -> Tuple[int, Union[int, None]]:
        size_default = 4                    #  For the classic '15 puzzle', use a grid with a dimension of 4.
        size_max = 31                       #  Grids with dimension >31 have >1000 tiles, would require re-formatting.
        size_min = 3                        #  Grids with dimension <3 are not functionally playable.
        size = size_default
        seed = None
        print("\nWelcome to the console version of the game 'Fifteen'! Goal: slide the game tiles ")
        print("into the open position, 1-by-1, to re-order them. [Or (q)uit at any prompt]")
        print("To play game, ", end="")
        valid_input = False
        while not valid_input:
            grid_size = input(f"simply choose a grid size from {size_min} to {size_max} [default: {size_default}] to begin: ")
            grid_size = Controller.command_check(grid_size)  # Updated this line
            if grid_size == "":             #  Default value selected.
                valid_input = True
            elif type(grid_size) == tuple:  #  Reject WASD input; unrelated to game_size
                pass
            elif grid_size.isdigit():
                size = int(grid_size)
                if size_min <= size <= size_max:
                    valid_input = True

        # Ask for optional seed for deterministic shuffles
        seed_input = input("Enter a seed for deterministic shuffles (optional, press Enter to skip): ")
        seed_input = Controller.command_check(seed_input)
        if seed_input != "" and type(seed_input) != tuple and seed_input.isdigit():
            seed = int(seed_input)

        print()
        return size, seed

    def input_shuffle(self, game: Game) -> None:
        print("*** Congratulations, you solved the puzzle! ***\n")
        print(game)
        shuffled = False
        while not shuffled:
            shuffles = input(f"How many tile-moves would you like to shuffle the board? [default: {game.shuffle_steps}] \n")
            shuffles = self.command_check(shuffles)
            if shuffles == "":              #  Default value selected.
                # Create new game with same parameters but increment seed for variety
                new_seed = game.seed + 1 if game.seed is not None else None
                new_game = Game(game.breadth, True, new_seed)
                game.tiles = new_game.tiles
                game.blank_position = new_game.blank_position
                game.seed = new_seed
                shuffled = True
                pass
            elif type(shuffles) == tuple:   #  Reject WASD input; unrelated to shuffling
                pass
            elif not shuffles.isdigit():    #  Reject non-integer input; has no meaning
                pass
            else:
                shuffled = game.shuffle(int(shuffles))

    def input_turn(self, game: Game) -> None:
        print(game)
        player_move = input("Please, enter the label (#) of the tile you would like to push into the gap.\n" +
                            "{Alternatively, (using W, A, S, or D) enter a direction to 'slide' the tile}\n" +
                            f"Tiles free to move this turn: {game.get_valid_moves()} ")
        player_move = self.command_check(player_move)
        self.process_turn(game, player_move)

    def play(self) -> None:
        while True:
            self.input_turn(self.game)
            if self.game.is_solved():
                self.input_shuffle(self.game)

    def process_turn(self, game: Game, player_move: Union[str, Tuple[int, int]]) -> None:
        print()
        if type(player_move) == tuple:
            wasd_label = game.get_ordinal_label(player_move)
            if game.get_valid_moves().__contains__(wasd_label):
                game.slide_tile(int(wasd_label))
            else:
                print(" ** Unable to move that direction...\n")
        elif not player_move.isdigit():
            print(" ** Please, input a valid tile number (or WASD direction) to move...\n")
        elif not game.player_move(int(player_move)):
            print(f" ** Unable to move tile {player_move}...\n")

    def quit_game() -> None:
        print("\nThank you for playing 'fifteen'. Have a nice day!\n")
        quit()
        

if __name__ == '__main__':
    game_size, seed = Controller.input_game_size()
    game = Game(game_size, True, seed)
    controller = Controller(game)
    controller.play()
