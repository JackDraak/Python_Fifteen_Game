# console_controller.py -- a controller for the console version of the game.
from Game import Game
from typing import Union, Tuple

def command_check(command: str) -> Union[str, Tuple[int, int]]:
    if command == "":                   # Default behaviour selected.
        return command
    else:
        direction = (0, 0)
        check_command = command.lower()[0]
        if check_command == "q":        # Quit.
            quit_game()
        elif check_command == "a":      # Move blank left.  (tile on left takes blank position)
            direction = (0, -1)
        elif check_command == "d":      # Move blank right. (tile on right takes blank position)
            direction = (0, 1)
        elif check_command == "w":      # Move blank up.    (tile above takes blank position)
            direction = (-1, 0)
        elif check_command == "s":      # Move blank down.  (tile below takes blank position)
            direction = (1, 0)
        if direction != (0, 0):
            return direction
        return command

def input_game_size() -> int:
    size_default = 4                    # For the classic '15 puzzle', use a grid with a dimension of 4.
    size_max = 31                       # Grids with dimension >31 have >1000 tiles, would require re-formatting.
    size_min = 3                        # Grids with dimension <3 are not functionally playable.
    size = size_default
    print("\nGoal: slide the game tiles into the open position, 1-by-1, to re-order them. [Or (q)uit at any prompt]")
    print("To play the classic tile game, '15', ", end="")
    valid_input = False
    while not valid_input:
        grid_size = input(f"please choose a grid size from {size_min} to {size_max} [default: {size_default}] ")
        grid_size = command_check(grid_size)
        if grid_size == "":             # Default value selected.
            valid_input = True
        elif type(grid_size) == tuple:  # Reject WASD input; unrelated to game_size
            pass
        elif grid_size.isdigit():
            size = int(grid_size)
            if size_min <= size <= size_max:
                valid_input = True
    print()
    return size

def input_shuffle(game: Game) -> None:
    print("*** Congratulations, you solved the puzzle! ***\n")
    print(game)
    shuffled = False
    while not shuffled:
        shuffles = input(f"How many times would you like to shuffle? [default: {game.shuffle_default}] \n")
        shuffles = command_check(shuffles)
        if shuffles == "":              # Default value selected.
            game.shuffle(game.shuffle_default)
            shuffled = True
            pass
        elif type(shuffles) == tuple:   # Reject WASD input; unrelated to shuffling
            pass
        elif not shuffles.isdigit():    # Reject non-integer input; has no meaning
            pass
        else:
            shuffled = game.shuffle(int(shuffles))

def input_turn(game: Game) -> None:
    print(game)
    player_move = input("Please, enter the label of the tile you would like to push into the gap.\n" +
                        "{Alternatively, enter a direction to 'move' the blank tile (using WASD)}\n" +
                        f"Valid tiles to move: {game.get_valid_moves()} ")
    player_move = command_check(player_move)
    process_turn(game, player_move)

def play(game: Game) -> None:
    while True:
        input_turn(game)
        if game.is_solved():
            input_shuffle(game)

def process_turn(game: Game, player_move: Union[str, Tuple[int, int]]) -> None:
    print()
    if type(player_move) == tuple:
        wasd_label = game.get_ordinal_label(player_move)
        if game.get_valid_moves().__contains__(wasd_label):
            game.slide_tile(int(wasd_label))
        else:
            print("Unable to move that direction...\n")
    elif not player_move.isdigit():
        print("Please, input a valid tile number to move...\n")
    elif not game.slide_tile(int(player_move)):
        print(f"Unable to move tile {player_move}...\n")


def quit_game() -> None:
    print("\nThank you for playing 'fifteen'. Have a nice day! ")
    quit()


if __name__ == '__main__':
    play(Game(input_game_size(), True)) # Start a game based on user selected size, shuffled.
