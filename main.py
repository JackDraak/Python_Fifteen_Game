# Playing around with Python 3, continued...
# the classic game "Fifteen", for the console:
# (C) 2021 Jack Draak

import Game
import Tile


def input_game_size():
    size_default = 4        # for the classic '15 puzzle'
    size_max = 31           # grids with dimension >31 have >1000 tiles, requires re-formatting
    size_min = 3            # grids with dimension <3 are not playable
    size = size_default
    print("\nTo play the classic tile game, '15', ", end="")
    invalid_input = True
    while invalid_input:

        grid_size = input(f"please choose a grid size from {size_min} to {size_max} [default: {size_default}] " +
                          "\n(the goal of the game is to slide the game tiles into the 'open' position, 1-by-1, " +
                          "until the tiles are in ascending order.) ")
        if grid_size == "":
            invalid_input = False
        elif grid_size.isdigit():
            size = int(grid_size)
            if size_min <= size <= size_max:
                invalid_input = False
    print()
    return size


def input_shuffle(game):
    print("Congratulations, you solved the puzzle! \n")
    print(game)
    shuffles = ""
    while not shuffles.isdigit():
        shuffles = input(f"How many times would you like to shuffle? [default: {game.shuffle_default}] \n")
        if shuffles == "":
            game.shuffle(game.shuffle_default)
            break
        elif not shuffles.isdigit():
            pass
        else:
            game.shuffle(int(shuffles))


def input_turn(game):
    print(game)
    player_move = input("Please, enter the label of the tile you would like to push into the gap.\n" +
                        f"Valid tiles to move: {game.get_valid_moves()} ")
    print()
    if not player_move.isdigit():
        print("Input valid tile number to move...\n")
    elif not game.slide_tile(int(player_move)):
        print(f"Unable to move tile {player_move}...\n")


def play(game):
    while True:
        input_turn(game)
        if game.is_solved():
            input_shuffle(game)


if __name__ == '__main__':
    play(Game(input_game_size()))
