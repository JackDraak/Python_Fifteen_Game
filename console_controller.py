from Game import Game


def command_check(command: str):
    direction = (0, 0)
    if command == "":                   # Select default behaviour:
        return ""
    elif command.lower()[0] == "q":     # Quit.
        quit_game()
    elif command.lower()[0] == "a":     # Move blank left.  (tile on left takes blank position)
        direction = (0, -1)
    elif command.lower()[0] == "d":     # Move blank right. (tile on right takes blank position)
        direction = (0, 1)
    elif command.lower()[0] == "w":     # Move blank up.    (tile above takes blank position)
        direction = (-1, 0)
    elif command.lower()[0] == "s":     # Move blank down.  (tile below takes blank position)
        direction = (1, 0)
    if direction != (0, 0):
        return direction
    return command


def input_game_size():
    size_default = 4                    # for the classic '15 puzzle', use a grid with a dimension of 4
    size_max = 31                       # grids with dimension >31 have >1000 tiles, would require re-formatting
    size_min = 3                        # grids with dimension <3 are not functionally playable
    size = size_default
    print("\nGoal: slide the game tiles into the open position, 1-by-1, to re-order them. [Or (q)uit at any prompt]")
    print("To play the classic tile game, '15', ", end="")
    valid_input = False
    while not valid_input:
        grid_size = input(f"please choose a grid size from {size_min} to {size_max} [default: {size_default}] ")
        grid_size = command_check(grid_size)
        if grid_size == "":
            valid_input = True
        elif type(grid_size) == tuple:
            pass
        elif grid_size.isdigit():
            size = int(grid_size)
            if size_min <= size <= size_max:
                valid_input = True
    print()
    return size


def input_shuffle(game: Game):
    print("*** Congratulations, you solved the puzzle! ***\n")
    print(game)
    shuffled = False
    while not shuffled:
        shuffles = input(f"How many times would you like to shuffle? [default: {game.shuffle_default}] \n")
        shuffles = command_check(shuffles)
        if shuffles == "":
            game.shuffle(game.shuffle_default)
            shuffled = True
            pass
        elif type(shuffles) == tuple:
            pass
        elif not shuffles.isdigit():
            pass
        else:
            shuffled = game.shuffle(int(shuffles))


def input_turn(game: Game):
    print(game)
    player_move = input("Please, enter the label of the tile you would like to push into the gap.\n" +
                        f"Valid tiles to move: {game.get_valid_moves()} ")
    player_move = command_check(player_move)
    process_turn(game, player_move)


def play(game: Game):
    while True:
        input_turn(game)
        if game.is_solved():
            input_shuffle(game)


def process_turn(game, player_move):
    print()
    if type(player_move) == tuple:
        wasd_move = game.directional_move(player_move)
        if game.get_valid_moves().__contains__(wasd_move):
            if not game.slide_tile(int(wasd_move)):
                print(f"Unable to WASD-move tile {player_move}...\n")  # This should never happen
        else:
            print("Unable to move that direction...\n")

    elif not player_move.isdigit():
        print("Please, input a valid tile number to move...\n")
    elif not game.slide_tile(int(player_move)):
        print(f"Unable to move tile {player_move}...\n")


def quit_game():
    print("\nThank you for playing 'fifteen'. Have a nice day! ")
    quit()


if __name__ == '__main__':
    play(Game(input_game_size(), True))
