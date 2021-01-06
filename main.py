# Playing around with Python 3, continued:
# the classic game "Fifteen", for the console:
# (C) 2021 Jack Draak


import random


class Game:
    def __init__(self, size):
        entropy_factor = 100
        self.blank_label = str(size * size)
        self.blank_position = size - 1, size - 1
        self.shuffle_default = size * entropy_factor
        self.size = size

        tiles = []
        label = 0
        for row in range(size):
            for column in range(size):
                label += 1
                tiles.append(Tile(label, row, column))
        self.tiles = tiles

        self.solution = self.get_tile_set()

        self.shuffle(self.shuffle_default)

    def __repr__(self):
        print_string = str()
        for x in range(self.size):
            print_string += "\t"
            for y in range(self.size):
                label = self.get_tile_label(x, y)
                if label != self.blank_label:
                    print_string += f"\t{label}"
                else:
                    print_string += "\t"
            print_string += "\n"
        print_string += str(self.get_inversion_count())
        return print_string

    def get_inversion_count(self):
        inversions = 0
        tile_set = self.get_tile_set()
        for fore in range(len(tile_set)):
            for aft in range(fore + 1, len(tile_set)):
                if fore < aft and int(tile_set[fore]) > int(tile_set[aft]):
                    inversions += 1
        return inversions

    def get_tile_label(self, row, column):
        for tile in self.tiles:
            if tile.row == row and tile.column == column:
                return tile.label

    def get_tile_position(self, label):
        for tile in self.tiles:
            if tile.label == label:
                return tile.row, tile.column
        return False

    def get_tile_set(self):
        tile_set = []
        for x in range(self.size):
            for y in range(self.size):
                tile_set.append(self.get_tile_label(x, y))
        return tile_set

    def get_valid_moves(self):
        valid_moves = []
        blank_row, blank_column = self.blank_position
        for tile in self.tiles:
            if tile.row == blank_row:
                if tile.column + 1 == blank_column or tile.column - 1 == blank_column:
                    valid_moves.append(tile.label)
            if tile.column == blank_column:
                if tile.row + 1 == blank_row or tile.row - 1 == blank_row:
                    valid_moves.append(tile.label)
        if valid_moves.__contains__(self.blank_label):
            valid_moves.remove(self.blank_label)
        return valid_moves

    def is_solved(self):
        return self.solution == self.get_tile_set()

    def is_valid(self):
        # If grid is odd, return true if inversion count is even.
        pass

    def set_tile_position(self, label, row, column):
        for tile in self.tiles:
            if tile.label == label:
                tile.move_to(row, column)
                return True
        return False

    def shuffle(self, cycles):
        last_move = str()
        while cycles > 0:
            options = self.get_valid_moves()
            if options.__contains__(last_move):
                options.remove(last_move)
            random_move = options[random.randint(0, len(options) - 1)]
            self.slide_tile(random_move)
            last_move = random_move
            cycles -= 1

    def slide_tile(self, label):
        if self.get_valid_moves().__contains__(label):
            swap_free_pos = self.blank_position
            swap_tile_pos = self.get_tile_position(label)
            self.set_tile_position(label, swap_free_pos[0], swap_free_pos[1])
            self.set_tile_position(self.blank_label, swap_tile_pos[0], swap_tile_pos[1])
            self.blank_position = swap_tile_pos[0], swap_tile_pos[1]
            return True
        return False


class Tile:
    def __init__(self, label, row, column):
        self.label = str(label)
        self.row = row
        self.column = column

    def __repr__(self):
        return str(f"<Tile> label:{self.label}, position: {self.row}, {self.column}")

    def move_to(self, row, column):
        self.row = row
        self.column = column


def input_game_size():
    size_default = 4
    size_max = 31
    size_min = 3
    size = size_default
    print("\nTo play the classic tile game, '15', ", end="")
    invalid_input = True
    while invalid_input:
        grid_size = input(f"please chose a grid size from {size_min} to {size_max} [default: {size_default}] " +
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
    shuffles_default = str(game.shuffle_default)
    while not shuffles.isdigit():
        shuffles = input(f"How many times would you like to shuffle? [default: {shuffles_default}] \n")
        if shuffles == "":
            shuffles = shuffles_default
    game.shuffle(int(shuffles))


def input_turn(game):
    print(game)
    input_string = \
        str("Please, enter the label of the tile you would like to push into the gap.\n" +
            f"valid plays: {game.get_valid_moves()} ")
    player_move = input(input_string)
    print()
    if not game.slide_tile(player_move):
        print("Input not understood...\n")


def play(game):
    while True:
        input_turn(game)
        if game.is_solved():
            input_shuffle(game)


if __name__ == '__main__':
    play(Game(input_game_size()))
