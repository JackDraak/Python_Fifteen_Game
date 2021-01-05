# Playing around with Python 3, continued:
# the game "Fifteen", for the console:
# (C) 2021 Jack Draak


import random  # will be used soon for shuffling the board and/or seeking a solution


class Game:
    def __init__(self, size):
        self.free_label = str(size * size)
        self.free_position = size - 1, size - 1
        self.size = size
        tiles = []
        this_label = 0
        for row in range(size):
            for column in range(size):
                this_label += 1
                tiles.append(Tile(this_label, row, column))
        self.tiles = tiles
        self.solution = self.get_tile_set()
        self.shuffle(50)  # starting with a "light" shuffle

    def __repr__(self):
        print_string = str()
        for x in range(self.size):
            print_string += "\t"
            for y in range(self.size):
                label = self.get_tile_label(x, y)
                if label is None or label and int(label) is self.size * self.size:
                    print_string += "\t"
                else:
                    print_string += f"\t{self.get_tile_label(x, y)}"
            print_string += "\n"
        # print_string += str(f"Solved: {self.is_solved()}")  # debug
        return print_string

    def get_tile_label(self, row, column):
        for tile in self.tiles:
            if tile.row == row and tile.column == column:
                return tile.label

    def get_tile_set(self):
        tile_set = []
        for x in range(self.size):
            for y in range(self.size):
                tile_set.append(self.get_tile_label(x, y))
        return tile_set

    def get_tile_position(self, label):
        for tile in self.tiles:
            if tile.label == label:
                return tile.row, tile.column
        return False

    def get_valid_moves(self):
        valid_moves = []
        free_row, free_column = self.free_position
        for tile in self.tiles:
            if tile.row == free_row:
                if tile.column + 1 == free_column or tile.column - 1 == free_column:
                    valid_moves.append(tile.label)
            if tile.column == free_column:
                if tile.row + 1 == free_row or tile.row - 1 == free_row:
                    valid_moves.append(tile.label)
        if valid_moves.__contains__(self.free_label):
            valid_moves.remove(self.free_label)
        return valid_moves

    def is_solved(self):
        return self.get_tile_set() == self.solution

    def set_tile_position(self, label, row, column):
        for tile in self.tiles:
            if tile.label == label:
                tile.move_to(row, column)
                return True
        return False

    def shuffle(self, cycles):
        while cycles > 0:
            options = self.get_valid_moves()
            random_move = options[random.randint(0, len(options) - 1)]
            self.slide_tile(random_move)
            cycles -= 1

    def slide_tile(self, label):
        valid_moves = self.get_valid_moves()
        if valid_moves.__contains__(label):
            swap_free_pos = self.free_position
            swap_tile_pos = self.get_tile_position(label)
            self.set_tile_position(label, swap_free_pos[0], swap_free_pos[1])
            self.set_tile_position(self.free_label, swap_tile_pos[0], swap_tile_pos[1])
            self.free_position = swap_tile_pos[0], swap_tile_pos[1]
            return True
        return False


class Tile:
    def __init__(self, label, row, column):
        self.label = str(label)
        self.row = row
        self.column = column

    def __repr__(self):
        print_string = str(f"<Tile> label:{self.label}, position: {self.row}, {self.column}")
        return print_string

    def move_to(self, row, column):
        self.row = row
        self.column = column


def input_game_size():
    size = int()
    print("To play the classic tile game, '15', ", end="")
    no_intention = True
    while no_intention:
        intention = input("please enter a grid size greater than 2, and less than 12 [default: 4] ")
        if intention == "":
            size = 4  # default
            no_intention = False
        elif intention.isdigit():
            size = int(intention)
            if 2 < size < 12:
                no_intention = False
    print()
    return size


def play(game):
    # Move the tiles, one-by-one, until player get bored:
    while True:
        print(game)
        # print(game.get_tile_set())  # debug
        # print(game.get_valid_moves())  # debug
        input_string = \
            str(f"Please, enter the label of the tile you would like to move\nvalid plays: {game.get_valid_moves()} ")
        player_move = input(input_string)
        print()
        if not game.slide_tile(player_move):
            print("Input not understood...\n")
        if game.is_solved():
            print("Congratulations, you solved the puzzle!")
            game.shuffle(int(input("How many times would you like to shuffle? ")))  # TODO validate user input


if __name__ == '__main__':
    play(Game(input_game_size()))
