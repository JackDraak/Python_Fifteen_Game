# Playing around with Python 3, continued:
# the classic game "Fifteen", for the console:
# (C) 2021 Jack Draak

import random
import time


class Game:
    def __init__(self, dimension):
        entropy_factor = 100
        self.dimension = dimension
        self.blank_label = str(dimension * dimension)
        self.blank_position = dimension - 1, dimension - 1
        self.shuffle_default = 5  # dimension * entropy_factor + 1  # TODO restore equation for a proper default shuffle
        self.tiles = self.reset_game(dimension)
        self.solution = self.get_tile_layout()
        self.shuffle(self.shuffle_default)

    def __repr__(self):
        print_string = str()
        for x in range(self.dimension):
            print_string += "\t"
            for y in range(self.dimension):
                label = self.get_label(x, y)
                if label != self.blank_label:
                    print_string += f"\t{label}"
                else:
                    print_string += "\t"
            print_string += "\n"
        return print_string

    def duplicate(self):
        a_game = Game(self.dimension)
        a_game.import_tiles(self.export_tiles())
        return a_game

    def export_tiles(self):
        tiles = []
        for tile in self.tiles:
            tiles.append(Tile(tile.label, tile.row, tile.column, tile.dimension))
        return tiles

    def get_label(self, row, column):
        for tile in self.tiles:
            if tile.row == row and tile.column == column:
                return tile.label

    def get_position(self, label):
        for tile in self.tiles:
            if tile.label == label:
                return tile.row, tile.column
        return False

    def get_tile_layout(self):
        tiles = []
        for row in range(self.dimension):
            for column in range(self.dimension):
                tiles.append(self.get_label(row, column))
        return tiles

    def get_tile_position(self, label):
        for tile in self.tiles:
            if tile.label == label:
                return tile.row, tile.column
        return False

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

    def h(self):
        sum_h = 0
        for tile in self.tiles:
            sum_h += tile.h
        return int(sum_h)

    def import_tiles(self, tiles):
        self.tiles = tiles

    def is_solved(self):
        return self.solution == self.get_tile_layout()

    @staticmethod
    def reset_game(dimension):
        tiles = []
        label = 0
        for row in range(dimension):
            for column in range(dimension):
                label += 1
                tiles.append(Tile(label, row, column, dimension))
        return tiles

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


class Node:
    def __init__(self, sequence, h, label, row, column):
        self.g = sequence
        self.h = h
        self.label = label
        self.row = row
        self.column = column
        self.sequence = sequence


class HopScore:
    def __init__(self, label, h, sequence):
        self.h = int(h)
        self.label = label
        self.sequence = sequence

    def __lt__(self, other):
        return self.h < other

    def __repr__(self):
        return f"HopScore-{self.sequence}: #{self.label}({self.h}) "


class Tile:
    def __init__(self, label, row, column, dimension):
        self.column = column
        self.dimension = dimension
        self.label = str(label)
        self.row = row
        self.h = abs(int(self.label) - 1 - ((self.row * self.dimension) + self.column))

    def __repr__(self):
        return str(f"<Tile> label:{self.label}, position: {self.row}, {self.column} H: {self.h()}")

    def move_to(self, row, column):
        self.row = row
        self.column = column

    def set(self, label, row, column):
        self.column = column
        self.label = str(label)
        self.row = row


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


def auto_play(game):
    blank_position = game.blank_position
    this_game = game.duplicate()
    last_move = ""
    nodes = list()
    turn = 1

    while this_game.h() != 0:
        open_moves = list()
        for move in this_game.get_valid_moves():
            if move is not last_move:
                open_moves.append(move)

        scores = list()
        for move in open_moves:
            test_game = this_game.duplicate()
            test_game.slide_tile(move)
            scores.append(HopScore(move, int(test_game.h()), turn))
            print(f"adding score: {test_game.h()} for move: {move}")
        for score in scores:
            print(score)
            print(score.h)

        low_score = 99999
        move = ""
        for score in scores:
            if score < low_score:
                low_score = score.h
                move = score.label
        game.slide_tile(move)

        # TODO use blank-position for A* pathing?
        nodes.append(Node(turn, game.h, move, blank_position[0], blank_position[1]))
        blank_position = game.blank_position
        last_move = move
        turn += 1
        print(game)
        time.sleep(0.5)


def play(game):
    while True:
        input_turn(game)
        if game.is_solved():
            input_shuffle(game)


if __name__ == '__main__':
    my_dimension = 3
    # auto_play(Game(my_dimension))  # TODO - this is work in progress
    play(Game(input_game_size()))
