# Playing around with Python 3, continued:
# the game "Fifteen", for the console:

import random  # Not strictly required; used to randomize the starting-grid


class Grid:
    def __init__(self, size):
        self.size = size
        tile_labels = []
        for label in range(1, size * size):
            tile_labels.append(label)
        self.tiles = []
        for row in range(size):
            for column in range(size):
                if len(tile_labels) > 0:
                    random_label = tile_labels.pop(random.randint(0, len(tile_labels) - 1))
                    self.tiles.append(Tile(random_label, row, column))
        self.free_position = size - 1, size - 1

    def get_tile_position(self, value):
        for this_tile in self.tiles:
            if this_tile.label == value:
                return this_tile.row, this_tile.column
        return False

    def get_tile_label(self, row, column):
        for tile in self.tiles:
            if tile.row == row and tile.column == column:
                return tile.label

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
        return valid_moves

    def move_tile(self, value):
        valid_moves = self.get_valid_moves()
        if valid_moves.__contains__(value):
            old_free_position = self.free_position
            self.free_position = self.get_tile_position(value)
            if self.set_tile_position(value, old_free_position[0], old_free_position[1]):
                return True
            else:
                self.free_position = old_free_position
        return False

    def print(self):
        for x in range(self.size):
            print("\t", end="")
            for y in range(self.size):
                label = self.get_tile_label(x, y)
                if label is None:
                    print(f"\t", end="")
                else:
                    print(f"\t{self.get_tile_label(x, y)}", end="")
            print()

    def set_tile_position(self, label, row, column):
        for tile in self.tiles:
            if tile.label == label:
                tile.set_position(row, column)
                return True
        return False


class Tile:
    def __init__(self, label, row, column):
        self.label = str(label)
        self.row = row
        self.column = column

    def set_position(self, row, column):
        self.row = row
        self.column = column

    # only used for debugging purposes: view tile meta-data
    def print(self):
        print(f"Tile: {self.label}, position: {self.row}, {self.column}")


def input_grid_size():
    size = 0
    print("To play the classic tile game, '15', ", end="")
    no_intention = True
    while no_intention:
        intention = input("please enter a grid size greater than 2, and less than 12 [default: 4] ")
        if intention == "":
            size = 4
            no_intention = False
        elif intention.isdigit():
            size = int(intention)
            if 2 < size < 12:
                no_intention = False
    return size


if __name__ == '__main__':
    # Initialize and display a new Grid-class game object of the specified size:
    game = Grid(input_grid_size())
    game.print()

    # Move the tiles, one-by-one, until you get bored:
    while True:
        while game.move_tile(input("enter the label of the tile you would like to move: ")):
            game.print()
