# Playing around with Python 3, continued:
# the game "Fifteen", for the console:

import random  # Not strictly required; used to randomize the starting-grid


class Grid:
    def __init__(self, grid_size):
        # Generate the set of tile labels:
        self.size = grid_size
        tile_range = []
        for a_tile in range(1, grid_size * grid_size):
            tile_range.append(a_tile)

        # Randomly label the tiles, and assign unique positions:
        self.tiles = []
        for row in range(grid_size):
            for column in range(grid_size):
                if len(tile_range) > 0:
                    self.tiles.append(Tile(tile_range.pop(random.randint(0, len(tile_range) - 1)), row, column))

        # Note the initially "open" position (the final cell) on the grid:
        self.free_position = grid_size - 1, grid_size - 1

    def get_tile_position(self, value):
        for this_tile in self.tiles:
            if this_tile.value == value:
                return this_tile.row, this_tile.column
        return False

    def get_tile_value(self, row, column):
        for this_tile in self.tiles:
            if this_tile.row == row and this_tile.column == column:
                return this_tile.value

    def get_valid_plays(self):
        plays = []
        free_row, free_column = self.free_position
        for this_tile in self.tiles:
            if this_tile.row == free_row:
                if this_tile.column + 1 == free_column or this_tile.column - 1 == free_column:
                    plays.append(this_tile.value)
            if this_tile.column == free_column:
                if this_tile.row + 1 == free_row or this_tile.row - 1 == free_row:
                    plays.append(this_tile.value)
        return plays

    def move_tile(self, value):
        valid_plays = self.get_valid_plays()
        if valid_plays.__contains__(value):
            old_free_position = self.free_position
            self.free_position = self.get_tile_position(value)
            if self.set_tile_position(value, old_free_position[0], old_free_position[1]):
                return True
        return False

    def print(self):
        for x in range(self.size):
            print("\t", end="")
            for y in range(self.size):
                this_cell = self.get_tile_value(x, y)
                if this_cell is None:
                    print(f"\t", end="")
                else:
                    print(f"\t{self.get_tile_value(x, y)}", end="")
            print()

    def set_tile_position(self, value, row, column):
        for this_tile in self.tiles:
            if this_tile.value == value:
                this_tile.set_position(row, column)
                return True
        return False


class Tile:
    def __init__(self, value, row, column):
        self.value = str(value)
        self.row = row
        self.column = column

    def set_position(self, row, column):
        self.row = row
        self.column = column

    def print(self):
        print(f"Tile: {self.value}, position: {self.row}, {self.column}")


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
    game_size = input_grid_size()
    this_game = Grid(game_size)

    this_game.print()
    while True:
        while this_game.move_tile(input("enter the number of the tile you would like to move: ")):
            this_game.print()
