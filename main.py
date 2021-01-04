# Playing around with Python 3, continued:
# the game "Fifteen", for the console:

import random


class Grid:
    def __init__(self):
        # Generate set of tile labels:
        tile_range = []
        for a_tile in range(1, grid_size * grid_size):
            tile_range.append(a_tile)

        # Randomly label the tiles, and assign unique positions:
        self.tiles = []
        for row in range(grid_size):
            for column in range(grid_size):
                if len(tile_range) > 0:
                    self.tiles.append(Tile(tile_range.pop(random.randint(0, len(tile_range) - 1)), row, column))

        # Note the "open" position on the grid:
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
        # ex. 2,2 == 2,1 || 1,2 || 2,3 || 3,2
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
            temp = self.free_position
            self.free_position = self.get_tile_position(value)
            self.set_tile_position(value, temp[0], temp[1])

    def print(self):
        for x in range(grid_size):
            print("\t", end="")
            for y in range(grid_size):
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


if __name__ == '__main__':
    grid_size = 4
    # setup a new game of 15:
    this_grid = Grid()

    # # print game meta-data to console:
    # for tile in this_grid.tiles:
    #     tile.print()

    # # test methods:
    # print(this_grid.free_position)
    # print(this_grid.get_tile_position("10"))
    # print(this_grid.get_tile_position(10))
    # print(this_grid.get_tile_value(1, 1))

    while True:
        this_grid.print()
        print(this_grid.get_valid_plays())
        this_grid.move_tile(input("Tile to move: "))
