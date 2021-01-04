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
        self.free_position = grid_size, grid_size

    def get_tile_position(self, value):
        for this_tile in self.tiles:
            if this_tile.value == value:
                return this_tile.row, this_tile.column
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


if __name__ == '__main__':
    grid_size = 4

    # setup a new game of 15:
    this_grid = Grid()

    # print game meta-data to console:
    for tile in this_grid.tiles:
        tile.print()

    # test get_position method:
    print(this_grid.get_tile_position("10"))
    print(this_grid.get_tile_position(10))
