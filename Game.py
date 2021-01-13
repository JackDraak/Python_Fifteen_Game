# Playing around with Python 3, continued...
# the classic game "Fifteen", for the console:
# (C) 2021 Jack Draak

from Tile import Tile
import random


class Game:
    def __init__(self, dimension):
        entropy_factor = 100
        self.dimension = dimension
        self.blank_label = dimension * dimension
        self.blank_position = dimension - 1, dimension - 1
        self.shuffle_default = dimension * entropy_factor
        self.tiles = self.generate_tiles(dimension)     # populate a fresh set of game tiles
        self.solution = self.get_labels_as_list()       # store the win state (the un-shuffled matrix)
        self.shuffle(self.shuffle_default)              # give the tile-grid a shuffle

    def __repr__(self):
        print_string = ""
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
        duplicate_game = Game(self.dimension)
        duplicate_game.import_tiles(self.export_tiles())
        return duplicate_game

    def export_tiles(self):
        tiles = list()
        for tile in self.tiles:
            tiles.append(Tile(tile.label, tile.row, tile.column, tile.dimension))
        return tiles

    @staticmethod
    def generate_tiles(dimension):
        tiles = list()
        label = 0
        for row in range(dimension):
            for column in range(dimension):
                label += 1
                tiles.append(Tile(label, row, column, dimension))
        return tiles

    def get_h_by_label(self, label):
        for tile in self.tiles:
            if tile.label == label:
                row_dimension = tile.row * tile.dimension
                return abs(tile.label - tile.column - row_dimension - 1)
        return False

    def get_label(self, row, column):
        for tile in self.tiles:
            if tile.row == row and tile.column == column:
                return tile.label

    def get_label_h_pairs(self):
        label_pairs = list()
        for row in range(self.dimension):
            for column in range(self.dimension):
                pair = list()
                label = self.get_label(row, column)
                this_pair = label, self.get_h_by_label(label)
                pair.append(this_pair)
                label_pairs.append(pair)
        return label_pairs

    def get_labels_as_list(self):                       # return tile-set labels as a 1D array
        tiles = list()
        for row in range(self.dimension):
            for column in range(self.dimension):
                tiles.append(self.get_label(row, column))
        return tiles

    def get_labels_as_matrix(self):                     # return tile-set labels as a 2D array
        tiles = list()
        for row in range(self.dimension):
            rows = list()
            for column in range(self.dimension):
                rows.append(self.get_label(row, column))
            tiles.append(rows)
        return tiles

    def get_position(self, label):
        for tile in self.tiles:
            if tile.label == label:
                return tile.row, tile.column

    def get_valid_moves(self):
        valid_moves = list()
        blank_row, blank_column = self.blank_position
        for tile in self.tiles:

            if tile.row == blank_row:                   # Select horizontal neighbors
                if tile.column + 1 == blank_column or tile.column - 1 == blank_column:
                    valid_moves.append(tile.label)

            if tile.column == blank_column:             # Select vertical neighbors
                if tile.row + 1 == blank_row or tile.row - 1 == blank_row:
                    valid_moves.append(tile.label)

        if valid_moves.__contains__(self.blank_label):  # Trim blank-tile from set
            valid_moves.remove(self.blank_label)
        return valid_moves

    def h(self):
        return sum(self.get_h_by_label(tile.label) for tile in self.tiles)

    def import_tiles(self, tiles):
        self.tiles = tiles

    def is_solved(self):
        return self.solution == self.get_labels_as_list()

    def print_tile_set(self):
        for tile in self.tiles:
            lab = tile.label
            car = tile.cardinal
            dim = tile.dimension
            row = tile.row
            col = tile.column
            h = self.get_h_by_label(tile.label)
            print(f"<Tile> label:{lab}({car}), position:({dim}){row},{col} H:{h}")

    def set_tile_position(self, label, row, column):
        for tile in self.tiles:
            if tile.label == label:
                tile.move_to(row, column)
                return True
        return False

    def shuffle(self, cycles):
        last_move = int()
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
            this_blank_position = self.blank_position
            this_tile_pos = self.get_position(label)
            if not self.set_tile_position(label, this_blank_position[0], this_blank_position[1]):   # set pos of tile
                print(f"\n{self}Game.set_tile_position({label},{this_blank_position[0]},{this_blank_position[1]}) FAIL")
                return False
            if not self.set_tile_position(self.blank_label, this_tile_pos[0], this_tile_pos[1]):    # set pos of blank
                print(f"\n{self}Game.set_tile_position({self.blank_label},{this_tile_pos[0]},{this_tile_pos[1]}) FAIL")
                return False
            else:
                self.blank_position = this_tile_pos[0], this_tile_pos[1]  # update self.blank_position
                return True
        return False
