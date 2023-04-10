import numpy as np
import random
import usage

class Game:
    def __init__(self, breadth: int, shuffled: bool) -> None:
        entropy_factor = 100
        self.blank_label = breadth * breadth
        self.blank_position = breadth - 1, breadth - 1
        self.breadth = breadth 
        self.tiles = self.generate_tiles(breadth)
        self.shuffle_steps = breadth * entropy_factor
        if shuffled:
            self.shuffle(self.shuffle_steps)
            # self.shuffle(3)                           #  TODO revert to above line after testing

    def __repr__(self):
        print_string = ""
        for x in range(self.breadth):
            print_string += "\t"
            for y in range(self.breadth):
                label = self.get_label(x, y)
                if label != self.blank_label:
                    print_string += f"\t{label}"
                else:
                    print_string += "\t"
            print_string += "\n"
        return print_string

    @staticmethod
    def generate_tiles(breadth: int):
        tiles = list()
        label = 0
        for row in range(breadth):
            for column in range(breadth):
                label += 1
                tiles.append(Tile(label, row, column, breadth))
        return tiles

    def get_distance_by_label(self, label: int):
        for tile in self.tiles:
            if tile.label == label:
                return tile.distance()
        return False

    def get_distance_set(self):
        label_pairs = np.zeros((self.breadth, self.breadth, 2), dtype=int)
        for row in range(self.breadth):
            for column in range(self.breadth):
                label = self.get_label(row, column)
                this_pair = label, self.get_distance_by_label(label)
                label_pairs[row, column] = this_pair
        return label_pairs

    def get_distance_sum(self):
        return np.sum([tile.distance() for tile in self.tiles])
    
    def get_label(self, row: int, column: int):
        for tile in self.tiles:
            if tile.row == row and tile.column == column:
                return tile.label

    def get_labels_as_list(self):                       #  Return tile-set labels as a 1D array.
        tiles = list()
        for row in range(self.breadth):
            for column in range(self.breadth):
                tiles.append(self.get_label(row, column))
        return tiles

    def get_labels_as_matrix(self):                     #  Return tile-set labels as a 2D array.
        tiles = list()
        for row in range(self.breadth):
            rows = list()
            for column in range(self.breadth):
                rows.append(self.get_label(row, column))
            tiles.append(rows)
        return tiles

    def get_ordinal_label(self, direction: tuple):
        delta = (direction[0] + self.blank_position[0]), (direction[1] + self.blank_position[1])
        return self.get_label(delta[0], delta[1])       #  Return tile.label based on position delta:blank

    def get_position(self, label: int):
        for tile in self.tiles:
            if tile.label == label:
                return tile.row, tile.column

    def get_state(self):
        tiles = list()
        for row in range(self.breadth):
            for column in range(self.breadth):
                tiles.append(self.get_label(row, column))
        return tiles

    def get_valid_moves(self):
        valid_moves = list()
        blank_row, blank_column = self.blank_position
        for tile in self.tiles:
            if tile.row == blank_row:                   #  Select horizontal neighbors.
                if tile.column + 1 == blank_column or tile.column - 1 == blank_column:
                    valid_moves.append(tile.label)
            if tile.column == blank_column:             #  Select vertical neighbors.
                if tile.row + 1 == blank_row or tile.row - 1 == blank_row:
                    valid_moves.append(tile.label)
        if valid_moves.__contains__(self.blank_label):  #  Trim blank-tile from set.
            valid_moves.remove(self.blank_label)
        return valid_moves
    
    def is_solved(self):
        return list(range(1, self.blank_label + 1)) == self.get_labels_as_list()

    def print_tile_set(self):
        for tile in self.tiles:
            lab = tile.label
            ord = tile.ordinal
            dim = tile.breadth
            row = tile.row
            col = tile.column
            dis = self.get_distance_by_label(tile.label)
            print(f"<Tile> label:{lab}({ord}), position:({dim}){row},{col} distance:{dis}")

    def set_tile_position(self, label: int, row: int, column: int):
        for tile in self.tiles:
            if tile.label == label:
                tile.move_to(row, column)
                return True
        return False

    def shuffle(self, moves: int):
        last_move = int()
        while moves > 0:
            options = self.get_valid_moves()
            if options.__contains__(last_move):
                options.remove(last_move)
            random_move = options[random.randint(0, len(options) - 1)]
            self.slide_tile(random_move)
            last_move = random_move
            moves -= 1
        return True

    def slide_tile(self, label: int):                   #  Swap tagret tile with blank tile.
        if self.get_valid_moves().__contains__(label):
            this_blank_position = self.blank_position
            this_tile_pos = self.get_position(label)
                                                        #  Set x, y position of target tile.
            if not self.set_tile_position(label, this_blank_position[0], this_blank_position[1]):   
                print(f"\n{self}Game.set_tile_position({label},{this_blank_position[0]},{this_blank_position[1]}) FAIL")
                return False
                                                        #  Set x, y position of blank.
            if not self.set_tile_position(self.blank_label, this_tile_pos[0], this_tile_pos[1]):   
                print(f"\n{self}Game.set_tile_position({self.blank_label},{this_tile_pos[0]},{this_tile_pos[1]}) FAIL")
                return False
            else:
                self.blank_position = this_tile_pos[0], this_tile_pos[1]
                return True
        return False


class Tile:
    def __init__(self, label: int, row: int, column: int, dimension: int):
        self.ordinal = label
        self.label = label
        self.row = row
        self.column = column
        self.dimension = dimension

    def __repr__(self):
        lab = self.label
        ord = self.ordinal
        dim = self.dimension
        row = self.row
        col = self.column
        dis = self.distance()
        return f"<Tile> label:{lab}({ord}), position:({dim}){row},{col} distance:{dis}"

    def distance(self):
        lab = self.label
        dim = self.dimension
        row = self.row
        col = self.column
        row_dimension = row * dim
        return abs(lab - col - row_dimension - 1)
        
    def get_tile_by_label(self, label: int):
        for tile in self.tiles:
            if tile.label == label:
                return tile

    def move_to(self, row: int, column: int):
        self.row = row
        self.column = column

    def set(self, ordinal: int, label: int, row: int, column: int):
        self.ordinal = ordinal
        self.label = label
        self.row = row
        self.column = column


if __name__ == '__main__':
    from unit_tests import TestGame                     #  If this module is run as a script, 
    import unittest                                     #  run unit tests,
    import usage                                        
    suite = unittest.TestLoader().loadTestsFromTestCase(TestGame)
    unittest.TextTestRunner().run(suite)
    print()
    usage.explain()                                     #  then explain usage.
