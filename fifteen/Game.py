# Game.py (with Tile class)
# Enhanced Game.py with multi-tile movement support with Claude Sonnet 4

import numpy as np
import random
import usage


class Game:

     ## GAME STATE METHODS

    def __init__(self, breadth: int, shuffled: bool, seed: int = None) -> None:
        entropy_factor = 100
        self.blank_label = breadth * breadth
        self.blank_position = breadth - 1, breadth - 1
        self.breadth = breadth
        self.copy_count = 0
        self.seed = seed
        self.tiles = self.generate_tiles(breadth)
        self.shuffle_steps = breadth * entropy_factor
        if shuffled:
            self.shuffle(self.shuffle_steps)
            # self.shuffle(3)                           #  TODO revert to above line after testing

    def copy(self):
        self.copy_count += 1
        copied_game = Game(self.breadth, False, self.seed) # Generate a Game object with same seed
        copied_game.tiles = [tile.copy() for tile in self.tiles]  # Deep copy of the tiles
        copied_game.blank_position = self.blank_position
        if self.copy_count % 100 == 0:
            print(f"Copy count: {self.copy_count}")
            print(self)
            print(copied_game)
            print()
        return copied_game

    def get_state(self):
        tiles = list()
        for row in range(self.breadth):
            for column in range(self.breadth):
                tiles.append(self.get_label(row, column))
        return tiles

    def set_state(self, state: list) -> bool:
        """
        Restore game state from a serialized list of tile labels.
        Returns True if successful, False if invalid state.
        """
        if len(state) != self.breadth * self.breadth:
            return False

        # Validate that all expected labels are present
        expected_labels = set(range(1, self.blank_label + 1))
        if set(state) != expected_labels:
            return False

        # Set tile positions based on the state
        for row in range(self.breadth):
            for column in range(self.breadth):
                index = row * self.breadth + column
                label = state[index]
                if not self.set_tile_position(label, row, column):
                    return False
                if label == self.blank_label:
                    self.blank_position = row, column

        return True

    def is_solved(self):
        return list(range(1, self.blank_label + 1)) == self.get_labels_as_list()

    ## BOARD OPERATION METHODS

    @staticmethod
    def generate_tiles(breadth: int):
        tiles = list()
        label = 0
        for row in range(breadth):
            for column in range(breadth):
                label += 1
                tiles.append(Tile(label, row, column, breadth))
        return tiles

    def slide_tile(self, label: int):                   #  Swap tagret tile with blank tile.
        if self._get_adjacent_moves().__contains__(label):  # Use adjacent-only check for slide_tile
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

    def player_move(self, label: int):
        """
        High-level player move function that can handle both single and multi-tile moves.
        Returns True if successful, False otherwise.
        """
        sequence = self.get_move_sequence(label)
        if not sequence:
            return False
    
        # Execute the sequence using existing slide_tile method
        for tile_label in sequence:
            if not self.slide_tile(tile_label):
                return False  # If any move fails, return failure
        return True

    ## HELPER METHODS FOR AI_TRAINER_CONTROLLER

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

    ## POSITION & LABEL METHODS

    def get_label(self, row: int, column: int):
        for tile in self.tiles:
            if tile.row == row and tile.column == column:
                return tile.label

    def get_position(self, label: int):
        for tile in self.tiles:
            if tile.label == label:
                return tile.row, tile.column

    def get_ordinal_label(self, direction: tuple):
        delta = (direction[0] + self.blank_position[0]), (direction[1] + self.blank_position[1])
        return self.get_label(delta[0], delta[1])       #  Return tile.label based on position delta:blank
    
    def set_tile_position(self, label: int, row: int, column: int):
        for tile in self.tiles:
            if tile.label == label:
                tile.move_to(row, column)
                return True
        return False

    def shuffle(self, moves: int):
        # Set random seed if provided for deterministic shuffles
        if self.seed is not None:
            random.seed(self.seed)

        last_move = int()
        while moves > 0:
            # Use original adjacent-only logic for shuffling to ensure solvability
            options = self._get_adjacent_moves()
            if options.__contains__(last_move):
                options.remove(last_move)
            random_move = options[random.randint(0, len(options) - 1)]
            self.slide_tile(random_move)
            last_move = random_move
            moves -= 1
        return True
    
    ## MOVE VALIDATION METHODS

    def get_valid_moves(self):
        """Return all tiles that can be moved: adjacent tiles + tiles in same row/column as blank."""
        valid_moves = list()
        blank_row, blank_column = self.blank_position
        
        # Get all tiles (excluding the blank itself)
        for tile in self.tiles:
            if tile.label == self.blank_label:
                continue
                
            tile_row, tile_col = tile.row, tile.column
            
            # Check if tile can move (adjacent OR same row/column as blank)
            is_adjacent = False
            is_same_line = False
            
            # Adjacent check (original logic)
            if tile_row == blank_row:
                if tile_col + 1 == blank_column or tile_col - 1 == blank_column:
                    is_adjacent = True
            if tile_col == blank_column:
                if tile_row + 1 == blank_row or tile_row - 1 == blank_row:
                    is_adjacent = True
            
            # Same row/column check (new logic)
            if tile_row == blank_row or tile_col == blank_column:
                is_same_line = True
            
            # Add to valid moves if either condition is met
            if is_adjacent or is_same_line:
                valid_moves.append(tile.label)
        
        return valid_moves

    def get_move_sequence(self, label: int):
        """
        Get the sequence of tiles that need to be moved to achieve the desired move.
        Returns a list of tile labels to pass to slide_tile() in order.
        """
        if label == self.blank_label:
            return []
        
        tile_pos = self.get_position(label)
        if not tile_pos:
            return []
            
        tile_row, tile_col = tile_pos
        blank_row, blank_col = self.blank_position
        
        # Check if move is valid
        if label not in self.get_valid_moves():
            return []
        
        # Direct adjacent move - single tile
        if (tile_row == blank_row and abs(tile_col - blank_col) == 1) or \
           (tile_col == blank_col and abs(tile_row - blank_row) == 1):
            return [label]
        
        # Multi-tile move - same row
        if tile_row == blank_row:
            sequence = []
            if tile_col < blank_col:  # Target is left of blank, move tiles right
                for col in range(blank_col - 1, tile_col - 1, -1):
                    move_label = self.get_label(tile_row, col)
                    if move_label and move_label != self.blank_label:
                        sequence.append(move_label)
            else:  # Target is right of blank, move tiles left
                for col in range(blank_col + 1, tile_col + 1):
                    move_label = self.get_label(tile_row, col)
                    if move_label and move_label != self.blank_label:
                        sequence.append(move_label)
            return sequence
        
        # Multi-tile move - same column
        if tile_col == blank_col:
            sequence = []
            if tile_row < blank_row:  # Target is above blank, move tiles down
                for row in range(blank_row - 1, tile_row - 1, -1):
                    move_label = self.get_label(row, tile_col)
                    if move_label and move_label != self.blank_label:
                        sequence.append(move_label)
            else:  # Target is below blank, move tiles up
                for row in range(blank_row + 1, tile_row + 1):
                    move_label = self.get_label(row, tile_col)
                    if move_label and move_label != self.blank_label:
                        sequence.append(move_label)
            return sequence
        
        return []
    
    def _get_adjacent_moves(self):
        """Helper method that returns only adjacent moves (original get_valid_moves logic)."""
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

    ## DISPLAY METHODS

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
    
    def print_tile_set(self):
        for tile in self.tiles:
            lab = tile.label
            ord = tile.ordinal
            dim = tile.breadth
            row = tile.row
            col = tile.column
            dis = self.get_distance_by_label(tile.label)
            print(f"<Tile> label:{lab}({ord}), position:({dim}){row},{col} distance:{dis}")

    def set_state_from_list(self, state_list: list):
        """
        Set game state from a flat list representation.
        Used for testing specific configurations.

        Args:
            state_list: List of tile labels in row-major order
        """
        if len(state_list) != self.breadth * self.breadth:
            raise ValueError(f"State list must have {self.breadth * self.breadth} elements")

        # Clear current tiles
        self.tiles.clear()

        # Create tiles from list
        for i, label in enumerate(state_list):
            row = i // self.breadth
            col = i % self.breadth

            if label == self.blank_label:
                self.blank_position = (row, col)
            else:
                tile = Tile(label, row, col, self.breadth)
                self.tiles.append(tile)


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
    
    def copy(self):
        return Tile(self.label, self.row, self.column, self.dimension)

    def distance(self):
        lab = self.label
        dim = self.dimension
        row = self.row
        col = self.column
        row_dimension = row * dim
        return abs(lab - col - row_dimension - 1)

    # return the set of tile labels as a 1D array of integers, paired with their distance from the goal
    def get_distance_scores(self):
        tiles = (list(), list())
        for row in range(self.breadth):
            for column in range(self.breadth):
                tiles.append(self.get_label(row, column), self.get_distance(row, column))
        return tiles
    
    # return the set of tile labels as a 2D array of integers
    def get_labels_as_matrix(self):
        tiles = list()
        for row in range(self.breadth):
            rows = list()
            for column in range(self.breadth):
                rows.append(self.get_label(row, column))
            tiles.append(rows)
        return tiles
        
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
