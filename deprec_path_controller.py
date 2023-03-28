from Game import Game
import random


class Frame:
    def __init__(self, game: Game, turn: int, last_move: int):
        self.sequence = turn
        self.solved = [game.is_solved(), game.get_distance_sum()]
        self.label_list = game.get_labels_as_list()
        self.distance_list = game.get_distance_set()
        self.matrix = game.get_labels_as_matrix()
        self.valid_moves = game.get_valid_moves()
        self.last_move = last_move

    def __repr__(self):
        if self.solved[0]:
            s_string = ">---SOLVED---<"
        else:
            s_string = "(not solved)"
        return_string = f"Turn: {self.sequence}, Move: {self.last_move}, "
        return_string += f"{s_string}, Net Distance: {self.solved[1]}\n"
        return_string += f"Label Order: {self.label_list}\n"
        return_string += f"Distance List: {self.distance_list[:3]} ... {self.distance_list[-3:]}\n"
        return_string += f"Matrix: {self.matrix}\n"
        return_string += f"Valid Moves: {self.valid_moves}\n"
        return return_string


def generate_training_set(dimension: int, turns: int):
    path = list()
    last_move = 0
    game = Game(dimension, False)
    while turns > 0:
        path.append(Frame(game, turns, last_move))
        turns -= 1
        options = game.get_valid_moves()
        if options.__contains__(last_move):
            options.remove(last_move)
        semi_random_move = options[random.randint(0, len(options) - 1)]
        if game.slide_tile(semi_random_move):
            last_move = semi_random_move
    return path


if __name__ == '__main__':
    print("""'path_controller.py' -- Use this module to generate frames from 15-puzzle matrix shuffles for M.L. training
    Usage:
        import path_controller as pc
        
        dimension = 4  # matrix size, 4 for a 'standard' game of fifteen
        turn = 10      # the number of turns to shuffle into the path
        path = pc.generate_training_set(dimension, turn)
        
        # 'Replay' the set from the final shuffle back to the first;
        # Ostensibly, train the MLA what solving the puzzle looks like.
        while len(path) > 0:
            frame = path.pop()
            print(frame)
        
        Turn: 1, Solved: False, Net Distance: 90
        Label Order: [7, 8, 6, 16, 14, 2, 1, 12, 5, 11, 15, 3, 13, 9, 4, 10]
        Distance List: [[(7, 6)], [(8, 6)], [(6, 3)]] ... [[(9, 5)], [(4, 11)], [(10, 6)]]
        Matrix: [[7, 8, 6, 16], [14, 2, 1, 12], [5, 11, 15, 3], [13, 9, 4, 10]]
        Valid Moves: [6, 12]
        ...
        Turn: 100, Solved: True, Net Distance: 0
        Label Order: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        Distance List: [[(1, 0)], [(2, 0)], [(3, 0)]] ... [[(14, 0)], [(15, 0)], [(16, 0)]]
        Matrix: [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
        Valid Moves: [12, 15]
""")

