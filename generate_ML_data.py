from Game import Game
import numpy
import random
import time

# TODO this is one giant WIP (work in progress) but the intention is to have a tool for generating M.L. training data
# i.e. play-by-play data-sets of matrix shuffles, reversed, to represent solutions, for the purpose of training and
# validating machine-learning models against the "15 puzzle" and, ostensibly, puzzles of any dimension >2.
#
# ie. ie... it's not doing anything useful at the moment.

if __name__ == '__main__':
    game = Game(3, False)
    print([game.is_solved(), game.get_distance_sum()])
    print(game.get_distance_set())
    print(game.get_valid_moves())
    print(game.get_labels_as_matrix())

    print("\nTraining-data Generator for the game '15 Puzzle'. Please define parameters for generation...")
    dimension = input("Dimension: ")
    entropy = input("Entropy: ")
    cycles = input("Cycles: ")
    this_time = time.time()

    training_set = numpy.ndarray((1, int(cycles), int(entropy), 4))
    local_cycles = int(cycles)
    while int(local_cycles) > 0:
        game = Game(int(dimension), False)
        frame_set = numpy.ndarray((int(cycles), int(entropy), 4))
        last_move = 0
        local_entropy = int(entropy)
        while int(local_entropy) > 0:
            frame = []  # numpy.ndarray((int(entropy), 10))
            del frame
            frame = list()
            frame[0:] = game.is_solved(), game.get_distance_sum()
            frame[len(frame):] = game.get_distance_set()
            frame[len(frame):] = game.get_labels_as_matrix()
            print(frame)
            # frame[n] = game.get_valid_moves()

            valid_moves = game.get_valid_moves()
            if valid_moves.__contains__(last_move):
                valid_moves.remove(last_move)
            semi_random_move = valid_moves[random.randint(0, len(valid_moves))]
            if game.slide_tile(semi_random_move):
                last_move = semi_random_move
            local_entropy -= 1
            frame_set.__add__(frame)
        training_set.__add__(frame_set)
        local_cycles -= 1
    print(training_set)
