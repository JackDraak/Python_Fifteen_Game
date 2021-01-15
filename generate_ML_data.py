# TODO this is one giant WIP (work in progress) but the intention is to have a tool for generating M.L. training data
# i.e. play-by-play data-sets of matrix shuffles, reversed, to represent solutions, for the purpose of training and
# validating machine-learning models against the "15 puzzle" and, ostensibly, puzzles of any dimension >2.
#
# ie. ie... it's not doing anything useful at the moment.

import usage
# import path_controller as pc
#
# dimension = 4  # matrix size, 4 for a 'standard' game of fifteen
# turn = 100     # the number of turns to shuffle into the path
# path = pc.generate_training_set(dimension, turn)
#
# # 'Replay' the set from the final shuffle back to the first;
# # Ostensibly, train the MLA what solving the puzzle looks like.
# while len(path) > 0:
#     frame = path.pop()
#     print(frame)


if __name__ == '__main__':
    usage.explain()
