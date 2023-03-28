from Game import Game
import itertools


class Path:
    def __init__(self):
        self.path = list()

    def __eq__(self, other):
        return self.path == other

    def append(self, new_move: int):
        self.path.append(new_move)

    def recall(self, index: int):
        if len(self.path) > index:
            return self.path[index]
        else:
            return None

    def last_node(self):
        path_size = len(self.path)
        if path_size > 0:
            return self.path[path_size - 1]
        else:
            return None


def e_generator():
    current_factorial = 1
    current_sum = 1
    for i in itertools.count(start=1):
        current_factorial *= i
        current_sum += 1 / current_factorial
        yield current_sum


def get_e(gen, max_iter=10000, precision=0):
    def my_wrap(*args, **kwargs):
        instance_gen = gen(*args, **kwargs)
        prev_value = next(instance_gen)
        for n, value in enumerate(instance_gen):
            if (value - prev_value < precision) or (n > max_iter):
                return value
            else:
                prev_value = value
    return my_wrap


# Normalize input: positive numbers under 100 approach a value of 1, negative numbers approach 0; f(0) = 0.5
def normalized_sigmoid(n):
    return 1 / (1 + get_e(e_generator)() ** - n)


if __name__ == '__main__':

    # e = get_e(e_generator)()
    print(normalized_sigmoid(0))
    print(normalized_sigmoid(.25))
    print(normalized_sigmoid(.5))
    print(normalized_sigmoid(.75))
    print(normalized_sigmoid(1))
    print(normalized_sigmoid(-1))
    print(normalized_sigmoid(-10))
    print(normalized_sigmoid(-111))

    paths = [Path]
    depth = 0
    prior_move = 0
    dimension = 4
    shuffled = True
    game = Game(dimension, shuffled)

    # initialize paths from current state
    for move in game.get_valid_moves():
        outset = list()
        if move == prior_move:
            outset.append(move)
            for path in paths:
                in_set = list()
                if path.last_node() == prior_move:
                    in_set.append(path)
