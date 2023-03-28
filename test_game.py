# test_game.py
import unittest
from Game import Game

class TestGame(unittest.TestCase):
    def test_initialization(self):
        game = Game(4, False)
        self.assertEqual(game.dimension, 4)
        self.assertEqual(game.blank_label, 16)
        self.assertEqual(game.blank_position, (3, 3))
        self.assertEqual(game.solution, list(range(1, 17)))

    def test_repr(self):
        game = Game(4, False)
        expected_repr = "\n\t1\t2\t3\t4\n\t5\t6\t7\t8\n\t9\t10\t11\t12\n\t13\t14\t15\t\n"
        self.assertEqual(repr(game), expected_repr)

    def test_duplicate(self):
        game = Game(4, False)
        duplicate_game = game.duplicate()
        self.assertEqual(game.dimension, duplicate_game.dimension)
        self.assertEqual(game.blank_label, duplicate_game.blank_label)
        self.assertEqual(game.blank_position, duplicate_game.blank_position)
        self.assertEqual(game.solution, duplicate_game.solution)

    # Add more tests to cover other methods in the Game class


if __name__ == "__main__":
    unittest.main()
    