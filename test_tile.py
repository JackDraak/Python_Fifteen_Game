# test_tile.py
import unittest
from Tile import Tile

class TestTile(unittest.TestCase):
    def test_initialization(self):
        tile = Tile(1, 0, 0, 4)
        self.assertEqual(tile.cardinal, 1)
        self.assertEqual(tile.label, 1)
        self.assertEqual(tile.row, 0)
        self.assertEqual(tile.column, 0)
        self.assertEqual(tile.dimension, 4)

    def test_repr(self):
        tile = Tile(1, 0, 0, 4)
        expected_repr = "<Tile> label:1(1), position:(4)0,0 distance:0"
        self.assertEqual(repr(tile), expected_repr)

    def test_distance(self):
        tile = Tile(1, 0, 0, 4)
        self.assertEqual(tile.distance(), 0)

    def test_move_to(self):
        tile = Tile(1, 0, 0, 4)
        tile.move_to(1, 1)
        self.assertEqual(tile.row, 1)
        self.assertEqual(tile.column, 1)

    # Add more tests to cover other methods in the Tile class


if __name__ == "__main__":
    unittest.main()
