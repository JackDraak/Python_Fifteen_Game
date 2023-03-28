import unittest
from Tile import Tile

class TestTile(unittest.TestCase):
    def test_distance(self):
        tile = Tile(1, 0, 0, 3)
        self.assertEqual(tile.distance(), 0)
        tile = Tile(9, 0, 0, 3)
        self.assertEqual(tile.distance(), 8)

    def test_move_to(self):
        tile = Tile(1, 0, 0, 3)
        tile.move_to(1, 0)
        self.assertEqual(tile.row, 1)
        self.assertEqual(tile.column, 0)
        self.assertNotEqual(tile.dimension, 4)

    def test_set(self):
        tile = Tile(1, 0, 0, 3)
        tile.set(0, 9, 8, 7)
        self.assertEqual(tile.label, 9)
        self.assertEqual(tile.row, 8)
        self.assertEqual(tile.column, 7)
        self.assertNotEqual(tile.cardinal, 2)

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

if __name__ == "__main__":
    unittest.main()
