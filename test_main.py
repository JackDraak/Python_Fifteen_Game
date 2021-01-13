# Playing around with Python 3, continued...
# the classic game "Fifteen", for the console:
# (C) 2021 Jack Draak

from unittest import TestCase
from Tile import Tile
from Game import Game


class TestTile(TestCase):
    def test_h(self):
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


class TestGame(TestCase):
    def test_duplicate(self):
        game_a = Game(3, True)
        game_b = Game(3, True)
        self.assertFalse(game_a.get_labels_as_list() == game_b.get_labels_as_list())
        game_b = game_a.duplicate()
        self.assertTrue(game_a.get_labels_as_list() == game_b.get_labels_as_list())

    def test_get_label_h_pairs(self):
        ordered_tiles = [[(1, 0)], [(2, 0)], [(3, 0)], [(4, 0)], [(5, 0)], [(6, 0)], [(7, 0)], [(8, 0)], [(9, 0)]]
        game = Game(3, False)
        self.assertEqual(game.get_distance_set(), ordered_tiles)
        game = Game(3, True)
        self.assertNotEqual(game.get_distance_set(), ordered_tiles)

    def test_h(self):
        game = Game(3, False)
        self.assertEqual(game.net_distance(), 0)
        game = Game(3, True)
        self.assertNotEqual(game.net_distance(), 0)

    def test_is_solved(self):
        game = Game(3, True)
        self.assertFalse(game.is_solved())
        game = Game(3, False)
        self.assertTrue(game.is_solved())

    def test_slide_tile(self):
        game = Game(3, False)                   # init game as a new (pre-shuffled) Game object, of dimension = 3
        self.assertFalse(game.slide_tile(1))    # from an ordered matrix (of any size) tile 1 is always locked-in
        self.assertFalse(game.slide_tile(2))    # from an ordered matrix (of any size) tile 2 is always locked-in
        self.assertFalse(game.slide_tile(3))    # from an ordered matrix (of any size) tile 3 is always locked-in
        self.assertFalse(game.slide_tile(4))    # from an ordered matrix (of any size) tile 4 is always locked-in
        self.assertFalse(game.slide_tile(5))    # from an ordered matrix (of any size) tile 5 is always locked-in
        self.assertTrue(game.slide_tile(6))     # from an ordered matrix (dimension = 3) tile 6 is always free
        self.assertTrue(game.slide_tile(6))     # any tile that can slide, can technically slide back to it's origin
        self.assertTrue(game.slide_tile(8))     # ...
        self.assertTrue(game.slide_tile(7))
        self.assertTrue(game.slide_tile(7))
        self.assertTrue(game.slide_tile(8))
        self.assertTrue(game.slide_tile(6))
        self.assertTrue(game.slide_tile(5))
        self.assertTrue(game.slide_tile(2))
        self.assertTrue(game.slide_tile(1))
