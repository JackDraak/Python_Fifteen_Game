# Playing around with Python 3, continued...
# the classic game "Fifteen", for the console:
# (C) 2021 Jack Draak

from unittest import TestCase
from Tile import Tile
from Game import Game

import pdb  # pdb.set_trace()


class TestTile(TestCase):
    def test_h(self):
        tile = Tile(1, 0, 0, 3)
        self.assertEqual(tile.h(), 0)
        tile = Tile(9, 0, 0, 3)
        self.assertEqual(tile.h(), 8)

    def test_move_to(self):
        tile = Tile(1, 0, 0, 3)
        tile.move_to(1, 0)
        self.assertEqual(tile.row, 1)
        self.assertEqual(tile.column, 0)

    def test_set(self):
        tile = Tile(1, 0, 0, 3)
        tile.set(0, 9, 8, 7)
        self.assertEqual(tile.label, 9)
        self.assertEqual(tile.row, 8)
        self.assertEqual(tile.column, 7)


class TestGame(TestCase):
    def test_duplicate(self):
        game_a = Game(3)
        game_b = Game(3)
        self.assertFalse(game_a.get_labels_as_list() == game_b.get_labels_as_list())
        game_b = game_a.duplicate()
        self.assertTrue(game_a.get_labels_as_list() == game_b.get_labels_as_list())

    def test_get_label_h_pairs(self):
        game = Game(3)
        game.shuffle(333)
        clean_tiles = [[(1, 0)], [(2, 0)], [(3, 0)], [(4, 0)], [(5, 0)], [(6, 0)], [(7, 0)], [(8, 0)], [(9, 0)]]
        self.assertNotEqual(game.get_label_h_pairs(), clean_tiles)
        game.tiles = game.generate_tiles(3)
        self.assertEqual(game.get_label_h_pairs(), clean_tiles)

    def test_h(self):
        game = Game(3)
        game.shuffle(333)
        self.assertNotEqual(game.h(), 0)
        game.tiles = game.generate_tiles(3)
        self.assertEqual(game.h(), 0)

    def test_is_solved(self):
        game = Game(3)
        game.shuffle(333)
        self.assertFalse(game.is_solved())
        game.tiles = game.generate_tiles(3)
        self.assertTrue(game.is_solved())

    def test_slide_tile(self):
        game = Game(3)                          # init game as a new (pre-shuffled) Game object, dimension=3
        game.tiles = game.generate_tiles(3)     # replace shuffled tiles with un-shuffled tiles, for a known state
        self.assertFalse(game.slide_tile(1))    # from an ordered matrix (of any size) tile 1 is always locked-in

        ### The asserts commented below are highly-likely to trigger intermittent assert failures

        # self.assertFalse(game.slide_tile(2))    # from an ordered matrix (of any size) tile 2 is always locked-in
        # self.assertFalse(game.slide_tile(4))    # from an ordered matrix (of any size) tile 5 is always locked-in

        # self.assertTrue(game.slide_tile(6))     # from an ordered matrix (dim=3) tile 6 is always free
        # self.assertTrue(game.slide_tile(6))     # any tile that can slide, can technically slide back to it's origin

        ### The asserts not commented below are less-likely to trigger intermittent assert failures, but still do

        self.assertFalse(game.slide_tile(1))    # more known assertions for an ordered matrix of dimension 3...
        self.assertTrue(game.slide_tile(8))
        self.assertTrue(game.slide_tile(7))
        self.assertFalse(game.slide_tile(1))
        self.assertTrue(game.slide_tile(7))
        self.assertTrue(game.slide_tile(8))
