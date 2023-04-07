import unittest
from unittest import TestCase
from Tile import Tile
from Game import Game
import console_controller


class test_console_controller(TestCase):
    def test_command_check(self):
        self.assertEqual(console_controller.command_check(""), "")
        self.assertEqual(console_controller.command_check("a"), (0, -1))
        self.assertEqual(console_controller.command_check("d"), (0, 1))
        self.assertEqual(console_controller.command_check("s"), (1, 0))
        self.assertEqual(console_controller.command_check("w"), (-1, 0))
        self.assertNotEqual(console_controller.command_check("foo"), "bar")
        self.assertEqual(console_controller.command_check("foobar"), "foobar")
        self.assertEqual(console_controller.command_check("13"), "13")


class test_Game_class(TestCase):
    def test_get_cardinal_label(self):
        game = Game(3, False)
        self.assertEqual(game.get_ordinal_label((-1, -1)), 5)
        self.assertEqual(game.get_ordinal_label((0, 0)), 9)
        self.assertNotEqual(game.get_ordinal_label((-1, 0)), 1)

    def test_get_distance_by_label(self):
        game = Game(3, False)
        game.slide_tile(6)
        self.assertEqual(game.get_distance_by_label(6), 3)
        self.assertEqual(game.get_distance_by_label(9), 3)
        self.assertNotEqual(game.get_distance_by_label(1), 9)

    def test_get_distance_set(self):
        ordered_tiles = [[(1, 0)], [(2, 0)], [(3, 0)], [(4, 0)], [(5, 0)], [(6, 0)], [(7, 0)], [(8, 0)], [(9, 0)]]
        game = Game(3, False)
        self.assertEqual(game.get_distance_set(), ordered_tiles)
        game = Game(3, True)
        self.assertNotEqual(game.get_distance_set(), ordered_tiles)
        self.assertNotEqual(game.get_distance_set(), [[(1, 0)], [(2, 0)], [(3, 0)], [(4, 0)], [(5, 0)], [(6, 0)], [(7, 0)], [(8, 0)], [(9, 0)]])

    def test_get_distance_sum(self):
        game = Game(3, False)
        self.assertEqual(game.get_distance_sum(), 0)
        game = Game(3, True)
        self.assertNotEqual(game.get_distance_sum(), 0)

    def test_get_label(self):
        game = Game(3, False)
        self.assertEqual(game.get_label(0, 0), 1)
        self.assertEqual(game.get_label(1, 1), 5)
        self.assertEqual(game.get_label(2, 2), 9)
        self.assertNotEqual(game.get_label(2, 0), 2)

    def test_get_labels_as_list(self):
        game = Game(3, False)
        ordered_tiles = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.assertEqual(game.get_labels_as_list(), ordered_tiles)

    def test_get_labels_as_matrix(self):
        game = Game(3, False)
        ordered_tiles = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        self.assertEqual(game.get_labels_as_matrix(), ordered_tiles)

    def test_get_position(self):
        game = Game(3, False)
        self.assertEqual(game.get_position(1), (0, 0))
        self.assertEqual(game.get_position(5), (1, 1))
        self.assertEqual(game.get_position(9), (2, 2))

    def test_get_valid_moves(self):
        game = Game(3, False)
        self.assertEqual(game.get_valid_moves(), [6, 8])
        game.slide_tile(6)
        self.assertEqual(game.get_valid_moves(), [3, 5, 6])

    def test_is_solved(self):
        game = Game(3, True)
        self.assertFalse(game.is_solved())
        game = Game(3, False)
        self.assertTrue(game.is_solved())

    def test_slide_tile(self):
        game = Game(3, False)                   # init game as a new (un-shuffled) Game object, of dimension = 3
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


class test_Tile_class(TestCase):
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
        self.assertNotEqual(tile.ordinal, 2)



if __name__ == '__main__':
    unittest.main()
