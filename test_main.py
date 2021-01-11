from unittest import TestCase
from main import Tile
from main import Game


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
        tile.set(9, 8, 7)
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

    def test_export_tiles(self):
        pass  # TODO

    def test_get_h_by_label(self):
        pass

    def test_get_label(self):
        pass

    def test_get_label_h_pairs(self):
        pass

    def test_get_labels_as_list(self):
        pass

    def test_get_labels_as_matrix(self):
        pass

    def test_get_position(self):
        pass

    def test_get_valid_moves(self):
        pass

    def test_h(self):
        pass

    def test_import_tiles(self):
        pass

    def test_is_solved(self):
        pass

    def test_reset_game(self):
        pass

    def test_set_tile_position(self):
        pass

    def test_shuffle(self):
        pass

    def test_slide_tile(self):
        pass
