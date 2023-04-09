import unittest
from Game import Game

class TestGame(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.game = Game(4, False)

    def test_get_label(self): #  Non-destructive test
        self.assertEqual(self.game.get_label(0, 0), 1)
        self.assertEqual(self.game.get_label(0, 1), 2)
        self.assertEqual(self.game.get_label(0, 2), 3)
        self.assertEqual(self.game.get_label(1, 0), 5)
        self.assertEqual(self.game.get_label(1, 1), 6)
        self.assertEqual(self.game.get_label(1, 2), 7)
        self.assertEqual(self.game.get_label(3, 3), 16)

    def test_get_position(self): #  Non-destructive test
        self.assertEqual(self.game.get_position(1), (0, 0))
        self.assertEqual(self.game.get_position(2), (0, 1))
        self.assertEqual(self.game.get_position(3), (0, 2))
        self.assertEqual(self.game.get_position(5), (1, 0))
        self.assertEqual(self.game.get_position(6), (1, 1))
        self.assertEqual(self.game.get_position(7), (1, 2))
        self.assertEqual(self.game.get_position(16), (3, 3))

    def test_slide_tile(self):
        self.assertFalse(self.game.slide_tile(1))   #  Tile cannot be moved from starting position
        self.assertTrue(self.game.slide_tile(12))   #  Tile can be moved to adjacent empty space
        self.assertEqual(self.game.get_position(12), (3, 3))
        self.assertEqual(self.game.get_position(self.game.blank_label), (2, 3))
        self.assertTrue(self.game.slide_tile(12))   #  Tile moved back to original position for next test

    def test_is_solved(self):
        self.assertTrue(self.game.is_solved())
        self.game.shuffle(10) #  Shuffle the tiles
        self.assertFalse(self.game.is_solved())


if __name__ == '__main__':
    unittest.main()
