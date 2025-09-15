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

    def test_deterministic_shuffle(self):
        """Test that shuffles with the same seed produce identical results"""
        game1 = Game(4, False, seed=42)
        game2 = Game(4, False, seed=42)

        # Both games should start in solved state
        self.assertEqual(game1.get_state(), game2.get_state())

        # Shuffle both with same number of moves
        game1.shuffle(10)
        game2.shuffle(10)

        # Should have identical states after shuffling
        self.assertEqual(game1.get_state(), game2.get_state())
        self.assertEqual(game1.blank_position, game2.blank_position)

    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different shuffle results"""
        game1 = Game(4, False, seed=42)
        game2 = Game(4, False, seed=123)

        game1.shuffle(10)
        game2.shuffle(10)

        # Should have different states (highly likely with different seeds)
        self.assertNotEqual(game1.get_state(), game2.get_state())

    def test_get_set_state(self):
        """Test state serialization and restoration"""
        game = Game(4, False)
        original_state = game.get_state()

        # Shuffle the game
        game.shuffle(5)
        shuffled_state = game.get_state()

        # State should be different after shuffle
        self.assertNotEqual(original_state, shuffled_state)

        # Restore original state
        self.assertTrue(game.set_state(original_state))
        restored_state = game.get_state()

        # Should match original state
        self.assertEqual(original_state, restored_state)
        self.assertTrue(game.is_solved())

    def test_set_state_validation(self):
        """Test that set_state properly validates input"""
        game = Game(4, False)

        # Test invalid length
        self.assertFalse(game.set_state([1, 2, 3]))  # Too short

        # Test invalid labels
        self.assertFalse(game.set_state([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17]))  # Missing 16, has 17

        # Test duplicate labels
        self.assertFalse(game.set_state([1, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]))  # Duplicate 1


if __name__ == '__main__':
    unittest.main()
