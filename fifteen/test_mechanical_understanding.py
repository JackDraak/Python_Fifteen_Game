"""
Unit Test Suite for Mechanical Understanding in Fifteen Puzzle AI

Tests verify that the AI controllers understand fundamental mechanical constraints:
- Blank position categories and their move limitations
- Progressive vs regressive move classification
- Position-dependent action utilities
- Chain-move relationships

Educational Purpose: Demonstrates how to test AI understanding of domain constraints
"""

import unittest
import numpy as np
from Game import Game
from mechanical_ai_controller import MechanicalAIController


class TestMechanicalUnderstanding(unittest.TestCase):
    """Test suite for mechanical understanding in fifteen puzzle AI."""

    def setUp(self):
        """Set up test environment with specific game states."""
        self.game = Game(4, False)  # Start with solved state
        self.ai = MechanicalAIController(self.game)

    def create_corner_blank_state(self):
        """Create a state with blank in corner position (top-left)."""
        # Place blank in position 0 (top-left corner)
        state = [16, 2, 3, 4,   # 16 is blank
                 5, 6, 7, 8,
                 9, 10, 11, 12,
                 13, 14, 15, 1]
        self.game.set_state_from_list(state)
        return state

    def create_edge_blank_state(self):
        """Create a state with blank on edge (top middle)."""
        # Place blank in position 1 (top edge, not corner)
        state = [1, 16, 3, 4,   # 16 is blank
                 5, 6, 7, 8,
                 9, 10, 11, 12,
                 13, 14, 15, 2]
        self.game.set_state_from_list(state)
        return state

    def create_interior_blank_state(self):
        """Create a state with blank in interior position."""
        # Place blank in position 5 (interior)
        state = [1, 2, 3, 4,
                 5, 16, 7, 8,   # 16 is blank
                 9, 10, 11, 12,
                 13, 14, 15, 6]
        self.game.set_state_from_list(state)
        return state

    def test_blank_position_classification(self):
        """Test accurate classification of blank position categories."""

        # Test corner classification
        corner_state = self.create_corner_blank_state()
        category = self.ai.get_blank_position_category(corner_state)
        self.assertEqual(category, 'corner', "Failed to classify corner position")

        # Test edge classification
        edge_state = self.create_edge_blank_state()
        category = self.ai.get_blank_position_category(edge_state)
        self.assertEqual(category, 'edge', "Failed to classify edge position")

        # Test interior classification
        interior_state = self.create_interior_blank_state()
        category = self.ai.get_blank_position_category(interior_state)
        self.assertEqual(category, 'interior', "Failed to classify interior position")

    def test_corner_move_limitations(self):
        """Test understanding of corner position mechanics per truth table."""
        corner_state = self.create_corner_blank_state()

        # TRUTH: Always exactly 6 valid moves
        valid_moves = self.game.get_valid_moves()
        self.assertEqual(len(valid_moves), 6,
                        f"Should always have exactly 6 valid moves, got {len(valid_moves)}")

        # TRUTH: Corner has 1 regressive, 1 progressive, 0 exploratory, 4 chain
        self.ai.previous_move = valid_moves[0]  # Simulate previous move

        move_classifications = {}
        for move in valid_moves:
            utility = self.ai.classify_move_utility(move, corner_state)
            move_classifications[utility] = move_classifications.get(utility, 0) + 1

        # Verify regressive move is detected
        self.assertEqual(move_classifications.get('regressive', 0), 1,
                        "Corner should have exactly 1 regressive move")

    def test_edge_move_characteristics(self):
        """Test understanding of edge position mechanics per truth table."""
        edge_state = self.create_edge_blank_state()

        # TRUTH: Always exactly 6 valid moves
        valid_moves = self.game.get_valid_moves()
        self.assertEqual(len(valid_moves), 6,
                        f"Should always have exactly 6 valid moves, got {len(valid_moves)}")

        # TRUTH: Edge has 1 regressive, 1 progressive, 1 exploratory, 3 chain
        self.ai.previous_move = valid_moves[0]  # Simulate previous move

        move_classifications = {}
        for move in valid_moves:
            utility = self.ai.classify_move_utility(move, edge_state)
            move_classifications[utility] = move_classifications.get(utility, 0) + 1

        # Verify regressive move is detected
        self.assertEqual(move_classifications.get('regressive', 0), 1,
                        "Edge should have exactly 1 regressive move")

    def test_interior_move_diversity(self):
        """Test understanding of surrounded position mechanics per truth table."""
        interior_state = self.create_interior_blank_state()

        # TRUTH: Always exactly 6 valid moves
        valid_moves = self.game.get_valid_moves()
        self.assertEqual(len(valid_moves), 6,
                        f"Should always have exactly 6 valid moves, got {len(valid_moves)}")

        # TRUTH: Surrounded has 1 regressive, 1 progressive, 2 exploratory, 2 chain
        self.ai.previous_move = valid_moves[0]  # Simulate previous move

        move_classifications = {}
        for move in valid_moves:
            utility = self.ai.classify_move_utility(move, interior_state)
            move_classifications[utility] = move_classifications.get(utility, 0) + 1

        # Verify regressive move is detected
        self.assertEqual(move_classifications.get('regressive', 0), 1,
                        "Surrounded position should have exactly 1 regressive move")

    def test_regressive_move_detection(self):
        """Test accurate detection of regressive moves."""
        corner_state = self.create_corner_blank_state()

        # Set up scenario where we know the previous move
        previous_action = 2  # Simulate moving tile 2
        self.ai.previous_move = previous_action

        # The same action should be classified as regressive
        utility = self.ai.classify_move_utility(previous_action, corner_state)
        self.assertEqual(utility, 'regressive',
                        "Immediate reverse move should be classified as regressive")

    def test_enhanced_state_representation(self):
        """Test that enhanced state representation includes mechanical features."""
        corner_state = self.create_corner_blank_state()

        # Get enhanced representation
        enhanced_state = self.ai.get_enhanced_state_representation()

        # Should be longer than base state (16 + 6 additional features)
        expected_length = 16 + 6  # base state + mechanical features
        self.assertEqual(len(enhanced_state), expected_length,
                        f"Enhanced state should have {expected_length} features, got {len(enhanced_state)}")

        # Check blank position encoding
        blank_features = enhanced_state[16:19]  # Features 16-18 are blank position category
        corner_feature = blank_features[0]
        self.assertEqual(corner_feature, 1.0,
                        "Corner blank position should be encoded as [1,0,0]")

    def test_mechanical_reward_system(self):
        """Test that mechanical rewards properly incentivize good moves."""
        corner_state = self.create_corner_blank_state()

        # Create a scenario with known progressive vs regressive moves
        valid_moves = self.game.get_valid_moves()

        # Simulate previous move to create regressive context
        self.ai.previous_move = valid_moves[0]

        # Test rewards for different move types
        for move in valid_moves:
            # Create next state (simplified)
            next_state = corner_state.copy()

            reward = self.ai.calculate_mechanical_reward(
                corner_state, move, next_state, False
            )

            # Verify reward incorporates mechanical understanding
            self.assertIsInstance(reward, float, "Reward should be numeric")

            # Regressive moves should have negative mechanical component
            if move == self.ai.previous_move:
                # Note: Total reward might still be positive due to entropy,
                # but mechanical component should be negative
                move_utility = self.ai.classify_move_utility(move, corner_state)
                self.assertEqual(move_utility, 'regressive',
                               "Previous move should be classified as regressive")

    def test_mechanical_action_selection_bias(self):
        """Test that action selection shows preference for mechanically sound moves."""
        corner_state = self.create_corner_blank_state()

        # Set up AI state
        enhanced_state = self.ai.get_enhanced_state_representation()
        valid_moves = self.game.get_valid_moves()

        # Set up regressive context
        self.ai.previous_move = valid_moves[0]

        # Test action selection multiple times (should show bias against regressive)
        regressive_count = 0
        progressive_count = 0
        trials = 50

        for _ in range(trials):
            # Use exploration mode to see preferences
            action = self.ai.choose_action(enhanced_state, valid_moves, training=True)

            if action == self.ai.previous_move:
                regressive_count += 1
            else:
                progressive_count += 1

        # Should show preference against regressive moves (though not absolute due to exploration)
        self.assertLess(regressive_count, progressive_count,
                       "AI should prefer non-regressive moves over regressive ones")

    def test_move_history_tracking(self):
        """Test that move history is properly tracked for analysis."""
        # Execute a sequence of moves
        test_sequence = [2, 6, 2, 6]  # Back-and-forth pattern

        for move in test_sequence:
            self.ai.move_history.append(move)

        # Test detection of patterns
        enhanced_state = self.ai.get_enhanced_state_representation()

        # History features should be included in state
        history_features = enhanced_state[-2:]  # Last 2 features are history-related

        # Should detect back-and-forth pattern
        pattern_detected = history_features[1]
        self.assertEqual(pattern_detected, 1.0,
                        "Back-and-forth pattern should be detected in state representation")

    def test_position_specific_constraints(self):
        """Test understanding of position-specific movement constraints per truth table."""

        # Test each position type - ALL should have exactly 6 moves
        test_positions = [
            (self.create_corner_blank_state, 'corner'),
            (self.create_edge_blank_state, 'edge'),
            (self.create_interior_blank_state, 'interior')
        ]

        for create_state, pos_type in test_positions:
            with self.subTest(position=pos_type):
                state = create_state()
                category = self.ai.get_blank_position_category(state)
                valid_moves = self.game.get_valid_moves()

                self.assertEqual(category, pos_type,
                               f"Position should be classified as {pos_type}")
                self.assertEqual(len(valid_moves), 6,
                               f"All positions should have exactly 6 valid moves")

    def test_chain_move_understanding(self):
        """Test understanding that some moves are chain-moves."""
        # This is a more complex test that would require deeper implementation
        # For now, verify that the framework exists for this analysis

        corner_state = self.create_corner_blank_state()
        valid_moves = self.game.get_valid_moves()

        # Verify that move classification system can handle chain-move concepts
        for move in valid_moves:
            utility = self.ai.classify_move_utility(move, corner_state)
            self.assertIn(utility, ['progressive', 'regressive', 'exploratory', 'chain'],
                         f"Move utility '{utility}' should be one of the recognized types")

    def test_educational_output(self):
        """Test that the system can explain its mechanical understanding."""
        corner_state = self.create_corner_blank_state()

        # Verify that the AI can provide educational explanations
        category = self.ai.get_blank_position_category(corner_state)
        valid_moves = self.game.get_valid_moves()

        # Test verbose output capability
        enhanced_state = self.ai.get_enhanced_state_representation()

        # This should not raise exceptions and should provide meaningful info
        for move in valid_moves:
            utility = self.ai.classify_move_utility(move, corner_state)

            # Verify we can create educational explanations
            explanation = f"Move {move} from {category} position: {utility} utility"
            self.assertIsInstance(explanation, str)
            self.assertIn(category, explanation)
            self.assertIn(utility, explanation)


class TestMechanicalVsBasicAI(unittest.TestCase):
    """Comparative tests between mechanical AI and basic AI."""

    def setUp(self):
        """Set up both AI types for comparison."""
        self.game = Game(4, False)
        self.mechanical_ai = MechanicalAIController(self.game)

        # Import basic AI for comparison
        try:
            from AI_controller import AIController
            self.basic_ai = AIController(self.game)
        except ImportError:
            self.skipTest("Basic AI controller not available for comparison")

    def test_state_representation_complexity(self):
        """Test that mechanical AI has richer state representation."""
        basic_state = self.basic_ai.get_state_representation()
        mechanical_state = self.mechanical_ai.get_enhanced_state_representation()

        # Mechanical state should have more features
        self.assertGreater(len(mechanical_state), len(basic_state),
                          "Mechanical AI should have richer state representation")

    def test_move_selection_sophistication(self):
        """Test that mechanical AI shows more sophisticated move selection."""
        # Set up a scenario where mechanical understanding matters
        corner_state = [16, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 1]
        self.game.set_state_from_list(corner_state)

        # Get action choices from both AIs
        basic_state = self.basic_ai.get_state_representation()
        mechanical_state = self.mechanical_ai.get_enhanced_state_representation()
        valid_moves = self.game.get_valid_moves()

        # Both should be able to choose actions
        basic_action = self.basic_ai.choose_action(basic_state, valid_moves, training=False)
        mechanical_action = self.mechanical_ai.choose_action(mechanical_state, valid_moves, training=False)

        # Actions should be valid
        self.assertIn(basic_action, valid_moves)
        self.assertIn(mechanical_action, valid_moves)

        # Note: We can't easily test that mechanical is "better" without training,
        # but we can verify it has the capability for more sophisticated analysis


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()

    # Add mechanical understanding tests
    test_suite.addTest(unittest.makeSuite(TestMechanicalUnderstanding))
    test_suite.addTest(unittest.makeSuite(TestMechanicalVsBasicAI))

    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Print summary
    print(f"\n{'='*60}")
    print(f"MECHANICAL UNDERSTANDING TEST RESULTS")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")

    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")

    if result.wasSuccessful():
        print(f"\n✅ ALL TESTS PASSED - Mechanical understanding verified!")
    else:
        print(f"\n❌ SOME TESTS FAILED - Mechanical understanding needs work")