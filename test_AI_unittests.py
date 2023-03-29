import unittest
from Game import Game
from AI_trainer_controller import QNetwork, AI_trainer_controller


class TestGameMethods(unittest.TestCase):
    def test_game_methods(self):
        game = Game(3, True)
        self.assertIsNotNone(game.get_state())
        self.assertIsNotNone(game.get_valid_moves())
        self.assertIsNotNone(game.slide_tile(1))
        self.assertIsNotNone(game.is_solved())


class TestQNetworkMethods(unittest.TestCase):
    def test_qnetwork_methods(self):
        qnetwork = QNetwork(9, 64, 9)
        state = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 0], dtype=torch.float32).unsqueeze(0)
        output = qnetwork(state)
        self.assertIsNotNone(output)


class TestAITrainerControllerMethods(unittest.TestCase):
    def test_ai_trainer_controller_methods(self):
        ai_trainer = AI_trainer_controller(3, 0.001, 0.95, 1.0)
        game = Game(3, True)

        state_tensor = ai_trainer._game_state_to_tensor(game)
        self.assertIsNotNone(state_tensor)

        action = ai_trainer._choose_action(game)
        self.assertIsNotNone(action)

        next_game = Game(3, True)
        ai_trainer._store_transition(game, action, -1, next_game, False)
        self.assertTrue(len(ai_trainer.memory) > 0)

        ai_trainer._learn_from_memory()

        ai_trainer.train(10)
        ai_trainer.play()


if __name__ == "__main__":
    unittest.main()