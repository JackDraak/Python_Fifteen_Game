"""
AI Controller for the Fifteen-style sliding puzzle, (this class written by ChatGPT5)

Features
- Epsilon-greedy Q-style learning keyed by flattened board state (tuple of labels).
- Lightweight state simulation that does not mutate the real Game instance.
- Reward signal based on reduction in "distance sum" (1D absolute index distance to goal).
- Persistence: save/load learned Q-values to a JSON file.
- Simple anti-repetition heuristic to avoid immediate back-and-forth moves.
- Tunable parameters: exploration_rate (epsilon), learning_rate, discount, exploration_decay.

Usage
- Create a Game from your Game class and pass it to AIController(game).
- Call ai.play_episode(max_steps=1000) to run a single episode that learns as it goes.
- Use ai.save_learning(path) and ai.load_learning(path) to persist the knowledge.

Notes
- This controller treats each board state as a tuple(game.get_labels_as_list()).
- It simulates candidate moves using the tile-label swap sequence from Game.get_move_sequence()
  without changing the real Game object.

"""
import json
import random
import math
from typing import Dict, Tuple, List, Optional


class AIController:
    def __init__(self,
                 game,
                 exploration_rate: float = 0.2,
                 learning_rate: float = 0.5,
                 discount: float = 0.9,
                 exploration_decay: float = 0.9999,
                 min_exploration: float = 0.01,
                 seed: Optional[int] = None):
        """
        game: instance of Game from the repo
        exploration_rate: initial epsilon for epsilon-greedy
        learning_rate: alpha for Q updates
        discount: gamma for Q updates
        exploration_decay: multiplicative decay applied after each step
        """
        if seed is not None:
            random.seed(seed)

        self.game = game
        self.breadth = game.breadth
        self.blank_label = game.blank_label

        # Q-table: maps state (tuple of labels) -> dict move_label -> q-value
        self.Q: Dict[Tuple[int, ...], Dict[int, float]] = dict()

        # parameters
        self.epsilon = exploration_rate
        self.alpha = learning_rate
        self.gamma = discount
        self.exploration_decay = exploration_decay
        self.min_exploration = min_exploration

        self.last_move: Optional[int] = None
        self.episode_steps = 0

    # -------------------- State helpers --------------------
    def _state_from_game(self, game) -> Tuple[int, ...]:
        return tuple(game.get_labels_as_list())

    @staticmethod
    def _distance_sum_from_labels(labels: List[int]) -> int:
        """Simple proxy for board "entropy". For label L at index i, goal index is L-1.
        We sum absolute differences of indices. This is cheap and consistent with the
        existing distance spirit in the code base.
        """
        total = 0
        for i, lab in enumerate(labels):
            # goal index for label lab is lab - 1
            total += abs((lab - 1) - i)
        return total

    # -------------------- Simulation helpers --------------------
    def _simulate_sequence(self, labels: List[int], sequence: List[int]) -> List[int]:
        """Apply sequence of moves (list of tile labels) to a flat labels list.
        Each move is a swap of the tile with the blank tile. We operate on copies.
        """
        labels_copy = list(labels)
        for move_label in sequence:
            try:
                blank_index = labels_copy.index(self.blank_label)
                move_index = labels_copy.index(move_label)
            except ValueError:
                # invalid label in sequence, return current state
                return labels_copy
            # swap
            labels_copy[blank_index], labels_copy[move_index] = labels_copy[move_index], labels_copy[blank_index]
        return labels_copy

    # -------------------- Q-table helpers --------------------
    def _ensure_state(self, state: Tuple[int, ...], valid_moves: List[int]):
        if state not in self.Q:
            self.Q[state] = {m: 0.0 for m in valid_moves}
        else:
            # make sure all current valid moves have keys
            for m in valid_moves:
                if m not in self.Q[state]:
                    self.Q[state][m] = 0.0

    # -------------------- Action selection --------------------
    def choose_move(self) -> Optional[int]:
        """Choose a move for the current game using epsilon-greedy policy.
        The controller will prefer moves with higher Q value, but also uses a cheap
        simulated immediate distance reduction heuristic to break ties.
        """
        valid_moves = self.game.get_valid_moves()
        if not valid_moves:
            return None

        state = self._state_from_game(self.game)
        self._ensure_state(state, valid_moves)

        # Exploration
        if random.random() < self.epsilon:
            # prefer moves that are not immediate reversal of last_move if possible
            choices = [m for m in valid_moves if m != self.last_move]
            if not choices:
                choices = valid_moves
            choice = random.choice(choices)
            return choice

        # Exploitation: pick move with highest expected Q + small simulated heuristic
        best_move = None
        best_score = -math.inf
        current_labels = list(state)
        current_distance = self._distance_sum_from_labels(current_labels)

        for move in valid_moves:
            q_value = self.Q[state].get(move, 0.0)
            seq = self.game.get_move_sequence(move)
            future_labels = self._simulate_sequence(current_labels, seq)
            new_distance = self._distance_sum_from_labels(future_labels)
            # heuristic: distance reduction
            dist_reward = current_distance - new_distance
            # combined score: Q-value plus a small factor of dist_reward to bias toward improvements
            score = q_value + 0.5 * dist_reward

            # avoid immediate reversal: penalize the same label as last move
            if self.last_move is not None and move == self.last_move:
                score -= 0.1

            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    # -------------------- Learning update --------------------
    def learn_from_transition(self,
                              state: Tuple[int, ...],
                              action: int,
                              reward: float,
                              next_state: Tuple[int, ...],
                              next_valid_moves: List[int]):
        self._ensure_state(state, next_valid_moves)  # ensure both states exist in Q table
        self._ensure_state(next_state, next_valid_moves)

        old_q = self.Q[state].get(action, 0.0)
        # estimate of optimal future
        next_max = 0.0
        if self.Q.get(next_state):
            next_max = max(self.Q[next_state].values()) if self.Q[next_state] else 0.0

        # Q-learning update
        new_q = old_q + self.alpha * (reward + self.gamma * next_max - old_q)
        self.Q[state][action] = new_q

    # -------------------- Episode / Play control --------------------
    def step(self) -> bool:
        """Take one action in the real game, learn from it, and return True if moved.
        Returns False if no move could be made.
        """
        state = self._state_from_game(self.game)
        valid_moves = self.game.get_valid_moves()
        if not valid_moves:
            return False

        action = self.choose_move()
        if action is None:
            return False

        # simulate to compute reward without changing the real game
        current_labels = list(state)
        current_distance = self._distance_sum_from_labels(current_labels)
        seq = self.game.get_move_sequence(action)
        next_labels = self._simulate_sequence(current_labels, seq)
        next_distance = self._distance_sum_from_labels(next_labels)

        reward = float(current_distance - next_distance)

        # Now execute the action on the real game
        success = self.game.player_move(action)
        if not success:
            # punish invalid execution
            reward = -1.0

        next_state = tuple(self.game.get_labels_as_list())
        next_valid_moves = self.game.get_valid_moves()

        # learn
        self.learn_from_transition(state, action, reward, next_state, next_valid_moves)

        # update bookkeeping
        self.last_move = action
        self.episode_steps += 1

        # decay exploration slightly
        self.epsilon = max(self.min_exploration, self.epsilon * self.exploration_decay)

        return success

    def play_episode(self, max_steps=100, verbose=False):
      """
      Run up to max_steps moves in the game, learning as we go.
      Returns (total_reward, steps_taken).
      """
      total_reward = 0
      steps_taken = 0
  
      for _ in range(max_steps):
          # Record current distance for reward shaping
          before_distance = self._distance_sum_from_labels(self.game.get_labels_as_list())
  
          moved = self.step()  # chooses move, executes, updates Q
  
          if not moved:
              break  # no valid moves
  
          after_distance = self._distance_sum_from_labels(self.game.get_labels_as_list())
          reward = before_distance - after_distance
          total_reward += reward
          steps_taken += 1
  
          if verbose:
              print(f"Step {steps_taken}, Reward: {reward:.2f}, Epsilon: {self.epsilon:.3f}")
  
          # Optional: check if solved
          if after_distance == 0:
              break
  
      return total_reward, steps_taken

  
    # -------------------- Persistence --------------------
    def save_learning(self, path: str):
        # Convert Q to a JSON-serializable structure
        serial = {','.join(map(str, state)): moves for state, moves in self.Q.items()}
        with open(path, 'w') as f:
            json.dump({
                'Q': serial,
                'epsilon': self.epsilon,
                'alpha': self.alpha,
                'gamma': self.gamma
            }, f)

    def load_learning(self, path: str):
        with open(path, 'r') as f:
            data = json.load(f)
        serial = data.get('Q', {})
        self.Q = {}
        for state_str, moves in serial.items():
            state = tuple(int(x) for x in state_str.split(',')) if state_str else tuple()
            # map keys back to ints
            self.Q[state] = {int(k): float(v) for k, v in moves.items()}
        self.epsilon = data.get('epsilon', self.epsilon)
        self.alpha = data.get('alpha', self.alpha)
        self.gamma = data.get('gamma', self.gamma)

    def reset_learning(self):
        self.Q = {}
        self.epsilon = 0.2
        self.last_move = None
        self.episode_steps = 0


# -------------------- Example helper function --------------------
def run_example_episode(game_class, board_size=4, shuffled=True, verbose=False):
    """Utility for quick manual testing. Creates a Game instance and runs the AI.
    game_class should be the Game class object imported from the codebase.
    """
    game = game_class(board_size, shuffled)
    ai = AIController(game)
    solved = ai.play_episode(max_steps=2000, verbose=verbose)
    return ai, solved


# If run directly, run a quick smoke test when the Game class is available
if __name__ == '__main__':
    try:
        from Game import Game
        ai, solved = run_example_episode(Game, board_size=4, shuffled=True, verbose=True)
        print('Solved:', solved)
    except Exception as e:
        print('Smoke test failed. Make sure Game.py is importable from this directory.')
        print(e)
