# API Documentation

Technical reference for the Fifteen Puzzle AI controllers and supporting systems.

## Core Game Engine

### `Game` Class

The fundamental puzzle implementation providing state management and move validation.

#### Constructor
```python
Game(breadth: int, shuffled: bool, seed: int = None)
```

**Parameters:**
- `breadth`: Grid size (typically 4 for 4×4 puzzle)
- `shuffled`: Whether to shuffle the puzzle on creation
- `seed`: Random seed for reproducible puzzles

**Example:**
```python
# Create solved 4x4 puzzle
game = Game(4, False)

# Create reproducible shuffled puzzle
game = Game(4, True, seed=42)
```

#### Key Methods

##### `get_state() -> List[int]`
Returns flattened list representation of current puzzle state.

**Returns:** List of tile labels (1-15, 16 for blank)

##### `player_move(tile_label: int) -> bool`
Attempts to move specified tile into blank space.

**Parameters:**
- `tile_label`: Label of tile to move (1-16)

**Returns:** `True` if move successful, `False` if invalid

##### `get_valid_moves() -> List[int]`
Returns list of tiles that can legally move.

**Returns:** List of valid tile labels

##### `is_solved() -> bool`
Checks if puzzle is in solved state.

**Returns:** `True` if solved, `False` otherwise

##### `get_distance_sum() -> int`
Calculates Manhattan distance sum (entropy measure).

**Returns:** Sum of distances from goal positions

##### `shuffle(moves: int) -> None`
Applies random valid moves to shuffle puzzle.

**Parameters:**
- `moves`: Number of random moves to apply

---

## Reinforcement Learning AI

### `AIController` Class

Deep Q-Network implementation for educational reinforcement learning demonstrations.

#### Constructor
```python
AIController(game: Game,
           learning_rate: float = 0.001,
           discount_factor: float = 0.95,
           epsilon_start: float = 1.0,
           epsilon_decay: float = 0.995,
           epsilon_min: float = 0.01,
           memory_size: int = 10000,
           batch_size: int = 32,
           target_update_freq: int = 100,
           reward_scale: float = 1.0,
           entropy_reward_scale: float = 0.1,
           use_tensorflow: bool = True,
           anti_loop_penalty: float = 0.1)
```

**Key Parameters:**
- `learning_rate`: Neural network learning rate
- `epsilon_start`: Initial exploration rate (1.0 = fully random)
- `epsilon_decay`: Exploration decay rate per episode
- `anti_loop_penalty`: Penalty multiplier for repetitive actions

#### Core Methods

##### `play_episode(max_steps: int = 300, training: bool = True, verbose: bool = False) -> Tuple[float, int, bool]`
Executes one complete puzzle-solving attempt.

**Parameters:**
- `max_steps`: Maximum moves before timeout
- `training`: Whether to update model weights
- `verbose`: Whether to print step-by-step progress

**Returns:**
- `total_reward`: Cumulative reward earned
- `steps_taken`: Number of moves made
- `solved`: Whether puzzle was solved

**Example:**
```python
ai = AIController(game)
reward, steps, solved = ai.play_episode(max_steps=100, verbose=True)
print(f"Result: {'Solved' if solved else 'Failed'} in {steps} steps")
```

##### `train(episodes: int, verbose: bool = True, save_freq: int = 100) -> None`
Runs training loop for specified number of episodes.

**Parameters:**
- `episodes`: Number of training episodes
- `verbose`: Whether to print progress updates
- `save_freq`: Episodes between model saves

##### `evaluate(episodes: int = 100, verbose: bool = True) -> Dict[str, Any]`
Evaluates trained model on fresh puzzles without training.

**Returns:** Dictionary with performance statistics

##### `attach_statistics_tracker(tracker: StatisticsTracker) -> None`
Attaches statistics tracker for detailed performance analysis.

#### Anti-Loop System

The RL AI includes sophisticated measures to prevent educational disruption from infinite loops:

- **Consecutive Move Limiting**: Blocks actions repeated more than `max_consecutive_moves` times
- **Stagnation Detection**: Tracks episodes without entropy improvement
- **Forced Exploration**: Triggers random moves when stagnation exceeds threshold

**Configuration:**
```python
ai.max_consecutive_moves = 3        # Max same action repetitions
ai.force_exploration_threshold = 5  # Stagnation episodes before random moves
```

---

## Optimal Search AI

### `OptimalAI` Class

IDA* implementation guaranteeing mathematically shortest solutions.

#### Constructor
```python
OptimalAI(game: Game)
```

#### Core Methods

##### `solve(max_moves: int = 80, timeout: float = 30.0, verbose: bool = True) -> Optional[List[int]]`
Finds optimal solution using IDA* search.

**Parameters:**
- `max_moves`: Maximum solution length to search
- `timeout`: Search timeout in seconds
- `verbose`: Whether to show search progress

**Returns:**
- List of tile labels representing optimal solution
- `None` if no solution found within limits

**Example:**
```python
optimal_ai = OptimalAI(game)
solution = optimal_ai.solve(max_moves=60, timeout=20.0)
if solution:
    print(f"Optimal solution: {len(solution)} moves")
    print(f"Moves: {solution}")
```

##### `execute_solution(solution: List[int], verbose: bool = True) -> bool`
Executes solution moves on the game.

**Parameters:**
- `solution`: List of moves from `solve()`
- `verbose`: Whether to show each move

**Returns:** `True` if execution successful

##### `heuristic(state: List[int]) -> int`
Calculates heuristic estimate for given state.

**Returns:** Lower bound estimate of moves to solution

##### `manhattan_distance(state: List[int]) -> int`
Calculates Manhattan distance component of heuristic.

##### `linear_conflict(state: List[int]) -> int`
Calculates linear conflict component of heuristic.

#### Heuristic Details

The optimal AI uses a hybrid heuristic combining:

1. **Manhattan Distance**: Sum of taxi-cab distances from current to goal positions
2. **Linear Conflict**: Additional moves needed when tiles are in correct row/column but wrong order

**Mathematical Properties:**
- **Admissible**: Never overestimates actual distance
- **Consistent**: Satisfies triangle inequality
- **Informative**: Provides tight lower bounds

---

## Statistics and Analytics

### `StatisticsTracker` Class

Comprehensive performance analysis and learning progress tracking.

#### Constructor
```python
StatisticsTracker(save_directory: str = "ai_statistics")
```

#### Core Methods

##### `record_episode(ai_controller, episode_num: int, reward: float, steps: int, solved: bool, training_time: float = 0.0) -> EpisodeStats`
Records detailed statistics for a single episode.

##### `generate_learning_report(last_n_episodes: int = 100) -> Dict[str, Any]`
Generates comprehensive learning analysis report.

**Returns:** Dictionary containing:
- Basic metrics (solve rate, average steps, rewards)
- Learning trends (improvement over time)
- Behavioral patterns (exploration rates, stagnation)

##### `record_comparison(puzzle_id: str, shuffle_count: int, initial_entropy: int, rl_result: Dict, optimal_result: Dict) -> ComparisonResult`
Records comparative results between RL and Optimal AI.

##### `save_statistics(filename: str = None) -> str`
Persists all tracked data to JSON file.

##### `load_statistics(filename: str) -> bool`
Loads previously saved statistics.

#### Data Structures

##### `EpisodeStats`
```python
@dataclass
class EpisodeStats:
    episode_num: int
    solved: bool
    steps: int
    reward: float
    final_entropy: int
    epsilon: float
    training_time: float
    stagnation_count: int
    forced_exploration: bool
    action_diversity: float
    entropy_reduction: int
```

##### `ComparisonResult`
```python
@dataclass
class ComparisonResult:
    puzzle_id: str
    shuffle_count: int
    initial_entropy: int
    rl_result: Dict[str, Any]
    optimal_result: Dict[str, Any]
    timestamp: float
```

---

## Educational Demo System

### `IntegratedDemo` Class

Orchestrates educational comparisons between AI approaches.

#### Constructor
```python
IntegratedDemo()
```

#### Key Methods

##### `run_single_comparison(shuffle_count: int = 20, max_steps: int = 100, show_moves: bool = True, seed: int = None) -> Dict[str, Any]`
Runs side-by-side comparison on identical puzzle.

**Educational Output:**
- Methodology explanations for each approach
- Real-time performance metrics
- Comparative analysis and insights
- Trade-off discussions

##### `run_performance_suite(difficulties: List[int] = None, trials_per_difficulty: int = 3) -> Dict`
Comprehensive evaluation across multiple difficulty levels.

**Returns:** Detailed performance breakdown by difficulty

---

## Usage Patterns

### Basic Educational Demo
```python
from integrated_demo import IntegratedDemo

demo = IntegratedDemo()
result = demo.run_single_comparison(
    shuffle_count=15,
    show_moves=True,
    seed=42  # Reproducible for classroom use
)
```

### Advanced Analysis
```python
from statistics_tracker import StatisticsTracker
from AI_controller import AIController

# Setup tracking
tracker = StatisticsTracker()
ai = AIController(game, epsilon_start=0.2)
ai.attach_statistics_tracker(tracker)

# Training loop with analysis
for episode in range(100):
    reward, steps, solved = ai.play_episode(training=True)
    if episode % 10 == 0:
        report = tracker.generate_learning_report(10)
        print(f"Episode {episode}: {report['basic_metrics']['solve_rate']} solve rate")

# Save session data
tracker.save_statistics(f"training_session_{episode}.json")
```

### Performance Benchmarking
```python
from Game import Game
from OptimalAI import OptimalAI

# Test optimal AI efficiency
difficulties = [5, 10, 15, 20, 25, 30]
for diff in difficulties:
    game = Game(4, False, seed=42)
    game.shuffle(diff)

    optimal_ai = OptimalAI(game)
    solution = optimal_ai.solve(verbose=False)

    if solution:
        print(f"Difficulty {diff}: {len(solution)} moves, {optimal_ai.nodes_generated} nodes")
```

---

## Error Handling and Edge Cases

### Common Exceptions

- **`InvalidMoveError`**: Attempting to move unmovable tile
- **`TimeoutError`**: Search/training exceeds time limits
- **`MemoryError`**: Insufficient memory for neural networks
- **`ImportError`**: Missing TensorFlow (auto-fallback to Q-table)

### Graceful Degradation

- **No TensorFlow**: Automatic fallback to Q-table implementation
- **No GPU**: TensorFlow uses CPU (with warnings)
- **Memory Constraints**: Reduced batch sizes and memory limits
- **Timeout Handling**: Search algorithms respect time limits

### Debugging Support

All classes support verbose output for educational transparency:

```python
# Enable detailed logging
ai.play_episode(verbose=True)         # Show each move decision
optimal_ai.solve(verbose=True)        # Show search progress
demo.run_single_comparison(show_moves=True)  # Full comparison details
```

---

## Performance Characteristics

### Computational Complexity

| Algorithm | Time Complexity | Space Complexity | Optimality |
|-----------|----------------|------------------|------------|
| **RL AI (DQN)** | O(k) per episode | O(memory_size) | Sub-optimal |
| **Optimal AI (IDA*)** | O(b^d) worst case | O(d) | Guaranteed optimal |

Where:
- k = episode length (typically 50-300 steps)
- b = branching factor (~3 average for fifteen puzzle)
- d = solution depth (optimal moves needed)

### Practical Performance

**RL AI:**
- Episode time: ~0.1-0.2 seconds
- Memory usage: ~100MB with TensorFlow
- Success rate: 0-80% depending on training

**Optimal AI:**
- Easy puzzles (≤20 moves): <0.001 seconds
- Medium puzzles (20-35 moves): 0.001-0.1 seconds
- Hard puzzles (35-50 moves): 0.1-10 seconds
- Very hard puzzles (50+ moves): May timeout

---

## Configuration Reference

### Environment Variables

```bash
# Disable TensorFlow GPU warnings
export TF_ENABLE_ONEDNN_OPTS=0
export TF_CPP_MIN_LOG_LEVEL=2

# Set memory limits
export CUDA_VISIBLE_DEVICES=""  # Force CPU-only
```

### Default Parameters

```python
# RL AI Defaults
DEFAULT_RL_PARAMS = {
    'learning_rate': 0.001,
    'epsilon_start': 1.0,
    'epsilon_decay': 0.995,
    'epsilon_min': 0.01,
    'memory_size': 10000,
    'batch_size': 32,
    'max_consecutive_moves': 3,
    'force_exploration_threshold': 5
}

# Optimal AI Defaults
DEFAULT_OPTIMAL_PARAMS = {
    'max_moves': 80,
    'timeout': 30.0,
    'search_algorithm': 'IDA*',
    'heuristic': 'manhattan_distance + linear_conflict'
}
```

This API documentation provides the technical foundation for understanding and extending the educational AI system. For usage examples and educational context, see the main README.md.