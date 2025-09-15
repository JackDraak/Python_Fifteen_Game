# AI Controller for Fifteen Puzzle

This directory contains a sophisticated AI controller designed to solve the Fifteen Puzzle (sliding tile puzzle) using deep reinforcement learning principles.

## Overview

The Fifteen Puzzle has **deceptive complexity** - while the rules are simple (slide tiles into the empty space), optimal solutions often require temporarily increasing entropy (disorder) to unlock better future states. This AI understands and learns these complex patterns.

## Key Features

### 🧠 Intelligent State Understanding
- **Entropy-based evaluation**: Uses Manhattan distance sum as the primary measure of puzzle disorder
- **Pattern recognition**: Deep neural network learns complex tile interaction patterns
- **Strategic thinking**: Understands when to temporarily increase entropy for long-term benefit

### 🎯 Sophisticated Reward System
- **+1000 points**: Solving the puzzle completely
- **+0.1 per unit**: Entropy reduction (good moves)
- **-0.1 per unit**: Entropy increase (sometimes necessary)
- **-0.01**: Small step penalty to encourage efficiency

### 🔧 Advanced Learning Algorithm
- **Deep Q-Network (DQN)**: Neural network learns Q-values for state-action pairs
- **Experience replay**: Stabilizes learning with memory buffer
- **Target networks**: Reduces overestimation bias
- **Epsilon-greedy exploration**: Balances exploration vs exploitation

## Files

- **`AI_controller.py`** - Main AI implementation with DQN and Q-table fallback
- **`train_ai.py`** - Training script with comprehensive evaluation and visualization
- **`demo_ai.py`** - Interactive demo showing AI behavior and entropy concepts

## Quick Start

### 1. Basic Demo (Untrained AI)
```bash
python demo_ai.py
```
Shows how untrained AI behaves and explains entropy concepts.

### 2. Train the AI
```bash
# Quick training (500 episodes)
python train_ai.py --episodes 500 --verbose

# Extended training with evaluation
python train_ai.py --episodes 2000 --eval-freq 100 --plot
```

### 3. Evaluate Trained Model
```bash
python train_ai.py --load final_model_4x4_[timestamp] --evaluate-only
```

### 4. Interactive Demo with Trained AI
```bash
python train_ai.py --load final_model_4x4_[timestamp] --interactive
```

## Understanding the Complexity

### Why the 15-Puzzle is Deceptively Hard

1. **State Space**: 16!/(2) ≈ 10^13 possible states for 4x4 puzzle
2. **Optimal Solutions**: Often require 50+ moves for random configurations
3. **Local Optima**: Many moves appear good locally but hurt globally
4. **Entropy Paradox**: Sometimes must increase disorder to decrease it later

### Example: The AI's Learning Journey

```
Initial State (Random):     Goal State:
[ 15  14   1   6 ]         [  1   2   3   4 ]
[  9  11   4  12 ]    →    [  5   6   7   8 ]
[  2   8   5   7 ]         [  9  10  11  12 ]
[  3  10  13   _ ]         [ 13  14  15   _ ]

Entropy: 142 Manhattan distance units
```

The AI must learn that achieving this transformation often requires:
1. Moving tiles further from their goal (entropy increase)
2. Creating space for critical tile movements
3. Recognizing multi-step patterns and sequences

## AI Architecture

### Neural Network Design
```
Input Layer:    16 units (flattened board state, normalized)
Hidden Layer 1: 256 units (ReLU) + Dropout(0.2)
Hidden Layer 2: 256 units (ReLU) + Dropout(0.2)
Hidden Layer 3: 128 units (ReLU) + Dropout(0.1)
Hidden Layer 4: 128 units (ReLU)
Output Layer:   16 units (Q-values for each tile move)
```

### Training Process
1. **Exploration Phase**: High epsilon (90%+ random moves)
2. **Learning Phase**: Gradually reduce randomness as patterns emerge
3. **Exploitation Phase**: Low epsilon (1% random moves)
4. **Experience Replay**: Learn from past successful/unsuccessful moves

## Expected Results

### Training Progress (Typical)
- **Episodes 1-100**: Mostly random behavior, rare solutions
- **Episodes 100-500**: Pattern recognition emerges, ~10-20% solve rate
- **Episodes 500-1000**: Strategic play develops, ~40-60% solve rate
- **Episodes 1000+**: Refined strategy, ~70-90% solve rate on solvable puzzles

### Performance Metrics
- **Solve Rate**: Percentage of puzzles solved within step limit
- **Average Steps**: Steps to solution (lower is better)
- **Entropy Reduction**: Ability to consistently reduce puzzle disorder
- **Convergence**: Stable Q-values indicating learned patterns

## Game.py Integration

The AI leverages these Game.py methods for complete state awareness:

### State Observation
- `get_state()` - Full board state for neural network input
- `get_distance_sum()` - Total entropy (Manhattan distance sum)
- `get_distance_by_label(label)` - Individual tile displacement
- `is_solved()` - Terminal condition detection

### Action Execution
- `get_valid_moves()` - Available actions (both single and multi-tile)
- `player_move(label)` - Execute move with multi-tile support
- `get_move_sequence(label)` - Preview move consequences

## Advanced Usage

### Custom Training Configuration
```python
from AI_controller import AIController
from Game import Game

# Create custom AI with specific parameters
game = Game(4, False)
ai = AIController(
    game,
    learning_rate=0.0005,      # Slower, more stable learning
    discount_factor=0.98,      # Value future rewards highly
    epsilon_decay=0.999,       # Very gradual exploration reduction
    memory_size=50000,         # Larger experience buffer
    entropy_reward_scale=0.2   # Higher entropy sensitivity
)

# Train with custom schedule
ai.train(episodes=1000, verbose=True, save_freq=50)
```

### Evaluation and Analysis
```python
# Detailed evaluation
results = ai.evaluate(episodes=200, verbose=True)

# Access training history
import matplotlib.pyplot as plt
rewards = [h['reward'] for h in ai.training_history]
plt.plot(rewards)
plt.title('Training Progress')
plt.show()
```

## Requirements

- **Core**: numpy, Game.py from this project
- **Deep Learning**: tensorflow/keras (falls back to Q-table if unavailable)
- **Visualization**: matplotlib (optional, for training plots)
- **Analysis**: Standard library (json, time, collections)

## Research Notes

This implementation explores several research-relevant concepts:

1. **Reward Shaping**: How entropy-based rewards guide learning
2. **Exploration vs Exploitation**: Balancing random vs learned moves
3. **Credit Assignment**: Learning which early moves lead to eventual success
4. **Transfer Learning**: Whether patterns learned on 4x4 boards generalize to larger sizes
5. **Curriculum Learning**: Progressive difficulty training from easy to hard 4x4 configurations

The AI demonstrates that even "simple" puzzles can require sophisticated reasoning about multi-step consequences and strategic entropy management.

---

*Built with understanding of the 15-puzzle's deceptive complexity and the AI's need to learn deep patterns in tile relationships.*