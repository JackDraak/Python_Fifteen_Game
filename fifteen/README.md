# Fifteen Puzzle AI Educational Project

An educational implementation of the classic fifteen puzzle with two distinct AI approaches for teaching pathfinding, reinforcement learning, and optimal search algorithms.

## 🎯 Project Purpose

This project is designed for **educational purposes** to help students understand:
- **Reinforcement Learning**: Trial-and-error learning through Q-Networks
- **Optimal Search**: Mathematical guarantees using IDA* and heuristics
- **Algorithm Trade-offs**: Performance vs learning, exploration vs exploitation
- **AI Methodology**: When to use learning vs analytical approaches

## 🚀 Quick Start

### Requirements
- Python 3.8+
- NumPy (required for all functionality)
- TensorFlow 2.x (optional, for RL AI - falls back to Q-table)
- Matplotlib (optional, for statistics visualization)
- pygame (optional, for GUI interface)
- tkinter (optional, alternative GUI interface)

**Note**: pygame and tkinter are NOT required for console operation or AI models

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd fifteen

# Install required dependencies
pip install numpy

# Install optional dependencies (choose what you need)
pip install tensorflow matplotlib  # For RL AI and statistics
pip install pygame  # For GUI interface

# Run the educational demo
python integrated_demo.py
```

## 🧩 The Fifteen Puzzle

The fifteen puzzle is a sliding puzzle consisting of numbered tiles in a 4×4 grid with one missing space. The goal is to arrange tiles in numerical order by sliding them into the empty space.

```
Initial State:        Goal State:
[ 5][ 1][ 2][ 4]     [ 1][ 2][ 3][ 4]
[ 9][ 6][ 3][ 8]     [ 5][ 6][ 7][ 8]
[13][10][ 7][11]     [ 9][10][11][12]
[14][15][12][  ]     [13][14][15][  ]
```

## 🤖 AI Approaches

### 1. Reinforcement Learning AI (`AI_controller.py`)
**Educational Focus**: Learning through experience

- **Algorithm**: Deep Q-Network (DQN) with experience replay
- **Learning**: Starts random, gradually learns better strategies
- **Key Features**:
  - Epsilon-greedy exploration with decay
  - Anti-loop measures prevent infinite repetition
  - Stagnation detection forces exploration when stuck
  - Detailed statistics tracking for analysis

**When to use**: Teaching machine learning, neural networks, trial-and-error learning

### 2. Optimal Search AI (`OptimalAI.py`)
**Educational Focus**: Mathematical guarantees

- **Algorithm**: IDA* (Iterative Deepening A*) search
- **Heuristics**: Manhattan Distance + Linear Conflict
- **Key Features**:
  - Guarantees shortest possible solution
  - Memory-efficient search
  - Fast execution for reasonable puzzles
  - Solvability verification

**When to use**: Teaching graph search, heuristics, optimal algorithms

## 📚 Educational Demonstrations

### Interactive Comparison Demo
```bash
python integrated_demo.py
```

Features:
- Side-by-side comparison on identical puzzles
- Performance analysis across difficulty levels
- Educational insights and methodology explanations
- Session data collection and analysis

### Individual AI Demos
```bash
# Test Reinforcement Learning AI
python AI_controller.py

# Test Optimal Search AI
python OptimalAI.py

# Enhanced RL demonstration
python demo_ai.py
```

### Interactive Game Controllers
**Experience the puzzle yourself first!**

```bash
# Play in terminal with simple text interface
python console_controller.py

# Play with colorful terminal interface (requires ncurses)
python ncurses_controller.py

# Play with graphical interface (requires tkinter)
python GUI_controller.py
```

**Educational Value**: Understanding the puzzle's challenge level and solution strategies through hands-on play provides crucial context before studying AI approaches. Students often gain better insights into why certain AI techniques work by first experiencing the puzzle's difficulty themselves.

## 📊 Performance Comparison

| Approach | Success Rate | Speed | Solution Quality | Learning Curve |
|----------|-------------|--------|------------------|----------------|
| **RL AI** | Variable (0-80%) | Moderate | Sub-optimal | Shows improvement |
| **Optimal AI** | 100%* | Very Fast | Guaranteed optimal | N/A (no learning) |

*For puzzles within complexity limits (≤50 optimal moves)

## 🔧 Advanced Usage

### Statistics Tracking
```python
from statistics_tracker import StatisticsTracker
from AI_controller import AIController

# Create tracker and attach to AI
tracker = StatisticsTracker()
ai = AIController(game)
ai.attach_statistics_tracker(tracker)

# Train and analyze
for episode in range(100):
    ai.play_episode(training=True)

# Generate learning report
report = tracker.generate_learning_report()
print(report)
```

### Custom Puzzles
```python
from Game import Game

# Create reproducible puzzle
game = Game(4, True, seed=42)  # 4x4, shuffled, deterministic seed

# Create specific difficulty
game = Game(4, False)  # Start solved
game.shuffle(15)  # Apply exactly 15 shuffle moves
```

### Performance Testing
```python
from integrated_demo import IntegratedDemo

demo = IntegratedDemo()
results = demo.run_performance_suite(
    difficulties=[5, 10, 20, 30],
    trials_per_difficulty=3
)
```

## 🎓 Educational Concepts Demonstrated

### Reinforcement Learning Concepts
- **Exploration vs Exploitation**: Epsilon-greedy policy balancing
- **Experience Replay**: Learning from past experiences
- **Target Networks**: Stable Q-value estimation
- **Reward Shaping**: Encouraging desired behaviors
- **Overfitting Prevention**: Regularization through exploration

### Search Algorithm Concepts
- **Heuristic Functions**: Manhattan distance and linear conflict
- **Admissibility**: Never overestimating remaining cost
- **Graph Search**: State space exploration
- **Memory Efficiency**: IDA* vs A* trade-offs
- **Optimality Guarantees**: Mathematical proofs vs empirical results

### General AI Concepts
- **Problem Formulation**: State space, actions, goals
- **Algorithm Selection**: When to use learning vs analytical approaches
- **Performance Metrics**: Success rate, efficiency, optimality
- **Trade-off Analysis**: Speed vs accuracy, memory vs time

## 📁 Project Structure

```
/fifteen/
├── Game.py                 # Core puzzle implementation
├── console_controller.py   # Text-based interactive game
├── ncurses_controller.py   # Colorful terminal game interface
├── GUI_controller.py       # Graphical tkinter game interface
├── AI_controller.py        # Reinforcement Learning AI
├── OptimalAI.py           # IDA* Optimal Search AI
├── integrated_demo.py     # Educational comparison demo
├── statistics_tracker.py  # Performance analytics system
├── demo_ai.py             # Enhanced RL demonstration
├── docs/
│   ├── README.md          # This file
│   ├── API.md             # Technical API documentation
│   └── STUDENT_GUIDE.md   # Usage guide for students
└── .gitignore             # Git ignore rules
```

## 🐛 Troubleshooting

### Common Issues

**RL AI seems "stupid" initially**
- ✅ This is expected! It starts with random moves and learns over time
- Try running training episodes to see improvement

**TensorFlow warnings/errors**
- ✅ Project falls back to Q-table implementation automatically
- Install TensorFlow 2.x for full neural network features

**Optimal AI timeouts on hard puzzles**
- ✅ This is expected for very difficult puzzles (50+ optimal moves)
- Use easier puzzles (10-30 shuffles) for reliable demonstrations

**Infinite loops or repetitive moves**
- ✅ Anti-loop measures should prevent this automatically
- If still occurring, check `max_consecutive_moves` and `force_exploration_threshold` settings

### Performance Guidelines

**For Educational Demonstrations**:
- Easy puzzles: 5-15 shuffles
- Medium puzzles: 15-25 shuffles
- Hard puzzles: 25-35 shuffles
- Expert puzzles: 35+ shuffles (may timeout)

**For RL Training**:
- Start with easy puzzles for faster learning
- Gradually increase difficulty as performance improves
- Use statistics tracker to monitor learning progress

## 🤝 Contributing

This is an educational project. Contributions should prioritize:
1. **Educational Value**: Clear demonstrations of AI concepts
2. **Robustness**: No crashes or infinite loops during student use
3. **Documentation**: Explaining the "why" behind implementation choices
4. **Accessibility**: Working across different system configurations

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

This project is designed for educational purposes to teach AI and machine learning concepts through hands-on experimentation with classic puzzles.

---

**For Students**: Start with `integrated_demo.py` to see both approaches in action!

**For Instructors**: Use the statistics tracking system to analyze student experiments and demonstrate learning progression.

**For Developers**: See `docs/API.md` for technical implementation details.