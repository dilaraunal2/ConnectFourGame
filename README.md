# Connect Four AI — Minimax with Alpha-Beta Pruning

An interactive Connect Four game featuring an AI agent powered by the Minimax algorithm with Alpha-Beta Pruning. Developed as part of the Introduction to Artificial Intelligence course at Ankara Medipol University.

## 🎮 Demo
- Click any column to drop your piece
- AI responds automatically
- Live timer tracks game duration
- Winning four pieces highlighted in gold

## 📋 Features

### Game Modes
- **Human vs AI** — Play against the Minimax AI (click to play)
- **AI vs AI** — Watch two AI agents compete at different depths
- **Experiment Mode** — Compare Alpha-Beta vs Pure Minimax performance
- **Summary Mode** — View all game results from results.csv

### Technical Features
- Minimax algorithm with Alpha-Beta Pruning
- Pure Minimax (without pruning) for comparison
- Configurable search depth (default: 4)
- Node counter for performance measurement
- Automatic CSV result logging
- Live timer (runs in separate thread during AI computation)
- Winning cell highlight
- Vibrant color scheme (Pink = Player, Purple = AI)

## 🚀 How to Run

### Requirements
```
pip install numpy matplotlib
```

### Human vs AI
```
python connect_four.py
```

### Depth Comparison Experiment
```
python connect_four.py experiment
```
Generates `depth_comparison.png` comparing nodes expanded and computation time.

### AI vs AI Simulation
```
python connect_four.py aivai
```
Runs 5 games between depth-3 and depth-5 agents.

### Game Results Summary
```
python connect_four.py summary
```
Prints win/loss statistics from `results.csv`.

## 📊 Experimental Results

### Human vs AI (33 games, depth=4)
| Winner | Games | Percentage |
|--------|-------|------------|
| AI | 32 | 97% |
| Player | 1 | 3% |
| Draw | 0 | 0% |

### Alpha-Beta vs Pure Minimax
| Depth | AB Nodes | Pure Nodes | Speedup | AB Time | Pure Time |
|-------|----------|------------|---------|---------|-----------|
| 2 | ~15 | ~60 | 4x | 0.001s | 0.008s |
| 3 | ~75 | ~350 | 4.7x | 0.008s | 0.050s |
| 4 | ~220 | ~2400 | 11x | 0.040s | 0.360s |
| 5 | ~930 | ~2400 | 2.6x | 0.150s | 0.370s |

### AI vs AI (depth 3 vs depth 5, 5 games)
| Result | Count |
|--------|-------|
| Depth 5 wins | 5 |
| Depth 3 wins | 0 |
| Draws | 0 |

## 🗂️ Project Structure
```
connect-four-minimax-ai/
├── connect_four.py        # Main program
├── results.csv            # Game results log
├── depth_comparison.png   # Experiment chart
├── board_final.png        # Last game board
└── README.md
```

## 🧮 Algorithms

### Minimax
Minimax is a recursive adversarial search algorithm. It assumes the AI maximizes its score while the opponent minimizes it. The algorithm explores the full game tree up to a given depth and selects the move with the highest score.

### Alpha-Beta Pruning
Alpha-Beta Pruning is an optimization of Minimax that eliminates branches that cannot affect the final decision. It maintains two values — alpha (best for maximizer) and beta (best for minimizer) — and prunes branches where alpha ≥ beta. This dramatically reduces the number of nodes evaluated without changing the result.

### Heuristic Scoring
The board is evaluated using a scoring function that rewards:
- 4 in a row: +100 points
- 3 in a row with 1 open: +5 points
- 2 in a row with 2 open: +2 points
- Center column control: +3 points per piece
- Opponent's 3 in a row blocked: -4 points

## 🛠️ Technologies
- Python 3.x
- NumPy — board representation
- Matplotlib — visualization and interactive click events
- Threading — AI computation runs in separate thread
- CSV — result logging

## 📚 Academic Context
**Course:** CPE4221305 — Introduction to Artificial Intelligence  
**Semester:** Spring 2026  
**University:** Ankara Medipol University  
**Topic:** Adversarial Search & Game AI (Topic 8)  

