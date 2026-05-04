import numpy as np
import time
import csv
import os
import threading
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

# ─────────────────────────────────────────
# BOARD CONSTANTS
# ─────────────────────────────────────────
ROWS = 6
COLS = 7
EMPTY = 0
AI = 1
PLAYER = 2

RESULTS_FILE = 'results.csv'

COLOR_BG        = '#1A1A2E'
COLOR_BOARD     = '#16213E'
COLOR_EMPTY     = '#0F3460'
COLOR_PLAYER    = '#E94560'   # Pink = Player
COLOR_AI        = '#A855F7'   # Purple = AI
COLOR_HIGHLIGHT = '#F0C040'   # Gold = Winning cells

# ─────────────────────────────────────────
# PART 1: BOARD FUNCTIONS
# ─────────────────────────────────────────

def create_board():
    """Create and return an empty 6x7 board."""
    return np.zeros((ROWS, COLS), dtype=int)

def drop_piece(board, row, col, piece):
    """Place a piece on the board at the given position."""
    board[row][col] = piece

def is_valid_location(board, col):
    """Check if a column has at least one empty cell."""
    return board[0][col] == EMPTY

def get_next_open_row(board, col):
    """Return the lowest empty row in the given column."""
    for r in range(ROWS - 1, -1, -1):
        if board[r][col] == EMPTY:
            return r

def get_valid_locations(board):
    """Return a list of columns that are not full."""
    return [c for c in range(COLS) if is_valid_location(board, c)]

# ─────────────────────────────────────────
# PART 2: WIN / TERMINAL CONDITIONS
# ─────────────────────────────────────────

def winning_move(board, piece):
    """Return True if the given piece has four in a row (any direction)."""
    for c in range(COLS - 3):
        for r in range(ROWS):
            if all(board[r][c + i] == piece for i in range(4)):
                return True
    for c in range(COLS):
        for r in range(ROWS - 3):
            if all(board[r + i][c] == piece for i in range(4)):
                return True
    for c in range(COLS - 3):
        for r in range(ROWS - 3):
            if all(board[r + i][c + i] == piece for i in range(4)):
                return True
    for c in range(COLS - 3):
        for r in range(3, ROWS):
            if all(board[r - i][c + i] == piece for i in range(4)):
                return True
    return False

def get_winning_cells(board, piece):
    """Return list of (row, col) tuples that form the winning four."""
    for c in range(COLS - 3):
        for r in range(ROWS):
            if all(board[r][c + i] == piece for i in range(4)):
                return [(r, c + i) for i in range(4)]
    for c in range(COLS):
        for r in range(ROWS - 3):
            if all(board[r + i][c] == piece for i in range(4)):
                return [(r + i, c) for i in range(4)]
    for c in range(COLS - 3):
        for r in range(ROWS - 3):
            if all(board[r + i][c + i] == piece for i in range(4)):
                return [(r + i, c + i) for i in range(4)]
    for c in range(COLS - 3):
        for r in range(3, ROWS):
            if all(board[r - i][c + i] == piece for i in range(4)):
                return [(r - i, c + i) for i in range(4)]
    return []

def is_terminal_node(board):
    """Return True if the game is over (win or draw)."""
    return (winning_move(board, AI) or
            winning_move(board, PLAYER) or
            len(get_valid_locations(board)) == 0)

# ─────────────────────────────────────────
# PART 3: SCORING & MINIMAX WITH ALPHA-BETA
# ─────────────────────────────────────────

def score_window(window, piece):
    """Score a window of 4 cells for the given piece."""
    opp = PLAYER if piece == AI else AI
    score = 0
    if window.count(piece) == 4:
        score += 100
    elif window.count(piece) == 3 and window.count(EMPTY) == 1:
        score += 5
    elif window.count(piece) == 2 and window.count(EMPTY) == 2:
        score += 2
    if window.count(opp) == 3 and window.count(EMPTY) == 1:
        score -= 4
    return score

def score_position(board, piece):
    """Evaluate the full board position and return a heuristic score."""
    score = 0
    center = [int(i) for i in list(board[:, COLS // 2])]
    score += center.count(piece) * 3
    for r in range(ROWS):
        row = [int(i) for i in list(board[r, :])]
        for c in range(COLS - 3):
            score += score_window(row[c:c + 4], piece)
    for c in range(COLS):
        col = [int(i) for i in list(board[:, c])]
        for r in range(ROWS - 3):
            score += score_window(col[r:r + 4], piece)
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            score += score_window([board[r + i][c + i] for i in range(4)], piece)
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            score += score_window([board[r + 3 - i][c + i] for i in range(4)], piece)
    return score

def minimax(board, depth, alpha, beta, maximizing, node_counter=None):
    """
    Minimax algorithm with Alpha-Beta pruning.
    Returns (best_column, score).
    """
    if node_counter is not None:
        node_counter[0] += 1

    valid_locations = get_valid_locations(board)
    terminal = is_terminal_node(board)

    if depth == 0 or terminal:
        if terminal:
            if winning_move(board, AI):       return (None,  1000000)
            elif winning_move(board, PLAYER): return (None, -1000000)
            else:                             return (None, 0)
        return (None, score_position(board, AI))

    if maximizing:
        value = -float('inf')
        best_col = valid_locations[0]
        for col in valid_locations:
            row = get_next_open_row(board, col)
            temp = board.copy()
            drop_piece(temp, row, col, AI)
            new_score = minimax(temp, depth - 1, alpha, beta, False, node_counter)[1]
            if new_score > value:
                value = new_score
                best_col = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return best_col, value
    else:
        value = float('inf')
        best_col = valid_locations[0]
        for col in valid_locations:
            row = get_next_open_row(board, col)
            temp = board.copy()
            drop_piece(temp, row, col, PLAYER)
            new_score = minimax(temp, depth - 1, alpha, beta, True, node_counter)[1]
            if new_score < value:
                value = new_score
                best_col = col
            beta = min(beta, value)
            if alpha >= beta:
                break
        return best_col, value

def minimax_no_pruning(board, depth, maximizing, node_counter=None):
    """Pure Minimax WITHOUT Alpha-Beta pruning — used for comparison."""
    if node_counter is not None:
        node_counter[0] += 1

    valid_locations = get_valid_locations(board)
    terminal = is_terminal_node(board)

    if depth == 0 or terminal:
        if terminal:
            if winning_move(board, AI):       return (None,  1000000)
            elif winning_move(board, PLAYER): return (None, -1000000)
            else:                             return (None, 0)
        return (None, score_position(board, AI))

    if maximizing:
        value = -float('inf')
        best_col = valid_locations[0]
        for col in valid_locations:
            row = get_next_open_row(board, col)
            temp = board.copy()
            drop_piece(temp, row, col, AI)
            new_score = minimax_no_pruning(temp, depth - 1, False, node_counter)[1]
            if new_score > value:
                value = new_score
                best_col = col
        return best_col, value
    else:
        value = float('inf')
        best_col = valid_locations[0]
        for col in valid_locations:
            row = get_next_open_row(board, col)
            temp = board.copy()
            drop_piece(temp, row, col, PLAYER)
            new_score = minimax_no_pruning(temp, depth - 1, True, node_counter)[1]
            if new_score < value:
                value = new_score
                best_col = col
        return best_col, value

# ─────────────────────────────────────────
# PART 4: RESULT LOGGING
# ─────────────────────────────────────────

def get_game_number():
    """Return the next game number based on existing CSV rows."""
    if not os.path.isfile(RESULTS_FILE):
        return 1
    with open(RESULTS_FILE, 'r') as f:
        return sum(1 for line in f)

def save_result(winner, total_time, total_moves, depth, total_nodes):
    """Append one game result to results.csv."""
    game_num = get_game_number()
    file_exists = os.path.isfile(RESULTS_FILE)
    with open(RESULTS_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Game', 'Winner', 'Time(s)', 'Moves', 'Depth', 'AI_Nodes'])
        writer.writerow([game_num, winner, f"{total_time:.2f}",
                         total_moves, depth, total_nodes])
    print(f"\nResult saved → {RESULTS_FILE}  "
          f"(Game {game_num} | Winner: {winner} | "
          f"Time: {total_time:.2f}s | Moves: {total_moves} | Nodes: {total_nodes:,})")

def print_summary():
    """Print a summary table of all saved game results."""
    if not os.path.isfile(RESULTS_FILE):
        print("No results file found.")
        return
    print("\n" + "="*60)
    print("  GAME RESULTS SUMMARY")
    print("="*60)
    with open(RESULTS_FILE, 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                print(f"  {'  '.join(f'{v:>10}' for v in row)}")
                print("  " + "-"*55)
            else:
                print(f"  {'  '.join(f'{v:>10}' for v in row)}")
    wins = {'Player': 0, 'AI': 0, 'Draw': 0}
    with open(RESULTS_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            w = row['Winner']
            if w in wins:
                wins[w] += 1
    total = sum(wins.values())
    print(f"\n  Player wins : {wins['Player']} / {total}")
    print(f"  AI wins     : {wins['AI']} / {total}")
    print(f"  Draws       : {wins['Draw']} / {total}")
    print("="*60)

# ─────────────────────────────────────────
# PART 5: VISUALIZATION
# ─────────────────────────────────────────

def draw_board(ax, board, status="", winning_cells=None, elapsed=0.0):
    """Redraw the board with vivid colors, timer, and optional winning highlight."""
    ax.clear()
    ax.set_facecolor(COLOR_BOARD)

    for r in range(ROWS):
        for c in range(COLS):
            piece = board[ROWS - 1 - r][c]
            bg = patches.Circle((c, r), 0.48, color=COLOR_BG, zorder=1)
            ax.add_patch(bg)
            color = COLOR_AI if piece == AI else \
                    COLOR_PLAYER if piece == PLAYER else COLOR_EMPTY
            circle = patches.Circle((c, r), 0.40, color=color, zorder=2)
            ax.add_patch(circle)
            if winning_cells:
                board_r = ROWS - 1 - r
                if (board_r, c) in winning_cells:
                    glow = patches.Circle((c, r), 0.44,
                                          color=COLOR_HIGHLIGHT,
                                          fill=False, linewidth=3, zorder=3)
                    ax.add_patch(glow)

    ax.set_xlim(-0.5, COLS - 0.5)
    ax.set_ylim(-0.7, ROWS + 0.3)
    ax.set_xticks(range(COLS))
    ax.set_xticklabels([str(i) for i in range(COLS)],
                       color='white', fontsize=13, fontweight='bold')
    ax.set_yticks([])
    ax.set_facecolor(COLOR_BOARD)
    ax.text(-0.45, ROWS + 0.1, f"⏱  {elapsed:.1f}s",
            color='#A0A0C0', fontsize=11, va='top')
    ax.text(COLS - 0.5, ROWS + 0.1,
            "● You (Pink)   ● AI (Purple)",
            color='#C0C0E0', fontsize=9, ha='right', va='top')
    ax.set_title(status, color='white', fontsize=13, pad=6, fontweight='bold')
    ax.figure.canvas.draw()

# ─────────────────────────────────────────
# PART 6: EXPERIMENT — DEPTH COMPARISON
# ─────────────────────────────────────────

def run_depth_comparison():
    """Compare Alpha-Beta vs Pure Minimax across depths. Saves chart."""
    print("\n" + "="*60)
    print("  EXPERIMENT: Minimax vs Alpha-Beta — Depth Comparison")
    print("="*60)

    test_board = create_board()
    for r, c, p in [(5,3,PLAYER),(5,4,AI),(5,2,PLAYER),(5,5,AI),
                    (4,3,PLAYER),(4,4,AI),(5,1,PLAYER)]:
        test_board[r][c] = p

    depths = [2, 3, 4, 5]
    ab_nodes, ab_times, pure_nodes, pure_times = [], [], [], []

    header = f"{'Depth':>6} | {'AB Nodes':>10} | {'AB Time':>9} | {'Pure Nodes':>12} | {'Pure Time':>10} | {'Speedup':>8}"
    print(header)
    print("-" * len(header))

    for d in depths:
        nc_ab = [0]
        t0 = time.time()
        minimax(test_board.copy(), d, -float('inf'), float('inf'), True, nc_ab)
        t_ab = time.time() - t0
        ab_nodes.append(nc_ab[0])
        ab_times.append(t_ab)

        nc_pure = [0]
        t0 = time.time()
        minimax_no_pruning(test_board.copy(), min(d, 4), True, nc_pure)
        t_pure = time.time() - t0
        pure_nodes.append(nc_pure[0])
        pure_times.append(t_pure)

        speedup = nc_pure[0] / nc_ab[0] if nc_ab[0] > 0 else 0
        print(f"{d:>6} | {nc_ab[0]:>10,} | {t_ab:>8.3f}s | "
              f"{nc_pure[0]:>12,} | {t_pure:>9.3f}s | {speedup:>7.1f}x")

    print("="*60)

    x = np.arange(len(depths))
    width = 0.35
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(COLOR_BG)

    for ax in (ax1, ax2):
        ax.set_facecolor(COLOR_BOARD)
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444466')

    ax1.bar(x - width/2, ab_nodes,   width, label='Alpha-Beta',   color=COLOR_AI,     alpha=0.9)
    ax1.bar(x + width/2, pure_nodes, width, label='Pure Minimax', color=COLOR_PLAYER, alpha=0.9)
    ax1.set_xlabel('Search Depth'); ax1.set_ylabel('Nodes Expanded')
    ax1.set_title('Nodes Expanded: Alpha-Beta vs Pure Minimax')
    ax1.set_xticks(x); ax1.set_xticklabels(depths)
    ax1.legend(facecolor=COLOR_BG, labelcolor='white')

    ax2.bar(x - width/2, ab_times,   width, label='Alpha-Beta',   color=COLOR_AI,     alpha=0.9)
    ax2.bar(x + width/2, pure_times, width, label='Pure Minimax', color=COLOR_PLAYER, alpha=0.9)
    ax2.set_xlabel('Search Depth'); ax2.set_ylabel('Time (seconds)')
    ax2.set_title('Computation Time: Alpha-Beta vs Pure Minimax')
    ax2.set_xticks(x); ax2.set_xticklabels(depths)
    ax2.legend(facecolor=COLOR_BG, labelcolor='white')

    plt.tight_layout()
    plt.savefig('depth_comparison.png', dpi=150, facecolor=COLOR_BG)
    print("Chart saved → depth_comparison.png")
    plt.show()

# ─────────────────────────────────────────
# PART 7: AI vs AI MODE
# ─────────────────────────────────────────

def ai_vs_ai(depth1=3, depth2=5, games=5):
    """Run AI vs AI simulation and print win statistics."""
    print(f"\n{'='*50}")
    print(f"  AI vs AI  —  depth {depth1} vs depth {depth2}  ({games} games)")
    print(f"{'='*50}")

    wins = {1: 0, 2: 0, 'draw': 0}

    for g in range(games):
        board = create_board()
        turn = 1
        game_over = False

        while not game_over:
            depth = depth1 if turn == 1 else depth2
            piece = AI if turn == 1 else PLAYER
            maximizing = (turn == 1)
            col, _ = minimax(board, depth, -float('inf'), float('inf'), maximizing)
            row = get_next_open_row(board, col)
            drop_piece(board, row, col, piece)

            if winning_move(board, piece):
                wins[turn] += 1
                print(f"  Game {g+1}: AI-{turn} wins (depth {depth})")
                game_over = True
            elif len(get_valid_locations(board)) == 0:
                wins['draw'] += 1
                print(f"  Game {g+1}: Draw")
                game_over = True
            else:
                turn = 2 if turn == 1 else 1

    print(f"\n  Results after {games} games:")
    print(f"  AI-1 (depth {depth1}) wins : {wins[1]}")
    print(f"  AI-2 (depth {depth2}) wins : {wins[2]}")
    print(f"  Draws                : {wins['draw']}")
    print(f"{'='*50}\n")
    return wins

# ─────────────────────────────────────────
# PART 8: MAIN GAME — CLICK TO PLAY
# ─────────────────────────────────────────

def play_game(depth=4):
    """
    Interactive Connect Four. Click a column to drop your piece.
    Pink = Player  |  Purple = AI
    Timer runs continuously — AI runs in a separate thread so timer never freezes.
    Results are automatically saved to results.csv after each game.
    """
    board = create_board()
    game_over = False
    turn = PLAYER
    start_time = time.time()
    total_moves = [0]
    total_nodes = [0]
    ai_thinking = [False]   # Lock to prevent double-clicks during AI turn

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(COLOR_BG)
    plt.subplots_adjust(bottom=0.06, top=0.92)

    draw_board(ax, board, status="Your turn — click a column!", elapsed=0.0)

    # ── Live timer — runs independently of AI computation ──
    def update_timer(frame):
        if not game_over:
            elapsed = time.time() - start_time
            for txt in ax.texts:
                if txt.get_text().startswith("⏱"):
                    txt.set_text(f"⏱  {elapsed:.1f}s")
                    fig.canvas.draw_idle()
                    break

    ani = animation.FuncAnimation(fig, update_timer, interval=100,
                                  cache_frame_data=False)

    def end_game(winner, wc=None):
        nonlocal game_over
        elapsed = time.time() - start_time
        status_map = {
            'Player': "You win! Congratulations! 🎉",
            'AI':     "AI wins!",
            'Draw':   "It's a draw!"
        }
        draw_board(ax, board, status=status_map[winner],
                   winning_cells=wc, elapsed=elapsed)
        plt.savefig("board_final.png", dpi=150, facecolor=COLOR_BG)
        save_result(winner, elapsed, total_moves[0], depth, total_nodes[0])
        game_over = True

    def run_ai():
        """AI computation runs in a separate thread so timer keeps ticking."""
        nonlocal turn

        nc = [0]
        t0 = time.time()
        ai_col, score = minimax(board, depth, -float('inf'), float('inf'), True, nc)
        t_ai = time.time() - t0
        total_nodes[0] += nc[0]
        total_moves[0] += 1
        elapsed = time.time() - start_time

        print(f"AI → col {ai_col} | {t_ai:.3f}s | score={score} | nodes={nc[0]:,}")

        ai_row = get_next_open_row(board, ai_col)
        drop_piece(board, ai_row, ai_col, AI)

        ai_thinking[0] = False

        if winning_move(board, AI):
            end_game('AI', get_winning_cells(board, AI))
            return
        if len(get_valid_locations(board)) == 0:
            end_game('Draw')
            return

        turn = PLAYER
        draw_board(ax, board,
                   status="Your turn — click a column!",
                   elapsed=elapsed)

    def on_click(event):
        nonlocal turn

        if game_over or event.inaxes != ax:
            return
        if turn != PLAYER or ai_thinking[0]:
            return

        col = int(round(event.xdata)) if event.xdata is not None else -1
        if col < 0 or col >= COLS:
            return
        if not is_valid_location(board, col):
            elapsed = time.time() - start_time
            draw_board(ax, board,
                       status="Column is full! Choose another.",
                       elapsed=elapsed)
            return

        # Player move
        row = get_next_open_row(board, col)
        drop_piece(board, row, col, PLAYER)
        total_moves[0] += 1
        elapsed = time.time() - start_time
        draw_board(ax, board, status="Your move...", elapsed=elapsed)

        if winning_move(board, PLAYER):
            end_game('Player', get_winning_cells(board, PLAYER))
            return
        if len(get_valid_locations(board)) == 0:
            end_game('Draw')
            return

        # Launch AI in separate thread
        turn = AI
        ai_thinking[0] = True
        elapsed = time.time() - start_time
        draw_board(ax, board,
                   status=f"AI is thinking (depth={depth})...",
                   elapsed=elapsed)
        threading.Thread(target=run_ai, daemon=True).start()

    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()

# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "experiment":
            run_depth_comparison()
        elif mode == "aivai":
            ai_vs_ai(depth1=3, depth2=5, games=5)
        elif mode == "summary":
            print_summary()
        else:
            print("Usage: python connect_four.py [experiment | aivai | summary]")
    else:
        play_game(depth=4)