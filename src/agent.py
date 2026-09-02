from board import(
    board_from_fen,
    move_to_uci,
    # generate_legal_moves,  <-- Import your move generator function when built
)
from search import get_best_move

# =============================================================================
# WARMUP / PRE-COMPUTATION (Runs once on import within the 60s setup budget)
# =============================================================================
# Trigger Numba's JIT compilation here so your bot doesn't lose time 
# compiling functions during the first move's clock.
# _dummy_board = board_from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
# Call your JIT search / movegen once to force compilation ahead of time
# _ = generate_legal_moves(_dummy_board)

# Import time runs once per game, inside a 60 second budget, before your clock starts.
# Load weights and build tables out here, not inside get_move.


# def get_move(fen: str, time_left_ms: int) -> str:
#     """Return a legal move in UCI notation.
#     fen           the position to move in; your colour is the side to move
#     time_left_ms  your clock before this move, in milliseconds
#     returns       "e2e4", or "e7e8q" for a promotion
#     The process stays alive between your moves, so state you keep on a module or in a
#     closure survives to the next call. It does not survive to the next game.
#     print() is safe. Your stdout is redirected away from the protocol stream, discarded
#     during rated games and shown back to you in the validation log.
    
#     fen: the current position
#     time_left_ms: remaining clock time in milliseconds
#     returns: "e2e4", "g1f3", "e7e8q", etc.
#     """
#     board = chess.Board(fen)
#     # Everything from here down is yours to replace. baselines/greedy searches one ply,
#     # baselines/minimax searches two. Neither is strong. Reading them is the fastest way
#     # to see the shape of a search, and beating them is the first real milestone.
#     return random.choice(list(board.legal_moves)).uci()
#     # 1. Parse the incoming FEN string into your Numba board array
#     board = board_from_fen(fen)
#     # 2. Generate legal moves for the current position
#     # legal_moves = generate_legal_moves(board)
#     # -------------------------------------------------------------------------
#     # 3. Search / Decision Logic (Placeholder)
#     # Pass 'board' and 'time_left_ms' into your minimax/alphabeta or NN evaluation.
#     # chosen_move = search(board, legal_moves, time_left_ms)
#     # -------------------------------------------------------------------------
    
#     # Example placeholder return until movegen is plugged in:
#     # return move_to_uci(chosen_move)
#     pass
def get_move(fen: str, time_left_ms: int) -> str:
    """Entry point for the tournament harness."""
    board_array = board_from_fen(fen)
    best_encoded_move = get_best_move(board_array, time_left_ms)

    return move_to_uci(best_encoded_move)