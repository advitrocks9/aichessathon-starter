import numpy as np
from numba import njit, uint64, int8, boolean

# =============================================================================
# CONSTANTS
# =============================================================================

# Piece indices (used as array indices)
# White: 0-5, Black: 6-11
W_PAWN, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING = 0, 1, 2, 3, 4, 5
B_PAWN, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN, B_KING = 6, 7, 8, 9, 10, 11

# Castling rights bitmask
CASTLE_WK = np.uint8(1)   # White kingside
CASTLE_WQ = np.uint8(2)   # White queenside
CASTLE_BK = np.uint8(4)   # Black kingside
CASTLE_BQ = np.uint8(8)   # Black queenside

# File/Rank masks (precomputed)
FILE_A = np.uint64(0x0101010101010101)
FILE_H = np.uint64(0x8080808080808080)
FILE_AB = np.uint64(0x0303030303030303)
FILE_GH = np.uint64(0xC0C0C0C0C0C0C0C0)
RANK_1  = np.uint64(0x00000000000000FF)
RANK_2  = np.uint64(0x000000000000FF00)
RANK_7  = np.uint64(0x00FF000000000000)
RANK_8  = np.uint64(0xFF00000000000000)
FULL    = np.uint64(0xFFFFFFFFFFFFFFFF)

# =============================================================================
# BOARD STATE (structured as a plain numpy array for Numba compatibility)
# Layout: float64 array of length 24
#   [0:12]  -> bitboards for each piece type (reinterpreted as uint64)
#   [12]    -> occupancy WHITE  (uint64)
#   [13]    -> occupancy BLACK  (uint64)
#   [14]    -> occupancy ALL    (uint64)
#   [15]    -> en passant square (int, 0-63, or 64 = none)
#   [16]    -> castling rights   (uint8 bitmask packed into float)
#   [17]    -> side to move      (0 = white, 1 = black)
#   [18]    -> halfmove clock
#   [19]    -> fullmove number
# =============================================================================

# We store everything as uint64 views for Numba
BOARD_SIZE = 20

def make_board() -> np.ndarray:
    """Allocate an empty board state array."""
    return np.zeros(BOARD_SIZE, dtype=np.uint64)

# =============================================================================
# BIT UTILITIES (Numba JIT)
# =============================================================================

@njit(cache=True)
def set_bit(bb: uint64, sq: int) -> uint64:
    return bb | (uint64(1) << uint64(sq))

@njit(cache=True)
def clear_bit(bb: uint64, sq: int) -> uint64:
    return bb & ~(uint64(1) << uint64(sq))

@njit(cache=True)
def get_bit(bb: uint64, sq: int) -> boolean:
    return (bb >> uint64(sq)) & uint64(1)

# @njit(cache=True)
# def lsb(bb: uint64) -> int:
#     """Index of the least significant bit (lowest set square)."""
#     return int(np.uint64(bb & (~bb + uint64(1))).bit_length() - 1) \
#         if bb != uint64(0) else 64
@njit(cache=True)
def lsb(bb: uint64) -> int:
    """Index of the least significant bit (0-63, or 64 if empty)."""
    return np.cttz(bb) if bb != uint64(0) else 64

@njit(cache=True)
def pop_lsb(bb: uint64):
    """Returns (lsb_index, bb_with_lsb_cleared)."""
    sq = lsb(bb)
    return sq, bb & (bb - uint64(1))

@njit(cache=True)
def popcount(bb: uint64) -> int:
    """Count number of set bits (Brian Kernighan)."""
    count = 0
    while bb:
        bb = bb & (bb - uint64(1))
        count += 1
    return count

# =============================================================================
# BOARD ACCESSORS (Numba JIT)
# =============================================================================

@njit(cache=True)
def get_pieces(board: uint64[:], piece: int) -> uint64:
    return board[piece]

@njit(cache=True)
def get_white_occ(board: uint64[:]) -> uint64:
    return board[12]

@njit(cache=True)
def get_black_occ(board: uint64[:]) -> uint64:
    return board[13]

@njit(cache=True)
def get_all_occ(board: uint64[:]) -> uint64:
    return board[14]

@njit(cache=True)
def get_ep_square(board: uint64[:]) -> int:
    return int(board[15])   # 64 = no en passant

@njit(cache=True)
def get_castling(board: uint64[:]) -> int:
    return int(board[16])

@njit(cache=True)
def get_side(board: uint64[:]) -> int:
    return int(board[17])   # 0 = white, 1 = black

@njit(cache=True)
def get_halfmove(board: uint64[:]) -> int:
    return int(board[18])

@njit(cache=True)
def get_fullmove(board: uint64[:]) -> int:
    return int(board[19])

# =============================================================================
# BOARD MUTATORS (Numba JIT)
# =============================================================================

# @njit(cache=True)
# def place_piece(board: uint64[:], piece: int, sq: int):
#     """Place a piece on a square and update occupancy."""
#     board[piece] = set_bit(board[piece], sq)
#     if piece < 6:   # white
#         board[12] = set_bit(board[12], sq)
#     else:           # black
#         board[13] = set_bit(board[13], sq)
#     board[14] = board[12] | board[13]

# @njit(cache=True)
# def remove_piece(board: uint64[:], piece: int, sq: int):
#     """Remove a piece from a square and update occupancy."""
#     board[piece] = clear_bit(board[piece], sq)
#     if piece < 6:
#         board[12] = clear_bit(board[12], sq)
#     else:
#         board[13] = clear_bit(board[13], sq)
#     board[14] = board[12] | board[13]

@njit(cache=True)
def place_piece(board: uint64[:], piece: int, sq: int):
    board[piece] = set_bit(board[piece], sq)
    occ_idx = 12 if piece < 6 else 13
    board[occ_idx] = set_bit(board[occ_idx], sq)
    board[14] = set_bit(board[14], sq)

@njit(cache=True)
def remove_piece(board: uint64[:], piece: int, sq: int):
    board[piece] = clear_bit(board[piece], sq)
    occ_idx = 12 if piece < 6 else 13
    board[occ_idx] = clear_bit(board[occ_idx], sq)
    board[14] = clear_bit(board[14], sq)

# @njit(cache=True)
# def piece_on(board: uint64[:], sq: int) -> int:
#     """Return piece index (0-11) on square, or -1 if empty."""
#     for p in range(12):
#         if get_bit(board[p], sq):
#             return p
#     return -1

@njit(cache=True)
def piece_on(board: uint64[:], sq: int) -> int:
    """Return piece index (0-11) on square, or -1 if empty."""
    if not get_bit(board[14], sq):
        return -1
    for p in range(12):
        if get_bit(board[p], sq):
            return p
    return -1

# =============================================================================
# FEN PARSER (pure Python — called once at startup, NOT in hot loop)
# =============================================================================

# Map FEN characters to piece indices
_FEN_PIECE_MAP = {
    'P': W_PAWN,   'N': W_KNIGHT, 'B': W_BISHOP,
    'R': W_ROOK,   'Q': W_QUEEN,  'K': W_KING,
    'p': B_PAWN,   'n': B_KNIGHT, 'b': B_BISHOP,
    'r': B_ROOK,   'q': B_QUEEN,  'k': B_KING,
}

def board_from_fen(fen: str) -> np.ndarray:
    """
    Parse a FEN string into a board state array.
    This is pure Python — only called once per position hand-off.
    """
    board = make_board()
    parts = fen.strip().split()

    # --- Piece placement ---
    rank = 7
    file = 0
    for ch in parts[0]:
        if ch == '/':
            rank -= 1
            file = 0
        elif ch.isdigit():
            file += int(ch)
        else:
            sq = rank * 8 + file
            piece = _FEN_PIECE_MAP[ch]
            place_piece(board, piece, sq)
            file += 1

    # --- Side to move ---
    board[17] = np.uint64(0 if parts[1] == 'w' else 1)

    # --- Castling rights ---
    castling = np.uint64(0)
    if parts[2] != '-':
        if 'K' in parts[2]: castling |= np.uint64(CASTLE_WK)
        if 'Q' in parts[2]: castling |= np.uint64(CASTLE_WQ)
        if 'k' in parts[2]: castling |= np.uint64(CASTLE_BK)
        if 'q' in parts[2]: castling |= np.uint64(CASTLE_BQ)
    board[16] = castling

    # --- En passant ---
    if parts[3] != '-':
        ep_file = ord(parts[3][0]) - ord('a')
        ep_rank = int(parts[3][1]) - 1
        board[15] = np.uint64(ep_rank * 8 + ep_file)
    else:
        board[15] = np.uint64(64)   # sentinel = no EP

    # --- Clocks ---
    board[18] = np.uint64(int(parts[4])) if len(parts) > 4 else np.uint64(0)
    board[19] = np.uint64(int(parts[5])) if len(parts) > 5 else np.uint64(1)

    return board

# =============================================================================
# FEN SERIALISER (for debugging / UCI output)
# =============================================================================

_IDX_TO_FEN = ['P','N','B','R','Q','K','p','n','b','r','q','k']

def board_to_fen(board: np.ndarray) -> str:
    rows = []
    for rank in range(7, -1, -1):
        empty = 0
        row = ''
        for file in range(8):
            sq = rank * 8 + file
            p = piece_on(board, sq)
            if p == -1:
                empty += 1
            else:
                if empty:
                    row += str(empty)
                    empty = 0
                row += _IDX_TO_FEN[p]
        if empty:
            row += str(empty)
        rows.append(row)

    side   = 'w' if board[17] == 0 else 'b'
    castle = ''
    c = int(board[16])
    if c & CASTLE_WK: castle += 'K'
    if c & CASTLE_WQ: castle += 'Q'
    if c & CASTLE_BK: castle += 'k'
    if c & CASTLE_BQ: castle += 'q'
    if not castle:    castle  = '-'

    ep = int(board[15])
    ep_str = '-' if ep == 64 else (chr(ord('a') + ep % 8) + str(ep // 8 + 1))

    return f"{'/'.join(rows)} {side} {castle} {ep_str} {int(board[18])} {int(board[19])}"

# =============================================================================
# BOARD COPY (Numba JIT — used heavily in search)
# =============================================================================

@njit(cache=True)
def copy_board(board: uint64[:]) -> uint64[:]:
    return board.copy()

# =============================================================================
# PRETTY PRINT (debug only)
# =============================================================================

_SYMBOLS = {
    W_PAWN:'♙', W_KNIGHT:'♘', W_BISHOP:'♗',
    W_ROOK:'♖', W_QUEEN:'♕',  W_KING:'♔',
    B_PAWN:'♟', B_KNIGHT:'♞', B_BISHOP:'♝',
    B_ROOK:'♜', B_QUEEN:'♛',  B_KING:'♚',
}

def print_board(board: np.ndarray):
    print("  a b c d e f g h")
    for rank in range(7, -1, -1):
        row = f"{rank+1} "
        for file in range(8):
            sq = rank * 8 + file
            p = piece_on(board, sq)
            row += (_SYMBOLS[p] if p != -1 else '·') + ' '
        print(row)
    print(f"  Side: {'White' if board[17]==0 else 'Black'} | "
          f"EP: {int(board[15])} | Castling: {int(board[16]):04b}")

# MORE HELPERS


# Convert 0-63 square index to UCI notation (e.g., 12 -> "e2")
def sq_to_uci(sq: int) -> str:
    file = chr(ord('a') + (sq % 8))
    rank = str((sq // 8) + 1)
    return f"{file}{rank}"

# Convert UCI move notation to square index (e.g., "e2" -> 12)
def uci_to_sq(uci: str) -> int:
    file = ord(uci[0]) - ord('a')
    rank = int(uci[1]) - 1
    return rank * 8 + file

# Encode a move into a single 16-bit int (efficient for Numba search)
# Format: [4 bits promotion piece] [6 bits to_sq] [6 bits from_sq]
@njit(cache=True)
def encode_move(from_sq: int, to_sq: int, promo: int = 0) -> int:
    return (promo << 12) | (to_sq << 6) | from_sq

@njit(cache=True)
def decode_move(move: int):
    from_sq = move & 63
    to_sq = (move >> 6) & 63
    promo = (move >> 12) & 15
    return from_sq, to_sq, promo

_PROMO_CHARS = {1: 'n', 2: 'b', 3: 'r', 4: 'q', 7: 'n', 8: 'b', 9: 'r', 10: 'q'}

def move_to_uci(move: int) -> str:
    from_sq, to_sq, promo = decode_move(move)
    uci = sq_to_uci(from_sq) + sq_to_uci(to_sq)
    if promo != 0:
        uci += _PROMO_CHARS.get(promo, 'q')
    return uci