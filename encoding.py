"""Board and move encoding shared by training and the runtime agent.

Positions are always oriented so the side to move plays "up" the board:
when it is black's turn the board is mirrored vertically with colors
swapped (python-chess Board.mirror()), so the network always sees the
position from the mover's point of view.

Planes (19 x 8 x 8), indexed [plane, rank, file] with rank 0 = mover's
back rank:
  0-5   mover's P N B R Q K
  6-11  opponent's P N B R Q K
  12-15 castling rights: mover K-side, mover Q-side, opp K-side, opp Q-side
  16    en-passant file (whole file set)
  17    halfmove clock / 100 (clamped)
  18    constant ones
"""
import chess
import numpy as np

NUM_PLANES = 19
POLICY_SIZE = 73 * 64  # AlphaZero-style move encoding, defined further down.

_PIECE_ORDER = [chess.PAWN, chess.KNIGHT, chess.BISHOP,
                chess.ROOK, chess.QUEEN, chess.KING]


def orient(board: chess.Board) -> chess.Board:
    """Return the position from the mover's point of view (mover = white)."""
    return board if board.turn == chess.WHITE else board.mirror()


def board_to_array(board: chess.Board):
    """Compact storage form of the oriented position.

    Returns (board64, meta): board64 is 64 uint8 piece codes (0 empty,
    1-6 mover's P..K, 7-12 opponent's P..K); meta is
    [castling bits, ep file or 255, halfmove clamped to 100].
    """
    b = orient(board)
    board64 = np.zeros(64, dtype=np.uint8)
    for square, piece in b.piece_map().items():
        code = _PIECE_ORDER.index(piece.piece_type) + 1
        if piece.color != chess.WHITE:
            code += 6
        board64[square] = code
    castling = (
        (1 if b.has_kingside_castling_rights(chess.WHITE) else 0)
        | (2 if b.has_queenside_castling_rights(chess.WHITE) else 0)
        | (4 if b.has_kingside_castling_rights(chess.BLACK) else 0)
        | (8 if b.has_queenside_castling_rights(chess.BLACK) else 0)
    )
    ep_file = b.ep_square % 8 if b.ep_square is not None else 255
    halfmove = min(b.halfmove_clock, 100)
    return board64, np.array([castling, ep_file, halfmove], dtype=np.uint8)


def array_to_planes(board64, meta):
    planes = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)
    grid = board64.reshape(8, 8)
    for code in range(1, 13):
        planes[code - 1][grid == code] = 1.0
    castling, ep_file, halfmove = int(meta[0]), int(meta[1]), int(meta[2])
    for bit in range(4):
        if castling & (1 << bit):
            planes[12 + bit][:] = 1.0
    if ep_file != 255:
        planes[16][:, ep_file] = 1.0
    planes[17][:] = halfmove / 100.0
    planes[18][:] = 1.0
    return planes


def board_to_planes(board: chess.Board):
    return array_to_planes(*board_to_array(board))


# ---------------------------------------------------------------------------
# Move encoding: AlphaZero's 8x8x73 scheme, flattened to movetype*64 + from_sq
# (matching how the network's (73,8,8) policy output flattens).
#   movetype 0-55:  "queen" moves, 8 directions x 7 distances
#   movetype 56-63: knight moves
#   movetype 64-72: underpromotions (3 directions x N/B/R); queen
#                   promotions are encoded as plain forward/diagonal moves.
# Moves must be oriented (mover plays up the board) before encoding.
# ---------------------------------------------------------------------------

_DIRS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
_DIR_IDX = {d: i for i, d in enumerate(_DIRS)}
_KNIGHT = [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]
_KNIGHT_IDX = {d: i for i, d in enumerate(_KNIGHT)}
_UNDERPROMO = [chess.KNIGHT, chess.BISHOP, chess.ROOK]


def mirror_move(move: chess.Move) -> chess.Move:
    """Map a move between the real board and its mirrored orientation."""
    return chess.Move(chess.square_mirror(move.from_square),
                      chess.square_mirror(move.to_square),
                      promotion=move.promotion)


def encode_move(move: chess.Move) -> int:
    fr, fc = divmod(move.from_square, 8)
    tr, tc = divmod(move.to_square, 8)
    dr, dc = tr - fr, tc - fc
    if move.promotion is not None and move.promotion != chess.QUEEN:
        movetype = 64 + (dc + 1) * 3 + _UNDERPROMO.index(move.promotion)
    elif (dr, dc) in _KNIGHT_IDX:
        movetype = 56 + _KNIGHT_IDX[(dr, dc)]
    else:
        dist = max(abs(dr), abs(dc))
        step = (dr // dist if dr else 0, dc // dist if dc else 0)
        movetype = _DIR_IDX[step] * 7 + dist - 1
    return movetype * 64 + move.from_square


def decode_move(index: int, board: chess.Board) -> chess.Move:
    movetype, from_sq = divmod(index, 64)
    fr, fc = divmod(from_sq, 8)
    if movetype >= 64:
        u = movetype - 64
        dc, piece = u // 3 - 1, _UNDERPROMO[u % 3]
        return chess.Move(from_sq, (fr + 1) * 8 + fc + dc, promotion=piece)
    if movetype >= 56:
        dr, dc = _KNIGHT[movetype - 56]
    else:
        d, dist = divmod(movetype, 7)
        dr, dc = _DIRS[d][0] * (dist + 1), _DIRS[d][1] * (dist + 1)
    to_sq = (fr + dr) * 8 + fc + dc
    promotion = None
    if board.piece_type_at(from_sq) == chess.PAWN and to_sq >= 56:
        promotion = chess.QUEEN
    return chess.Move(from_sq, to_sq, promotion=promotion)
