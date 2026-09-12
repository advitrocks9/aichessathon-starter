# Where the strength comes from

A model is optional. Material plus piece-square tables is a legal entry, and the winning shape is
usually a search that calls a small evaluation, learned or not. This is what tends to matter,
roughly in order.

## Search

Negamax with alpha-beta is the whole game. The gap between the `minimax` baseline and something
respectable is mostly move ordering, because alpha-beta only pays off when good moves come first.

- Order captures before quiet moves, and order captures by MVV-LVA.
- Keep a transposition table. Even a plain dict keyed on `board._transposition_key()` and cleared
  between moves is worth a ply.
- Iterative deepening. Search depth 1, then 2, then 3, keeping the best move from each pass. It
  gives you ordering for free and, more importantly, something to return when time runs out.
- Quiescence search at the leaves, captures only. Without it your evaluation is measured in
  positions that are mid-exchange and it will be wrong.

You are on one core in Python, so node counts are small. Expect thousands, not millions. That
changes the trade. Depth is expensive, so evaluation quality and ordering buy more than they would
in a C engine.

numba closes most of that gap. The platform preinstalls it, and a jitted movegen and evaluation
reach node counts pure Python cannot. Warm every jitted function once at import so compilation
happens inside the init budget, and warm it with the argument types the real calls use, since
numba compiles per signature. `baselines/numba` shows the pattern. It scores barely better than
`baselines/minimax`, because jitting a two-ply search wins nothing on its own. The gain is the
depth the speed lets you afford.

## Evaluation

Material plus piece-square tables is a real evaluation and it beats both baselines. It is also the
thing to build first, because it gives you a reference to measure a model against.

- The base image ships torch and onnxruntime, so a small network is practical. Export to ONNX and
  run it with onnxruntime. Startup is faster than torch and inference on one core is competitive.
- Keep it small. A net you can evaluate thousands of times per move is worth more than a better
  net you can evaluate fifty times.
- Batch. Collect the leaf positions of a search pass and evaluate them in one call rather than one
  at a time.
- Size a learned evaluation like the CPU engine nets. An NNUE-style net quantised to int8 or int16
  lands between 1 and 40 MB and is fast enough to search with. A deep convolutional net at fp32
  manages a few hundred evaluations a second on one core, which is a policy model's budget, not a
  search evaluation's.

## Training data

You have no network at runtime, so everything ships in the zip inside the 50 MB cap, and data
gathering happens on your machine before you upload. Public game databases and self-play against
your own earlier versions are both reasonable starting points. Training data is unrestricted,
including positions annotated by an existing engine. What ships inside the zip is what the ban
covers. Whatever you train on, any network you ship is one you trained yourself.

## Time management

120 seconds plus 0.5 per move. A flag is a loss unless the other side cannot mate, and it is the
most common self-inflicted one.

- Budget per move from the clock you were handed, not from a constant. Something like
  `time_left_ms / max(20, expected_moves_left)` is enough to start. The increment lands after the
  move, so you can spend the one you are about to earn.
- Check the clock inside your search, not only between moves, and return the best move you have
  when the budget is gone. Iterative deepening makes that easy.
- Leave a margin. The referee measures wall time, and the watchdog does not forgive.
- Your machine is not the match machine. One core of an EPYC 9V74 at 2.60 GHz is slower than a
  laptop core, so depth you reach here is depth you may not reach in a rated game. Your validation
  log prints your real init time and your slowest move. Upload once and scale the budgets you
  tuned locally by what it tells you.

## What the position alone does not tell you

The process stays alive between your moves, so you can keep state. Two things are worth keeping.

- The positions you have been asked about. The referee claims threefold repetition automatically,
  so if you are winning and shuffling you can draw a won game without being told. Count from the
  first fen, not from move one. That fen is where the game starts, and its halfmove clock is where
  the fifty move count begins.
- Your own search results. A transposition table that survives across moves is a real gain.

An opening book is worth less here than it looks. Rated games start from curated positions, so a
book keyed on move one is already out of book. The eight in `harness/rules.py` are a sample, not
the set, so anything you tune on them is tuned on eight positions. Test with `make play FEN=...`
from positions you have not prepared, and spend the effort on the search.

The endgame is the other side of this. `chess.polyglot` and `chess.syzygy` are in the base image,
and 3 and 4 man syzygy fits inside the 50 MB cap while 5 man is far too big. Four man tables end
king and pawn races correctly. A shallow search does not.

## Measuring a change

Two games tell you nothing. `make arena` alternates colours, fixes the opening, seeds the opponent
and prints the 95% interval, so you can see when a result is real instead of guessing.

- A change worth 3% needs hundreds of games before that interval shrinks past it.
- Run several shells at once to get them, but score the final pass in one, since games sharing a
  machine stop measuring time management.
- The interval assumes games are independent, and eight openings means they stop being
  independent past sixteen, so read it as optimistic in a long run.
- Keep the previous version as an opponent. Better than my last one is the only comparison that
  matters.

## What loses games for free

- Flagging. See time management above.
- Crashing on an edge case. No legal moves, a promotion, an en passant capture. Play a few hundred
  games against a random baseline and the rare paths show up.
- Blowing the import budget loading weights, 90 seconds in the qualifier and 30 at the final.
  Warm jitted functions inside it, not on the clock.
- Writing anywhere but `/tmp`. Everything else is read-only, and `/tmp` is wiped between games, so
  nothing you write survives. numba's `cache=True` never hits.
- More threads than cores. `torch.set_num_threads(1)`.
- Splitting your agent across files and uploading only `agent.py`. `make zip` plays out of the zip
  it built, the cheapest place to find that out.
