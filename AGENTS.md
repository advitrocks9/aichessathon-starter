# Working in this repo

This is a starter for AI Chessathon, a chess-engine competition. The deliverable is one file,
`agent.py`, exposing `get_move(fen, time_left_ms) -> str`. It gets zipped and uploaded, and the
platform plays it against other people's agents on a fixed cadence.

## Read the rules from the source

The rules and the agent contract live on the site and change. Fetch them before you answer
anything about limits, deadlines or what is allowed, and before you rely on a number below.

- https://aichessathon.com/docs/agent-contract.md
- https://aichessathon.com/docs/rules.md

## The contract

- `agent.py` sits at the root of the zip, not inside a folder. The platform does `import agent`.
- `get_move(fen: str, time_left_ms: int) -> str` returns a UCI move, `e2e4` or `e7e8q`.
- Your colour is the side to move in the fen. There is no other input.
- One process per game, started fresh for every game. It stays alive between your moves, so
  module state survives to your next move in the same game, never to the next game.
- Importing your agent has its own budget before the clock starts, 90 s in the qualifier and
  30 s at the final. Load weights there.
- The clock is 120 s plus 0.5 s per move, per side, on wall time. `time_left_ms` is the clock
  before your move. The increment lands after it.
- One core of an AMD EPYC 9V74, measured at 2.60 GHz. 2 GB RAM. No network. No GPU. Identical
  for every game.
- An illegal move, malformed output, a crash, running out of memory or missing the init budget
  loses the game. A move payload over 4 KB counts as an illegal move.
- Losing on time loses unless the other side has no way to mate, and then the game is a draw.
- Draws follow FIDE rules. A game still running at 600 plies is a draw, and the opening position
  counts toward the 600.
- A submission is at most 50 MB unzipped. A team may upload 10 times a day. The latest upload
  that passed validation is the one that plays.
- Every game starts from a curated opening position that is close to level. The set is not
  published. The first fen is where the game starts, so repetition and fifty-move counts begin
  there, not at move one.
- Your process is suspended while your opponent thinks, so work you leave running between your
  own moves does not run. Search inside `get_move`. Both agents share one core in turns, so time
  you measure inside a move is yours alone. Two of your games can run at once, in separate
  containers.

## Things that break agents here

- The filesystem is read-only apart from 256 MB at `/tmp`. `HOME` and every cache path already
  point there. Do not write anywhere else. `/tmp` is wiped between games, so it is scratch and
  never a cache. numba's `cache=True` buys nothing, and the harness reproduces that.
- No network at all. Nothing downloads at runtime. Weights ship inside the zip.
- One core. `torch.set_num_threads(1)`. More threads lose time rather than winning it.
- Your zip is first on `sys.path`, so a file named after a module you import, like `chess.py` or
  `types.py`, shadows the real one. `random.py` too, and the failure will look unrelated.
- Python 3.12 with torch, numpy, python-chess, onnxruntime and numba preinstalled at fixed
  versions. Nothing else installs and a `requirements.txt` in the zip is ignored. The versions
  are torch 2.13 (CPU), numpy 2.5, python-chess 1.11, onnxruntime 1.29 and numba 0.67. An import
  outside that stack crashes on the platform even when it works locally. Ask
  hello@aichessathon.com for a package the stack lacks. Any addition is announced to every team.
- Native binaries in the zip are rejected. Model weights are not binaries, so `.onnx`,
  `.safetensors` and `.pt` are fine. Ship Python source. Any network you ship is one you trained
  yourself, and training data is unrestricted, including positions annotated by an existing
  engine.
- numba is how Python gets fast here. Warm every jitted function once at import so compilation
  lands in the init budget, not on the clock. Cython does not work on the platform.
- `print` is safe. The runner points file descriptor 1 at stderr before importing the agent, so
  nothing you write can corrupt the protocol. Your output is kept after validation and after
  every rated game, in a log only your team can read. Only the first 4 KB and the last 4 KB
  survive it, and so does the harness, so print what you want to read back.

## Do not

- Do not ship someone else's engine or someone else's network. Third party engines are
  prohibited. That covers Stockfish, Lc0, Maia, any wrapper around one and any port or
  translation of one. Starting from a published chess network is not allowed, so fine-tuning or
  re-exporting one counts as shipping it. This is checked after games are played, not only at
  upload.
- Do not ship a table that answers middlegame positions for the agent to look up while it plays.
  That is an engine in another shape. A table you ship and read during a game may answer the
  opening or the endgame. The middlegame you search yourself. The opening is a position whose
  move number is 20 or lower. The endgame is a position of at most 7 pieces, counting both kings.
  Opening books and endgame tablebases are fine whatever produced them, and `chess.polyglot` and
  `chess.syzygy` are in the base image to read them. Everything ships inside the 50 MB cap, so 3
  and 4 man syzygy fits and 5 man is far too big.
- Do not add network calls, subprocess calls to external binaries, or anything that reads outside
  the agent directory and `/tmp`. Your agent is one process and stays one process. On a single
  core `multiprocessing` cannot win you time anyway.
- Do not obfuscate. What you ship must be source a judge can read. Obfuscated agents are
  disqualified.
- Do not edit `harness/`. It mirrors the platform's protocol and clock. Changing it makes local
  results meaningless.

## Verify

```
make play      # one game against a baseline, real time control
make arena     # 16 fast games against a baseline, with a score and an interval
make zip       # build submission.zip, then smoke it the way the platform does
make gate      # ruff, mypy, and two games that have to finish cleanly
```

- `make zip` extracts what it built and plays out of it, so a module you never packaged fails
  there instead of costing an upload. Nothing here decides acceptance. The platform validates on
  upload and writes the log that is the authority.
- Local python is pinned to 3.12 to match the image. Everything else the container enforces, the
  read-only filesystem, the 2 GB cap and the missing network, is not reproduced here.
- The harness sets `HARNESS_SEED` on every agent it starts, which the baselines read so their
  tie-breaks replay. The platform sets nothing of the kind, so do not read it in your own agent.

## Style

Python 3.12, type-annotated, ruff and mypy strict clean. Keep `agent.py` readable. It is the thing
a judge reads if your games get flagged, and the thing you have to explain at the final.
