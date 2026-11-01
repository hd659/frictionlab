# Phase 6: performance work on the continuous order book (fl-039)

fl-038 gave us a benchmark harness whose checksums hash *results*. fl-039
uses it for the first optimisation, which is the continuous limit order book hot
path. The goal was faster matching with byte-identical outputs. Results stay
identical. The speedup is real when you time the book by itself and small
when you time a whole scenario. Both numbers are given below.

## Data-structure choice

Before (frozen verbatim in `tests/reference_order_book.py`), each side kept a
sorted Python list of price keys. A new level used `bisect.insort` and an
emptied level used `del keys[i]`, and both shift the list in O(L) for L live
levels.

After (`frictionlab/core/book_levels.py`, `BookSide`):

| piece | structure | cost |
|---|---|---|
| level queues | `dict[price_ticks] -> deque[Order]` | O(1) lookup, FIFO = time priority |
| level size | `dict[price_ticks] -> int`, kept eagerly | exact depth / `volume_at` at all times |
| best price | `heapq` of signed keys (bids negated) | O(1) peek, O(log L) push |
| level death | delete from the dicts only | O(1); heap key removed lazily |
| order index | `OrderBook._live: dict[order_id] -> Order` | O(1) cancel lookup |
| depth listing | sorted price cache, invalidated on level birth/death | repeated depth queries are a slice |

Prices are integer ticks (fl-002), so they make exact, hashable dict keys and
there is no float-key aliasing.

Lazy deletion works like this. When a level empties it leaves `qty`, but its key
stays in the heap. `best()` pops keys from the heap top until it finds one that
is still in `qty`. A set called `_in_heap` keeps at most one key per price. If a
level dies and is reborn while its old key is still buried, the old key is reused
and no duplicate is pushed, so the heap never grows past the number of
distinct prices ever seen. Stale keys are never returned. `best()` checks `qty`,
and `prices()` / `levels()` read from `qty` and never from the heap.

Inside a level, cancelled orders are still purged lazily from the deque head
(unchanged from Phase 1). A cancel looks the order up in `_live`, marks it,
and subtracts its lots from the level's `qty`. The deque entry is dropped the
next time `head()` reaches it.

Matching loop micro-optimisations: the crossing test is inlined against the
taker's limit, fills update `filled` directly (the invariant
`0 < lots <= remaining` holds by construction), and the buy/sell
`Trade` fields are built without an intermediate tuple swap. `Trade` itself is
unchanged and still validates every print.

## Optional vectorised depth and numba

`OrderBook.depth_array(side, levels)` returns an int64 `(k, 3)` array of
`[price, qty, cumulative_qty]` from best outward. It is meant for
MarketView-style depth queries (lots available within k ticks, impact of a
sweep). The cumulative column comes from a kernel picked once per process:

- default: `numpy.cumsum(dtype=int64)`;
- `FRICTIONLAB_NUMBA=1` and numba importable and compiling: an `njit` loop.

If numba is missing, or it fails to compile, the code falls back to numpy
without raising. Both kernels return the same integers, so the flag can never
change a result. numba is not a dependency. `pyproject.toml` still lists
only numpy.

## How correctness is guarded

1. **Frozen oracle.** `tests/test_order_book_equivalence.py` replays 50 seeded
   random streams through both books, using twin `Order` objects. The streams
   mix limits (some marketable), markets, full and partial cancels, and
   cancels of filled, cancelled or unknown ids, under both STP modes. After
   *every* operation the test checks that the trades match field for field,
   along with the cancel return value, best bid/ask, full depth of both sides
   and the set of live ids. At the end it checks the final `filled` /
   `cancelled` / `qty` of every order. As a sanity check, a LIFO-within-level
   mutant is caught on 40/40 seeds.
2. **Lazy-deletion property test.** 3000 random level births and deaths on
   one side, with `best()` and `levels()` compared to a brute-force model
   after each step. The test also asserts that the heap holds no duplicate keys.
3. **fl-038 checksums.** Every scenario's small-size checksum must still equal
   `benchmarks/baseline_small.json`. Running
   `frictionlab bench --compare benchmarks/baseline_small.json` reports
   `compare: ok`, and the medium checksums also match the pre-change tree.
4. **Unchanged regression suites.** The Phase 1 order book tests, the runner
   tests and the Phase 2-5 example regression tests pass without edits.

## Before / after

Same machine, interleaved runs (pre-change worktree, then working tree),
`warmup=1, repeat=7`, median wall seconds. Checksums were identical in every row.

End-to-end fl-038 scenarios:

| scenario | size | ops | before | after | speedup |
|---|---|---:|---:|---:|---:|
| book_random_flow | small | 3000 | 0.0269 | 0.0277 | 0.97x |
| runner_zi_market | small | 1500 | 0.0132 | 0.0136 | 0.97x |
| batch_auction_clear | small | 2000 | 0.0631 | 0.0632 | 1.00x |
| amm_swap_stream | small | 2000 | 0.0196 | 0.0197 | 0.99x |
| sweep_coase_small | small | 3303 | 0.0358 | 0.0372 | 0.96x |
| book_random_flow | medium | 30000 | 0.2699 | 0.2600 | 1.04x |
| runner_zi_market | medium | 15000 | 0.1206 | 0.1192 | 1.01x |
| batch_auction_clear | medium | 10000 | 0.7438 | 0.7383 | 1.01x |
| amm_swap_stream | medium | 20000 | 0.1956 | 0.1933 | 1.01x |
| sweep_coase_small | medium | 32309 | 0.3405 | 0.3427 | 0.99x |

Repeat pairs varied by about +-7% (one medium `book_random_flow` pair showed
1.23x). Treat any end-to-end change under 10% as noise.

Book only. Orders are pre-built, so only `submit` / `cancel` are timed.
The streams are 60% limits, 10% markets and 30% cancels, with the oracle
as "before":

| stream | ops | price width | before | after | speedup |
|---|---:|---:|---:|---:|---:|
| small | 3000 | 20 | 0.0044 | 0.0036 | 1.22x |
| medium | 30000 | 20 | 0.0398 | 0.0331 | 1.20x |
| medium, deep book | 30000 | 2000 | 0.0426 | 0.0341 | 1.25x |
| large, deep book | 100000 | 5000 | 0.1536 | 0.1167 | 1.32x |

As expected, the gain grows with the number of live levels: it is the O(L)
list shifts that go away.

## Why end-to-end barely moves

A cProfile of medium `book_random_flow` puts about 15% of the time in
`OrderBook._match` (cumulative). Most of the rest goes to `Ledger.settle`
(about 50%, with exact `Fraction` fee arithmetic), `Order.__post_init__`
validation (`_check_int` and ABC `isinstance` checks) and the scenario's own
RNG calls. A 1.25x book therefore buys only a few percent of total time.
Next targets, each one guarded by the same checksums:

- an integer-bps fast path in the fee stack, to avoid `Fraction` when bps
  divide evenly;
- a cheaper `_check_int` (an exact-`int` fast path before the `Integral` ABC check);
- the batch auction clear (the slowest scenario per op).

fl-040 below handles the first two items.

# Ledger and fee stack hot path (fl-040)

fl-039's profile showed that settlement, not matching, dominated the book
scenarios. fl-040 makes `Ledger.settle` and `FrictionStack.trade_fees` cheap
and changes no results.

## Integer basis-point fast path

Every built-in knob is linear in the fill, so
`raw_cost = (a * notional + b * qty + c) / den` with:

| knob | a | b | c | den |
|---|---:|---:|---:|---:|
| `MakerTakerFee` | bps(role) | 0 | 0 | 10000 |
| `HalfSpreadCost` | 0 | spread_ticks | 0 | 2 |
| `LatencyCost` | 0 | drift * delay | 0 | 1 |
| `FixedCost` | 0 | 0 | per_trade | 1 |

`LinearCost.int_terms(role)` returns this tuple when the parameter is an integer.
It returns `(0, 0, 0, 1)` when the role is not covered, and `None` otherwise
(0.5 bps, 3/2 spread). `FrictionStack` folds the integer tuples of all
components into one `(A, B, C, D)` per role over the lcm of the denominators.
It does this once at construction or `add`, not per trade. One fill then costs
`-(-(A * notional + B * qty + C) // D)`, which is exact integer ceil
division. This is the same single venue-favourable rounding of the same exact sum
that the Fraction path does, so a rebate still rounds toward zero and a fee
still rounds up. `fees.div_round` exposes ceil and floor for other
rounding modes.

Only components with no integer terms fall back to `Fraction`: non-integer
rates, `GasCost`, tiered `FeeComponent`, experiment-local knobs, and any
subclass that overrides `raw_cost`. The fast path is enabled per exact type, so a
subclass is never assumed to be linear. In a mixed stack the integer part is
added to the slow parts as `Fraction(num, D)` before the single rounding. If
something appends to `stack.models` directly, the stack recompiles its plan the
next time it is evaluated. `charge_fraction` keeps the old path available for
spot checks.

## Ledger

- `_int(name, v)` accepts `type(v) is int` with one identity test. Every other
  value (numpy ints, bool, float, numpy floats) is passed to `types._check_int`,
  so accepted values and error messages are byte-identical.
- `settle` books both legs inline. `price` and `qty` were already validated
  by `Trade.__post_init__`, and the fees by `_int`. It then checks
  conservation once per trade: the buyer, seller and venue cash deltas sum to zero
  and the positions net to zero. Before, every leg went through
  `Account.apply_fill`, which repeated four `_check_int` calls and a
  `Side(...)` coercion.
- Strict mode (`Ledger(strict=True)` or `FRICTIONLAB_STRICT_LEDGER=1`) keeps the
  per-leg path. It books through `apply_fill` and asserts each leg's cash,
  position and fee delta separately. `check_every` (full invariant scan)
  works the same as before in both modes.

## How correctness is guarded

1. **Frozen oracle.** `tests/reference_fees.py` freezes the pre-change
   Fraction classes and stack evaluation. `tests/test_fee_fast_path.py` runs
   40 seeds x 250 random fills (maker rebates down to `-taker`, odd qtys, and
   prices placed within one tick of the `10000/bps` rounding boundary).
   It checks `trade_fees`, per-fill `charge` and `round_trip` against the
   oracle, then runs the same checks for non-integer rates, gas and tiered
   components.
2. **Same errors.** Bool, float, `np.float64`, `np.float32`, str and None
   get the same `TypeError` text from `_int` as from `_check_int`, at every
   ledger entry point, and the ledger is left untouched.
3. **Strict vs default.** Seeded book flows and the `runner_zi_market`
   scenario give identical balances, fee and volume counters and journals.
   A leaking account is caught by both modes.
4. **fl-038 checksums.** All small checksums equal
   `benchmarks/baseline_small.json` in both modes, and the medium checksums
   equal the pre-change tree.

## Before / after

Same machine. Pre-change tree from a `git worktree` at fl-039 HEAD, runs
interleaved, `warmup=1, repeat=7`, median wall seconds averaged over two
pairs. Checksums were identical in every row.

| scenario | size | ops | before | after | speedup |
|---|---|---:|---:|---:|---:|
| book_random_flow | small | 3000 | 0.0284 | 0.0239 | 1.19x |
| runner_zi_market | small | 1500 | 0.0136 | 0.0122 | 1.11x |
| batch_auction_clear | small | 2000 | 0.0630 | 0.0580 | 1.09x |
| amm_swap_stream | small | 2000 | 0.0197 | 0.0140 | 1.41x |
| sweep_coase_small | small | 3303 | 0.0366 | 0.0367 | 1.00x |
| book_random_flow | medium | 30000 | 0.2566 | 0.1746 | 1.47x |
| runner_zi_market | medium | 15000 | 0.1194 | 0.1128 | 1.06x |
| batch_auction_clear | medium | 10000 | 0.7368 | 0.7419 | 0.99x |
| amm_swap_stream | medium | 20000 | 0.1940 | 0.1389 | 1.40x |
| sweep_coase_small | medium | 32309 | 0.3483 | 0.3469 | 1.00x |

In one after run, medium `sweep_coase_small` hit 0.42 s, a background-load
outlier, so that "after" cell is the median of three runs. As in fl-039, treat
changes under 10% as noise.

Hot path only: 30000 pre-built trades over 8 accounts, median of 5.

| path | stack | before | after | speedup |
|---|---|---:|---:|---:|
| `trade_fees` | 10 bps taker | 0.1204 | 0.0121 | 9.9x |
| `trade_fees` | -1/7 bps + spread 1 + fixed 2 | 0.2024 | 0.0124 | 16.3x |
| `Ledger.settle_all` | 10 bps taker | 0.2321 | 0.0495 | 4.7x |
| `Ledger.settle_all` | -1/7 bps + spread 1 + fixed 2 | 0.3198 | 0.0509 | 6.3x |

The fee hashes and venue revenue were identical before and after.

The gains show up where trades settle through the stack: the book scenario
and the AMM swap stream, whose pool fees go through `FrictionStack.charge`.
`sweep_coase_small` settles only about 6 trades, and the batch auction is
limited by its clearing loop, so both stay flat. The batch auction clear is
still the next target.
