"""Integer-tick indexed price levels for one side of the continuous book.

Data structure (fl-039):
- `queues`: dict price -> deque of resting orders (FIFO = time priority);
- `qty`:    dict price -> aggregate open lots, maintained eagerly, so depth
  and `volume_at` are exact at all times;
- `_heap`:  heapq of signed keys (bids negated, so the heap top is always the
  best price) with *lazy deletion*: an emptied level is dropped from the
  dicts in O(1) and its heap key is discarded only when it reaches the top.

`_in_heap` guarantees at most one heap key per price, so a level that dies
and is reborn while its stale key is still buried does not get pushed twice.
Insert of a new level and pop of a dead one cost O(log L); everything else
on the matching hot path (best price, head order, fill) is O(1) amortised,
versus O(L) list shifts for the previous sorted-list levels.

Depth snapshots for `MarketView`-style queries can be taken as an int64
numpy array (`depth_array`), with an optional numba kernel for cumulative
depth gated behind the FRICTIONLAB_NUMBA env flag. numba is never required:
if the flag is unset, or numba is missing / fails to compile, the pure
numpy kernel is used and the integer results are identical.
"""
from __future__ import annotations

import heapq
import os
from collections import deque
from typing import Callable, Deque, Dict, List, Optional, Set, Tuple

import numpy as np

from .types import Order, Side

NUMBA_ENV = "FRICTIONLAB_NUMBA"
_FLAG_ON = ("1", "true", "yes", "on")


class BookSide:
    """One side of the book: dict-of-deques levels plus a lazy best-price heap."""

    __slots__ = ("side", "_sign", "queues", "qty", "_heap", "_in_heap", "_sorted")

    def __init__(self, side: Side) -> None:
        self.side = Side(side)
        self._sign = -1 if self.side is Side.BUY else 1  # key = sign * price
        self.queues: Dict[int, Deque[Order]] = {}
        self.qty: Dict[int, int] = {}
        self._heap: List[int] = []
        self._in_heap: Set[int] = set()
        self._sorted: Optional[List[int]] = []  # best-outward cache; None = stale

    def best(self) -> Optional[int]:
        heap, qty = self._heap, self.qty
        while heap:
            price = heap[0] * self._sign
            if price in qty:
                return price
            heapq.heappop(heap)  # lazy deletion of an emptied level
            self._in_heap.discard(price)
        return None

    def add(self, order: Order) -> None:
        p = order.price
        q = self.queues.get(p)
        if q is None:
            q = self.queues[p] = deque()
            self.qty[p] = 0
            self._sorted = None
            if p not in self._in_heap:  # stale key still buried -> reuse it
                heapq.heappush(self._heap, self._sign * p)
                self._in_heap.add(p)
        q.append(order)
        self.qty[p] += order.remaining

    def head(self, price: int) -> Order:
        q = self.queues[price]
        while not q[0].is_active:  # lazy purge; qty > 0 guarantees a live order
            q.popleft()
        return q[0]

    def reduce(self, price: int, lots: int) -> None:
        left = self.qty[price] - lots
        if left:
            self.qty[price] = left
        else:  # O(1) level death; the heap key is purged lazily by best()
            del self.queues[price], self.qty[price]
            self._sorted = None

    def prices(self, n: Optional[int] = None) -> List[int]:
        """Live prices from best outward (stale heap keys never appear).

        Sorted once per change of the level *set* (not per fill), so repeated
        MarketView depth queries between level births/deaths are a slice.
        """
        s = self._sorted
        if s is None:
            s = self._sorted = sorted(self.qty, reverse=self.side is Side.BUY)
        return s[:] if n is None else s[:max(n, 0)]

    def levels(self, n: Optional[int] = None) -> List[Tuple[int, int]]:
        qty = self.qty
        return [(p, qty[p]) for p in self.prices(n)]

    def __len__(self) -> int:
        return len(self.qty)


# ---- numpy depth snapshots + optional numba kernel ----------------------
def _cumsum_numpy(qty: np.ndarray) -> np.ndarray:
    return np.cumsum(qty, dtype=np.int64)


def numba_requested() -> bool:
    return os.environ.get(NUMBA_ENV, "").strip().lower() in _FLAG_ON


def _build_numba() -> Optional[Callable[[np.ndarray], np.ndarray]]:
    try:
        import numba  # optional; absence is not an error
    except Exception:
        return None

    @numba.njit(cache=False)
    def _cumsum_jit(qty):  # pragma: no cover - only with numba installed
        out = np.empty(qty.shape[0], dtype=np.int64)
        acc = 0
        for i in range(qty.shape[0]):
            acc += qty[i]
            out[i] = acc
        return out

    try:  # compile eagerly so a broken toolchain falls back here, not mid-run
        _cumsum_jit(np.zeros(1, dtype=np.int64))
    except Exception:
        return None
    return _cumsum_jit


_KERNELS: Dict[bool, Callable[[np.ndarray], np.ndarray]] = {}


def cumulative_kernel() -> Callable[[np.ndarray], np.ndarray]:
    """numba kernel iff FRICTIONLAB_NUMBA is set and numba works, else numpy."""
    want = numba_requested()
    if want not in _KERNELS:
        _KERNELS[want] = (_build_numba() if want else None) or _cumsum_numpy
    return _KERNELS[want]


def reset_kernel_cache() -> None:
    _KERNELS.clear()


def depth_array(half: BookSide, n: Optional[int] = None) -> np.ndarray:
    """(k, 3) int64 array of [price, qty, cumulative qty] from best outward."""
    lv = half.levels(n)
    if not lv:
        return np.zeros((0, 3), dtype=np.int64)
    arr = np.empty((len(lv), 3), dtype=np.int64)
    arr[:, :2] = np.asarray(lv, dtype=np.int64)
    arr[:, 2] = cumulative_kernel()(np.ascontiguousarray(arr[:, 1]))
    return arr
