"""Data -> pixel mapping for the dependency-free SVG plots.

- `extent`: finite (min, max) of one or more series; NaN/inf ignored, empty
  input falls back to (0, 1) and a constant series is padded symmetrically.
- `nice_ticks`: Heckbert "nice numbers" (1, 2, 5 x 10^k) tick values whose
  first tick is <= lo and last tick is >= hi, so the axis brackets the data.
- `LinearScale`: affine map from a data domain onto a pixel range (inverted
  ranges are allowed, e.g. SVG y grows downwards).
- `fmt` / `format_tick`: fixed-precision number formatting with no "-0" and
  integer ticks printed without a decimal point, so SVG output is
  byte-identical across runs and platforms.
- `ramp`: monotone single-hue color ramp (light -> dark) for heatmaps.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

PRECISION = 2
_LIGHT = (247, 251, 255)
_DARK = (8, 48, 107)


def fmt(x: float, prec: int = PRECISION) -> str:
    """Round to `prec` decimals and strip trailing zeros; never emit '-0'."""
    v = round(float(x), prec)
    if v == 0.0:
        return "0"
    s = f"{v:.{prec}f}".rstrip("0").rstrip(".") if prec > 0 else f"{v:.0f}"
    return "0" if s in ("-0", "") else s


def _finite(values: Iterable[float]) -> list[float]:
    out = []
    for v in values:
        f = float(v)
        if math.isfinite(f):
            out.append(f)
    return out


def extent(*series: Sequence[float], pad: float = 1.0) -> tuple[float, float]:
    """Finite min/max over all series; degenerate cases yield a usable range."""
    vals = [v for s in series for v in _finite(s)]
    if not vals:
        return 0.0, 1.0
    lo, hi = min(vals), max(vals)
    if lo == hi:
        d = abs(lo) * 0.1 if lo != 0 else pad
        return lo - d, hi + d
    return lo, hi


def nice_number(x: float, round_: bool) -> float:
    """Nearest (round_) or ceiling 1/2/5 x 10^k to a positive `x`."""
    if x <= 0 or not math.isfinite(x):
        return 1.0
    exp = math.floor(math.log10(x))
    f = x / 10 ** exp
    if round_:
        nf = 1 if f < 1.5 else 2 if f < 3 else 5 if f < 7 else 10
    else:
        nf = 1 if f <= 1 else 2 if f <= 2 else 5 if f <= 5 else 10
    return nf * 10 ** exp


def tick_step(lo: float, hi: float, n: int = 5) -> float:
    span = nice_number(hi - lo, False) if hi > lo else 1.0
    return nice_number(span / max(n - 1, 1), True)


def _decimals(step: float) -> int:
    return max(0, -math.floor(math.log10(step))) if step > 0 else 0


def nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    """Nice tick values with ticks[0] <= lo and ticks[-1] >= hi."""
    lo, hi = float(lo), float(hi)
    if lo > hi:
        lo, hi = hi, lo
    if lo == hi:
        lo, hi = extent([lo])
    step = tick_step(lo, hi, n)
    dec = _decimals(step) + 1
    k0 = math.floor(round(lo / step, 9))
    k1 = math.ceil(round(hi / step, 9))
    return [round(k * step, dec) + 0.0 for k in range(k0, k1 + 1)]


def format_tick(v: float, step: float | None = None) -> str:
    """Integer-valued ticks print as ints; otherwise enough decimals for `step`."""
    if float(v).is_integer() and (step is None or float(step).is_integer()):
        return str(int(v))
    dec = _decimals(step) if step else PRECISION
    return fmt(v, max(dec, 1))


@dataclass(frozen=True)
class LinearScale:
    """Affine map domain (d0, d1) -> range (r0, r1); constant domain maps to midpoint."""

    d0: float
    d1: float
    r0: float
    r1: float

    def __call__(self, v: float) -> float:
        if self.d1 == self.d0:
            return (self.r0 + self.r1) / 2.0
        t = (float(v) - self.d0) / (self.d1 - self.d0)
        return self.r0 + t * (self.r1 - self.r0)

    def ticks(self, n: int = 5) -> list[float]:
        return nice_ticks(min(self.d0, self.d1), max(self.d0, self.d1), n)

    @classmethod
    def nice(cls, lo: float, hi: float, r0: float, r1: float, n: int = 5) -> "LinearScale":
        """Scale whose domain is widened to the bracketing nice ticks."""
        t = nice_ticks(lo, hi, n)
        return cls(t[0], t[-1], r0, r1)


def ramp(t: float) -> str:
    """Hex color on a light->dark ramp; every RGB channel is monotone in t."""
    t = min(1.0, max(0.0, float(t)))
    rgb = (round(a + (b - a) * t) for a, b in zip(_LIGHT, _DARK))
    return "#" + "".join(f"{c:02x}" for c in rgb)


def luminance(hex_color: str) -> float:
    """Relative (unweighted-gamma) luminance of '#rrggbb', for ramp checks."""
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b
