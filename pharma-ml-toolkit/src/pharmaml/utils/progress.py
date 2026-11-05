r"""Progress-rate machinery: the tqdm EMA and the display formatters.

Round 173 absorbed ``tqdm/tqdm``'s rate bookkeeping (the bar rendering
itself is GUI-specific and stays out):

* **The bias-corrected EMA** (``tqdm/std.py`` 211-239): ``last =
  alpha*x + (1-alpha)*last`` with the early-call de-bias ``last /
  (1 - (1-alpha)^calls)``; the displayed rate is ``ema_dn / ema_dt``
  fed from the per-refresh deltas, and the final frame falls back to
  the raw overall ``n / elapsed``.
* **``format_interval``** ([H:]MM:SS), **``format_num``** (.3g with
  exponent-zero stripping, shortest-representation rule) and
  **``format_sizeof``** (SI k/M/G/T with the 999.5 / 99.95 / 9.995
  formatting tiers).

Examples
--------
>>> from pharmaml.utils.progress import Ema, format_interval, format_sizeof
>>> ema = Ema(smoothing=0.3)
>>> round(ema(10.0), 6)
10.0
>>> ema(20.0) > 13.0        # biased toward the first sample still
True
>>> format_interval(3723)
'1:02:03'
>>> format_interval(65)
'01:05'
>>> format_sizeof(15_000_000)
'15.0M'
"""

from __future__ import annotations

import math

__all__ = [
    "Ema",
    "format_interval",
    "format_num",
    "format_sizeof",
]


class Ema:
    """The bias-corrected exponential moving average of tqdm's std.py.

    ``alpha`` = the smoothing (0.3 default -- "0 = average speed, 1 =
    instantaneous"); the value read back before enough calls have
    accumulated is de-biased by ``1 - (1-alpha)^calls``.

    Examples
    --------
    >>> ema = Ema()
    >>> [round(ema(v), 2) for v in (10.0, 10.0, 10.0)]
    [10.0, 10.0, 10.0]
    """

    def __init__(self, smoothing: float = 0.3) -> None:
        if not 0 <= smoothing <= 1:
            raise ValueError("smoothing must lie in [0, 1]")
        self.alpha = smoothing
        self.last = 0.0
        self.calls = 0

    def __call__(self, value: float | None = None) -> float:
        beta = 1 - self.alpha
        if value is not None:
            self.last = self.alpha * value + beta * self.last
            self.calls += 1
        if self.calls == 0:
            return self.last
        return self.last / (1 - beta**self.calls)


def format_interval(seconds: float) -> str:
    """The [H:]MM:SS clock format of tqdm's format_interval.

    Examples
    --------
    >>> format_interval(59.6)
    '00:59'
    >>> format_interval(-65)
    '-01:05'
    """
    sign = "-" if seconds < 0 else ""
    total = int(abs(seconds))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{sign}{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{sign}{minutes:02d}:{secs:02d}"


def format_num(value: float) -> str:
    """tqdm's .3g number format with exponent-zero stripping.

    Examples
    --------
    >>> format_num(1234.5)
    '1234.5'
    >>> format_num(0.5)
    '0.5'
    """
    text = f"{value:.3g}".replace("e+0", "e+").replace("e-0", "e-")
    plain = str(value)
    return text if len(text) < len(plain) else plain


def format_sizeof(value: float, divisor: float = 1000.0) -> str:
    """The SI size format (k / M / G / T ...) with the tqdm tiers.

    999.5 -> ``{:3.0f}``, 99.95 -> ``{:2.1f}``, 9.995 -> ``{:1.2f}``.

    Examples
    --------
    >>> format_sizeof(999.4)
    '999'
    >>> format_sizeof(1234.0)
    '1.23k'
    >>> format_sizeof(15_000_000)
    '15.0M'
    """
    units = ("", "k", "M", "G", "T", "P", "E", "Z", "Y")
    if math.isnan(value) or math.isinf(value):
        return "?"
    if value is None or value == 0 or divisor == 1:
        return str(value)
    power = math.log(value, divisor)
    index = min(max(math.floor(power), 0), len(units) - 1)
    scaled = value / divisor**index
    if scaled >= 999.5:
        index = min(index + 1, len(units) - 1)
        scaled = value / divisor**index
    if scaled >= 99.95:
        return f"{scaled:3.0f}{units[index]}"
    if scaled >= 9.995:
        return f"{scaled:2.1f}{units[index]}"
    return f"{scaled:1.2f}{units[index]}"
