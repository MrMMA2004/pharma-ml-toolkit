"""seaborn's palette generators, absorbed natively (100x-170).

``mwaskom/seaborn`` was reviewed file by file; everything in
``seaborn/palettes.py`` that is *math* rather than matplotlib plumbing
lands here as pure Python/NumPy:

* the vendored HUSL colorspace (``seaborn/external/husl.py``, itself
  HUSL v2.1.0 by Alexei Boronine, MIT) -- the full conversion chain
  HUSL -> LCh -> Luv -> XYZ -> sRGB with the ``max_chroma`` boundary
  solver, reimplemented here as :func:`husl_to_rgb` /
  :func:`rgb_to_husl` (verified numerically against the vendored
  original; see ``scratch/probe_vizlab_170.py``);
* :func:`husl_palette` / :func:`hls_palette` -- the hue-ladder
  generators (``linspace(0, 1, n + 1)[:-1] + h`` mod 1);
* :func:`cubehelix_palette` -- Green (2011)'s helix, the exact channel
  functions seaborn copied from matplotlib's private module
  (``xg = x**gamma``, ``a = hue·xg·(1-xg)/2``,
  ``phi = 2π(start/3 + rot·x)`` with the
  ``(-0.14861, 1.78277) / (-0.29227, -0.90649) / (1.97294, 0)``
  deviation vectors);
* :func:`blend_palette` / :func:`dark_palette` / :func:`light_palette`
  -- seaborn's interpolation semantics with
  ``LinearSegmentedColormap.from_list`` replaced by direct per-channel
  linear interpolation over equally spaced anchors (identical values);
  the dark/light endpoints are HUSL grays at L=15/95 carrying 15 % of
  the anchor's saturation, exactly seaborn's choice;
* :func:`diverging_palette` -- the two HUSL half-ramps joined through
  ``sep`` near-gray midpoints (``n_half = 128 - sep // 2``, the same
  construction as upstream);
* :func:`color_palette` -- the dispatcher: seaborn's named cycles
  (deep/muted/pastel/bright/dark/colorblind and their ``6`` subsets),
  ``hls``/``husl``, the ``ch:``/``light:``/``dark:``/``blend:``
  mini-grammar (``_r`` reverses), and the embedded ColorBrewer /
  matplotlib tables with seaborn's two sampling rules -- qualitative
  families take exact indexed colors, continuous maps take the
  interior positions ``linspace(0, 1, n + 2)[1:-1]`` "to provide
  better contrast at the extremes".  ``jet`` is refused with
  seaborn's exact ``ValueError("No.")``.
* :func:`desaturate` and :func:`saturate` (``seaborn/utils.py``'s
  colorsys round-trip) -- what seaborn's categorical ``saturation``
  argument applies.

Examples
--------
>>> from pharmaml.viz.palettes import (
...     blend_palette, color_palette, cubehelix_palette, dark_palette,
...     diverging_palette, husl_palette, light_palette,
... )
>>> husl_palette(3, h=0.0)
('#f77090', '#5faf31', '#3ba4e4')
>>> cubehelix_palette(3, start=0.0, rot=0.0)
('#d7d5f2', '#7c78b2', '#242240')
>>> blend_palette(("#000000", "#ffffff"), 3)
('#000000', '#808080', '#ffffff')
>>> dark_palette("#ff0000", 3)
('#312222', '#981111', '#ff0000')
>>> light_palette("#ff0000", 3)
('#f3f0f0', '#f97878', '#ff0000')
>>> pal = diverging_palette(10, 220, n=5)
>>> len(pal)
5
>>> color_palette("deep6")[0] == color_palette("deep")[0]
True
>>> color_palette("Set1", 3)
('#e41a1c', '#377eb8', '#4daf4a')
>>> color_palette("viridis", 4)[0]
'#414487'
>>> try:
...     color_palette("jet")
... except ValueError as exc:
...     str(exc)
'No.'
"""

from __future__ import annotations

import colorsys
import math
from typing import Any

import numpy as np

from pharmaml.viz._palette_data import (
    BREWER,
    PERCEPTUAL_CMAPS,
    SEABORN_PALETTES,
    TAB_CMAPS,
)
from pharmaml.viz.colors import rgb_to_hex, to_rgb

__all__ = [
    "blend_palette",
    "color_palette",
    "cubehelix_palette",
    "dark_palette",
    "desaturate",
    "diverging_palette",
    "hls_palette",
    "husl_palette",
    "husl_to_rgb",
    "light_palette",
    "rgb_to_husl",
    "saturate",
]

# ---------------------------------------------------------------------------
# HUSL (seaborn/external/husl.py -- HUSL v2.1.0, Alexei Boronine, MIT)
# ---------------------------------------------------------------------------

_M_XYZ_TO_RGB = (
    (3.2406, -1.5372, -0.4986),
    (-0.9689, 1.8758, 0.0415),
    (0.0557, -0.2040, 1.0570),
)
_M_RGB_TO_XYZ = (
    (0.4124, 0.3576, 0.1805),
    (0.2126, 0.7152, 0.0722),
    (0.0193, 0.1192, 0.9505),
)
_REF_U = 0.19784
_REF_V = 0.46834
_LAB_E = 0.008856
_LAB_K = 903.3


def _from_linear(c: float) -> float:
    if c <= 0.0031308:
        return 12.92 * c
    return 1.055 * (c ** (1.0 / 2.4)) - 0.055


def _to_linear(c: float) -> float:
    if c > 0.04045:
        return ((c + 0.055) / 1.055) ** 2.4
    return c / 12.92


def _dot(row: tuple[float, float, float], v: tuple[float, float, float]) -> float:
    return row[0] * v[0] + row[1] * v[1] + row[2] * v[2]


def _xyz_to_rgb(xyz: tuple[float, float, float]) -> tuple[float, float, float]:
    raw = (
        _dot(_M_XYZ_TO_RGB[0], xyz),
        _dot(_M_XYZ_TO_RGB[1], xyz),
        _dot(_M_XYZ_TO_RGB[2], xyz),
    )
    return (_from_linear(raw[0]), _from_linear(raw[1]), _from_linear(raw[2]))


def _rgb_to_xyz(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    lin = (_to_linear(rgb[0]), _to_linear(rgb[1]), _to_linear(rgb[2]))
    return (
        _dot(_M_RGB_TO_XYZ[0], lin),
        _dot(_M_RGB_TO_XYZ[1], lin),
        _dot(_M_RGB_TO_XYZ[2], lin),
    )


def _f(t: float) -> float:
    if t > _LAB_E:
        return t ** (1.0 / 3.0)
    return 7.787 * t + 16.0 / 116.0


def _f_inv(t: float) -> float:
    if t**3 > _LAB_E:
        return t**3
    return (116.0 * t - 16.0) / _LAB_K


def _xyz_to_luv(xyz: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = xyz
    if x == y == z == 0.0:
        return (0.0, 0.0, 0.0)
    denom = x + 15.0 * y + 3.0 * z
    var_u = 4.0 * x / denom
    var_v = 9.0 * y / denom
    l = 116.0 * _f(y) - 16.0
    if l == 0.0:
        return (0.0, 0.0, 0.0)
    return (l, 13.0 * l * (var_u - _REF_U), 13.0 * l * (var_v - _REF_V))


def _luv_to_xyz(luv: tuple[float, float, float]) -> tuple[float, float, float]:
    l, u, v = luv
    if l == 0.0:
        return (0.0, 0.0, 0.0)
    var_y = _f_inv((l + 16.0) / 116.0)
    var_u = u / (13.0 * l) + _REF_U
    var_v = v / (13.0 * l) + _REF_V
    y = var_y
    x = 0.0 - (9.0 * y * var_u) / ((var_u - 4.0) * var_v - var_u * var_v)
    z = (9.0 * y - 15.0 * var_v * y - var_v * x) / (3.0 * var_v)
    return (x, y, z)


def _luv_to_lch(luv: tuple[float, float, float]) -> tuple[float, float, float]:
    l, u, v = luv
    c = math.hypot(u, v)
    if c < 1e-8:
        h = 0.0
    else:
        h = math.degrees(math.atan2(v, u)) % 360.0
    return (l, c, h)


def _lch_to_luv(lch: tuple[float, float, float]) -> tuple[float, float, float]:
    l, c, h = lch
    hrad = math.radians(h)
    return (l, math.cos(hrad) * c, math.sin(hrad) * c)


def _max_chroma(l: float, h: float) -> float:
    """The maximum chroma for lightness/hue inside the sRGB gamut."""
    hrad = math.radians(h)
    sin_h, cos_h = math.sin(hrad), math.cos(hrad)
    sub1 = ((l + 16.0) ** 3) / 1560896.0
    sub2 = sub1 if sub1 > _LAB_E else l / _LAB_K
    result = float("inf")
    for row in _M_XYZ_TO_RGB:
        top = (0.99915 * row[0] + 1.05122 * row[1] + 1.14460 * row[2]) * sub2
        rbottom = 0.86330 * row[2] - 0.17266 * row[1]
        lbottom = 0.12949 * row[2] - 0.38848 * row[0]
        bottom = (rbottom * sin_h + lbottom * cos_h) * sub2
        for t in (0.0, 1.0):
            c = l * (top - 1.05122 * t) / (bottom + 0.17266 * sin_h * t)
            if 0.0 < c < result:
                result = c
    return result


def _husl_to_lch(husl: tuple[float, float, float]) -> tuple[float, float, float]:
    h, s, l = husl
    if l > 99.9999999:
        return (100.0, 0.0, h)
    if l < 1e-8:
        return (0.0, 0.0, h)
    c = _max_chroma(l, h) / 100.0 * s
    return (l, c, h)


def _lch_to_husl(lch: tuple[float, float, float]) -> tuple[float, float, float]:
    l, c, h = lch
    if l > 99.9999999:
        return (h, 0.0, 100.0)
    if l < 1e-8:
        return (h, 0.0, 0.0)
    mx = _max_chroma(l, h)
    s = c / mx * 100.0 if mx > 0 else 0.0
    return (h, s, l)


def husl_to_rgb(h: float, s: float, l: float) -> tuple[float, float, float]:
    """HUSL (h in 0-360, s/l in 0-100) -> sRGB 0-1 triple."""
    lch = _husl_to_lch((h, s, l))
    luv = _lch_to_luv(lch)
    xyz = _luv_to_xyz(luv)
    rgb = _xyz_to_rgb(xyz)
    return (
        min(max(rgb[0], 0.0), 1.0),
        min(max(rgb[1], 0.0), 1.0),
        min(max(rgb[2], 0.0), 1.0),
    )


def rgb_to_husl(rgb: tuple[float, float, float]) -> tuple[float, float, float]:
    """sRGB 0-1 triple -> HUSL (h 0-360, s/l 0-100)."""
    xyz = _rgb_to_xyz(rgb)
    luv = _xyz_to_luv(xyz)
    lch = _luv_to_lch(luv)
    return _lch_to_husl(lch)


def husl_palette(
    n_colors: int = 6, h: float = 0.01, s: float = 0.9, l: float = 0.65
) -> tuple[str, ...]:
    """seaborn's ``husl_palette``: evenly spaced HUSL hues as hex."""
    hues = np.linspace(0.0, 1.0, int(n_colors) + 1)[:-1]
    hues += h
    hues %= 1.0
    return tuple(
        rgb_to_hex(husl_to_rgb(float(hue) * 359.0, s * 99.0, l * 99.0)) for hue in hues
    )


def hls_palette(
    n_colors: int = 6, h: float = 0.01, l: float = 0.6, s: float = 0.65
) -> tuple[str, ...]:
    """seaborn's ``hls_palette``: evenly spaced HLS hues as hex."""
    hues = np.linspace(0.0, 1.0, int(n_colors) + 1)[:-1]
    hues += h
    hues %= 1.0
    return tuple(
        rgb_to_hex(colorsys.hls_to_rgb(float(hue), l, s)) for hue in hues
    )


# ---------------------------------------------------------------------------
# cubehelix (Green 2011 -- seaborn's copy of matplotlib's channel functions)
# ---------------------------------------------------------------------------

_CUBEHelix_VECTORS = {
    "red": (-0.14861, 1.78277),
    "green": (-0.29227, -0.90649),
    "blue": (1.97294, 0.0),
}


def cubehelix_palette(
    n_colors: int = 6,
    start: float = 0.0,
    rot: float = 0.4,
    gamma: float = 1.0,
    hue: float = 0.8,
    light: float = 0.85,
    dark: float = 0.15,
    reverse: bool = False,
) -> tuple[str, ...]:
    """Green's cubehelix ramp sampled at ``n_colors`` hex points.

    The channel functions are matplotlib's (copied into seaborn's
    ``palettes.py`` verbatim); ``x`` runs from ``light`` to ``dark``
    exactly as upstream samples it, and ``reverse`` flips the result.
    """
    x = np.linspace(light, dark, int(n_colors))

    def channel(p0: float, p1: float) -> np.ndarray:
        xg = x**gamma
        a = hue * xg * (1.0 - xg) / 2.0
        phi = 2.0 * np.pi * (start / 3.0 + rot * x)
        return xg + a * (p0 * np.cos(phi) + p1 * np.sin(phi))

    rgb = np.column_stack(
        [channel(*_CUBEHelix_VECTORS["red"]), channel(*_CUBEHelix_VECTORS["green"]),
         channel(*_CUBEHelix_VECTORS["blue"])]
    )
    hexes = [rgb_to_hex(tuple(np.clip(row, 0.0, 1.0))) for row in rgb]
    if reverse:
        hexes = hexes[::-1]
    return tuple(hexes)


# ---------------------------------------------------------------------------
# blend / dark / light / diverging (seaborn's interpolation semantics)
# ---------------------------------------------------------------------------


def blend_palette(colors: Any, n_colors: int = 6) -> tuple[str, ...]:
    """Interpolate through anchor colors (``LinearSegmentedColormap.from_list``).

    Anchors sit at equal positions; the result is per-channel linear
    interpolation sampled at ``linspace(0, 1, n)`` -- identical values
    to seaborn's ``blend_palette`` for equally spaced anchors.
    """
    anchors = [to_rgb(c) for c in colors]
    if len(anchors) < 2:
        raise ValueError("blend_palette needs at least two anchor colors")
    pos = np.linspace(0.0, 1.0, int(n_colors))
    span = np.linspace(0.0, 1.0, len(anchors))
    rgb = np.asarray(anchors, dtype=float)
    blended = np.column_stack([np.interp(pos, span, rgb[:, ch]) for ch in range(3)])
    return tuple(rgb_to_hex(tuple(row)) for row in blended)


def dark_palette(color: object, n_colors: int = 6, reverse: bool = False) -> tuple[str, ...]:
    """seaborn's ``dark_palette``: anchor -> HUSL gray at L=15 ramp."""
    rgb = to_rgb(color)
    hue, sat, _light = rgb_to_husl(rgb)
    gray = husl_to_rgb(hue, 0.15 * sat, 15.0)
    anchors = [rgb, gray] if reverse else [gray, rgb]
    return blend_palette(anchors, n_colors)


def light_palette(color: object, n_colors: int = 6, reverse: bool = False) -> tuple[str, ...]:
    """seaborn's ``light_palette``: anchor -> HUSL gray at L=95 ramp."""
    rgb = to_rgb(color)
    hue, sat, _light = rgb_to_husl(rgb)
    gray = husl_to_rgb(hue, 0.15 * sat, 95.0)
    anchors = [rgb, gray] if reverse else [gray, rgb]
    return blend_palette(anchors, n_colors)


def diverging_palette(
    h_neg: float,
    h_pos: float,
    s: float = 75,
    l: float = 50,
    sep: int = 1,
    n: int = 6,
    center: str = "light",
) -> tuple[str, ...]:
    """seaborn's ``diverging_palette``: two HUSL half-ramps + gray middle.

    ``h_neg``/``h_pos`` are HUSL hues (0-360); each half ramps toward a
    near-white (``center="light"``) or near-dark (``"dark"``) HUSL gray
    at ``n_half = 128 - sep // 2`` resolution, ``sep`` midpoint rows are
    inserted, and the concatenation is blended down to ``n`` colors.
    """
    if center not in ("light", "dark"):
        raise ValueError("center must be 'light' or 'dark'")
    palfunc = light_palette if center == "light" else dark_palette
    n_half = int(128 - sep // 2)
    neg = palfunc((h_neg, s, l), n_half, reverse=True)
    pos = palfunc((h_pos, s, l), n_half)
    mid_rgb = (0.95, 0.95, 0.95) if center == "light" else (0.133, 0.133, 0.133)
    mid = [rgb_to_hex(mid_rgb)] * int(sep)
    return blend_palette([*neg, *mid, *pos], n)


# ---------------------------------------------------------------------------
# desaturation (seaborn/utils.py)
# ---------------------------------------------------------------------------


def desaturate(color: object, prop: float) -> str:
    """Decrease a color's HLS saturation channel by ``prop`` (seaborn)."""
    if not 0.0 <= prop <= 1.0:
        raise ValueError("prop must be in [0, 1]")
    rgb = to_rgb(color)
    h, li, s = colorsys.rgb_to_hls(*rgb)
    return rgb_to_hex(colorsys.hls_to_rgb(h, li, s * prop))


def saturate(color: object) -> str:
    """Return a fully saturated variant (seaborn's ``saturate``)."""
    rgb = to_rgb(color)
    h, li, _s = colorsys.rgb_to_hls(*rgb)
    return rgb_to_hex(colorsys.hls_to_rgb(h, li, 1.0))


# ---------------------------------------------------------------------------
# the dispatcher (seaborn's color_palette)
# ---------------------------------------------------------------------------

#: families whose colors are exact-indexed (seaborn's MPL_QUAL_PALS).
_QUALITATIVE_FAMILIES = {
    "tab10": 10,
    "tab20": 20,
    "Set1": 9,
    "Set2": 8,
    "Set3": 12,
    "Paired": 12,
    "Accent": 8,
    "Dark2": 8,
    "Pastel1": 9,
    "Pastel2": 8,
}


def _interior_sample(hex_table: tuple[str, ...], n_colors: int) -> tuple[str, ...]:
    """seaborn's continuous-map sampling: ``linspace(0, 1, n + 2)[1:-1]``."""
    if n_colors <= 0:
        raise ValueError("n_colors must be >= 1")
    positions = np.linspace(0.0, 1.0, n_colors + 2)[1:-1]
    idx = np.floor(positions * (len(hex_table) - 1) + 0.5).astype(int)
    return tuple(hex_table[int(i)] for i in idx)


def _table_palette(name: str, n_colors: int) -> tuple[str, ...]:
    if name in SEABORN_PALETTES:
        return tuple(SEABORN_PALETTES[name])
    if name in _QUALITATIVE_FAMILIES:
        source = TAB_CMAPS if name in TAB_CMAPS else BREWER
        full = source[name]
        bins = np.linspace(0.0, 1.0, _QUALITATIVE_FAMILIES[name])[:n_colors]
        idx = np.floor(bins * (len(full) - 1) + 0.5).astype(int)
        return tuple(full[int(i)] for i in idx)
    for source in (PERCEPTUAL_CMAPS, TAB_CMAPS, BREWER):
        if name in source:
            return _interior_sample(source[name], n_colors)
    raise ValueError(
        f"unknown palette {name!r}; embedded families: seaborn cycles "
        f"({', '.join(sorted(SEABORN_PALETTES))}), perceptual "
        f"({', '.join(PERCEPTUAL_CMAPS)}), brewer ({', '.join(sorted(BREWER))}), "
        f"tab10/tab20/coolwarm -- plus 'hls', 'husl', 'ch:', 'light:', "
        "'dark:' and 'blend:' strings"
    )


def _parse_cubehelix_args(text: str) -> tuple[float, float, float, float, float, float, float]:
    """Parse seaborn's ``ch:`` mini-grammar (positional floats + keywords).

    Returns ``(start, rot, gamma, hue, light, dark, reverse)``.
    """
    body = text[3:]
    reverse = body.endswith("_r")
    if reverse:
        body = body[:-2]
    start, rot, gamma, hue, light, dark = 0.0, 0.4, 1.0, 0.8, 0.85, 0.15
    tokens = [t for t in body.split() if t]
    positionals: list[float] = []
    for token in tokens:
        if token.startswith(("s=", "r=", "g=", "h=", "l=", "d=")):
            key, _, raw = token.partition("=")
            parsed = float(raw)
            if key == "s=":
                start = parsed
            elif key == "r=":
                rot = parsed
            elif key == "g=":
                gamma = parsed
            elif key == "h=":
                hue = parsed
            elif key == "l=":
                light = parsed
            else:
                dark = parsed
        else:
            positionals.append(float(token))
    for slot, value in zip(("start", "rot", "gamma", "hue"), positionals):
        if slot == "start":
            start = value
        elif slot == "rot":
            rot = value
        elif slot == "gamma":
            gamma = value
        else:
            hue = value
    return start, rot, gamma, hue, light, dark, 1.0 if reverse else 0.0  # type: ignore[return-value]


def color_palette(
    palette: object = None, n_colors: int | None = None, desat: float | None = None
) -> tuple[str, ...]:
    """seaborn's ``color_palette`` dispatcher over the embedded tables.

    Accepts a palette name (seaborn cycle, ``hls``/``husl``, ``ch:``/
    ``light:``/``dark:``/``blend:`` strings with optional ``_r``
    reversal, brewer/perceptual/tab families) or a plain color
    sequence.  Qualitative families return exact indexed colors;
    continuous families use seaborn's interior sampling.  ``jet`` is
    refused with seaborn's exact error.
    """
    if palette is None:
        palette = "deep"
    if isinstance(palette, (tuple, list)):
        colors = [to_rgb(c) for c in palette]
        if n_colors is None:
            n_colors = len(colors)
        hexes = [rgb_to_hex(c) for c in colors]
        cycled = [hexes[i % len(hexes)] for i in range(int(n_colors))]
        out = tuple(cycled)
    else:
        name = str(palette)
        reversed_ = name.endswith("_r")
        stem = name[:-2] if reversed_ else name
        if stem == "jet":
            raise ValueError("No.")
        if stem in SEABORN_PALETTES and n_colors is None:
            out = tuple(SEABORN_PALETTES[stem])
        elif stem == "hls":
            out = hls_palette(n_colors or 6)
        elif stem == "husl":
            out = husl_palette(n_colors or 6)
        elif stem.startswith("ch:"):
            start, rot, gamma, hue, light, dark, rev = _parse_cubehelix_args(name)
            out = cubehelix_palette(
                n_colors or 6, start=start, rot=rot, gamma=gamma, hue=hue,
                light=light, dark=dark, reverse=bool(rev or reversed_),
            )
        elif stem.startswith(("light:", "dark:", "blend:")):
            kind, _, args = stem.partition(":")
            if kind == "blend":
                anchors = [a.strip() for a in args.split(",")]
                out = blend_palette(anchors, n_colors or 6)
            else:
                func = light_palette if kind == "light" else dark_palette
                out = func(args.strip(), n_colors or 6, reverse=reversed_)
        else:
            table = _table_palette(stem, n_colors or 6)
            out = table
        if reversed_ and not stem.startswith(("ch:", "light:", "dark:")):
            if stem in SEABORN_PALETTES and n_colors is None:
                out = tuple(reversed(out))
            else:
                out = tuple(reversed(out))
    if desat is not None:
        out = tuple(desaturate(c, desat) for c in out)
    return out
