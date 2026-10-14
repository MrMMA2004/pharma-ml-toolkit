"""Visual pharmacokinetic output — profiles and dosing regimens.

The original archive produced *no* pharma figures at all (PK numbers
were printed to the console), so this module adds the two classic
visuals every PK report shows:

* :func:`plot_concentration_profile` — the concentration-time curve on
  linear and semi-logarithmic panels, with the trapezoidal AUC shaded
  and Cmax/Tmax annotated;
* :func:`plot_dose_accumulation` — the one-compartment superposition of
  repeated doses approaching steady state, with the Cmax/Cmin
  steady-state envelope and the accumulation ratio.

Both functions return matplotlib figures (Agg backend); save them as
png/svg/pdf via :func:`pharmaml.viz.save_figure`.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from pharmaml.exceptions import DataValidationError
from pharmaml.viz._compat import PALETTE, new_figure

__all__ = ["plot_concentration_profile", "plot_dose_accumulation"]


def plot_concentration_profile(
    times: np.ndarray | list[float],
    conc: np.ndarray | list[float],
    summary: Any | None = None,
    auc: float | None = None,
    title: str = "Concentration-time profile",
) -> Any:
    """Plot one PK profile: linear + semi-log panels with AUC shading.

    Parameters
    ----------
    times, conc:
        Paired sample series (validated like every other PK function).
    summary:
        Optional :class:`~pharmaml.pharma.pk.PKSummary` — when given,
        AUC/Cmax/Tmax annotations come from it and the shaded area uses
        its AUC.
    auc:
        Explicit AUC override (``summary`` takes precedence).

    Returns
    -------
    matplotlib.figure.Figure
    """
    from pharmaml.utils.validation import validate_concentration_series

    t, c = validate_concentration_series(times, conc, require_positive_conc=False)

    if summary is not None:
        auc_value = float(summary.auc)
        c_max = float(summary.c_max)
        t_max = float(summary.t_max)
        half_life = summary.half_life_terminal
    else:
        auc_value = float(auc) if auc is not None else float(np.trapezoid(c, t))
        c_max = float(np.max(c))
        t_max = float(t[int(np.argmax(c))])
        half_life = None

    fig, (ax_lin, ax_log) = new_figure(nrows=1, ncols=2, figsize=(11.4, 4.8))

    # --- linear panel with trapezoid shading -----------------------
    ax_lin.fill_between(t, c, step=None, alpha=0.18, color=PALETTE[0],
                        label=f"AUC = {auc_value:.4g}")
    ax_lin.plot(t, c, marker="o", markersize=5, lw=1.8, color=PALETTE[0],
                markeredgecolor="white", markeredgewidth=0.5)
    ax_lin.scatter([t_max], [c_max], s=70, color=PALETTE[1], zorder=5,
                   edgecolor="white", linewidth=0.8)
    ax_lin.annotate(f"Cmax {c_max:.4g} @ t={t_max:g}",
                    xy=(t_max, c_max), xytext=(8, -12),
                    textcoords="offset points", fontsize=9, color=PALETTE[1])
    ax_lin.set_xlabel("time (h)")
    ax_lin.set_ylabel("concentration")
    ax_lin.set_title("linear scale")
    ax_lin.legend(fontsize=9, loc="upper right")

    # --- semi-log panel (positive samples only) --------------------
    positive = c > 0
    if positive.sum() >= 2:
        ax_log.plot(t[positive], c[positive], marker="o", markersize=5, lw=1.8,
                    color=PALETTE[0], markeredgecolor="white", markeredgewidth=0.5)
        ax_log.set_yscale("log")
        if half_life is not None and math.isfinite(half_life):
            ax_log.set_title(f"semi-log (terminal t½ ≈ {half_life:.3g} h)")
        else:
            ax_log.set_title("semi-log scale")
    else:
        ax_log.text(0.5, 0.5, "no positive concentrations\nfor a log axis",
                    ha="center", va="center", transform=ax_log.transAxes, fontsize=10)
        ax_log.set_title("semi-log scale")
    ax_log.set_xlabel("time (h)")
    ax_log.set_ylabel("concentration (log)")

    fig.suptitle(title, y=1.02)
    return fig


def plot_dose_accumulation(
    dose: float,
    tau: float,
    t_half: float | None = None,
    ke: float | None = None,
    vd: float | None = None,
    f: float = 1.0,
    n_doses: int = 6,
    title: str = "Repeated dosing — approach to steady state",
) -> Any:
    """One-compartment superposition of an intermittent IV regimen.

    Each bolus adds ``F·dose/Vd · e^(−ke·(t − t_i))`` after its dose
    time ``t_i``; the summed curve is drawn together with the
    steady-state peak/trough envelope
    ``Cmax_ss = C0/(1 − e^(−ke·τ))`` and ``Cmin_ss = Cmax_ss·e^(−ke·τ)``,
    the accumulation ratio R, and the ~5·t½ steady-state marker.

    Parameters
    ----------
    dose:
        Administered amount per dose.
    tau:
        Dosing interval (same time unit as ``t_half``/``ke``).
    t_half:
        Terminal half-life (mutually exclusive with ``ke``).
    ke:
        Elimination rate constant (mutually exclusive with ``t_half``).
    vd:
        Volume of distribution — required to convert dose to
        concentration.
    f:
        Bioavailability fraction.
    n_doses:
        Number of doses to draw.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from pharmaml.pharma.pk import accumulation_ratio, elimination_rate_constant, half_life

    if (t_half is None) == (ke is None):
        raise DataValidationError("pass exactly one of t_half or ke")
    if ke is None:
        ke = elimination_rate_constant(t_half)
    if vd is None:
        raise DataValidationError("vd is required to convert dose to concentration")
    if n_doses < 1:
        raise DataValidationError(f"n_doses must be >= 1, got {n_doses}")

    ke = float(ke)
    c0 = f * dose / vd
    horizon = n_doses * tau + 3 * (math.log(2) / ke)
    grid = np.linspace(0.0, horizon, max(600, 40 * n_doses))

    total = np.zeros_like(grid)
    for index in range(n_doses):
        t_i = index * tau
        total += np.where(grid >= t_i, c0 * np.exp(-ke * (grid - t_i)), 0.0)

    c_max_ss = c0 / (1.0 - math.exp(-ke * tau))
    c_min_ss = c_max_ss * math.exp(-ke * tau)
    ratio = accumulation_ratio(tau, math.log(2) / ke)
    t_ss = 5.0 * half_life(ke)

    fig, ax = new_figure(figsize=(9.6, 5.4))
    for index in range(n_doses):
        ax.axvline(index * tau, color="#D1D5DB", lw=0.8, ls=":",
                   ymax=0.97)
    ax.plot(grid, total, lw=2.0, color=PALETTE[0], label="C(t) — superposition")
    ax.axhline(c_max_ss, ls="--", color=PALETTE[1], lw=1.6,
               label=f"Cmax,ss = {c_max_ss:.4g}")
    ax.axhline(c_min_ss, ls="--", color=PALETTE[3], lw=1.6,
               label=f"Cmin,ss = {c_min_ss:.4g}")
    if t_ss <= horizon:
        ax.axvline(t_ss, ls="--", color="#6B7280", lw=1.4,
                   label=f"≈ steady state (5·t½ = {t_ss:.3g} h)")
    ax.set_xlabel(f"time (h) — {n_doses} doses every {tau:g} h")
    ax.set_ylabel("concentration")
    ax.set_title(f"{title}\naccumulation ratio R = {ratio:.3g}")
    ax.legend(fontsize=9, loc="best")
    return fig
