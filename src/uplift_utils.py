"""Reusable causal metrics and policy utilities for the uplift case."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


EPS = 1e-12


@dataclass(frozen=True)
class DifferenceInMeans:
    estimate: float
    standard_error: float
    ci_low: float
    ci_high: float
    treated_mean: float
    control_mean: float
    n_treated: int
    n_control: int


def difference_in_means(
    outcome: Iterable[float], treatment: Iterable[int], z_value: float = 1.959963984540054
) -> DifferenceInMeans:
    """Estimate the randomized ITT and a normal-approximation confidence interval."""
    y = np.asarray(outcome, dtype=float)
    t = np.asarray(treatment, dtype=int)
    treated = y[t == 1]
    control = y[t == 0]
    if treated.size < 2 or control.size < 2:
        raise ValueError("Both treatment arms need at least two observations.")
    estimate = float(treated.mean() - control.mean())
    standard_error = float(
        np.sqrt(treated.var(ddof=1) / treated.size + control.var(ddof=1) / control.size)
    )
    return DifferenceInMeans(
        estimate=estimate,
        standard_error=standard_error,
        ci_low=estimate - z_value * standard_error,
        ci_high=estimate + z_value * standard_error,
        treated_mean=float(treated.mean()),
        control_mean=float(control.mean()),
        n_treated=int(treated.size),
        n_control=int(control.size),
    )


def uplift_at_fraction(
    outcome: Iterable[float], treatment: Iterable[int], score: Iterable[float], fraction: float
) -> float:
    """Difference in randomized arm means inside the highest-score fraction."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1].")
    y = np.asarray(outcome, dtype=float)
    t = np.asarray(treatment, dtype=int)
    s = np.asarray(score, dtype=float)
    n_top = max(2, int(np.ceil(len(y) * fraction)))
    selected = np.argsort(-s, kind="mergesort")[:n_top]
    return difference_in_means(y[selected], t[selected]).estimate


def qini_curve(
    outcome: Iterable[float],
    treatment: Iterable[int],
    score: Iterable[float],
    n_points: int = 101,
) -> pd.DataFrame:
    """Build a cumulative incremental-gain curve under randomized assignment.

    The raw gain at a prefix is treated outcomes minus reweighted control outcomes.
    Areas are later divided by population size, which keeps metrics comparable
    across temporal samples with different row counts.
    """
    y = np.asarray(outcome, dtype=float)
    t = np.asarray(treatment, dtype=int)
    s = np.asarray(score, dtype=float)
    if not (len(y) == len(t) == len(s)):
        raise ValueError("outcome, treatment and score must have equal lengths.")
    if len(y) < 4:
        raise ValueError("At least four observations are required.")

    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    t_sorted = t[order]
    cum_t = np.cumsum(t_sorted)
    cum_c = np.cumsum(1 - t_sorted)
    cum_y_t = np.cumsum(y_sorted * t_sorted)
    cum_y_c = np.cumsum(y_sorted * (1 - t_sorted))

    checkpoints = np.unique(
        np.r_[0, np.ceil(np.linspace(1, len(y), n_points - 1)).astype(int)]
    )
    rows = []
    for n_selected in checkpoints:
        if n_selected == 0:
            rows.append((0.0, 0.0, 0, 0, 0))
            continue
        idx = n_selected - 1
        n_t = int(cum_t[idx])
        n_c = int(cum_c[idx])
        if n_t == 0 or n_c == 0:
            gain = np.nan
        else:
            gain = float(cum_y_t[idx] - cum_y_c[idx] * n_t / n_c)
        rows.append((n_selected / len(y), gain, int(n_selected), n_t, n_c))

    curve = pd.DataFrame(
        rows,
        columns=["population_fraction", "incremental_gain", "n_selected", "n_treated", "n_control"],
    )
    curve["incremental_gain"] = curve["incremental_gain"].interpolate().fillna(0.0)
    final_gain = float(curve["incremental_gain"].iloc[-1])
    curve["random_gain"] = curve["population_fraction"] * final_gain
    curve["gain_per_customer"] = curve["incremental_gain"] / len(y)
    curve["random_gain_per_customer"] = curve["random_gain"] / len(y)
    return curve


def qini_metrics(curve: pd.DataFrame) -> dict[str, float]:
    """Return AUUC and Qini area using per-customer cumulative gain."""
    x = curve["population_fraction"].to_numpy(dtype=float)
    gain = curve["gain_per_customer"].to_numpy(dtype=float)
    random_gain = curve["random_gain_per_customer"].to_numpy(dtype=float)
    auuc = float(np.trapezoid(gain, x))
    qini = float(np.trapezoid(gain - random_gain, x))
    return {"auuc": auuc, "qini_coefficient": qini}


def exact_top_fraction(score: Iterable[float], fraction: float) -> np.ndarray:
    """Select exactly ceil(fraction * n) rows using a stable score ordering."""
    s = np.asarray(score, dtype=float)
    n_top = int(np.ceil(len(s) * fraction))
    selected = np.zeros(len(s), dtype=bool)
    selected[np.argsort(-s, kind="mergesort")[:n_top]] = True
    return selected


def evaluate_policy(
    frame: pd.DataFrame,
    selected: Iterable[bool],
    policy_name: str,
    treatment_col: str = "treatment_assigned",
    outcome_col: str = "conversion_30d",
    revenue_col: str = "revenue_30d",
    margin_col: str = "margin_30d",
    cost_col: str = "campaign_cost",
) -> dict[str, float | int | str]:
    """Estimate policy value from randomized treated/control contrasts.

    Scores used to build ``selected`` must be functions of pre-treatment data only.
    The economic projection treats every selected customer and therefore scales
    arm differences to the full selected audience.
    """
    mask = np.asarray(selected, dtype=bool)
    chosen = frame.loc[mask]
    if chosen.empty:
        raise ValueError("Policy selected no customers.")
    treated = chosen[chosen[treatment_col] == 1]
    control = chosen[chosen[treatment_col] == 0]
    if len(treated) < 2 or len(control) < 2:
        raise ValueError("Selected audience needs observations in both randomized arms.")

    n_selected = len(chosen)
    conv_t = float(treated[outcome_col].mean())
    conv_c = float(control[outcome_col].mean())
    uplift = conv_t - conv_c
    revenue_uplift = float(treated[revenue_col].mean() - control[revenue_col].mean())
    margin_uplift = float(treated[margin_col].mean() - control[margin_col].mean())
    expected_cost_per_contact = float(treated[cost_col].mean())
    incremental_conversions = uplift * n_selected
    incremental_revenue = revenue_uplift * n_selected
    incremental_margin = margin_uplift * n_selected
    campaign_cost = expected_cost_per_contact * n_selected
    net_incremental_value = incremental_margin - campaign_cost
    roi = net_incremental_value / max(campaign_cost, EPS)
    return {
        "policy": policy_name,
        "n_selected": int(n_selected),
        "audience_share": float(n_selected / len(frame)),
        "n_treated_observed": int(len(treated)),
        "n_control_observed": int(len(control)),
        "treated_conversion_rate": conv_t,
        "control_conversion_rate": conv_c,
        "uplift": uplift,
        "incremental_conversions": incremental_conversions,
        "incremental_revenue": incremental_revenue,
        "incremental_margin": incremental_margin,
        "expected_cost_per_contact": expected_cost_per_contact,
        "campaign_cost": campaign_cost,
        "net_incremental_value": net_incremental_value,
        "roi": roi,
        "policy_value_per_eligible_customer": net_incremental_value / len(frame),
    }


def standardized_mean_differences(
    frame: pd.DataFrame,
    treatment_col: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> pd.DataFrame:
    """Calculate numeric and level-wise categorical standardized differences."""
    rows: list[dict[str, float | str]] = []
    t = frame[treatment_col].astype(int)
    for col in numeric_cols:
        values = pd.to_numeric(frame[col], errors="coerce")
        treated = values[t == 1]
        control = values[t == 0]
        pooled = np.sqrt((treated.var(ddof=1) + control.var(ddof=1)) / 2)
        smd = 0.0 if not np.isfinite(pooled) or pooled < EPS else (treated.mean() - control.mean()) / pooled
        rows.append({"feature": col, "level": "numeric", "smd": float(smd)})
        missing = values.isna().astype(float)
        p_t = missing[t == 1].mean()
        p_c = missing[t == 0].mean()
        p_pool = (p_t + p_c) / 2
        denom = np.sqrt(max(p_pool * (1 - p_pool), EPS))
        rows.append({"feature": col, "level": "missing", "smd": float((p_t - p_c) / denom)})

    for col in categorical_cols:
        values = frame[col].fillna("__MISSING__").astype(str)
        for level in sorted(values.unique()):
            binary = (values == level).astype(float)
            p_t = binary[t == 1].mean()
            p_c = binary[t == 0].mean()
            p_pool = (p_t + p_c) / 2
            denom = np.sqrt(max(p_pool * (1 - p_pool), EPS))
            rows.append({"feature": col, "level": level, "smd": float((p_t - p_c) / denom)})
    result = pd.DataFrame(rows)
    result["abs_smd"] = result["smd"].abs()
    return result.sort_values("abs_smd", ascending=False, ignore_index=True)


def numeric_psi(reference: pd.Series, comparison: pd.Series, bins: int = 10) -> float:
    """Population Stability Index using reference quantile bins."""
    ref = pd.to_numeric(reference, errors="coerce")
    cmp = pd.to_numeric(comparison, errors="coerce")
    valid_ref = ref.dropna().to_numpy(dtype=float)
    valid_cmp = cmp.dropna().to_numpy(dtype=float)
    if valid_ref.size == 0 or valid_cmp.size == 0:
        return float("nan")
    edges = np.unique(np.quantile(valid_ref, np.linspace(0, 1, bins + 1)))
    if edges.size < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_hist = np.histogram(valid_ref, bins=edges)[0].astype(float)
    cmp_hist = np.histogram(valid_cmp, bins=edges)[0].astype(float)
    ref_pct = np.clip(ref_hist / max(ref_hist.sum(), 1), 1e-6, None)
    cmp_pct = np.clip(cmp_hist / max(cmp_hist.sum(), 1), 1e-6, None)
    return float(np.sum((cmp_pct - ref_pct) * np.log(cmp_pct / ref_pct)))


def uplift_by_quantile(
    frame: pd.DataFrame,
    score_col: str,
    treatment_col: str = "treatment_assigned",
    outcome_col: str = "conversion_30d",
    n_quantiles: int = 10,
) -> pd.DataFrame:
    """Observed randomized uplift by descending score quantile."""
    ranked = frame.copy()
    ranked["rank"] = ranked[score_col].rank(method="first", ascending=False)
    ranked["quantile"] = pd.qcut(ranked["rank"], q=n_quantiles, labels=False) + 1
    rows = []
    for quantile, group in ranked.groupby("quantile", observed=True):
        estimate = difference_in_means(group[outcome_col], group[treatment_col])
        rows.append(
            {
                "quantile": int(quantile),
                "n": int(len(group)),
                "treated_rate": estimate.treated_mean,
                "control_rate": estimate.control_mean,
                "observed_uplift": estimate.estimate,
                "ci_low": estimate.ci_low,
                "ci_high": estimate.ci_high,
                "mean_estimated_cate": float(group[score_col].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("quantile", ignore_index=True)
