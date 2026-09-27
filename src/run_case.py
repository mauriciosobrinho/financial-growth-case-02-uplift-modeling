"""Execute the end-to-end uplift modeling case.

The public pipeline never reads the private DGP audit file. The OOT wave is
scored only after the champion and policy are frozen on Train/Validation.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.uplift_utils import (
    difference_in_means,
    evaluate_policy,
    exact_top_fraction,
    numeric_psi,
    qini_curve,
    qini_metrics,
    standardized_mean_differences,
    uplift_at_fraction,
    uplift_by_quantile,
)


SEED = 20260927
ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "case_02_uplift_modeling_raw.csv"
DICTIONARY_PATH = ROOT / "data" / "raw" / "case_02_data_dictionary.csv"
FIGURES = ROOT / "outputs" / "figures"
TABLES = ROOT / "outputs" / "tables"
METRICS = ROOT / "outputs" / "metrics"
MODELS = ROOT / "outputs" / "models"

TRAIN_WAVES = ["2026-01", "2026-02", "2026-03", "2026-04"]
VALIDATION_WAVES = ["2026-05"]
OOT_WAVES = ["2026-06"]

POST_TREATMENT_COLUMNS = [
    "message_delivered",
    "channel",
    "campaign_cost",
    "post_campaign_product_page_views_7d",
]
OUTCOME_COLUMNS = ["conversion_30d", "revenue_30d", "margin_30d"]
IDENTIFIER_COLUMNS = ["customer_id", "reference_date", "campaign_wave"]


def ensure_directories() -> None:
    for directory in [FIGURES, TABLES, METRICS, MODELS]:
        directory.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def model_parameters(seed: int) -> dict:
    return {
        "n_estimators": 260,
        "learning_rate": 0.045,
        "num_leaves": 15,
        "max_depth": 5,
        "min_child_samples": 250,
        "subsample": 0.85,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.20,
        "reg_lambda": 1.00,
        "random_state": seed,
        "n_jobs": -1,
        "verbosity": -1,
        "importance_type": "gain",
    }


def make_classifier(seed: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(objective="binary", **model_parameters(seed))


def make_regressor(seed: int) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(objective="regression_l2", **model_parameters(seed))


def load_public_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(RAW_PATH)
    dictionary = pd.read_csv(DICTIONARY_PATH)
    frame["reference_date"] = pd.to_datetime(frame["reference_date"])
    return frame, dictionary


def build_covariates(dictionary: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    timing = dictionary[["column", "measurement_timing", "description"]].copy()
    valid = timing["measurement_timing"].eq("pre_treatment")
    covariates = timing.loc[valid, "column"].tolist()
    covariates = [column for column in covariates if column not in IDENTIFIER_COLUMNS]
    review = timing.copy()
    review["model_role"] = np.select(
        [
            review["column"].isin(covariates),
            review["column"].eq("treatment_assigned"),
            review["column"].isin(POST_TREATMENT_COLUMNS),
            review["column"].isin(OUTCOME_COLUMNS),
            review["column"].isin(IDENTIFIER_COLUMNS),
        ],
        ["pre-treatment covariate", "treatment", "excluded post-treatment", "outcome", "identifier/time"],
        default="excluded",
    )
    return covariates, review


def engineer_features(frame: pd.DataFrame, base_covariates: list[str]) -> pd.DataFrame:
    result = frame[base_covariates].copy()
    sentinel_columns = ["days_since_last_card_txn", "days_since_last_app_session"]
    for column in sentinel_columns:
        if column in result:
            result[f"{column}_sentinel_flag"] = result[column].isin([999, 9999]).astype(int)
            result[column] = result[column].replace({999: np.nan, 9999: np.nan})

    result["txn_count_ratio_30_90"] = result["txn_count_30d"] / (result["txn_count_90d"] + 1.0)
    result["txn_value_ratio_30_90"] = result["txn_value_30d"] / (result["txn_value_90d"] + 1.0)
    result["app_session_ratio_30_90"] = result["app_sessions_30d"] / (result["app_sessions_90d"] + 1.0)
    result["prior_campaign_click_rate"] = result["campaign_clicks_90d"] / (
        result["campaigns_received_90d"] + 1.0
    )
    result["pix_count_30d"] = result["pix_in_count_30d"] + result["pix_out_count_30d"]
    result["pix_value_30d"] = result["pix_in_value_30d"] + result["pix_out_value_30d"]
    result["balance_range_30d"] = result["max_balance_30d"] - result["min_balance_30d"]

    log_columns = [
        "txn_value_30d",
        "txn_value_90d",
        "pix_in_value_30d",
        "pix_out_value_30d",
        "avg_balance_30d",
        "max_balance_30d",
        "salary_inflow_avg_90d",
        "card_spend_30d",
        "credit_limit",
    ]
    for column in log_columns:
        result[f"log1p_{column}"] = np.log1p(result[column].clip(lower=0))
    return result


def make_preprocessor(features: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    categorical = features.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numeric = [column for column in features.columns if column not in categorical]
    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", min_frequency=100, sparse_output=True),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        [("num", numeric_pipeline, numeric), ("cat", categorical_pipeline, categorical)],
        sparse_threshold=1.0,
    )
    return preprocessor, numeric, categorical


def fit_model_suite(
    x_train: sparse.spmatrix,
    y_train: np.ndarray,
    treatment_train: np.ndarray,
) -> dict:
    models: dict[str, object] = {}

    response = make_classifier(SEED + 1)
    response.fit(x_train[treatment_train == 1], y_train[treatment_train == 1])
    models["propensity_response"] = response

    s_matrix = sparse.hstack([x_train, treatment_train.reshape(-1, 1)], format="csr")
    s_model = make_classifier(SEED + 2)
    s_model.fit(s_matrix, y_train)
    models["S-Learner"] = {"model": s_model}

    mu0 = make_classifier(SEED + 3)
    mu1 = make_classifier(SEED + 4)
    mu0.fit(x_train[treatment_train == 0], y_train[treatment_train == 0])
    mu1.fit(x_train[treatment_train == 1], y_train[treatment_train == 1])
    models["T-Learner"] = {"mu0": mu0, "mu1": mu1}

    strata = treatment_train.astype(str) + "_" + y_train.astype(str)
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED + 5)
    mu0_oof = np.zeros(len(y_train), dtype=float)
    mu1_oof = np.zeros(len(y_train), dtype=float)
    for fold, (fit_idx, holdout_idx) in enumerate(folds.split(np.zeros(len(strata)), strata), start=1):
        fold_t = treatment_train[fit_idx]
        fold_y = y_train[fit_idx]
        fold_mu0 = make_classifier(SEED + 10 + fold)
        fold_mu1 = make_classifier(SEED + 20 + fold)
        fold_mu0.fit(x_train[fit_idx][fold_t == 0], fold_y[fold_t == 0])
        fold_mu1.fit(x_train[fit_idx][fold_t == 1], fold_y[fold_t == 1])
        mu0_oof[holdout_idx] = fold_mu0.predict_proba(x_train[holdout_idx])[:, 1]
        mu1_oof[holdout_idx] = fold_mu1.predict_proba(x_train[holdout_idx])[:, 1]

    control = treatment_train == 0
    treated = treatment_train == 1
    d0 = np.clip(mu1_oof[control] - y_train[control], -1.0, 1.0)
    d1 = np.clip(y_train[treated] - mu0_oof[treated], -1.0, 1.0)
    tau0 = make_regressor(SEED + 30)
    tau1 = make_regressor(SEED + 31)
    tau0.fit(x_train[control], d0)
    tau1.fit(x_train[treated], d1)
    models["X-Learner"] = {
        "tau0": tau0,
        "tau1": tau1,
        "treatment_probability": float(treatment_train.mean()),
        "cross_fitting_folds": 3,
    }
    return models


def predict_uplift(model_name: str, model: dict, matrix: sparse.spmatrix) -> np.ndarray:
    if model_name == "S-Learner":
        ones = np.ones((matrix.shape[0], 1), dtype=float)
        zeros = np.zeros((matrix.shape[0], 1), dtype=float)
        p1 = model["model"].predict_proba(sparse.hstack([matrix, ones], format="csr"))[:, 1]
        p0 = model["model"].predict_proba(sparse.hstack([matrix, zeros], format="csr"))[:, 1]
        return np.clip(p1 - p0, -1.0, 1.0)
    if model_name == "T-Learner":
        p1 = model["mu1"].predict_proba(matrix)[:, 1]
        p0 = model["mu0"].predict_proba(matrix)[:, 1]
        return np.clip(p1 - p0, -1.0, 1.0)
    if model_name == "X-Learner":
        propensity = model["treatment_probability"]
        tau0 = model["tau0"].predict(matrix)
        tau1 = model["tau1"].predict(matrix)
        return np.clip(propensity * tau0 + (1 - propensity) * tau1, -1.0, 1.0)
    raise KeyError(f"Unknown uplift model: {model_name}")


def validation_metrics(frame: pd.DataFrame, score: np.ndarray, model_name: str) -> tuple[dict, pd.DataFrame]:
    curve = qini_curve(frame["conversion_30d"], frame["treatment_assigned"], score)
    metrics = qini_metrics(curve)
    metrics.update(
        {
            "model": model_name,
            "uplift_at_10pct": uplift_at_fraction(
                frame["conversion_30d"], frame["treatment_assigned"], score, 0.10
            ),
            "uplift_at_20pct": uplift_at_fraction(
                frame["conversion_30d"], frame["treatment_assigned"], score, 0.20
            ),
            "mean_estimated_cate": float(np.mean(score)),
            "predicted_negative_effect_share": float(np.mean(score < 0)),
        }
    )
    curve["model"] = model_name
    return metrics, curve


def random_policy_average(frame: pd.DataFrame, fraction: float = 0.20, repetitions: int = 200) -> dict:
    rng = np.random.default_rng(SEED + 100)
    estimates = []
    n_selected = int(np.ceil(len(frame) * fraction))
    for repetition in range(repetitions):
        selected = np.zeros(len(frame), dtype=bool)
        selected[rng.choice(len(frame), size=n_selected, replace=False)] = True
        estimates.append(evaluate_policy(frame, selected, f"Random {fraction:.0%}"))
    numeric = pd.DataFrame(estimates).select_dtypes(include="number")
    result = numeric.mean().to_dict()
    result["policy"] = f"Random {fraction:.0%}"
    result["random_repetitions"] = repetitions
    result["uplift_std_across_draws"] = float(numeric["uplift"].std(ddof=1))
    return result


def aggregate_feature_importance(
    model_name: str, model: dict, transformed_feature_names: np.ndarray
) -> pd.DataFrame:
    if model_name == "X-Learner":
        importance = (model["tau0"].feature_importances_ + model["tau1"].feature_importances_) / 2
    elif model_name == "T-Learner":
        importance = (model["mu0"].feature_importances_ + model["mu1"].feature_importances_) / 2
    else:
        importance = model["model"].feature_importances_[:-1]
    table = pd.DataFrame({"transformed_feature": transformed_feature_names, "importance": importance})
    table["feature"] = (
        table["transformed_feature"]
        .str.replace(r"^(num|cat)__", "", regex=True)
        .str.replace(r"_(__MISSING__|Mass|Emerging|MicroMerchant|Affluent|SMB)$", "", regex=True)
    )
    table["feature"] = table["feature"].str.split("_").str[:5].str.join("_")
    result = table.groupby("feature", as_index=False)["importance"].sum()
    total = result["importance"].sum()
    result["importance_share"] = result["importance"] / max(total, 1e-12)
    return result.sort_values("importance", ascending=False, ignore_index=True)


def create_development_outputs(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    dictionary: pd.DataFrame,
    base_covariates: list[str],
    timing_review: pd.DataFrame,
) -> dict:
    timing_review.to_csv(TABLES / "variable_timing_and_leakage_review.csv", index=False)

    summary = {
        "rows_total": int(len(train) + len(validation)),
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "train_unique_customers": int(train["customer_id"].nunique()),
        "validation_unique_customers": int(validation["customer_id"].nunique()),
        "duplicate_customer_wave_train": int(train.duplicated(["customer_id", "campaign_wave"]).sum()),
        "public_columns": int(len(dictionary)),
        "pre_treatment_covariates": int(len(base_covariates)),
    }
    pd.DataFrame([summary]).to_csv(TABLES / "development_data_quality_summary.csv", index=False)

    wave_rows = []
    for wave, group in pd.concat([train, validation]).groupby("campaign_wave", sort=True):
        itt = difference_in_means(group["conversion_30d"], group["treatment_assigned"])
        wave_rows.append(
            {
                "campaign_wave": wave,
                "n": len(group),
                "treatment_rate": group["treatment_assigned"].mean(),
                "delivery_rate_among_assigned": group.loc[group["treatment_assigned"].eq(1), "message_delivered"].mean(),
                "control_conversion_rate": itt.control_mean,
                "treated_conversion_rate": itt.treated_mean,
                "itt": itt.estimate,
                "itt_ci_low": itt.ci_low,
                "itt_ci_high": itt.ci_high,
            }
        )
    wave_summary = pd.DataFrame(wave_rows)
    wave_summary.to_csv(TABLES / "development_experiment_summary_by_wave.csv", index=False)

    missing_cols = ["push_open_rate_90d", "email_open_rate_90d", "avg_balance_30d"]
    missing = (
        pd.concat([train, validation])
        .groupby("campaign_wave")[missing_cols]
        .agg(lambda series: series.isna().mean())
        .reset_index()
    )
    missing.to_csv(TABLES / "development_missing_rate_by_wave.csv", index=False)

    raw_features = train[base_covariates]
    categorical = raw_features.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numeric = [column for column in base_covariates if column not in categorical]
    balance = standardized_mean_differences(train, "treatment_assigned", numeric, categorical)
    balance.to_csv(TABLES / "train_randomization_balance_smd.csv", index=False)

    psi_rows = [
        {"feature": column, "psi_train_vs_validation": numeric_psi(train[column], validation[column])}
        for column in numeric
    ]
    psi = pd.DataFrame(psi_rows).sort_values("psi_train_vs_validation", ascending=False)
    psi.to_csv(TABLES / "psi_train_vs_validation.csv", index=False)

    non_compliance = pd.DataFrame(
        [
            {
                "metric": "assignment_rate",
                "value": pd.concat([train, validation])["treatment_assigned"].mean(),
                "interpretation": "Randomized assignment probability",
            },
            {
                "metric": "delivery_rate_among_assigned",
                "value": pd.concat([train, validation]).query("treatment_assigned == 1")["message_delivered"].mean(),
                "interpretation": "Operational compliance among assigned clients",
            },
            {
                "metric": "conversion_delivered_minus_not_delivered_descriptive",
                "value": pd.concat([train, validation]).query("treatment_assigned == 1 and message_delivered == 1")["conversion_30d"].mean()
                - pd.concat([train, validation]).query("treatment_assigned == 1 and message_delivered == 0")["conversion_30d"].mean(),
                "interpretation": "Descriptive only; delivery is not randomized",
            },
        ]
    )
    non_compliance.to_csv(TABLES / "development_non_compliance_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].plot(wave_summary["campaign_wave"], wave_summary["treatment_rate"], marker="o", label="Assigned")
    axes[0].plot(wave_summary["campaign_wave"], wave_summary["delivery_rate_among_assigned"], marker="o", label="Delivered | assigned")
    axes[0].axhline(0.5, color="gray", linestyle="--", linewidth=1)
    axes[0].set_title("Assignment balance and delivery")
    axes[0].set_ylabel("Rate")
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].legend()
    axes[1].plot(wave_summary["campaign_wave"], wave_summary["control_conversion_rate"], marker="o", label="Control")
    axes[1].plot(wave_summary["campaign_wave"], wave_summary["treated_conversion_rate"], marker="o", label="Treatment")
    axes[1].set_title("Conversion rate by randomized arm")
    axes[1].set_ylabel("Conversion rate")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "01_development_experiment_design.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    top_balance = balance.head(20).sort_values("abs_smd")
    axes[0].barh(top_balance["feature"] + " | " + top_balance["level"].astype(str), top_balance["smd"], color="#176B87")
    axes[0].axvline(-0.10, color="#D95555", linestyle="--", linewidth=1)
    axes[0].axvline(0.10, color="#D95555", linestyle="--", linewidth=1)
    axes[0].set_title("Largest absolute SMDs in Train")
    axes[0].set_xlabel("Standardized mean difference")
    for column in missing_cols:
        axes[1].plot(missing["campaign_wave"], missing[column], marker="o", label=column)
    axes[1].set_title("Missingness drift before OOT")
    axes[1].set_ylabel("Missing rate")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "02_balance_and_missing_drift.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    train_itt = difference_in_means(train["conversion_30d"], train["treatment_assigned"])
    validation_itt = difference_in_means(validation["conversion_30d"], validation["treatment_assigned"])
    return {
        "train_itt": train_itt,
        "validation_itt": validation_itt,
        "max_abs_smd": float(balance["abs_smd"].max()),
        "max_psi_train_validation": float(psi["psi_train_vs_validation"].max()),
    }


def plot_validation_results(
    curves: pd.DataFrame,
    validation_scored: pd.DataFrame,
    champion: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    for model_name, group in curves.groupby("model", sort=False):
        ax.plot(group["population_fraction"], group["gain_per_customer"], label=model_name)
    random_line = curves[curves["model"].eq(champion)]
    ax.plot(
        random_line["population_fraction"],
        random_line["random_gain_per_customer"],
        color="gray",
        linestyle="--",
        label="Random baseline",
    )
    ax.set_title("Validation Qini curves - wave 5")
    ax.set_xlabel("Targeted population share")
    ax.set_ylabel("Cumulative incremental conversions per eligible customer")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "03_validation_qini_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    quantiles = uplift_by_quantile(validation_scored, "estimated_cate")
    quantiles.to_csv(TABLES / "validation_uplift_by_decile.csv", index=False)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.bar(quantiles["quantile"], quantiles["observed_uplift"], color="#2A9D8F")
    ax.errorbar(
        quantiles["quantile"],
        quantiles["observed_uplift"],
        yerr=[
            quantiles["observed_uplift"] - quantiles["ci_low"],
            quantiles["ci_high"] - quantiles["observed_uplift"],
        ],
        fmt="none",
        ecolor="#161B22",
        capsize=3,
    )
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title(f"Validation observed uplift by {champion} decile")
    ax.set_xlabel("Decile (1 = highest estimated uplift)")
    ax.set_ylabel("Observed ITT uplift")
    fig.tight_layout()
    fig.savefig(FIGURES / "04_validation_uplift_deciles.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def create_oot_outputs(
    oot: pd.DataFrame,
    score: np.ndarray,
    propensity_score: np.ndarray,
    champion: str,
) -> dict:
    scored = oot.copy()
    scored["estimated_cate"] = score
    scored["propensity_response_score"] = propensity_score

    oot_itt = difference_in_means(scored["conversion_30d"], scored["treatment_assigned"])
    champion_metrics, curve = validation_metrics(scored, score, champion)
    champion_metrics["sample"] = "OOT wave 6"
    pd.DataFrame([champion_metrics]).to_csv(METRICS / "oot_champion_metrics.csv", index=False)
    curve.to_csv(TABLES / "oot_qini_curve.csv", index=False)

    quantiles = uplift_by_quantile(scored, "estimated_cate")
    quantiles.to_csv(TABLES / "oot_uplift_by_decile.csv", index=False)

    random_policy = random_policy_average(scored, 0.20, repetitions=200)
    propensity_policy = evaluate_policy(
        scored,
        exact_top_fraction(scored["propensity_response_score"], 0.20),
        "Top Propensity 20%",
    )
    uplift_policy = evaluate_policy(
        scored,
        exact_top_fraction(scored["estimated_cate"], 0.20),
        "Top Uplift 20%",
    )
    policies = pd.DataFrame([random_policy, propensity_policy, uplift_policy])
    policies.to_csv(TABLES / "oot_policy_comparison.csv", index=False)

    segment_rows = []
    for segment_col in ["customer_segment", "merchant_flag", "region"]:
        for segment, group in scored.groupby(segment_col, observed=True):
            if group["treatment_assigned"].nunique() < 2 or len(group) < 200:
                continue
            effect = difference_in_means(group["conversion_30d"], group["treatment_assigned"])
            segment_rows.append(
                {
                    "segment_variable": segment_col,
                    "segment": str(segment),
                    "n": len(group),
                    "observed_itt": effect.estimate,
                    "ci_low": effect.ci_low,
                    "ci_high": effect.ci_high,
                    "mean_estimated_cate": group["estimated_cate"].mean(),
                    "predicted_negative_effect_share": (group["estimated_cate"] < 0).mean(),
                }
            )
    segments = pd.DataFrame(segment_rows)
    segments.to_csv(TABLES / "oot_segment_effects.csv", index=False)

    scored[
        [
            "customer_id",
            "campaign_wave",
            "treatment_assigned",
            "conversion_30d",
            "revenue_30d",
            "margin_30d",
            "estimated_cate",
            "propensity_response_score",
        ]
    ].to_csv(TABLES / "oot_scored_population.csv", index=False)

    psi_candidates = [
        "txn_value_30d",
        "avg_balance_30d",
        "app_sessions_30d",
        "push_open_rate_90d",
        "email_open_rate_90d",
        "previous_offer_acceptance_rate",
    ]
    # PSI reference is loaded only after the champion is frozen.
    train_reference = pd.read_csv(RAW_PATH, usecols=["campaign_wave", *psi_candidates])
    train_reference = train_reference[train_reference["campaign_wave"].isin(TRAIN_WAVES)]
    psi_oot = pd.DataFrame(
        [
            {"feature": column, "psi_train_vs_oot": numeric_psi(train_reference[column], scored[column])}
            for column in psi_candidates
        ]
    ).sort_values("psi_train_vs_oot", ascending=False)
    psi_oot.to_csv(TABLES / "psi_train_vs_oot_after_freeze.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    axes[0].plot(curve["population_fraction"], curve["gain_per_customer"], color="#176B87", label=champion)
    axes[0].plot(curve["population_fraction"], curve["random_gain_per_customer"], color="gray", linestyle="--", label="Random baseline")
    axes[0].set_title("OOT Qini curve - wave 6")
    axes[0].set_xlabel("Targeted population share")
    axes[0].set_ylabel("Cumulative incremental conversions per eligible customer")
    axes[0].legend()
    axes[1].bar(quantiles["quantile"], quantiles["observed_uplift"], color="#2A9D8F")
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_title("OOT observed uplift by score decile")
    axes[1].set_xlabel("Decile (1 = highest estimated uplift)")
    axes[1].set_ylabel("Observed ITT uplift")
    fig.tight_layout()
    fig.savefig(FIGURES / "05_oot_qini_and_deciles.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    colors = ["#8D99AE", "#E9A23B", "#2A9D8F"]
    axes[0].bar(policies["policy"], policies["uplift"], color=colors)
    axes[0].set_title("Observed uplift at 20% capacity")
    axes[0].set_ylabel("ITT uplift")
    axes[0].tick_params(axis="x", rotation=18)
    axes[1].bar(policies["policy"], policies["net_incremental_value"], color=colors)
    axes[1].set_title("Projected net incremental value")
    axes[1].set_ylabel("Value units in OOT population")
    axes[1].tick_params(axis="x", rotation=18)
    fig.tight_layout()
    fig.savefig(FIGURES / "06_oot_policy_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    sample = scored.sample(min(12000, len(scored)), random_state=SEED)
    hb = ax.hexbin(
        sample["propensity_response_score"],
        sample["estimated_cate"],
        gridsize=42,
        cmap="Blues",
        mincnt=1,
    )
    ax.axhline(0, color="#D95555", linestyle="--", linewidth=1)
    ax.set_title("Propensity and uplift rank different behaviors")
    ax.set_xlabel("Predicted conversion under treatment")
    ax.set_ylabel("Estimated incremental effect")
    fig.colorbar(hb, ax=ax, label="Customers")
    fig.tight_layout()
    fig.savefig(FIGURES / "07_propensity_vs_uplift.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    segment_plot = segments[segments["segment_variable"].eq("customer_segment")].sort_values("observed_itt")
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.barh(segment_plot["segment"], segment_plot["observed_itt"], color="#176B87")
    ax.errorbar(
        segment_plot["observed_itt"],
        segment_plot["segment"],
        xerr=[
            segment_plot["observed_itt"] - segment_plot["ci_low"],
            segment_plot["ci_high"] - segment_plot["observed_itt"],
        ],
        fmt="none",
        color="#161B22",
        capsize=3,
    )
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("OOT ITT by customer segment")
    ax.set_xlabel("Observed uplift")
    fig.tight_layout()
    fig.savefig(FIGURES / "08_oot_segment_effects.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    score_correlation = float(scored["estimated_cate"].corr(scored["propensity_response_score"], method="spearman"))
    top_uplift = exact_top_fraction(scored["estimated_cate"], 0.20)
    top_propensity = exact_top_fraction(scored["propensity_response_score"], 0.20)
    top_overlap = float(np.mean(top_uplift[top_propensity]))
    return {
        "oot_itt": oot_itt,
        "oot_metrics": champion_metrics,
        "policies": policies,
        "score_spearman": score_correlation,
        "top20_overlap_share_of_propensity_audience": top_overlap,
        "max_psi_train_oot": float(psi_oot["psi_train_vs_oot"].max()),
    }


def main() -> None:
    started = time.time()
    ensure_directories()
    plt.style.use("seaborn-v0_8-whitegrid")

    frame, dictionary = load_public_data()
    base_covariates, timing_review = build_covariates(dictionary)
    train = frame[frame["campaign_wave"].isin(TRAIN_WAVES)].copy()
    validation = frame[frame["campaign_wave"].isin(VALIDATION_WAVES)].copy()
    oot = frame[frame["campaign_wave"].isin(OOT_WAVES)].copy()

    development = create_development_outputs(train, validation, dictionary, base_covariates, timing_review)

    x_train_raw = engineer_features(train, base_covariates)
    x_validation_raw = engineer_features(validation, base_covariates)
    preprocessor, numeric_features, categorical_features = make_preprocessor(x_train_raw)
    x_train = preprocessor.fit_transform(x_train_raw)
    x_validation = preprocessor.transform(x_validation_raw)
    y_train = train["conversion_30d"].to_numpy(dtype=int)
    t_train = train["treatment_assigned"].to_numpy(dtype=int)

    models = fit_model_suite(x_train, y_train, t_train)
    propensity_validation = models["propensity_response"].predict_proba(x_validation)[:, 1]

    metric_rows = []
    curve_rows = []
    validation_scores: dict[str, np.ndarray] = {}
    for model_name in ["S-Learner", "T-Learner", "X-Learner"]:
        score = predict_uplift(model_name, models[model_name], x_validation)
        validation_scores[model_name] = score
        metrics, curve = validation_metrics(validation, score, model_name)
        metric_rows.append(metrics)
        curve_rows.append(curve)
    comparison = pd.DataFrame(metric_rows).sort_values(
        ["qini_coefficient", "auuc", "uplift_at_20pct"], ascending=False, ignore_index=True
    )
    comparison.to_csv(METRICS / "validation_model_comparison.csv", index=False)
    all_curves = pd.concat(curve_rows, ignore_index=True)
    all_curves.to_csv(TABLES / "validation_qini_curves.csv", index=False)
    champion = str(comparison.loc[0, "model"])
    champion_score_validation = validation_scores[champion]

    validation_scored = validation.copy()
    validation_scored["estimated_cate"] = champion_score_validation
    validation_scored["propensity_response_score"] = propensity_validation
    plot_validation_results(all_curves, validation_scored, champion)

    feature_names = preprocessor.get_feature_names_out()
    importance = aggregate_feature_importance(champion, models[champion], feature_names)
    importance.to_csv(TABLES / "champion_feature_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    top_importance = importance.head(18).sort_values("importance_share")
    ax.barh(top_importance["feature"], top_importance["importance_share"], color="#176B87")
    ax.set_title(f"{champion} global feature importance")
    ax.set_xlabel("Share of model gain")
    fig.tight_layout()
    fig.savefig(FIGURES / "09_champion_feature_importance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    availability = pd.DataFrame(
        [
            {"method": "S-Learner", "implemented": True, "reason": "LightGBM outcome model with treatment indicator"},
            {"method": "T-Learner / Two-model", "implemented": True, "reason": "Separate LightGBM outcome models by randomized arm"},
            {"method": "X-Learner", "implemented": True, "reason": "Three-fold cross-fitted imputed treatment effects"},
            {"method": "Causal Forest", "implemented": False, "reason": "econml not available in the execution environment"},
            {"method": "Uplift Tree", "implemented": False, "reason": "causalml not available in the execution environment"},
        ]
    )
    availability.to_csv(TABLES / "model_availability.csv", index=False)

    freeze_record = {
        "frozen_before_oot": True,
        "champion": champion,
        "selection_sample": "Validation wave 2026-05",
        "selection_rule": "Highest validation Qini coefficient; AUUC and Uplift@20% as tie-breakers",
        "capacity": 0.20,
        "treatment": "treatment_assigned",
        "outcome": "conversion_30d",
        "estimand": "Intention-to-Treat CATE",
        "train_waves": TRAIN_WAVES,
        "validation_waves": VALIDATION_WAVES,
        "oot_waves": OOT_WAVES,
        "base_covariates": base_covariates,
        "engineered_feature_count": int(x_train_raw.shape[1]),
        "transformed_feature_count": int(x_train.shape[1]),
        "post_treatment_exclusions": POST_TREATMENT_COLUMNS,
        "model_parameters": model_parameters(SEED),
    }
    save_json(METRICS / "champion_freeze_record.json", freeze_record)
    joblib.dump(
        {
            "champion": champion,
            "preprocessor": preprocessor,
            "model": models[champion],
            "base_covariates": base_covariates,
            "train_waves": TRAIN_WAVES,
            "validation_waves": VALIDATION_WAVES,
            "oot_waves": OOT_WAVES,
        },
        MODELS / "champion_uplift_bundle.joblib",
        compress=3,
    )
    joblib.dump(
        {
            "preprocessor": preprocessor,
            "model": models["propensity_response"],
            "base_covariates": base_covariates,
        },
        MODELS / "propensity_response_bundle.joblib",
        compress=3,
    )

    # OOT is accessed only after the freeze record and serialized models exist.
    x_oot_raw = engineer_features(oot, base_covariates)
    x_oot = preprocessor.transform(x_oot_raw)
    oot_score = predict_uplift(champion, models[champion], x_oot)
    propensity_oot = models["propensity_response"].predict_proba(x_oot)[:, 1]
    oot_results = create_oot_outputs(oot, oot_score, propensity_oot, champion)

    split_summary = pd.DataFrame(
        [
            {"split": "Train", "waves": ", ".join(TRAIN_WAVES), "rows": len(train), "event_rate": train["conversion_30d"].mean()},
            {"split": "Validation", "waves": ", ".join(VALIDATION_WAVES), "rows": len(validation), "event_rate": validation["conversion_30d"].mean()},
            {"split": "OOT Final Test", "waves": ", ".join(OOT_WAVES), "rows": len(oot), "event_rate": oot["conversion_30d"].mean()},
        ]
    )
    split_summary.to_csv(TABLES / "split_summary_after_oot_open.csv", index=False)

    policies = oot_results["policies"].set_index("policy")
    top_uplift = policies.loc["Top Uplift 20%"]
    top_propensity = policies.loc["Top Propensity 20%"]
    random_policy = policies.loc["Random 20%"]
    executive_summary = {
        "dataset": {
            "rows": int(len(frame)),
            "waves": int(frame["campaign_wave"].nunique()),
            "unique_customers": int(frame["customer_id"].nunique()),
        },
        "experimental_design": {
            "treatment_rate": float(frame["treatment_assigned"].mean()),
            "delivery_rate_among_assigned": float(frame.query("treatment_assigned == 1")["message_delivered"].mean()),
            "train_max_abs_smd": development["max_abs_smd"],
        },
        "development": {
            "train_itt": development["train_itt"].estimate,
            "validation_itt": development["validation_itt"].estimate,
            "champion": champion,
            "validation_qini": float(comparison.loc[0, "qini_coefficient"]),
            "validation_auuc": float(comparison.loc[0, "auuc"]),
            "validation_uplift_at_20pct": float(comparison.loc[0, "uplift_at_20pct"]),
        },
        "oot": {
            "itt": oot_results["oot_itt"].estimate,
            "itt_ci_low": oot_results["oot_itt"].ci_low,
            "itt_ci_high": oot_results["oot_itt"].ci_high,
            "qini": oot_results["oot_metrics"]["qini_coefficient"],
            "auuc": oot_results["oot_metrics"]["auuc"],
            "uplift_at_20pct": oot_results["oot_metrics"]["uplift_at_20pct"],
            "predicted_negative_effect_share": oot_results["oot_metrics"]["predicted_negative_effect_share"],
            "propensity_uplift_spearman": oot_results["score_spearman"],
            "top20_audience_overlap": oot_results["top20_overlap_share_of_propensity_audience"],
        },
        "policy": {
            "random_20_net_incremental_value": float(random_policy["net_incremental_value"]),
            "top_propensity_20_net_incremental_value": float(top_propensity["net_incremental_value"]),
            "top_uplift_20_net_incremental_value": float(top_uplift["net_incremental_value"]),
            "top_uplift_20_incremental_conversions": float(top_uplift["incremental_conversions"]),
            "top_uplift_20_roi": float(top_uplift["roi"]),
            "uplift_value_vs_propensity": float(top_uplift["net_incremental_value"] - top_propensity["net_incremental_value"]),
            "uplift_value_vs_random": float(top_uplift["net_incremental_value"] - random_policy["net_incremental_value"]),
        },
        "methodological_conclusion": (
            "Propensity ranks likely converters; uplift ranks customers whose behavior is expected to change because of treatment. "
            "Production value must be confirmed by a fresh randomized policy experiment."
        ),
        "runtime": {
            "seconds": time.time() - started,
            "python": platform.python_version(),
            "lightgbm": lgb.__version__,
        },
    }
    save_json(METRICS / "executive_summary.json", executive_summary)
    print(json.dumps(executive_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
