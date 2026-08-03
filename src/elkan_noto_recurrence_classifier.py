"""
Elkan-Noto positive-unlabeled recurrence classifier.

Example:
    python src/elkan_noto_recurrence_classifier.py \
        --study pdac \
        --modality all \
        --horizon-days 1825 \
        --k-best 20

Target definition:
    s = 1: documented recurrence on or before the chosen horizon
    s = 0: unlabeled (every other patient)

The model first estimates P(s=1 | x). Elkan-Noto then estimates
c = P(s=1 | y=1) from out-of-fold predictions on the known positives and uses

    P(y=1 | x) = P(s=1 | x) / c

where y is the unobserved true recurrence class.
"""

from argparse import ArgumentParser

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from query_patients import process_cancer


EVENT_COL = "Recurrence"
TIME_COL = "Recurrence-free survival time, days"

MODALITY_TO_TAG = {
    "proteomics": "proteomics::",
    "phosphoproteomics": "phosphoproteomics::",
    "acetylproteomics": "acetylproteomics::",
}


def parse_event(value):
    """Convert common recurrence values to True, False, or missing."""
    if pd.isna(value):
        return pd.NA
    if value is True or value == 1:
        return True
    if value is False or value == 0:
        return False

    text = str(value).strip().lower()
    if text in {"true", "yes", "y", "1", "recurred", "recurrence", "relapse"}:
        return True
    if text in {"false", "no", "n", "0", "no recurrence", "disease free"}:
        return False
    return pd.NA


def selected_feature_tags(modality: str) -> tuple[str, ...]:
    if modality == "all":
        return tuple(MODALITY_TO_TAG.values())
    return (MODALITY_TO_TAG[modality],)


def build_pipeline(k_best: int) -> Pipeline:
    """One intentionally simple probabilistic classifier for P(s=1 | x)."""
    return Pipeline([
        ( "imputer", SimpleImputer(strategy="median", keep_empty_features=True),),
        ("feature_selector", SelectKBest(score_func=f_classif, k=k_best)),
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                C=0.1,
                solver="liblinear",
                max_iter=5000,
                random_state=42,
            ),
        ),
    ])


def elkan_noto_probabilities(p_s: np.ndarray, c: float, s: np.ndarray):
    """Return P(y=1|x) and P(y=1|x,s) under the Elkan-Noto assumptions."""
    p_y_given_x = np.clip(p_s / c, 0.0, 1.0)

    denominator = np.clip(1.0 - p_s, 1e-6, None)
    p_y_given_x_and_unlabeled = np.clip(
        ((1.0 - c) / c) * (p_s / denominator),
        0.0,
        1.0,
    )

    # A labeled positive is known to be positive. For s=0, use the posterior
    # probability of being a hidden positive among the unlabeled patients.
    p_y_given_x_and_s = np.where(s == 1, 1.0, p_y_given_x_and_unlabeled)
    return p_y_given_x, p_y_given_x_and_s


def fit_fold_model(
    X_train: pd.DataFrame,
    s_train: np.ndarray,
    k_best_requested: int,
    max_missing: float,
    inner_folds_requested: int,
    random_state: int,
):
    """Fit one outer-fold model and estimate c using inner out-of-fold predictions."""
    keep_feature = (
        (X_train.isna().mean() <= max_missing)
        & (X_train.nunique(dropna=True) > 1)
    )
    X_train = X_train.loc[:, keep_feature]

    if X_train.shape[1] == 0:
        raise ValueError("No features remain after training-fold filtering.")

    k_best = min(k_best_requested, X_train.shape[1])
    smallest_class = int(np.bincount(s_train).min())
    inner_folds = min(inner_folds_requested, smallest_class)
    if inner_folds < 2:
        raise ValueError("Not enough training examples for inner PU cross-validation.")

    inner_cv = StratifiedKFold(
        n_splits=inner_folds,
        shuffle=True,
        random_state=random_state,
    )

    c_pipeline = build_pipeline(k_best)
    train_oof_p_s = cross_val_predict(
        c_pipeline,
        X_train,
        s_train,
        cv=inner_cv,
        method="predict_proba",
    )[:, 1]

    c = float(train_oof_p_s[s_train == 1].mean())
    c = float(np.clip(c, 1e-6, 1.0))

    model = build_pipeline(k_best)
    model.fit(X_train, s_train)
    return model, c, X_train.columns, k_best, inner_folds


def main():
    parser = ArgumentParser(
        description="Elkan-Noto positive-unlabeled recurrence classifier."
    )
    parser.add_argument("--study", required=True, help="One CPTAC cancer, such as pdac.")
    parser.add_argument("--modality", choices=["all", *MODALITY_TO_TAG], default="all", help="Use all available omics or one modality.",)
    parser.add_argument("--horizon-days", type=float, default=1825.0, help="Reliable-positive recurrence horizon. Default: 1825 days (5 years).",)
    parser.add_argument(
        "--k-best",
        type=int,
        default=20,
        help="Number of omics features selected inside each training fold.",
    )
    parser.add_argument(
        "--outer-folds",
        type=int,
        default=5,
        help="Outer stratified folds used to evaluate held-out patients.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Inner stratified folds used to estimate Elkan-Noto c.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-missing", type=float, default=0.6)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-prefix", default=None, help="Output prefix. Default: <study>_elkan_noto.",)
    args = parser.parse_args()

    if args.horizon_days <= 0:
        raise ValueError("--horizon-days must be positive.")
    if args.k_best <= 0:
        raise ValueError("--k-best must be positive.")
    if args.outer_folds < 2:
        raise ValueError("--outer-folds must be at least 2.")
    if args.cv_folds < 2:
        raise ValueError("--cv-folds must be at least 2.")
    if not 0 <= args.max_missing < 1:
        raise ValueError("--max-missing must be in [0, 1).")
    if not 0 < args.threshold < 1:
        raise ValueError("--threshold must be between 0 and 1.")

    study_name = args.study.lower()
    print(f"Loading {study_name.upper()}")
    study_df = process_cancer(study_name)["combined"].copy()

    if EVENT_COL not in study_df.columns:
        raise ValueError(f"Missing required event column: {EVENT_COL!r}")
    if TIME_COL not in study_df.columns:
        raise ValueError(f"Missing required time column: {TIME_COL!r}")

    tags = selected_feature_tags(args.modality)
    feature_cols = [col for col in study_df.columns if isinstance(col, str) and col.startswith(tags)]
    if not feature_cols:
        raise ValueError(f"No {args.modality} features were found for {study_name.upper()}.")

    event = study_df[EVENT_COL].map(parse_event)
    event_time = pd.to_numeric(study_df[TIME_COL], errors="coerce")

    # s=1 is a reliable observed positive. Every other patient is unlabeled.
    observed_positive = (
        event.eq(True).fillna(False)
        & event_time.notna()
        & (event_time <= args.horizon_days)
    )
    s = observed_positive.astype(int)

    X = study_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    patient_ids = (
        study_df["Patient_ID"].astype(str)
        if "Patient_ID" in study_df.columns
        else pd.Series(study_df.index.astype(str), index=study_df.index)
    )

    if s.sum() < 3:
        raise ValueError(
            f"Only {int(s.sum())} reliable positives were found at "
            f"{args.horizon_days:g} days; at least 3 are required."
        )
    if (s == 0).sum() < 3:
        raise ValueError("At least 3 unlabeled patients are required.")

    print("\nPositive-unlabeled outcome:")
    print(f"Horizon: {args.horizon_days:g} days")
    print(f"All patients: {len(study_df)}")
    print(f"Reliable positives: {int(s.sum())}")
    print(f"Unlabeled: {int((s == 0).sum())}")

    s_array = s.to_numpy()
    smallest_class = int(np.bincount(s_array).min())
    outer_folds = min(args.outer_folds, smallest_class)
    if outer_folds < 2:
        raise ValueError("Not enough patients for outer cross-validation.")

    outer_cv = StratifiedKFold(
        n_splits=outer_folds,
        shuffle=True,
        random_state=args.random_state,
    )

    prediction_blocks = []
    fold_metric_rows = []
    selected_feature_rows = []

    for fold, (train_indices, test_indices) in enumerate(
        outer_cv.split(X, s_array),
        start=1,
    ):
        X_train = X.iloc[train_indices].copy()
        X_test = X.iloc[test_indices].copy()
        s_train = s_array[train_indices]
        s_test = s_array[test_indices]

        model, c, train_columns, k_best, inner_folds = fit_fold_model(
            X_train=X_train,
            s_train=s_train,
            k_best_requested=args.k_best,
            max_missing=args.max_missing,
            inner_folds_requested=args.cv_folds,
            random_state=args.random_state + fold,
        )

        X_test = X_test.loc[:, train_columns]
        test_p_s = model.predict_proba(X_test)[:, 1]
        test_p_y, test_p_y_given_s = elkan_noto_probabilities(
            test_p_s,
            c,
            s_test,
        )
        predicted_y = (test_p_y >= args.threshold).astype(int)
        predicted_y_given_s = (test_p_y_given_s >= args.threshold).astype(int)
        clipped_to_one = (test_p_s / c >= 1.0)

        observed_label_auc = roc_auc_score(s_test, test_p_s)
        observed_label_ap = average_precision_score(s_test, test_p_s)
        observed_positive_prevalence = s_test.mean()
        positive_recall = predicted_y[s_test == 1].mean()
        unlabeled_positive_rate = predicted_y[s_test == 0].mean()
        unlabeled_posterior_positive_rate = predicted_y_given_s[s_test == 0].mean()

        fold_metric_rows.append({
            "fold": fold,
            "train_patients": len(train_indices),
            "test_patients": len(test_indices),
            "inner_folds": inner_folds,
            "features_after_filtering": len(train_columns),
            "selected_features": k_best,
            "c": c,
            "observed_positive_prevalence": observed_positive_prevalence,
            "observed_label_roc_auc": observed_label_auc,
            "observed_label_average_precision": observed_label_ap,
            "known_positive_recall": positive_recall,
            "unlabeled_predicted_positive_rate": unlabeled_positive_rate,
            "unlabeled_posterior_positive_rate": unlabeled_posterior_positive_rate,
            "fraction_corrected_probabilities_clipped": clipped_to_one.mean(),
        })

        selector = model.named_steps["feature_selector"]
        selected_features = train_columns[selector.get_support()]
        coefficients = model.named_steps["classifier"].coef_[0]

        for feature, coefficient in zip(selected_features, coefficients):
            selected_feature_rows.append({
                "fold": fold,
                "feature": feature,
                "coefficient": coefficient,
                "absolute_coefficient": abs(coefficient),
            })

        prediction_blocks.append(pd.DataFrame({
            "row_index": test_indices,
            "fold": fold,
            "Patient_ID": patient_ids.iloc[test_indices].to_numpy(),
            "documented_recurrence": event.iloc[test_indices].to_numpy(),
            "documented_recurrence_time_days": event_time.iloc[test_indices].to_numpy(),
            "observed_PU_label_s": s_test,
            "P_s_equals_1_given_x": test_p_s,
            "elkan_noto_P_y_equals_1_given_x": test_p_y,
            "P_y_equals_1_given_x_and_observed_s": test_p_y_given_s,
            "predicted_latent_recurrence": predicted_y,
            "predicted_given_observed_s": predicted_y_given_s,
            "corrected_probability_clipped_to_one": clipped_to_one,
        }))

    prediction_output = (
        pd.concat(prediction_blocks, ignore_index=True)
        .sort_values("row_index")
        .reset_index(drop=True)
    )
    fold_metrics = pd.DataFrame(fold_metric_rows)
    selected_by_fold = pd.DataFrame(selected_feature_rows)

    feature_stability = (
        selected_by_fold.groupby("feature", as_index=False)
        .agg(
            selected_folds=("fold", "nunique"),
            mean_coefficient=("coefficient", "mean"),
            mean_absolute_coefficient=("absolute_coefficient", "mean"),
        )
    )
    feature_stability["selection_frequency"] = (
        feature_stability["selected_folds"] / outer_folds
    )
    feature_stability = feature_stability.sort_values(
        ["selection_frequency", "mean_absolute_coefficient"],
        ascending=False,
    )

    # Fit one full-data model for downstream interpretation. Its feature
    # coefficients are not used to report cross-validated performance.
    final_model, final_c, final_columns, final_k, final_inner_folds = fit_fold_model(
        X_train=X.copy(),
        s_train=s_array,
        k_best_requested=args.k_best,
        max_missing=args.max_missing,
        inner_folds_requested=args.cv_folds,
        random_state=args.random_state + 1000,
    )
    final_selector = final_model.named_steps["feature_selector"]
    final_features = final_columns[final_selector.get_support()]
    final_coefficients = final_model.named_steps["classifier"].coef_[0]
    final_feature_importance = pd.DataFrame({
        "feature": final_features,
        "coefficient": final_coefficients,
        "absolute_coefficient": np.abs(final_coefficients),
    }).sort_values("absolute_coefficient", ascending=False)

    pooled_s = prediction_output["observed_PU_label_s"].to_numpy()
    pooled_p_s = prediction_output["P_s_equals_1_given_x"].to_numpy()
    pooled_prediction = prediction_output["predicted_latent_recurrence"].to_numpy()

    pooled_auc = roc_auc_score(pooled_s, pooled_p_s)
    pooled_ap = average_precision_score(pooled_s, pooled_p_s)
    pooled_prevalence = pooled_s.mean()
    pooled_recall = pooled_prediction[pooled_s == 1].mean()
    pooled_unlabeled_positive_rate = pooled_prediction[pooled_s == 0].mean()
    pooled_unlabeled_posterior_rate = prediction_output.loc[
        pooled_s == 0,
        "predicted_given_observed_s",
    ].mean()
    pooled_clipped_fraction = prediction_output[
        "corrected_probability_clipped_to_one"
    ].mean()

    output_prefix = args.output_prefix or f"{study_name}_elkan_noto"
    prediction_path = f"{output_prefix}_cross_validated_predictions.csv"
    fold_metrics_path = f"{output_prefix}_fold_metrics.csv"
    stability_path = f"{output_prefix}_feature_stability.csv"
    final_feature_path = f"{output_prefix}_final_selected_features.csv"

    prediction_output.drop(columns="row_index").to_csv(prediction_path, index=False)
    fold_metrics.to_csv(fold_metrics_path, index=False)
    feature_stability.to_csv(stability_path, index=False)
    final_feature_importance.to_csv(final_feature_path, index=False)

    print("\nCross-validation architecture:")
    print(f"Outer evaluation folds: {outer_folds}")
    print(f"Inner c-estimation folds: up to {args.cv_folds}")
    print(f"Every patient received one outer held-out prediction: {len(prediction_output)}")

    print("\nPooled outer-cross-validated PU sanity metrics:")
    print(f"Observed-positive prevalence / no-skill AP: {pooled_prevalence:.4f}")
    print(f"Observed-label ROC-AUC: {pooled_auc:.4f}")
    print(f"Observed-label average precision: {pooled_ap:.4f}")
    print(f"Recall of reliable positives: {pooled_recall:.4f}")
    print(f"Unlabeled predicted-positive rate using P(y=1|x): {pooled_unlabeled_positive_rate:.4f}")
    print(f"Unlabeled positive rate using P(y=1|x,s=0): {pooled_unlabeled_posterior_rate:.4f}")
    print(f"Corrected probabilities clipped to 1: {pooled_clipped_fraction:.4f}")

    print("\nFold variability:")
    print(
        "Observed-label ROC-AUC: "
        f"{fold_metrics['observed_label_roc_auc'].mean():.4f} "
        f"+/- {fold_metrics['observed_label_roc_auc'].std():.4f}"
    )
    print(
        "Observed-label average precision: "
        f"{fold_metrics['observed_label_average_precision'].mean():.4f} "
        f"+/- {fold_metrics['observed_label_average_precision'].std():.4f}"
    )
    print(
        "Elkan-Noto c: "
        f"{fold_metrics['c'].mean():.4f} +/- {fold_metrics['c'].std():.4f}"
    )

    print("\nFull-data interpretation model:")
    print(f"Features after filtering: {len(final_columns)}")
    print(f"Selected features: {final_k}")
    print(f"Inner folds used for final c: {final_inner_folds}")
    print(f"Final c estimate: {final_c:.4f}")

    print("\nImportant interpretation:")
    print("These are cross-validated PU sanity metrics, not true recurrence accuracy.")
    print("Unlabeled patients do not provide confirmed negative outcomes.")

    print("\nTop final-model selected features:")
    print(final_feature_importance.head(20).to_string(index=False))
    print(f"\nSaved cross-validated predictions: {prediction_path}")
    print(f"Saved fold metrics: {fold_metrics_path}")
    print(f"Saved feature stability: {stability_path}")
    print(f"Saved final selected features: {final_feature_path}")


if __name__ == "__main__":
    main()