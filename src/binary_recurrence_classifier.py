from argparse import ArgumentParser

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from query_patients import ALL_STUDIES, process_cancer


EVENT_COL = "Recurrence"
TIME_COL = "Derived recurrence-free survival time, days"
CANCER_COL = "Cancer_Type"

PROTEOMICS_TAG = "proteomics::"
PHOSPHOPROTEOMICS_TAG = "phosphoproteomics::"
ACETYLPROTEOMICS_TAG = "acetylproteomics::"

MODALITY_TO_TAG = {
    "proteomics": PROTEOMICS_TAG,
    "proteome": PROTEOMICS_TAG,
    "phosphoproteomics": PHOSPHOPROTEOMICS_TAG,
    "phosphoproteome": PHOSPHOPROTEOMICS_TAG,
    "acetylproteomics": ACETYLPROTEOMICS_TAG,
    "acetylproteome": ACETYLPROTEOMICS_TAG,
}

FEATURE_COUNT_GRID = [5, 10, 20, 30]
C_GRID = [0.1, 1.0, 10.0]
L1_RATIO_GRID = [0.1, 0.5, 0.9]


class TrainingFeatureFilter(BaseEstimator, TransformerMixin):
    """
    Remove highly missing, empty, and constant features.

    Because this transformer is inside the Pipeline, it learns which
    columns to retain using only the training patients in the current
    cross-validation fold.
    """

    def __init__(self, missing_threshold=0.60):
        self.missing_threshold = missing_threshold

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "TrainingFeatureFilter expects a pandas DataFrame."
            )

        if not 0 < self.missing_threshold <= 1:
            raise ValueError(
                "missing_threshold must be greater than 0 and at most 1."
            )

        missingness = X.isna().mean()
        unique_values = X.nunique(dropna=True)

        keep = (
            (missingness < self.missing_threshold)
            & (unique_values > 1)
        )

        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.selected_columns_ = np.asarray(
            X.columns[keep],
            dtype=object,
        )

        if len(self.selected_columns_) == 0:
            raise ValueError(
                "No features remain after missingness and variance filtering."
            )

        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_in_)

        return X.loc[:, self.selected_columns_]

    def get_feature_names_out(self, input_features=None):
        return self.selected_columns_


def canonical(col: str) -> str:
    """
    Make feature names comparable across CPTAC studies.

    Proteomics keeps the gene name.
    PTM modalities keep the gene and modification site.
    """
    if not isinstance(col, str) or "::" not in col:
        return col

    modality, _, rest = col.partition("::")
    parts = [part for part in rest.split("|") if part]

    if not parts:
        return col

    if modality == "proteomics":
        return f"{modality}::{parts[0]}"

    return f"{modality}::" + "|".join(parts[:2])


def parse_binary_event(series: pd.Series) -> pd.Series:
    """
    Convert recurrence labels to 0/1 without treating the string
    'False' as True.
    """
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)

    numeric = pd.to_numeric(series, errors="coerce")

    text_mapping = {
        "true": 1.0,
        "false": 0.0,
        "yes": 1.0,
        "no": 0.0,
        "recurred": 1.0,
        "not recurred": 0.0,
    }

    text_values = (
        series.astype("string")
        .str.strip()
        .str.lower()
        .map(text_mapping)
    )

    return numeric.fillna(text_values)


def select_modalities(requested_modalities):
    requested_modalities = [
        modality.lower()
        for modality in requested_modalities
    ]

    if "all" in requested_modalities:
        return (
            PROTEOMICS_TAG,
            PHOSPHOPROTEOMICS_TAG,
            ACETYLPROTEOMICS_TAG,
        )

    invalid = [
        modality
        for modality in requested_modalities
        if modality not in MODALITY_TO_TAG
    ]

    if invalid:
        raise ValueError(
            f"Invalid modalities: {invalid}. "
            f"Valid options are {sorted(MODALITY_TO_TAG)} or all."
        )

    return tuple(
        MODALITY_TO_TAG[modality]
        for modality in requested_modalities
    )


def load_studies(studies, selected_tags):
    frames = []

    for study_name in studies:
        print(f"Loading {study_name.upper()}")

        output = process_cancer(study_name)
        df = output["combined"].copy()

        df.columns = [
            canonical(col)
            if isinstance(col, str) and col.startswith(selected_tags)
            else col
            for col in df.columns
        ]

        df = df.loc[:, ~df.columns.duplicated()]
        frames.append(df)

    return pd.concat(frames, ignore_index=True, sort=False)


def prepare_fixed_horizon_outcome(study_df, horizon_days):
    """
    Construct a fixed-horizon binary recurrence outcome.

    Positive:
        Recurrence occurred on or before the horizon.

    Negative:
        The patient remained recurrence-free through the horizon,
        including patients whose recurrence occurred after the horizon.

    Excluded:
        No recurrence was recorded, but follow-up ended before the horizon.
    """
    event = parse_binary_event(study_df[EVENT_COL])
    time = pd.to_numeric(study_df[TIME_COL], errors="coerce")

    valid = event.notna() & time.notna() & (time > 0)

    event = event.loc[valid].astype(bool)
    time = time.loc[valid]
    model_df = study_df.loc[valid].copy()

    known_outcome = event | (time >= horizon_days)

    model_df = model_df.loc[known_outcome].copy()
    event = event.loc[known_outcome]
    time = time.loc[known_outcome]

    target = (event & (time <= horizon_days)).astype(int)

    model_df = model_df.reset_index(drop=True)
    target = target.reset_index(drop=True)

    return model_df, target


def make_model(missing_threshold):
    """
    Complete fold-specific preprocessing and classification pipeline.
    """
    return Pipeline([
        (
            "feature_filter",
            TrainingFeatureFilter(
                missing_threshold=missing_threshold,
            ),
        ),
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "selector",
            SelectKBest(
                score_func=f_classif,
                k=20,
            ),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "classifier",
            LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                class_weight=None,
                max_iter=20000,
                tol=1e-3,
                random_state=42,
            ),
        ),
    ])


def make_parameter_grid(number_of_input_features):
    feature_counts = [
        count
        for count in FEATURE_COUNT_GRID
        if count <= number_of_input_features
    ]

    if not feature_counts:
        feature_counts = [number_of_input_features]

    return {
        "selector__k": feature_counts,
        "classifier__C": C_GRID,
        "classifier__l1_ratio": L1_RATIO_GRID,
    }


def make_stratified_cv(requested_folds, target, random_state):
    smallest_class = int(target.value_counts().min())
    number_of_folds = min(requested_folds, smallest_class)

    if number_of_folds < 2:
        raise ValueError(
            "At least two patients from each class are required "
            "for stratified cross-validation."
        )

    return StratifiedKFold(
        n_splits=number_of_folds,
        shuffle=True,
        random_state=random_state,
    )


def extract_feature_importance(fitted_model):
    """
    Match the final coefficients to the features retained by both
    filtering and SelectKBest.
    """
    filtered_features = fitted_model.named_steps[
        "feature_filter"
    ].get_feature_names_out()

    selector = fitted_model.named_steps["selector"]
    selected_features = filtered_features[selector.get_support()]

    coefficients = fitted_model.named_steps[
        "classifier"
    ].coef_[0]

    importance = pd.DataFrame({
        "feature": selected_features,
        "coefficient": coefficients,
        "odds_ratio": np.exp(np.clip(coefficients, -20, 20)),
        "absolute_coefficient": np.abs(coefficients),
    })

    return importance.sort_values(
        "absolute_coefficient",
        ascending=False,
    )


def main():
    parser = ArgumentParser()

    parser.add_argument(
        "--study",
        nargs="+",
        required=True,
        help="Cancer studies to use, such as luad, pdac, or all.",
    )

    parser.add_argument(
        "--modality",
        nargs="+",
        default=["all"],
        help=(
            "Use proteomics, phosphoproteomics, "
            "acetylproteomics, or all."
        ),
    )

    parser.add_argument(
        "--horizon-days",
        type=float,
        default=730,
        help="Prediction horizon in days. Default is 730 days.",
    )

    parser.add_argument(
        "--missing-threshold",
        type=float,
        default=0.60,
        help=(
            "Within each training fold, drop features missing in "
            "this fraction of patients."
        ),
    )

    parser.add_argument(
        "--outer-folds",
        type=int,
        default=5,
        help="Outer stratified folds used to evaluate performance.",
    )

    parser.add_argument(
        "--inner-folds",
        type=int,
        default=3,
        help="Inner stratified folds used for GridSearchCV.",
    )

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel jobs used by GridSearchCV.",
    )

    parser.add_argument(
        "--verbose",
        type=int,
        default=0,
        help="GridSearchCV verbosity.",
    )

    args = parser.parse_args()

    requested_studies = [
        study.lower()
        for study in args.study
    ]

    if "all" in requested_studies:
        studies = [
            study.lower()
            for study in ALL_STUDIES
        ]
    else:
        studies = requested_studies

    selected_tags = select_modalities(args.modality)
    study_df = load_studies(studies, selected_tags)

    features = [
        col
        for col in study_df.columns
        if isinstance(col, str) and col.startswith(selected_tags)
    ]

    if not features:
        raise ValueError(
            f"No features found for modality selection: {args.modality}"
        )

    model_df, target = prepare_fixed_horizon_outcome(
        study_df,
        args.horizon_days,
    )

    if target.nunique() < 2:
        raise ValueError(
            "The selected data do not contain both recurrence classes."
        )

    X = (
        model_df[features]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )

    cancer_types = (
        model_df[CANCER_COL]
        .astype(str)
        .reset_index(drop=True)
    )

    print("\nOutcome information:")
    print(f"Prediction horizon: {args.horizon_days:.0f} days")
    print(f"Usable patients: {len(model_df)}")
    print(
        target.value_counts()
        .rename({0: "No recurrence", 1: "Recurrence"})
    )

    print("\nPatients by cancer:")
    print(cancer_types.value_counts())

    print(f"\nInput features before fold-specific filtering: {X.shape[1]}")

    outer_cv = make_stratified_cv(
        requested_folds=args.outer_folds,
        target=target,
        random_state=42,
    )

    outer_strata = (
        cancer_types
        + "_"
        + target.astype(str)
    )

    if outer_strata.value_counts().min() < outer_cv.n_splits:
        outer_strata = target

    parameter_grid = make_parameter_grid(X.shape[1])

    outer_results = []
    out_of_fold_probability = np.full(len(target), np.nan)
    out_of_fold_prediction = np.full(len(target), -1, dtype=int)

    for fold, (train_index, test_index) in enumerate(
        outer_cv.split(X, outer_strata),
        start=1,
    ):
        print(
            f"\nOuter fold {fold}/{outer_cv.n_splits}: "
            f"train={len(train_index)}, test={len(test_index)}"
        )

        X_outer_train = X.iloc[train_index]
        X_outer_test = X.iloc[test_index]
        y_outer_train = target.iloc[train_index]
        y_outer_test = target.iloc[test_index]

        inner_cv = make_stratified_cv(
            requested_folds=args.inner_folds,
            target=y_outer_train,
            random_state=100 + fold,
        )

        search = GridSearchCV(
            estimator=make_model(args.missing_threshold),
            param_grid=parameter_grid,
            scoring="roc_auc",
            cv=inner_cv,
            n_jobs=args.n_jobs,
            refit=True,
            verbose=args.verbose,
            error_score="raise",
        )

        search.fit(X_outer_train, y_outer_train)

        best_model = search.best_estimator_
        probability = best_model.predict_proba(
            X_outer_test
        )[:, 1]
        prediction = (probability >= 0.5).astype(int)

        out_of_fold_probability[test_index] = probability
        out_of_fold_prediction[test_index] = prediction

        fold_auc = roc_auc_score(
            y_outer_test,
            probability,
        )
        fold_average_precision = average_precision_score(
            y_outer_test,
            probability,
        )
        fold_balanced_accuracy = balanced_accuracy_score(
            y_outer_test,
            prediction,
        )

        outer_results.append({
            "fold": fold,
            "train_samples": len(train_index),
            "test_samples": len(test_index),
            "roc_auc": fold_auc,
            "average_precision": fold_average_precision,
            "balanced_accuracy": fold_balanced_accuracy,
            "best_k": search.best_params_["selector__k"],
            "best_C": search.best_params_["classifier__C"],
            "best_l1_ratio": search.best_params_[
                "classifier__l1_ratio"
            ],
        })

        print(f"Best parameters: {search.best_params_}")
        print(f"Outer-fold ROC-AUC: {fold_auc:.4f}")
        print(
            "Outer-fold average precision: "
            f"{fold_average_precision:.4f}"
        )

    results_df = pd.DataFrame(outer_results)

    print("\nNested cross-validation fold results:")
    print(results_df.to_string(index=False))

    print("\nNested cross-validation summary:")
    print(
        "Mean outer ROC-AUC: "
        f"{results_df['roc_auc'].mean():.4f} "
        f"+/- {results_df['roc_auc'].std(ddof=1):.4f}"
    )
    print(
        "Mean outer average precision: "
        f"{results_df['average_precision'].mean():.4f} "
        f"+/- {results_df['average_precision'].std(ddof=1):.4f}"
    )
    print(
        "Mean outer balanced accuracy: "
        f"{results_df['balanced_accuracy'].mean():.4f} "
        f"+/- {results_df['balanced_accuracy'].std(ddof=1):.4f}"
    )

    pooled_auc = roc_auc_score(
        target,
        out_of_fold_probability,
    )
    pooled_average_precision = average_precision_score(
        target,
        out_of_fold_probability,
    )
    pooled_balanced_accuracy = balanced_accuracy_score(
        target,
        out_of_fold_prediction,
    )

    print("\nPooled out-of-fold results:")
    print(f"ROC-AUC: {pooled_auc:.4f}")
    print(
        "Average precision: "
        f"{pooled_average_precision:.4f}"
    )
    print(
        "Balanced accuracy: "
        f"{pooled_balanced_accuracy:.4f}"
    )

    print("\nPooled out-of-fold confusion matrix:")
    print(confusion_matrix(target, out_of_fold_prediction))

    print("\nPooled out-of-fold classification report:")
    print(
        classification_report(
            target,
            out_of_fold_prediction,
            digits=4,
            zero_division=0,
        )
    )

    if cancer_types.nunique() > 1:
        print("\nCancer-specific pooled out-of-fold ROC-AUC:")

        for cancer in sorted(cancer_types.unique()):
            cancer_mask = cancer_types == cancer
            cancer_target = target.loc[cancer_mask]

            if (
                len(cancer_target) < 4
                or cancer_target.nunique() < 2
            ):
                print(
                    f"{cancer.upper()}: insufficient outcomes "
                    f"(n={len(cancer_target)})"
                )
                continue

            cancer_auc = roc_auc_score(
                cancer_target,
                out_of_fold_probability[cancer_mask.to_numpy()],
            )

            print(
                f"{cancer.upper()}: "
                f"ROC-AUC={cancer_auc:.4f}, "
                f"n={len(cancer_target)}"
            )

    print(
        "\nFitting one final model on all patients for "
        "feature interpretation..."
    )

    final_inner_cv = make_stratified_cv(
        requested_folds=args.inner_folds,
        target=target,
        random_state=999,
    )

    final_search = GridSearchCV(
        estimator=make_model(args.missing_threshold),
        param_grid=parameter_grid,
        scoring="roc_auc",
        cv=final_inner_cv,
        n_jobs=args.n_jobs,
        refit=True,
        verbose=args.verbose,
        error_score="raise",
    )

    final_search.fit(X, target)
    final_model = final_search.best_estimator_

    print("\nFinal full-data model parameters:")
    print(final_search.best_params_)
    print(
        "Final inner validation ROC-AUC "
        "(tuning only, not final performance): "
        f"{final_search.best_score_:.4f}"
    )

    importance = extract_feature_importance(final_model)
    nonzero_importance = importance[
        importance["coefficient"] != 0
    ]

    print(
        "\nFinal selected features: "
        f"{len(importance)}"
    )
    print(
        "Final nonzero-coefficient features: "
        f"{len(nonzero_importance)}"
    )

    print("\nTop recurrence-associated features:")
    print(
        nonzero_importance[
            nonzero_importance["coefficient"] > 0
        ]
        .head(10)
        .to_string(index=False)
    )

    print("\nTop non-recurrence-associated features:")
    print(
        nonzero_importance[
            nonzero_importance["coefficient"] < 0
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()