from argparse import ArgumentParser

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
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
    requested_modalities = [m.lower() for m in requested_modalities]

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

    return tuple(MODALITY_TO_TAG[m] for m in requested_modalities)


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
    Construct a fixed-horizon binary outcome.

    Positive:
        Recurrence occurred on or before the horizon.

    Negative:
        No recurrence was observed through the horizon, or recurrence
        occurred after the horizon.

    Excluded:
        No recurrence, but follow-up ended before the horizon.
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

    # A late recurrence is negative for recurrence by this horizon.
    target = (event & (time <= horizon_days)).astype(int)

    return model_df, target


def make_model(C=1.0, l1_ratio=0.5, number_features=20):
    return Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "selector",
            SelectKBest(
                score_func=f_classif,
                k=number_features,
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
                C=C,
                l1_ratio=l1_ratio,
                class_weight=None,
                max_iter=20000,
                tol=1e-3,
                random_state=42,
            ),
        ),
    ])


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
        help="Drop features missing in this fraction of training samples.",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--C",
        type=float,
        default=0.1,
        help="Inverse regularization strength.",
    )

    parser.add_argument(
        "--l1-ratio",
        type=float,
        default=0.5,
        help="0 is ridge-like, 1 is lasso-like.",
    )

    parser.add_argument(
        "--number-features",
        type=int,
        default=20,
        help=(
            "Number of features selected before logistic regression "
            "when --tune is not used."
        ),
    )

    parser.add_argument(
        "--tune",
        action="store_true",
        help="Tune C and l1_ratio using training-set cross-validation.",
    )

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel jobs for GridSearchCV.",
    )

    args = parser.parse_args()

    requested_studies = [study.lower() for study in args.study]

    if "all" in requested_studies:
        studies = [study.lower() for study in ALL_STUDIES]
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

    print("\nOutcome information:")
    print(f"Prediction horizon: {args.horizon_days:.0f} days")
    print(f"Usable patients: {len(model_df)}")
    print(target.value_counts().rename({0: "No recurrence", 1: "Recurrence"}))

    print("\nPatients by cancer:")
    print(model_df[CANCER_COL].value_counts())

    X = model_df[features].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    cancer_types = model_df[CANCER_COL].astype(str)

    # Keep cancer and recurrence proportions similar in train and test.
    cancer_event_strata = cancer_types + "_" + target.astype(str)

    if cancer_event_strata.value_counts().min() >= 2:
        stratify = cancer_event_strata
    else:
        stratify = target

    (
        X_train,
        X_test,
        y_train,
        y_test,
        cancer_train,
        cancer_test,
    ) = train_test_split(
        X,
        target,
        cancer_types,
        test_size=args.test_size,
        random_state=42,
        stratify=stratify,
    )

    # Feature filtering is based only on the training set.
    training_missingness = X_train.isna().mean()
    training_unique_values = X_train.nunique(dropna=True)

    keep_features = (
        (training_missingness < args.missing_threshold)
        & (training_unique_values > 1)
    )

    X_train = X_train.loc[:, keep_features]
    X_test = X_test.loc[:, X_train.columns]

    if X_train.shape[1] == 0:
        raise ValueError(
            "No features remain after training-set feature filtering."
        )

    print(f"\nTraining patients: {X_train.shape[0]}")
    print(f"Testing patients: {X_test.shape[0]}")
    print(f"Features after filtering: {X_train.shape[1]}")

    if args.number_features < 1:
        raise ValueError("--number-features must be at least 1.")

    number_features = min(args.number_features, X_train.shape[1])

    model = make_model(
        C=args.C,
        l1_ratio=args.l1_ratio,
        number_features=number_features,
    )

    if args.tune:
        smallest_class = int(y_train.value_counts().min())
        cv_folds = min(3, smallest_class)

        if cv_folds < 2:
            raise ValueError(
                "Not enough patients in both classes for cross-validation."
            )

        cross_validation = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=42,
        )

        feature_counts = [
            count
            for count in [5, 10, 20, 30]
            if count <= X_train.shape[1]
        ]

        if not feature_counts:
            feature_counts = [X_train.shape[1]]

        parameter_grid = {
            "selector__k": feature_counts,
            "classifier__C": [0.1, 1.0, 10.0],
            "classifier__l1_ratio": [0.1, 0.5, 0.9],
        }

        search = GridSearchCV(
            estimator=model,
            param_grid=parameter_grid,
            scoring="roc_auc",
            cv=cross_validation,
            n_jobs=args.n_jobs,
            refit=True,
            verbose=1,
        )

        search.fit(X_train, y_train)

        fitted_model = search.best_estimator_

        print("\nBest parameters:")
        print(search.best_params_)
        print(f"Mean validation ROC-AUC: {search.best_score_:.4f}")

        tuning_results = pd.DataFrame(search.cv_results_)
        tuning_columns = [
            "param_selector__k",
            "param_classifier__C",
            "param_classifier__l1_ratio",
            "mean_test_score",
            "std_test_score",
            "rank_test_score",
        ]

        print("\nTop cross-validation configurations:")
        print(
            tuning_results[tuning_columns]
            .sort_values(
                ["rank_test_score", "std_test_score"],
                ascending=[True, True],
            )
            .head(15)
            .to_string(index=False)
        )

    else:
        fitted_model = model.fit(X_train, y_train)

    recurrence_probability = fitted_model.predict_proba(X_test)[:, 1]
    predictions = (recurrence_probability >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_test, recurrence_probability)
    average_precision = average_precision_score(
        y_test,
        recurrence_probability,
    )
    balanced_accuracy = balanced_accuracy_score(y_test, predictions)

    print("\nOverall test results:")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Average precision: {average_precision:.4f}")
    print(f"Balanced accuracy: {balanced_accuracy:.4f}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            digits=4,
            zero_division=0,
        )
    )

    # Evaluate each cancer separately in the pan-cancer test set.
    if cancer_test.nunique() > 1:
        print("\nCancer-specific test ROC-AUC:")

        for cancer in sorted(cancer_test.unique()):
            cancer_mask = cancer_test == cancer
            cancer_y = y_test.loc[cancer_mask]
            cancer_probability = recurrence_probability[cancer_mask.values]

            if len(cancer_y) < 4 or cancer_y.nunique() < 2:
                print(
                    f"{cancer.upper()}: insufficient test events "
                    f"(n={len(cancer_y)})"
                )
                continue

            cancer_auc = roc_auc_score(
                cancer_y,
                cancer_probability,
            )

            print(
                f"{cancer.upper()}: ROC-AUC={cancer_auc:.4f}, "
                f"n={len(cancer_y)}"
            )

    selector = fitted_model.named_steps["selector"]
    selected_features = X_train.columns[selector.get_support()]
    coefficients = fitted_model.named_steps["classifier"].coef_[0]

    importance = pd.DataFrame({
        "feature": selected_features,
        "coefficient": coefficients,
        "absolute_coefficient": np.abs(coefficients),
    })

    importance = importance[
        importance["coefficient"] != 0
    ].sort_values(
        "absolute_coefficient",
        ascending=False,
    )

    print("\nTop recurrence-associated features:")
    print(
        importance[importance["coefficient"] > 0]
        .head(10)
        .to_string(index=False)
    )

    print("\nTop non-recurrence-associated features:")
    print(
        importance[importance["coefficient"] < 0]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()