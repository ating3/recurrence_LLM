from argparse import ArgumentParser
from collections import defaultdict
import json
import re

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import (
    ParameterGrid,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from query_patients import ALL_STUDIES, process_cancer


EVENT_COL = "Recurrence"
TIME_COL = "Derived recurrence-free survival time, days"
CANCER_COL = "Cancer_Type"
SAMPLE_ID_COL = "_Evaluation_Sample_ID"

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

# Stronger regularization than the earlier [0.1, 1, 10] grid.
# Smaller C means stronger regularization.
C_GRID = [
    0.001,
    0.003,
    0.01,
    0.03,
    0.1,
    0.3,
    1.0,
]
L1_RATIO_GRID = [0.1, 0.5, 0.9]
CLASS_WEIGHT_GRID = [None, "balanced"]
FEATURE_COUNT_GRID = [5, 10, 20, 30]

# Exact normalized aliases only. Broad substring matching is intentionally
# avoided because it could accidentally include outcome-derived columns.
SAFE_CLINICAL_ALIASES = {
    "age",
    "age_at_diagnosis",
    "age_at_index",
    "age_at_initial_pathologic_diagnosis",
    "gender",
    "sex",
    "race",
    "ethnicity",
    "tumor_stage",
    "clinical_stage",
    "pathologic_stage",
    "pathological_stage",
    "ajcc_stage",
    "ajcc_pathologic_stage",
    "ajcc_clinical_stage",
    "tumor_grade",
    "histologic_grade",
    "histological_grade",
    "grade",
    "pathologic_t",
    "pathologic_n",
    "pathologic_m",
    "clinical_t",
    "clinical_n",
    "clinical_m",
    "ajcc_pathologic_t",
    "ajcc_pathologic_n",
    "ajcc_pathologic_m",
    "tumor_size",
    "tumor_size_cm",
    "smoking_status",
    "tobacco_smoking_history",
    "pack_years",
}

LEAKAGE_KEYWORDS = {
    "recurr",
    "relapse",
    "survival",
    "followup",
    "follow_up",
    "last_contact",
    "vital_status",
    "death",
    "deceased",
    "censor",
    "progression",
    "outcome",
    "days_to",
}


class TrainingFeatureFilter(BaseEstimator, TransformerMixin):
    """
    Remove highly missing, empty, and constant omics features.

    This transformer remains inside every pipeline, so the retained columns
    are learned using only the current training fold.
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

        self.feature_names_in_ = np.asarray(
            X.columns,
            dtype=object,
        )
        self.selected_columns_ = np.asarray(
            X.columns[keep],
            dtype=object,
        )

        if len(self.selected_columns_) == 0:
            raise ValueError(
                "No omics features remain after missingness and "
                "variance filtering."
            )

        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(
                X,
                columns=self.feature_names_in_,
            )

        return X.loc[:, self.selected_columns_]

    def get_feature_names_out(self, input_features=None):
        return self.selected_columns_


class SigmoidScoreCalibrator:
    """
    One-dimensional sigmoid calibration fitted to out-of-fold scores.

    The base recurrence model produces a log-odds decision score. This class
    learns a sigmoid mapping from that score to an empirically calibrated
    probability. It never sees an outer evaluation patient during fitting.
    """

    def __init__(self):
        self.model_ = None
        self.constant_probability_ = None

    def fit(self, scores, target):
        scores = np.asarray(scores, dtype=float).reshape(-1, 1)
        target = np.asarray(target, dtype=int)

        if not np.isfinite(scores).all():
            raise ValueError(
                "Calibration received a non-finite decision score."
            )

        if np.unique(scores).size < 2:
            self.constant_probability_ = float(target.mean())
            return self

        self.model_ = LogisticRegression(
            penalty="l2",
            C=1_000_000.0,
            solver="lbfgs",
            max_iter=5000,
        )
        self.model_.fit(scores, target)

        return self

    def predict_proba(self, scores):
        scores = np.asarray(scores, dtype=float).reshape(-1, 1)

        if self.model_ is None:
            probability = np.full(
                scores.shape[0],
                self.constant_probability_,
                dtype=float,
            )
        else:
            probability = self.model_.predict_proba(scores)[:, 1]

        return np.clip(probability, 1e-6, 1.0 - 1e-6)


def canonical(col: str) -> str:
    """
    Make feature names comparable across CPTAC studies.

    Proteomics retains the gene name. PTM modalities retain the gene and
    modification site.
    """
    if not isinstance(col, str) or "::" not in col:
        return col

    modality, _, rest = col.partition("::")
    parts = [
        part
        for part in rest.split("|")
        if part
    ]

    if not parts:
        return col

    if modality == "proteomics":
        return f"{modality}::{parts[0]}"

    return f"{modality}::" + "|".join(parts[:2])


def normalize_column_name(column_name) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(column_name).strip().lower(),
    ).strip("_")


def parse_binary_event(series: pd.Series) -> pd.Series:
    """
    Convert recurrence labels to 0/1 without treating the text "False"
    as truthy.
    """
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

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


def find_sample_ids(df, study_name):
    candidate_columns = [
        "Patient_ID",
        "patient_id",
        "case_id",
        "Case_ID",
        "Sample_ID",
        "sample_id",
    ]

    for column in candidate_columns:
        if column not in df.columns:
            continue

        values = df[column].astype("string")

        if values.notna().sum() >= max(2, int(0.5 * len(df))):
            return [
                f"{study_name.upper()}::{value}"
                for value in values.fillna("missing")
            ]

    if not isinstance(df.index, pd.RangeIndex):
        values = [
            "|".join(map(str, value))
            if isinstance(value, tuple)
            else str(value)
            for value in df.index
        ]
    else:
        values = [
            f"row_{position}"
            for position in range(len(df))
        ]

    return [
        f"{study_name.upper()}::{value}"
        for value in values
    ]


def load_studies(studies, selected_tags):
    frames = []

    for study_name in studies:
        print(f"Loading {study_name.upper()}")

        output = process_cancer(study_name)
        df = output["combined"].copy()
        df[SAMPLE_ID_COL] = find_sample_ids(
            df,
            study_name,
        )

        df.columns = [
            canonical(col)
            if isinstance(col, str) and col.startswith(selected_tags)
            else col
            for col in df.columns
        ]

        df = df.loc[:, ~df.columns.duplicated()]
        frames.append(df)

    return pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )


def prepare_fixed_horizon_outcome(study_df, horizon_days):
    """
    Construct a fixed-horizon binary recurrence outcome.

    Positive:
        Recurrence occurred on or before the horizon.

    Negative:
        The patient remained recurrence-free through the horizon, including
        patients whose recurrence occurred after the horizon.

    Excluded:
        No recurrence was recorded, but follow-up ended before the horizon.
    """
    event = parse_binary_event(study_df[EVENT_COL])
    time = pd.to_numeric(
        study_df[TIME_COL],
        errors="coerce",
    )

    valid = (
        event.notna()
        & time.notna()
        & (time > 0)
    )

    event = event.loc[valid].astype(bool)
    time = time.loc[valid]
    model_df = study_df.loc[valid].copy()

    known_outcome = event | (time >= horizon_days)

    model_df = model_df.loc[known_outcome].copy()
    event = event.loc[known_outcome]
    time = time.loc[known_outcome]

    target = (
        event
        & (time <= horizon_days)
    ).astype(int)

    model_df = model_df.reset_index(drop=True)
    target = target.reset_index(drop=True)

    return model_df, target


def column_is_leakage(column_name) -> bool:
    normalized = normalize_column_name(column_name)

    return any(
        keyword in normalized
        for keyword in LEAKAGE_KEYWORDS
    )


def resolve_clinical_columns(
    study_df,
    requested_columns,
    include_cancer_type,
):
    non_omics_columns = [
        column
        for column in study_df.columns
        if (
            isinstance(column, str)
            and not column.startswith((
                PROTEOMICS_TAG,
                PHOSPHOPROTEOMICS_TAG,
                ACETYLPROTEOMICS_TAG,
            ))
        )
    ]

    normalized_lookup = defaultdict(list)

    for column in non_omics_columns:
        normalized_lookup[
            normalize_column_name(column)
        ].append(column)

    selected = []

    if requested_columns:
        for requested in requested_columns:
            if requested in study_df.columns:
                matches = [requested]
            else:
                matches = normalized_lookup[
                    normalize_column_name(requested)
                ]

            if len(matches) == 0:
                raise ValueError(
                    f"Clinical column not found: {requested}"
                )

            if len(matches) > 1:
                raise ValueError(
                    f"Clinical column '{requested}' is ambiguous: "
                    f"{matches}"
                )

            selected.append(matches[0])
    else:
        for normalized_name in sorted(SAFE_CLINICAL_ALIASES):
            matches = normalized_lookup.get(
                normalized_name,
                [],
            )

            if len(matches) == 1:
                selected.append(matches[0])

    if (
        include_cancer_type
        and CANCER_COL in study_df.columns
    ):
        selected.append(CANCER_COL)

    selected = list(dict.fromkeys(selected))

    unsafe = [
        column
        for column in selected
        if column_is_leakage(column)
    ]

    if unsafe:
        raise ValueError(
            "The following clinical columns appear outcome-derived "
            f"and are blocked to prevent leakage: {unsafe}"
        )

    selected = [
        column
        for column in selected
        if (
            study_df[column].notna().sum() >= 2
            and study_df[column].nunique(dropna=True) > 1
        )
    ]

    return selected


def prepare_clinical_dataframe(
    model_df,
    clinical_columns,
):
    clinical_df = model_df[
        clinical_columns
    ].copy()

    numeric_columns = []
    categorical_columns = []

    for column in clinical_columns:
        original = clinical_df[column]
        nonmissing_count = int(original.notna().sum())
        numeric = pd.to_numeric(
            original,
            errors="coerce",
        )

        required_numeric = max(
            2,
            int(np.ceil(0.80 * nonmissing_count)),
        )

        if numeric.notna().sum() >= required_numeric:
            clinical_df[column] = numeric
            numeric_columns.append(column)
        else:
            categorical = original.map(
                lambda value: (
                    str(value)
                    if pd.notna(value)
                    else np.nan
                )
            )
            clinical_df[column] = categorical
            categorical_columns.append(column)

    return (
        clinical_df,
        numeric_columns,
        categorical_columns,
    )


def make_one_hot_encoder():
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )
    except TypeError:
        # Compatibility with older scikit-learn versions.
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
        )


def make_clinical_transformer(
    numeric_columns,
    categorical_columns,
):
    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ])
        transformers.append((
            "numeric",
            numeric_pipeline,
            numeric_columns,
        ))

    if categorical_columns:
        categorical_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "one_hot",
                make_one_hot_encoder(),
            ),
        ])
        transformers.append((
            "categorical",
            categorical_pipeline,
            categorical_columns,
        ))

    if not transformers:
        raise ValueError(
            "No usable clinical columns were available."
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=True,
    )


def make_omics_transformer(missing_threshold):
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
    ])


def make_omics_classifier():
    return LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        C=0.1,
        l1_ratio=0.5,
        class_weight="balanced",
        max_iter=100_000,
        tol=1e-3,
        random_state=42,
    )


def make_clinical_classifier():
    return LogisticRegression(
        penalty="l2",
        solver="liblinear",
        C=0.1,
        class_weight="balanced",
        max_iter=20_000,
        random_state=42,
    )


def make_estimator(
    model_name,
    omics_columns,
    clinical_columns,
    numeric_clinical_columns,
    categorical_clinical_columns,
    missing_threshold,
):
    if model_name == "omics":
        return Pipeline([
            (
                "omics",
                make_omics_transformer(
                    missing_threshold
                ),
            ),
            (
                "classifier",
                make_omics_classifier(),
            ),
        ])

    if model_name == "clinical":
        return Pipeline([
            (
                "clinical",
                make_clinical_transformer(
                    numeric_clinical_columns,
                    categorical_clinical_columns,
                ),
            ),
            (
                "classifier",
                make_clinical_classifier(),
            ),
        ])

    if model_name == "combined":
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "omics",
                    make_omics_transformer(
                        missing_threshold
                    ),
                    omics_columns,
                ),
                (
                    "clinical",
                    make_clinical_transformer(
                        numeric_clinical_columns,
                        categorical_clinical_columns,
                    ),
                    clinical_columns,
                ),
            ],
            remainder="drop",
            sparse_threshold=0.0,
            verbose_feature_names_out=False,
        )

        return Pipeline([
            (
                "preprocess",
                preprocessor,
            ),
            (
                "classifier",
                make_omics_classifier(),
            ),
        ])

    raise ValueError(
        f"Unknown model name: {model_name}"
    )


def make_parameter_grid(
    model_name,
    number_of_omics_features,
):
    if model_name == "clinical":
        return {
            "classifier__C": C_GRID,
            "classifier__class_weight": CLASS_WEIGHT_GRID,
        }

    feature_counts = [
        count
        for count in FEATURE_COUNT_GRID
        if count <= number_of_omics_features
    ]

    if not feature_counts:
        feature_counts = [number_of_omics_features]

    if model_name == "omics":
        selector_key = "omics__selector__k"
    else:
        selector_key = "preprocess__omics__selector__k"

    return {
        selector_key: feature_counts,
        "classifier__C": C_GRID,
        "classifier__l1_ratio": L1_RATIO_GRID,
        "classifier__class_weight": CLASS_WEIGHT_GRID,
    }


def make_stratified_splits(
    target,
    cancer_types,
    requested_folds,
    random_state,
):
    target = pd.Series(target).reset_index(drop=True)
    cancer_types = pd.Series(
        cancer_types
    ).astype(str).reset_index(drop=True)

    smallest_class = int(
        target.value_counts().min()
    )
    number_of_folds = min(
        requested_folds,
        smallest_class,
    )

    if number_of_folds < 2:
        raise ValueError(
            "At least two patients from each class are required "
            "for stratified cross-validation."
        )

    cancer_outcome_strata = (
        cancer_types
        + "_"
        + target.astype(str)
    )

    if (
        cancer_outcome_strata.value_counts().min()
        >= number_of_folds
    ):
        strata = cancer_outcome_strata
    else:
        strata = target

    splitter = StratifiedKFold(
        n_splits=number_of_folds,
        shuffle=True,
        random_state=random_state,
    )

    dummy_X = np.zeros(
        (len(target), 1),
        dtype=float,
    )

    return (
        list(splitter.split(dummy_X, strata)),
        number_of_folds,
    )


def fit_sigmoid_calibration(
    fitted_model,
    X_train,
    y_train,
    inner_splits,
    n_jobs,
    calibration,
):
    if calibration == "none":
        inner_probability = cross_val_predict(
            estimator=clone(fitted_model),
            X=X_train,
            y=y_train,
            cv=inner_splits,
            method="predict_proba",
            n_jobs=n_jobs,
        )[:, 1]

        return None, inner_probability

    inner_decision_score = cross_val_predict(
        estimator=clone(fitted_model),
        X=X_train,
        y=y_train,
        cv=inner_splits,
        method="decision_function",
        n_jobs=n_jobs,
    )

    calibrator = SigmoidScoreCalibrator()
    calibrator.fit(
        inner_decision_score,
        y_train,
    )

    inner_probability = calibrator.predict_proba(
        inner_decision_score
    )

    return calibrator, inner_probability


def predict_probability(
    fitted_model,
    calibrator,
    X,
):
    if calibrator is None:
        return fitted_model.predict_proba(X)[:, 1]

    decision_score = fitted_model.decision_function(X)

    return calibrator.predict_proba(
        decision_score
    )


def calculate_binary_counts(target, prediction):
    tn, fp, fn, tp = confusion_matrix(
        target,
        prediction,
        labels=[0, 1],
    ).ravel()

    return (
        int(tn),
        int(fp),
        int(fn),
        int(tp),
    )


def choose_sensitivity_threshold(
    target,
    probabilities,
    target_sensitivity,
):
    """
    Select the most specific threshold that reaches the requested sensitivity.

    This function is called only with inner out-of-fold training predictions.
    """
    target = np.asarray(
        target,
        dtype=int,
    )
    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    if not 0 < target_sensitivity <= 1:
        raise ValueError(
            "target_sensitivity must be greater than 0 and at most 1."
        )

    unique_probabilities = np.unique(
        probabilities
    )

    if unique_probabilities.size > 1:
        midpoints = (
            unique_probabilities[:-1]
            + unique_probabilities[1:]
        ) / 2.0
    else:
        midpoints = np.asarray([], dtype=float)

    candidates = np.unique(
        np.concatenate([
            np.asarray([0.0, 0.5, 1.0]),
            unique_probabilities,
            midpoints,
        ])
    )

    valid_records = []

    for threshold in candidates:
        prediction = (
            probabilities >= threshold
        ).astype(int)

        tn, fp, fn, tp = calculate_binary_counts(
            target,
            prediction,
        )

        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        balanced_accuracy = (
            sensitivity + specificity
        ) / 2.0

        if sensitivity + 1e-12 >= target_sensitivity:
            valid_records.append({
                "threshold": float(threshold),
                "sensitivity": float(sensitivity),
                "specificity": float(specificity),
                "balanced_accuracy": float(
                    balanced_accuracy
                ),
            })

    if not valid_records:
        raise RuntimeError(
            "No threshold reached the requested sensitivity."
        )

    valid_records.sort(
        key=lambda record: (
            -record["specificity"],
            -record["balanced_accuracy"],
            -record["threshold"],
        )
    )

    return valid_records[0]


def calculate_metrics(
    target,
    probabilities,
    predictions,
):
    target = np.asarray(target, dtype=int)
    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )
    predictions = np.asarray(
        predictions,
        dtype=int,
    )

    tn, fp, fn, tp = calculate_binary_counts(
        target,
        predictions,
    )

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )
    negative_predictive_value = (
        tn / (tn + fn)
        if (tn + fn) > 0
        else 0.0
    )

    return {
        "roc_auc": roc_auc_score(
            target,
            probabilities,
        ),
        "average_precision": average_precision_score(
            target,
            probabilities,
        ),
        "accuracy": accuracy_score(
            target,
            predictions,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            target,
            predictions,
        ),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "negative_predictive_value": (
            negative_predictive_value
        ),
        "predicted_recurrence_rate": float(
            predictions.mean()
        ),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
    }


def extract_selected_omics_features(
    fitted_model,
    model_name,
):
    if model_name == "clinical":
        return np.asarray([], dtype=object), np.asarray([])

    if model_name == "omics":
        omics_transformer = fitted_model.named_steps[
            "omics"
        ]
    else:
        preprocessor = fitted_model.named_steps[
            "preprocess"
        ]
        omics_transformer = preprocessor.named_transformers_[
            "omics"
        ]

    filtered_features = omics_transformer.named_steps[
        "feature_filter"
    ].get_feature_names_out()
    selector = omics_transformer.named_steps[
        "selector"
    ]
    selected_features = filtered_features[
        selector.get_support()
    ]

    coefficients = fitted_model.named_steps[
        "classifier"
    ].coef_[0]
    omics_coefficients = coefficients[
        :len(selected_features)
    ]

    return (
        np.asarray(
            selected_features,
            dtype=object,
        ),
        np.asarray(
            omics_coefficients,
            dtype=float,
        ),
    )


def extract_all_feature_importance(
    fitted_model,
    model_name,
):
    coefficients = fitted_model.named_steps[
        "classifier"
    ].coef_[0]

    if model_name == "omics":
        feature_names, _ = (
            extract_selected_omics_features(
                fitted_model,
                model_name,
            )
        )
    elif model_name == "clinical":
        clinical_transformer = fitted_model.named_steps[
            "clinical"
        ]
        feature_names = (
            clinical_transformer.get_feature_names_out()
        )
    else:
        preprocessor = fitted_model.named_steps[
            "preprocess"
        ]
        omics_names, _ = (
            extract_selected_omics_features(
                fitted_model,
                model_name,
            )
        )
        clinical_names = preprocessor.named_transformers_[
            "clinical"
        ].get_feature_names_out()
        feature_names = np.concatenate([
            omics_names,
            clinical_names,
        ])

    if len(feature_names) != len(coefficients):
        raise RuntimeError(
            f"Feature/coefficient mismatch for {model_name}: "
            f"{len(feature_names)} features versus "
            f"{len(coefficients)} coefficients."
        )

    importance = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients,
        "odds_ratio": np.exp(
            np.clip(
                coefficients,
                -20,
                20,
            )
        ),
        "absolute_coefficient": np.abs(coefficients),
    })

    return importance.sort_values(
        "absolute_coefficient",
        ascending=False,
    )


def model_input(
    model_name,
    omics_df,
    clinical_df,
):
    if model_name == "omics":
        return omics_df

    if model_name == "clinical":
        return clinical_df

    return pd.concat(
        [
            omics_df,
            clinical_df,
        ],
        axis=1,
    )


def run_repeated_nested_cv(
    model_names,
    omics_df,
    clinical_df,
    target,
    cancer_types,
    sample_ids,
    observed_times,
    omics_columns,
    clinical_columns,
    numeric_clinical_columns,
    categorical_clinical_columns,
    missing_threshold,
    outer_folds,
    outer_repeats,
    inner_folds,
    tuning_iterations,
    target_sensitivity,
    calibration,
    n_jobs,
    random_seed,
    print_progress,
    collect_stability,
):
    prediction_records = []
    fold_metric_records = []
    stability_records = []

    for repeat in range(1, outer_repeats + 1):
        repeat_seed = (
            random_seed
            + repeat * 10_000
        )

        outer_splits, actual_outer_folds = (
            make_stratified_splits(
                target=target,
                cancer_types=cancer_types,
                requested_folds=outer_folds,
                random_state=repeat_seed,
            )
        )

        if print_progress:
            print(
                f"\nRepeated nested CV: repetition "
                f"{repeat}/{outer_repeats}"
            )

        for fold, (
            train_index,
            evaluation_index,
        ) in enumerate(
            outer_splits,
            start=1,
        ):
            y_train = target.iloc[
                train_index
            ].reset_index(drop=True)
            y_evaluation = target.iloc[
                evaluation_index
            ].reset_index(drop=True)

            cancer_train = cancer_types.iloc[
                train_index
            ].reset_index(drop=True)

            inner_splits, _ = make_stratified_splits(
                target=y_train,
                cancer_types=cancer_train,
                requested_folds=inner_folds,
                random_state=repeat_seed + fold,
            )

            for model_position, model_name in enumerate(
                model_names
            ):
                X_full = model_input(
                    model_name,
                    omics_df,
                    clinical_df,
                )
                X_train = X_full.iloc[
                    train_index
                ].reset_index(drop=True)
                X_evaluation = X_full.iloc[
                    evaluation_index
                ].reset_index(drop=True)

                estimator = make_estimator(
                    model_name=model_name,
                    omics_columns=omics_columns,
                    clinical_columns=clinical_columns,
                    numeric_clinical_columns=(
                        numeric_clinical_columns
                    ),
                    categorical_clinical_columns=(
                        categorical_clinical_columns
                    ),
                    missing_threshold=missing_threshold,
                )

                parameter_grid = make_parameter_grid(
                    model_name=model_name,
                    number_of_omics_features=(
                        len(omics_columns)
                    ),
                )

                total_configurations = len(
                    list(ParameterGrid(parameter_grid))
                )
                number_of_iterations = min(
                    tuning_iterations,
                    total_configurations,
                )

                search = RandomizedSearchCV(
                    estimator=estimator,
                    param_distributions=parameter_grid,
                    n_iter=number_of_iterations,
                    scoring="roc_auc",
                    cv=inner_splits,
                    refit=True,
                    random_state=(
                        repeat_seed
                        + fold * 100
                        + model_position
                    ),
                    n_jobs=n_jobs,
                    verbose=0,
                    error_score="raise",
                )

                search.fit(
                    X_train,
                    y_train,
                )

                fitted_model = search.best_estimator_

                (
                    calibrator,
                    inner_probability,
                ) = fit_sigmoid_calibration(
                    fitted_model=fitted_model,
                    X_train=X_train,
                    y_train=y_train,
                    inner_splits=inner_splits,
                    n_jobs=n_jobs,
                    calibration=calibration,
                )

                threshold_record = (
                    choose_sensitivity_threshold(
                        target=y_train,
                        probabilities=inner_probability,
                        target_sensitivity=(
                            target_sensitivity
                        ),
                    )
                )

                evaluation_probability = (
                    predict_probability(
                        fitted_model=fitted_model,
                        calibrator=calibrator,
                        X=X_evaluation,
                    )
                )
                evaluation_prediction = (
                    evaluation_probability
                    >= threshold_record["threshold"]
                ).astype(int)

                fold_metrics = calculate_metrics(
                    target=y_evaluation,
                    probabilities=(
                        evaluation_probability
                    ),
                    predictions=(
                        evaluation_prediction
                    ),
                )
                fold_metrics.update({
                    "model": model_name,
                    "repeat": repeat,
                    "fold": fold,
                    "outer_folds": actual_outer_folds,
                    "train_samples": len(train_index),
                    "evaluation_samples": (
                        len(evaluation_index)
                    ),
                    "inner_tuning_roc_auc": (
                        search.best_score_
                    ),
                    "decision_threshold": (
                        threshold_record["threshold"]
                    ),
                    "inner_threshold_sensitivity": (
                        threshold_record["sensitivity"]
                    ),
                    "inner_threshold_specificity": (
                        threshold_record["specificity"]
                    ),
                    "best_parameters": json.dumps(
                        search.best_params_,
                        sort_keys=True,
                    ),
                })
                fold_metric_records.append(
                    fold_metrics
                )

                for local_position, global_index in enumerate(
                    evaluation_index
                ):
                    prediction_records.append({
                        "model": model_name,
                        "repeat": repeat,
                        "fold": fold,
                        "sample_position": int(
                            global_index
                        ),
                        "sample_id": sample_ids.iloc[
                            global_index
                        ],
                        "cancer_type": cancer_types.iloc[
                            global_index
                        ],
                        "observed_time_days": (
                            observed_times.iloc[
                                global_index
                            ]
                        ),
                        "true_value": int(
                            target.iloc[global_index]
                        ),
                        "predicted_value": int(
                            evaluation_prediction[
                                local_position
                            ]
                        ),
                        "recurrence_probability": float(
                            evaluation_probability[
                                local_position
                            ]
                        ),
                        "decision_threshold": float(
                            threshold_record[
                                "threshold"
                            ]
                        ),
                        "correct": bool(
                            evaluation_prediction[
                                local_position
                            ]
                            == target.iloc[global_index]
                        ),
                    })

                if (
                    collect_stability
                    and model_name != "clinical"
                ):
                    (
                        selected_features,
                        selected_coefficients,
                    ) = extract_selected_omics_features(
                        fitted_model,
                        model_name,
                    )

                    for feature, coefficient in zip(
                        selected_features,
                        selected_coefficients,
                    ):
                        stability_records.append({
                            "model": model_name,
                            "repeat": repeat,
                            "fold": fold,
                            "feature": feature,
                            "coefficient": float(
                                coefficient
                            ),
                            "nonzero": bool(
                                coefficient != 0
                            ),
                        })

                if print_progress:
                    print(
                        f"  {model_name:8s} "
                        f"fold {fold}/{actual_outer_folds}: "
                        f"AUC={fold_metrics['roc_auc']:.3f}, "
                        f"sensitivity="
                        f"{fold_metrics['sensitivity']:.3f}, "
                        f"specificity="
                        f"{fold_metrics['specificity']:.3f}, "
                        f"threshold="
                        f"{threshold_record['threshold']:.3f}"
                    )

    predictions_df = pd.DataFrame(
        prediction_records
    )
    fold_metrics_df = pd.DataFrame(
        fold_metric_records
    )
    stability_raw_df = pd.DataFrame(
        stability_records
    )

    return (
        predictions_df,
        fold_metrics_df,
        stability_raw_df,
    )


def calculate_repeat_metrics(predictions_df):
    records = []

    for (
        model_name,
        repeat,
    ), group in predictions_df.groupby([
        "model",
        "repeat",
    ]):
        group = group.sort_values(
            "sample_position"
        )

        metrics = calculate_metrics(
            target=group["true_value"],
            probabilities=group[
                "recurrence_probability"
            ],
            predictions=group[
                "predicted_value"
            ],
        )
        metrics.update({
            "model": model_name,
            "repeat": repeat,
        })
        records.append(metrics)

    return pd.DataFrame(records)


def summarize_model_metrics(repeat_metrics_df):
    metric_columns = [
        "roc_auc",
        "average_precision",
        "accuracy",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "precision",
        "negative_predictive_value",
        "predicted_recurrence_rate",
    ]

    records = []

    for model_name, group in repeat_metrics_df.groupby(
        "model"
    ):
        record = {
            "model": model_name,
            "repeats": len(group),
        }

        for metric in metric_columns:
            record[f"{metric}_mean"] = group[
                metric
            ].mean()
            record[f"{metric}_std"] = (
                group[metric].std(ddof=1)
                if len(group) > 1
                else 0.0
            )

        records.append(record)

    return pd.DataFrame(records).sort_values(
        "roc_auc_mean",
        ascending=False,
    )


def build_patient_summary(
    predictions_df,
    horizon_days,
):
    summary = (
        predictions_df
        .groupby(
            [
                "model",
                "sample_position",
                "sample_id",
                "cancer_type",
                "true_value",
            ],
            as_index=False,
        )
        .agg(
            observed_time_days=(
                "observed_time_days",
                "first",
            ),
            held_out_evaluations=(
                "predicted_value",
                "size",
            ),
            mean_recurrence_probability=(
                "recurrence_probability",
                "mean",
            ),
            probability_std=(
                "recurrence_probability",
                "std",
            ),
            mean_decision_threshold=(
                "decision_threshold",
                "mean",
            ),
            predicted_recurrence_fraction=(
                "predicted_value",
                "mean",
            ),
        )
    )

    summary["probability_std"] = (
        summary["probability_std"].fillna(0.0)
    )
    summary["consensus_predicted_value"] = (
        summary["predicted_recurrence_fraction"]
        >= 0.5
    ).astype(int)
    summary["correct"] = (
        summary["consensus_predicted_value"]
        == summary["true_value"]
    )

    horizon_text = f"{horizon_days:.0f} days"

    summary["true_outcome"] = summary[
        "true_value"
    ].map({
        0: f"No recurrence by {horizon_text}",
        1: f"Recurrence by {horizon_text}",
    })
    summary["predicted_outcome"] = summary[
        "consensus_predicted_value"
    ].map({
        0: f"No recurrence by {horizon_text}",
        1: f"Recurrence by {horizon_text}",
    })

    ordered_columns = [
        "model",
        "sample_position",
        "sample_id",
        "cancer_type",
        "observed_time_days",
        "held_out_evaluations",
        "true_value",
        "true_outcome",
        "consensus_predicted_value",
        "predicted_outcome",
        "mean_recurrence_probability",
        "probability_std",
        "mean_decision_threshold",
        "predicted_recurrence_fraction",
        "correct",
    ]

    return summary[ordered_columns]


def summarize_feature_stability(
    stability_raw_df,
    outer_repeats,
    outer_folds,
):
    if stability_raw_df.empty:
        return pd.DataFrame(
            columns=[
                "model",
                "feature",
                "selection_count",
                "selection_frequency",
                "nonzero_count",
                "nonzero_frequency",
                "mean_coefficient_when_selected",
                "mean_absolute_coefficient_when_selected",
            ]
        )

    stability = (
        stability_raw_df
        .assign(
            absolute_coefficient=lambda frame: (
                frame["coefficient"].abs()
            )
        )
        .groupby(
            ["model", "feature"],
            as_index=False,
        )
        .agg(
            selection_count=(
                "feature",
                "size",
            ),
            nonzero_count=(
                "nonzero",
                "sum",
            ),
            mean_coefficient_when_selected=(
                "coefficient",
                "mean",
            ),
            mean_absolute_coefficient_when_selected=(
                "absolute_coefficient",
                "mean",
            ),
        )
    )

    fitted_model_counts = (
        stability_raw_df[
            ["model", "repeat", "fold"]
        ]
        .drop_duplicates()
        .groupby("model")
        .size()
        .rename("number_of_outer_models")
        .reset_index()
    )
    stability = stability.merge(
        fitted_model_counts,
        on="model",
        how="left",
    )

    stability["selection_frequency"] = (
        stability["selection_count"]
        / stability["number_of_outer_models"]
    )
    stability["nonzero_frequency"] = (
        stability["nonzero_count"]
        / stability["number_of_outer_models"]
    )

    return stability.sort_values(
        [
            "model",
            "selection_frequency",
            "mean_absolute_coefficient_when_selected",
        ],
        ascending=[True, False, False],
    )


def print_consensus_confusion_matrices(
    patient_summary_df,
):
    print(
        "\nPatient-level consensus confusion matrices "
        "across repeated held-out predictions:"
    )

    for model_name, group in patient_summary_df.groupby(
        "model"
    ):
        matrix = confusion_matrix(
            group["true_value"],
            group["consensus_predicted_value"],
            labels=[0, 1],
        )

        print(f"\n{model_name.upper()}")
        print(
            "[[true negatives, false positives],\n"
            " [false negatives, true positives]]"
        )
        print(matrix)
        print(
            classification_report(
                group["true_value"],
                group[
                    "consensus_predicted_value"
                ],
                target_names=[
                    "No recurrence by horizon",
                    "Recurrence by horizon",
                ],
                digits=4,
                zero_division=0,
            )
        )


def fit_final_models(
    model_names,
    omics_df,
    clinical_df,
    target,
    cancer_types,
    omics_columns,
    clinical_columns,
    numeric_clinical_columns,
    categorical_clinical_columns,
    missing_threshold,
    inner_folds,
    tuning_iterations,
    target_sensitivity,
    calibration,
    n_jobs,
    output_prefix,
):
    final_records = []

    inner_splits, _ = make_stratified_splits(
        target=target,
        cancer_types=cancer_types,
        requested_folds=inner_folds,
        random_state=999,
    )

    for model_position, model_name in enumerate(
        model_names
    ):
        print(
            f"\nFitting final {model_name} model "
            "on all patients..."
        )

        X = model_input(
            model_name,
            omics_df,
            clinical_df,
        )
        estimator = make_estimator(
            model_name=model_name,
            omics_columns=omics_columns,
            clinical_columns=clinical_columns,
            numeric_clinical_columns=(
                numeric_clinical_columns
            ),
            categorical_clinical_columns=(
                categorical_clinical_columns
            ),
            missing_threshold=missing_threshold,
        )
        parameter_grid = make_parameter_grid(
            model_name=model_name,
            number_of_omics_features=(
                len(omics_columns)
            ),
        )
        total_configurations = len(
            list(ParameterGrid(parameter_grid))
        )

        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=parameter_grid,
            n_iter=min(
                tuning_iterations,
                total_configurations,
            ),
            scoring="roc_auc",
            cv=inner_splits,
            refit=True,
            random_state=999 + model_position,
            n_jobs=n_jobs,
            verbose=0,
            error_score="raise",
        )
        search.fit(X, target)

        fitted_model = search.best_estimator_
        calibrator, inner_probability = (
            fit_sigmoid_calibration(
                fitted_model=fitted_model,
                X_train=X,
                y_train=target,
                inner_splits=inner_splits,
                n_jobs=n_jobs,
                calibration=calibration,
            )
        )
        threshold_record = (
            choose_sensitivity_threshold(
                target=target,
                probabilities=inner_probability,
                target_sensitivity=(
                    target_sensitivity
                ),
            )
        )

        final_records.append({
            "model": model_name,
            "inner_tuning_roc_auc": (
                search.best_score_
            ),
            "deployment_threshold": (
                threshold_record["threshold"]
            ),
            "threshold_training_sensitivity": (
                threshold_record["sensitivity"]
            ),
            "threshold_training_specificity": (
                threshold_record["specificity"]
            ),
            "best_parameters": json.dumps(
                search.best_params_,
                sort_keys=True,
            ),
        })

        print(
            f"Best parameters: {search.best_params_}"
        )
        print(
            "Inner tuning ROC-AUC "
            "(not final performance): "
            f"{search.best_score_:.4f}"
        )
        print(
            "Deployment threshold: "
            f"{threshold_record['threshold']:.4f}"
        )

        importance = extract_all_feature_importance(
            fitted_model,
            model_name,
        )
        importance.to_csv(
            f"{output_prefix}_{model_name}"
            "_final_feature_importance.csv",
            index=False,
        )

        print("Top positive coefficients:")
        print(
            importance[
                importance["coefficient"] > 0
            ]
            .head(10)
            .to_string(index=False)
        )

        print("Top negative coefficients:")
        print(
            importance[
                importance["coefficient"] < 0
            ]
            .head(10)
            .to_string(index=False)
        )

    return pd.DataFrame(final_records)


def run_permutation_test(
    number_of_permutations,
    primary_model,
    observed_repeat_metrics,
    omics_df,
    clinical_df,
    target,
    cancer_types,
    sample_ids,
    observed_times,
    omics_columns,
    clinical_columns,
    numeric_clinical_columns,
    categorical_clinical_columns,
    missing_threshold,
    outer_folds,
    permutation_outer_repeats,
    inner_folds,
    permutation_tuning_iterations,
    target_sensitivity,
    calibration,
    n_jobs,
):
    if number_of_permutations <= 0:
        return pd.DataFrame(), np.nan

    observed_auc = observed_repeat_metrics.loc[
        observed_repeat_metrics["model"]
        == primary_model,
        "roc_auc",
    ].mean()

    rng = np.random.default_rng(2026)
    records = []

    print(
        f"\nRunning {number_of_permutations} full-label "
        f"permutations for {primary_model}..."
    )

    for permutation in range(
        1,
        number_of_permutations + 1,
    ):
        permuted_target = pd.Series(
            rng.permutation(target.to_numpy()),
            index=target.index,
        )

        (
            permuted_predictions,
            _,
            _,
        ) = run_repeated_nested_cv(
            model_names=[primary_model],
            omics_df=omics_df,
            clinical_df=clinical_df,
            target=permuted_target,
            cancer_types=cancer_types,
            sample_ids=sample_ids,
            observed_times=observed_times,
            omics_columns=omics_columns,
            clinical_columns=clinical_columns,
            numeric_clinical_columns=(
                numeric_clinical_columns
            ),
            categorical_clinical_columns=(
                categorical_clinical_columns
            ),
            missing_threshold=missing_threshold,
            outer_folds=outer_folds,
            outer_repeats=(
                permutation_outer_repeats
            ),
            inner_folds=inner_folds,
            tuning_iterations=(
                permutation_tuning_iterations
            ),
            target_sensitivity=(
                target_sensitivity
            ),
            calibration=calibration,
            n_jobs=n_jobs,
            random_seed=500_000 + permutation,
            print_progress=False,
            collect_stability=False,
        )

        permuted_metrics = calculate_repeat_metrics(
            permuted_predictions
        )
        permuted_auc = permuted_metrics[
            "roc_auc"
        ].mean()

        records.append({
            "permutation": permutation,
            "permuted_mean_roc_auc": (
                permuted_auc
            ),
        })

        print(
            f"  permutation "
            f"{permutation}/{number_of_permutations}: "
            f"AUC={permuted_auc:.4f}"
        )

    permutation_df = pd.DataFrame(records)
    empirical_p_value = (
        1
        + (
            permutation_df["permuted_mean_roc_auc"]
            >= observed_auc
        ).sum()
    ) / (number_of_permutations + 1)

    return permutation_df, empirical_p_value


def main():
    parser = ArgumentParser()

    parser.add_argument(
        "--study",
        nargs="+",
        required=True,
        help="Cancer studies such as pdac, luad, or all.",
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
        "--clinical-cols",
        nargs="*",
        default=None,
        help=(
            "Exact clinical columns for clinical and combined models. "
            "If omitted, a conservative safe alias list is used."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=[
            "all",
            "omics",
            "clinical",
            "combined",
        ],
        default=["all"],
        help="Models to compare. Default is all available models.",
    )
    parser.add_argument(
        "--primary-model",
        choices=[
            "auto",
            "omics",
            "clinical",
            "combined",
        ],
        default="auto",
        help=(
            "Model printed in the patient-level table and used for "
            "optional permutation testing."
        ),
    )
    parser.add_argument(
        "--horizon-days",
        type=float,
        default=730,
    )
    parser.add_argument(
        "--missing-threshold",
        type=float,
        default=0.60,
    )
    parser.add_argument(
        "--outer-folds",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--outer-repeats",
        type=int,
        default=3,
        help="Number of repeated outer cross-validation runs.",
    )
    parser.add_argument(
        "--inner-folds",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--tuning-iterations",
        type=int,
        default=24,
        help=(
            "Random hyperparameter configurations per inner search. "
            "Randomized search controls runtime."
        ),
    )
    parser.add_argument(
        "--target-sensitivity",
        type=float,
        default=0.80,
        help=(
            "Training-only recurrence sensitivity required when "
            "selecting the classification threshold."
        ),
    )
    parser.add_argument(
        "--calibration",
        choices=["sigmoid", "none"],
        default="sigmoid",
        help=(
            "Sigmoid maps inner out-of-fold scores to probabilities. "
            "Use none to retain raw logistic probabilities."
        ),
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=0,
        help=(
            "Full nested-CV label permutations. Expensive; start "
            "with 10 and use at least 100 for a final analysis."
        ),
    )
    parser.add_argument(
        "--permutation-outer-repeats",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--permutation-tuning-iterations",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--output-prefix",
        default="recurrence_evaluation",
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

    selected_tags = select_modalities(
        args.modality
    )
    study_df = load_studies(
        studies,
        selected_tags,
    )

    omics_columns = [
        column
        for column in study_df.columns
        if (
            isinstance(column, str)
            and column.startswith(selected_tags)
        )
    ]

    if not omics_columns:
        raise ValueError(
            "No omics features were found for the selected modalities."
        )

    model_df, target = prepare_fixed_horizon_outcome(
        study_df,
        args.horizon_days,
    )

    if target.nunique() < 2:
        raise ValueError(
            "Both recurrence outcome classes are required."
        )

    clinical_columns = resolve_clinical_columns(
        study_df=model_df,
        requested_columns=args.clinical_cols,
        include_cancer_type=(
            model_df[CANCER_COL].nunique() > 1
        ),
    )

    omics_df = (
        model_df[omics_columns]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .reset_index(drop=True)
    )

    if clinical_columns:
        (
            clinical_df,
            numeric_clinical_columns,
            categorical_clinical_columns,
        ) = prepare_clinical_dataframe(
            model_df,
            clinical_columns,
        )
        clinical_df = clinical_df.reset_index(
            drop=True
        )
    else:
        clinical_df = pd.DataFrame(
            index=model_df.index
        )
        numeric_clinical_columns = []
        categorical_clinical_columns = []

    cancer_types = (
        model_df[CANCER_COL]
        .astype(str)
        .reset_index(drop=True)
    )
    sample_ids = (
        model_df[SAMPLE_ID_COL]
        .astype(str)
        .reset_index(drop=True)
    )
    observed_times = (
        pd.to_numeric(
            model_df[TIME_COL],
            errors="coerce",
        )
        .reset_index(drop=True)
    )

    if "all" in args.models:
        model_names = ["omics"]

        if clinical_columns:
            model_names.extend([
                "clinical",
                "combined",
            ])
    else:
        model_names = list(
            dict.fromkeys(args.models)
        )

    if (
        not clinical_columns
        and any(
            model in {"clinical", "combined"}
            for model in model_names
        )
    ):
        raise ValueError(
            "Clinical or combined modeling was requested, but no safe "
            "clinical columns were found. Supply them with "
            "--clinical-cols."
        )

    if args.primary_model == "auto":
        primary_model = (
            "combined"
            if "combined" in model_names
            else "omics"
        )
    else:
        primary_model = args.primary_model

    if primary_model not in model_names:
        raise ValueError(
            f"Primary model '{primary_model}' is not in "
            f"the requested model list {model_names}."
        )

    print("\nAnalysis design:")
    print(
        f"Studies: {', '.join(study.upper() for study in studies)}"
    )
    print(
        f"Prediction horizon: {args.horizon_days:.0f} days"
    )
    print(f"Usable patients: {len(model_df)}")
    print(
        target.value_counts().rename({
            0: "No recurrence by horizon",
            1: "Recurrence by horizon",
        })
    )
    print(
        f"Omics features before fold filtering: "
        f"{len(omics_columns)}"
    )
    print(
        f"Clinical columns: "
        f"{clinical_columns if clinical_columns else 'none found'}"
    )
    print(f"Models: {model_names}")
    print(
        f"Outer CV: {args.outer_folds} folds x "
        f"{args.outer_repeats} repeats"
    )
    print(
        f"Inner CV: {args.inner_folds} folds"
    )
    print(
        f"Target recurrence sensitivity: "
        f"{args.target_sensitivity:.2f}"
    )
    print(f"Calibration: {args.calibration}")

    (
        predictions_df,
        fold_metrics_df,
        stability_raw_df,
    ) = run_repeated_nested_cv(
        model_names=model_names,
        omics_df=omics_df,
        clinical_df=clinical_df,
        target=target,
        cancer_types=cancer_types,
        sample_ids=sample_ids,
        observed_times=observed_times,
        omics_columns=omics_columns,
        clinical_columns=clinical_columns,
        numeric_clinical_columns=(
            numeric_clinical_columns
        ),
        categorical_clinical_columns=(
            categorical_clinical_columns
        ),
        missing_threshold=args.missing_threshold,
        outer_folds=args.outer_folds,
        outer_repeats=args.outer_repeats,
        inner_folds=args.inner_folds,
        tuning_iterations=args.tuning_iterations,
        target_sensitivity=(
            args.target_sensitivity
        ),
        calibration=args.calibration,
        n_jobs=args.n_jobs,
        random_seed=42,
        print_progress=True,
        collect_stability=True,
    )

    repeat_metrics_df = calculate_repeat_metrics(
        predictions_df
    )
    model_summary_df = summarize_model_metrics(
        repeat_metrics_df
    )
    patient_summary_df = build_patient_summary(
        predictions_df,
        args.horizon_days,
    )
    stability_df = summarize_feature_stability(
        stability_raw_df,
        outer_repeats=args.outer_repeats,
        outer_folds=args.outer_folds,
    )

    print("\nModel comparison across outer repetitions:")
    print(
        model_summary_df.to_string(index=False)
    )

    prevalence = float(target.mean())
    print("\nNo-skill references:")
    print("ROC-AUC: 0.5000")
    print("Balanced accuracy: 0.5000")
    print(
        f"Average precision: {prevalence:.4f}"
    )
    print(
        "Majority-class ordinary accuracy: "
        f"{max(prevalence, 1.0 - prevalence):.4f}"
    )

    print_consensus_confusion_matrices(
        patient_summary_df
    )

    if not stability_df.empty:
        print("\nTop stable omics features:")

        for model_name in stability_df[
            "model"
        ].unique():
            print(f"\n{model_name.upper()}")
            print(
                stability_df[
                    stability_df["model"]
                    == model_name
                ]
                .head(20)
                .to_string(index=False)
            )

    primary_patient_summary = patient_summary_df[
        patient_summary_df["model"]
        == primary_model
    ].copy()

    print(
        f"\nPatient-level repeated held-out predictions "
        f"for the {primary_model.upper()} model:"
    )
    print(
        primary_patient_summary.to_string(
            index=False,
            formatters={
                "mean_recurrence_probability": (
                    lambda value: f"{value:.4f}"
                ),
                "probability_std": (
                    lambda value: f"{value:.4f}"
                ),
                "mean_decision_threshold": (
                    lambda value: f"{value:.4f}"
                ),
                "predicted_recurrence_fraction": (
                    lambda value: f"{value:.4f}"
                ),
            },
        )
    )

    predictions_df.to_csv(
        f"{args.output_prefix}"
        "_held_out_predictions.csv",
        index=False,
    )
    patient_summary_df.to_csv(
        f"{args.output_prefix}"
        "_patient_summary.csv",
        index=False,
    )
    fold_metrics_df.to_csv(
        f"{args.output_prefix}"
        "_outer_fold_metrics.csv",
        index=False,
    )
    repeat_metrics_df.to_csv(
        f"{args.output_prefix}"
        "_repeat_metrics.csv",
        index=False,
    )
    model_summary_df.to_csv(
        f"{args.output_prefix}"
        "_model_comparison.csv",
        index=False,
    )
    stability_df.to_csv(
        f"{args.output_prefix}"
        "_feature_stability.csv",
        index=False,
    )

    final_models_df = fit_final_models(
        model_names=model_names,
        omics_df=omics_df,
        clinical_df=clinical_df,
        target=target,
        cancer_types=cancer_types,
        omics_columns=omics_columns,
        clinical_columns=clinical_columns,
        numeric_clinical_columns=(
            numeric_clinical_columns
        ),
        categorical_clinical_columns=(
            categorical_clinical_columns
        ),
        missing_threshold=args.missing_threshold,
        inner_folds=args.inner_folds,
        tuning_iterations=args.tuning_iterations,
        target_sensitivity=(
            args.target_sensitivity
        ),
        calibration=args.calibration,
        n_jobs=args.n_jobs,
        output_prefix=args.output_prefix,
    )
    final_models_df.to_csv(
        f"{args.output_prefix}"
        "_final_model_settings.csv",
        index=False,
    )

    (
        permutation_df,
        empirical_p_value,
    ) = run_permutation_test(
        number_of_permutations=args.permutations,
        primary_model=primary_model,
        observed_repeat_metrics=(
            repeat_metrics_df
        ),
        omics_df=omics_df,
        clinical_df=clinical_df,
        target=target,
        cancer_types=cancer_types,
        sample_ids=sample_ids,
        observed_times=observed_times,
        omics_columns=omics_columns,
        clinical_columns=clinical_columns,
        numeric_clinical_columns=(
            numeric_clinical_columns
        ),
        categorical_clinical_columns=(
            categorical_clinical_columns
        ),
        missing_threshold=args.missing_threshold,
        outer_folds=args.outer_folds,
        permutation_outer_repeats=(
            args.permutation_outer_repeats
        ),
        inner_folds=args.inner_folds,
        permutation_tuning_iterations=(
            args.permutation_tuning_iterations
        ),
        target_sensitivity=(
            args.target_sensitivity
        ),
        calibration=args.calibration,
        n_jobs=args.n_jobs,
    )

    if not permutation_df.empty:
        permutation_df.to_csv(
            f"{args.output_prefix}"
            "_permutation_test.csv",
            index=False,
        )
        print(
            f"\nPermutation-test empirical p-value "
            f"for {primary_model}: "
            f"{empirical_p_value:.4f}"
        )

    print("\nSaved output files with prefix:")
    print(args.output_prefix)


if __name__ == "__main__":
    main()