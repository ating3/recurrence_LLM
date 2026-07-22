from argparse import ArgumentParser

import numpy as np
import pandas as pd
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from query_patients import ALL_STUDIES, process_cancer


EVENT_COL = "Recurrence"
TIME_COL = "Derived recurrence-free survival time, days"

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
    Strip database IDs / peptide details so similar features can line up across studies.
    Proteomics keeps gene only. PTMs keep gene + site/peptide-level identifier.
    """
    if "::" not in col:
        return col

    modality, _, rest = col.partition("::")
    parts = [p for p in rest.split("|") if p]

    if not parts:
        return col

    if modality == "proteomics":
        return f"{modality}::{parts[0]}"

    return f"{modality}::" + "|".join(parts[:2])


def clean_numeric_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """
    Convert impossible numeric values to NaN.
    SimpleImputer can handle NaN but not inf/-inf.
    """
    X = X.replace([np.inf, -np.inf], np.nan)
    return X


def zscore_by_study_train_test(X_train, X_test, train_studies, test_studies):
    """
    Center/scale within cancer type using training data only.
    This avoids leaking test-set statistics into training.
    """
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    global_mean = X_train.mean()
    global_std = X_train.std()
    global_std = global_std.replace([0, np.inf, -np.inf], np.nan).fillna(1)

    train_studies = pd.Series(train_studies, index=X_train.index)
    test_studies = pd.Series(test_studies, index=X_test.index)

    for study in train_studies.dropna().unique():
        train_mask = train_studies == study
        test_mask = test_studies == study

        study_train = X_train.loc[train_mask]

        mean = study_train.mean()
        std = study_train.std()
        std = std.replace([0, np.inf, -np.inf], np.nan).fillna(1)

        X_train_scaled.loc[train_mask] = (study_train - mean) / std

        if test_mask.any():
            X_test_scaled.loc[test_mask] = (X_test.loc[test_mask] - mean) / std

    unseen_test_mask = ~test_studies.isin(train_studies.unique())
    if unseen_test_mask.any():
        X_test_scaled.loc[unseen_test_mask] = (
            X_test.loc[unseen_test_mask] - global_mean
        ) / global_std

    X_train_scaled = clean_numeric_matrix(X_train_scaled)
    X_test_scaled = clean_numeric_matrix(X_test_scaled)

    return X_train_scaled, X_test_scaled


def main():
    parser = ArgumentParser()
    parser.add_argument("--study", nargs="+", required=True)
    parser.add_argument(
        "--modality",
        nargs="+",
        default=["all"],
        help=(
            "Modalities to include. Use any of: proteomics, phosphoproteomics, "
            "acetylproteomics, or all."
        ),
    )
    args = parser.parse_args()

    studies = (
        ALL_STUDIES
        if "all" in [s.lower() for s in args.study]
        else [s.lower() for s in args.study]
    )

    requested_modalities = [m.lower() for m in args.modality]

    if "all" in requested_modalities:
        selected_tags = (PROTEOMICS_TAG, PHOSPHOPROTEOMICS_TAG, ACETYLPROTEOMICS_TAG)
    else:
        invalid = [m for m in requested_modalities if m not in MODALITY_TO_TAG]
        if invalid:
            raise ValueError(
                f"Invalid modality/modalities: {invalid}. "
                f"Valid options: {sorted(MODALITY_TO_TAG)} or all."
            )
        selected_tags = tuple(MODALITY_TO_TAG[m] for m in requested_modalities)

    frames = []

    for name in studies:
        output = process_cancer(name)
        df = output["combined"].copy()

        df.columns = [
            canonical(c) if isinstance(c, str) and c.startswith(selected_tags) else c
            for c in df.columns
        ]

        df = df.loc[:, ~df.columns.duplicated()]
        frames.append(df)

    study_df = pd.concat(frames, ignore_index=True)

    features_cols = [
        col for col in study_df.columns
        if isinstance(col, str) and col.startswith(selected_tags)
    ]

    if len(features_cols) == 0:
        raise ValueError(f"No features found for selected modalities: {requested_modalities}")

    study_model = study_df.dropna(subset=[EVENT_COL, TIME_COL]).copy()
    study_model[TIME_COL] = pd.to_numeric(study_model[TIME_COL], errors="coerce")
    study_model = study_model[study_model[TIME_COL] > 0]

    if study_model.empty:
        raise ValueError("No usable recurrence-labeled samples after filtering.")

    event = study_model[EVENT_COL].astype(bool)
    time = study_model[TIME_COL].astype(float)

    print("\nEvent/time debugging:")
    print(f"Study model shape: {study_model.shape}")
    print("EVENT_COL value counts:")
    print(study_model[EVENT_COL].value_counts(dropna=False))
    print("TIME_COL nonmissing:", study_model[TIME_COL].notna().sum())
    print("TIME_COL > 0:", (study_model[TIME_COL] > 0).sum())
    print("Cancer types:")
    print(study_model["Cancer_Type"].value_counts(dropna=False))

    if event.nunique() < 2:
        raise ValueError("Need both recurrence events and non-events/censored samples.")

    X = study_model[features_cols].apply(pd.to_numeric, errors="coerce")
    X = clean_numeric_matrix(X)

    X = X.loc[:, X.notna().any(axis=0)]
    X = X.loc[:, X.isna().mean(axis=0) < 0.6]

    if X.shape[1] == 0:
        raise ValueError("No usable features after missingness filtering.")

    stratify_labels = study_model["Cancer_Type"].astype(str) + "_" + event.astype(str)
    usable_strata = stratify_labels.value_counts()

    if usable_strata.min() < 2:
        stratify = event
    else:
        stratify = stratify_labels

    X_train, X_test, y_train, y_test, train_studies, test_studies = train_test_split(
        X,
        Surv.from_arrays(event=event, time=time),
        study_model["Cancer_Type"].values,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    train_non_empty = X_train.notna().any(axis=0)
    X_train = X_train.loc[:, train_non_empty]
    X_test = X_test.loc[:, X_train.columns]

    X_train, X_test = zscore_by_study_train_test(
        X_train,
        X_test,
        train_studies,
        test_studies,
    )

    train_non_empty_after_scaling = X_train.notna().any(axis=0)
    X_train = X_train.loc[:, train_non_empty_after_scaling]
    X_test = X_test.loc[:, X_train.columns]

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("coxnet", CoxnetSurvivalAnalysis(
            l1_ratio=0.8,
            alpha_min_ratio=0.01,
            n_alphas=100,
        )),
    ])

    model.fit(X_train, y_train)

    c_index = model.score(X_test, y_test)

    coefficients = model.named_steps["coxnet"].coef_[:, -1]

    feature_importance = pd.DataFrame({
        "feature": X_train.columns,
        "coefficient": coefficients,
    })

    feature_importance = feature_importance[feature_importance["coefficient"] != 0]
    feature_importance["abs_coefficient"] = feature_importance["coefficient"].abs()
    feature_importance = feature_importance.sort_values("abs_coefficient", ascending=False)

    important_features = feature_importance.head(10)

    print(f'Studies: {", ".join(s.upper() for s in studies)}')
    print(f"Selected modality tags: {selected_tags}")
    print(f"Number of train samples: {X_train.shape[0]}")
    print(f"Number of test samples: {X_test.shape[0]}")
    print(f"Number of features: {X_test.shape[1]}")
    print(f"Concordance index: {c_index:.4f}")
    print(f"Top 10 important features:\n{important_features.to_string(index=False)}")


if __name__ == "__main__":
    main()