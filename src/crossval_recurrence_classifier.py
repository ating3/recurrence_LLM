from argparse import ArgumentParser

import numpy as np
import pandas as pd
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from pathlib import Path
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

def compute_linear_shap_values(model, X_train, X_test):
    """
    Compute exact linear SHAP-style contributions for the CoxNet risk score.

    SHAP contribution:
        beta_j * (x_j - mean_train_j)

    where x_j is after the same imputer/scaler preprocessing used by the model.
    """
    preprocess = model[:-1]
    coxnet = model.named_steps["coxnet"]

    X_train_processed = preprocess.transform(X_train)
    X_test_processed = preprocess.transform(X_test)

    coefficients = coxnet.coef_.ravel()
    background = X_train_processed.mean(axis=0)

    shap_values = (X_test_processed - background) * coefficients
    base_value = float(background @ coefficients)

    shap_df = pd.DataFrame(
        shap_values,
        columns=X_train.columns,
        index=X_test.index,
    )

    return shap_df, base_value


def summarize_global_shap(shap_df):
    global_shap = pd.DataFrame({
        "feature": shap_df.columns,
        "mean_abs_shap": shap_df.abs().mean(axis=0).values,
        "mean_shap": shap_df.mean(axis=0).values,
    })

    return global_shap.sort_values("mean_abs_shap", ascending=False)


def summarize_patient_shap(shap_df, patient_index, top_n=10):
    patient_shap = shap_df.loc[patient_index]

    out = pd.DataFrame({
        "feature": patient_shap.index,
        "shap_value": patient_shap.values,
        "abs_shap": patient_shap.abs().values,
    })

    return out.sort_values("abs_shap", ascending=False).head(top_n)

def canonical(col: str) -> str:
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
    return X.replace([np.inf, -np.inf], np.nan)


def cindex_scorer(estimator, X, y):
    risk = estimator.predict(X)
    return concordance_index_censored(y["event"], y["time"], risk)[0]


def zscore_by_study_train_test(X_train, X_test, train_studies, test_studies):
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

    return clean_numeric_matrix(X_train_scaled), clean_numeric_matrix(X_test_scaled)


def main():
    parser = ArgumentParser()
    parser.add_argument("--study", nargs="+", required=True)
    parser.add_argument(
        "--modality",
        nargs="+",
        default=["all"],
        help="Modalities to include: proteomics, phosphoproteomics, acetylproteomics, or all.",
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
    stratify = event if usable_strata.min() < 2 else stratify_labels

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

    base_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("coxnet", CoxnetSurvivalAnalysis(max_iter=100000)),
    ])

    param_grid = {
        "coxnet__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        "coxnet__alphas": [[a] for a in np.logspace(-4, 1, 12)],
    }

    inner_labels = pd.Series(train_studies).astype(str) + "_" + pd.Series(y_train["event"]).astype(str)
    if inner_labels.value_counts().min() < 2:
        inner_labels = pd.Series(y_train["event"])

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring=cindex_scorer,
        cv=cv.split(X_train, inner_labels),
        n_jobs=-1,
        refit=True,
        error_score=np.nan,
        verbose=1,
    )

    grid.fit(X_train, y_train)

    model = grid.best_estimator_
    c_index = model.score(X_test, y_test)

    shap_df, shap_base_value = compute_linear_shap_values(model, X_train, X_test)
    global_shap = summarize_global_shap(shap_df)

    shap_output_dir = Path("outputs")
    shap_output_dir.mkdir(exist_ok=True)

    study_label = "_".join(studies)
    modality_label = "_".join(tag.replace("::", "") for tag in selected_tags)

    global_shap_path = shap_output_dir / f"{study_label}_{modality_label}_global_shap.csv"
    patient_shap_path = shap_output_dir / f"{study_label}_{modality_label}_patient_shap.csv"

    global_shap.to_csv(global_shap_path, index=False)
    shap_df.to_csv(patient_shap_path, index=True)

    top_global_shap = global_shap.head(10)

    coefficients = model.named_steps["coxnet"].coef_.ravel()

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
    print(f"Best inner-CV C-index: {grid.best_score_:.4f}")
    print(f"Best params: {grid.best_params_}")
    print(f"Test concordance index: {c_index:.4f}")
    print(f"Top 10 important features:\n{important_features.to_string(index=False)}")

    print(f"SHAP base risk value: {shap_base_value:.4f}")
    print(f"Saved global SHAP summary to: {global_shap_path}")
    print(f"Saved patient-level SHAP values to: {patient_shap_path}")
    print(f"Top 10 global SHAP features:\n{top_global_shap.to_string(index=False)}")

if __name__ == "__main__":
    main()