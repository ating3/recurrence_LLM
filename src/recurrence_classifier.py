from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from src.data_processing.preprocessing_patients import ALL_STUDIES, process_cancer
import pandas as pd

EVENT_COL = "Recurrence status (1, yes; 0, no)"
TIME_COL = "Recurrence-free survival from collection, days"
PROTEOMICS_TAG = "proteomics::"
PHOSPHOPROTEOMICS_TAG = "phosphoproteomics::"


def canonical(col):
    # Strip database IDs and peptide sequences so the same gene matches across studies.
    modality, _, rest = col.partition("::")
    parts = [p for p in rest.split("|") if p]
    return f"{modality}::" + (parts[0] if modality == "proteomics" else "|".join(parts[:2]))


def main():
    # Load
    parser = ArgumentParser()
    parser.add_argument("--study", nargs="+", required=True)
    args = parser.parse_args()

    studies = ALL_STUDIES if "all" in args.study else args.study

    frames = []
    for name in studies:
        df = process_cancer(name)["combined"]
        df.columns = [canonical(c) if c.startswith((PROTEOMICS_TAG, PHOSPHOPROTEOMICS_TAG))
                      else c for c in df.columns]
        frames.append(df.loc[:, ~df.columns.duplicated()])
    study_df = pd.concat(frames, ignore_index=True)

    #Extract features
    features_cols = []
    for measurement in study_df.columns:
        if measurement.startswith(PROTEOMICS_TAG) or measurement.startswith(PHOSPHOPROTEOMICS_TAG):
            features_cols.append(measurement)

    study_model = study_df.dropna(subset=[EVENT_COL, TIME_COL]).copy()
    study_model = study_model[study_model[TIME_COL].astype(float) > 0]

    X = study_model[features_cols].apply(pd.to_numeric, errors='coerce')
    # Drop features with no observed values
    X = X.loc[:, X.notna().any(axis=0)]
    # Optional but recommended: drop features missing in >80% of patients
    X = X.loc[:, X.isna().mean(axis=0) < 0.6]
    # Center and scale within each study so batch differences don't drive the model
    X = X.groupby(study_model["Cancer_Type"].values).transform(lambda g: (g - g.mean()) / g.std())
    # Survival data
    X_train, X_test, y_train, y_test = train_test_split(X, Surv.from_arrays(event=study_model[EVENT_COL].astype(float).astype(bool), time=study_model[TIME_COL].astype(float)), test_size=0.2, random_state=42, stratify=study_model["Cancer_Type"] + study_model[EVENT_COL].astype(str))


    train_non_empty = X_train.notna().any(axis=0)
    X_train = X_train.loc[:, train_non_empty]
    X_test = X_test.loc[:, X_train.columns]

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("coxnet", CoxnetSurvivalAnalysis(l1_ratio=0.8, alpha_min_ratio=0.01, n_alphas=100)),])

    model.fit(X_train, y_train)

    risk_scores = model.predict(X_test)
    c_index = model.score(X_test, y_test)

    # Getting most important phosphoproteomics and proteomics features
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
    print(f'Number of samples: {X_test.shape[0]}')
    print(f'Number of features: {X_test.shape[1]}')
    print(f'Concordance index: {c_index:.4f}')
    print(f'Top 10 important features:\n{important_features.to_string(index=False)}')


if __name__ == "__main__":
    main()