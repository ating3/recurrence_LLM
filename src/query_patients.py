"""
Build one-cancer CPTAC multi-omics recurrence tables.

Examples:
    python preprocessing_patients.py --study ucec -o processed/
    python preprocessing_patients.py --study coad --recurrence-column recurrence -o processed/
    python preprocessing_patients.py --study brca -o processed/ --save-modalities
    python preprocessing_patients.py --study pdac -o processed/ --include-normal

Repo usage:
    python src/data_processing/proteomics.py --study ucec -o processed/
    # writes to processed/Recurrence/UCEC/

Inside Python:
    output = process_cancer("ucec")
    combined = output["combined"]
    proteomics_only = output["proteomics"]
    phospho_only = output["phosphoproteomics"]
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import cptac
import numpy as np
import pandas as pd


ALL_STUDIES = [
    "luad", "ccrcc", "gbm", "ucec", "pdac", "ov", "brca",
    "hnscc", "lscc", "coad",
]

MODALITIES = ("proteomics", "phosphoproteomics")

RECURRENCE_STATUS_COLUMN = "Recurrence status (1, yes; 0, no)"

RECURRENCE_CLINICAL_COLS = [
    "diagnostic_evidence_of_recurrence_or_relapse",
    "Recurrence-free survival, days",
    "Recurrence-free survival from collection, days",
    RECURRENCE_STATUS_COLUMN,
]

SOURCE_BY_CANCER = {
    ("luad", "phosphoproteomics"): "umich",
    ("ccrcc", "phosphoproteomics"): "umich",
    ("gbm", "phosphoproteomics"): "umich",
    ("ucec", "phosphoproteomics"): "umich",
    ("pdac", "phosphoproteomics"): "umich",
    ("ov", "phosphoproteomics"): "bcm",
    ("brca", "phosphoproteomics"): "umich",
    ("hnscc", "phosphoproteomics"): "umich",
    ("lscc", "phosphoproteomics"): "umich",
    ("coad", "phosphoproteomics"): "bcm",
}

META_COLS = [
    "Patient_ID",
    "Base_Patient_ID",
    "Cancer_Type",
    "Tumor_Present",
    *RECURRENCE_CLINICAL_COLS,
    "Recurrence",
]


def _as_list(value) -> list:
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Index)):
        return list(value)
    return [value]


def _make_unique(names: list[str]) -> list[str]:
    seen = {}
    unique_names = []
    for name in names:
        count = seen.get(name, 0)
        unique_names.append(name if count == 0 else f"{name}__dup{count}")
        seen[name] = count + 1
    return unique_names


@dataclass
class MultiOmicsOutput:
    """Container for a combined table plus direct modality access."""

    patient_info: pd.DataFrame
    proteomics: pd.DataFrame
    phosphoproteomics: pd.DataFrame
    combined: pd.DataFrame

    def __getitem__(self, key: str) -> pd.DataFrame:
        aliases = {
            "metadata": "patient_info",
            "patient": "patient_info",
            "patients": "patient_info",
            "prot": "proteomics",
            "phos": "phosphoproteomics",
            "phospho": "phosphoproteomics",
            "all": "combined",
        }
        key = aliases.get(key, key)
        if not hasattr(self, key):
            valid = ["patient_info", "proteomics", "phosphoproteomics", "combined"]
            raise KeyError(f"Unknown output key {key!r}. Valid keys: {valid}")
        return getattr(self, key)


def _get_study(study_name: str):
    """
    Given a study name, return the CPTAC study object.
    """
    try:
        study_class = getattr(cptac, study_name.capitalize())
    except AttributeError as exc:
        raise ValueError(f"Unrecognized study name: {study_name}") from exc

    return study_class()


def _choose_source(study, data_type: str, study_name: str) -> str:
    """
    Pick the requested source when known; otherwise use the only available source.
    """
    data_sources = study.list_data_sources()
    available = (
        data_sources
        .loc[data_sources["Data type"] == data_type, "Available sources"]
        .values
    )
    if len(available) == 0:
        raise ValueError(f"{study_name.upper()} has no {data_type} data source listed.")

    available_sources = _as_list(available[0])
    preferred = SOURCE_BY_CANCER.get((study_name, data_type), "umich")

    if preferred in available_sources:
        return preferred
    if len(available_sources) == 1:
        return available_sources[0]

    raise ValueError(
        f"Multiple {data_type} sources found for {study_name.upper()}: "
        f"{available_sources}. Add one to SOURCE_BY_CANCER."
    )


def _get_clinical_data(study, study_name: str) -> pd.DataFrame:
    """
    Return the clinical table for a CPTAC study.
    """
    source = _choose_source(study, "clinical", study_name)
    clinical = study.get_clinical(source=source)
    clinical.index = clinical.index.astype(str)
    return clinical


def _collapse_duplicate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge duplicate feature columns by taking the first non-missing value per sample.
    """
    if not df.columns.duplicated().any():
        return df

    return df.T.groupby(level=list(range(df.columns.nlevels)), sort=False).first().T


def _remove_non_patient_samples(df: pd.DataFrame) -> pd.DataFrame:
    bad_pattern = r"pool|reference|ref|control"
    keep_mask = ~df["Base_Patient_ID"].astype(str).str.lower().str.contains(
        bad_pattern,
        regex=True,
        na=False,
    )
    return df[keep_mask].copy()


def _flatten_columns(cols) -> list[str]:
    """
    Collapse a possibly MultiIndex column axis into unique single-level strings.

    Proteomics columns are commonly (Name, Database_ID). Phosphoproteomics columns
    may have more levels. Non-empty levels are joined with "|", preserving enough
    information to recover the gene symbol with col.split("|")[0].
    """
    flat = []
    for col in cols:
        if isinstance(col, tuple):
            parts = [
                str(part)
                for part in col
                if part not in ("", None)
                and not (isinstance(part, float) and np.isnan(part))
            ]
            flat.append("|".join(parts))
        else:
            flat.append(str(col))
    return flat


def _find_recurrence_column(
    clinical: pd.DataFrame,
    recurrence_column: str | None = None,
) -> str | None:
    """
    Find the clinical recurrence label column, or validate the user-provided one.
    """
    if recurrence_column is not None:
        if recurrence_column not in clinical.columns:
            raise ValueError(
                f"Requested recurrence column {recurrence_column!r} was not found. "
                f"Available columns include: {list(clinical.columns[:20])}"
            )
        return recurrence_column

    if RECURRENCE_STATUS_COLUMN in clinical.columns:
        return RECURRENCE_STATUS_COLUMN

    exact_names = {
        "recurrence",
        "tumor_recurrence",
        "disease_recurrence",
        "progression_or_recurrence",
        "recurrence_status",
        "recurred",
        "relapse",
        "relapse_status",
    }
    normalized_to_original = {
        str(col).strip().lower().replace(" ", "_"): col
        for col in clinical.columns
    }
    for name in exact_names:
        if name in normalized_to_original:
            return normalized_to_original[name]

    fuzzy_matches = [
        col for col in clinical.columns
        if "recur" in str(col).lower() or "relapse" in str(col).lower()
    ]
    if len(fuzzy_matches) == 1:
        return fuzzy_matches[0]

    return None


def _normalize_recurrence_value(value):
    """
    Convert common recurrence labels to booleans while preserving unknown labels.
    """
    if pd.isna(value):
        return pd.NA

    if isinstance(value, (int, float, np.integer, np.floating)):
        if value == 1:
            return True
        if value == 0:
            return False

    text = str(value).strip().lower()
    positive = {"yes", "y", "true", "1", "recurred", "recurrence", "relapse"}
    negative = {
        "no", "n", "false", "0", "not_recurred", "non-recurrence",
        "no recurrence", "no_recurrence", "disease free", "disease_free",
    }

    if text in positive:
        return True
    if text in negative:
        return False
    return value


def _build_patient_info(
    sample_ids: pd.Index,
    study_name: str,
    clinical: pd.DataFrame,
    recurrence_col: str | None,
) -> pd.DataFrame:
    """
    Build the front metadata block shared by both modalities.
    """
    sample_ids = pd.Index(sample_ids.astype(str), name="Patient_ID")
    base_ids = sample_ids.str.replace(r"\.N$", "", regex=True)

    patient_info = pd.DataFrame({
        "Patient_ID": sample_ids,
        "Base_Patient_ID": base_ids,
        "Cancer_Type": study_name,
        "Tumor_Present": ~sample_ids.str.endswith(".N"),
    })

    for col in RECURRENCE_CLINICAL_COLS:
        if col in clinical.columns:
            clinical_values = clinical[col]
            if not clinical_values.index.is_unique:
                clinical_values = clinical_values.groupby(level=0).first()
            patient_info[col] = patient_info["Base_Patient_ID"].map(clinical_values)
        else:
            patient_info[col] = pd.NA

    if recurrence_col is not None:
        recurrence = clinical[recurrence_col].map(_normalize_recurrence_value)
        if not recurrence.index.is_unique:
            recurrence = recurrence.groupby(level=0).first()
        patient_info["Recurrence"] = patient_info["Base_Patient_ID"].map(recurrence)
    elif RECURRENCE_STATUS_COLUMN in patient_info.columns:
        patient_info["Recurrence"] = patient_info[RECURRENCE_STATUS_COLUMN].map(
            _normalize_recurrence_value
        )
    else:
        patient_info["Recurrence"] = pd.NA

    return patient_info[META_COLS]


def clean_abundance_data(
    abundance: pd.DataFrame,
    modality: str,
    study_name: str,
    clinical: pd.DataFrame,
    recurrence_col: str | None,
    include_normal: bool = False,
) -> pd.DataFrame:
    """
    Return one modality with metadata columns first and prefixed feature columns.
    """
    cleaned = _collapse_duplicate_features(abundance.copy())
    cleaned.index = cleaned.index.astype(str)
    cleaned.index.name = "Patient_ID"

    patient_info = _build_patient_info(
        cleaned.index,
        study_name=study_name,
        clinical=clinical,
        recurrence_col=recurrence_col,
    )

    features = cleaned.reset_index()
    features.columns = _make_unique(_flatten_columns(features.columns))
    features = features.drop(columns=["Patient_ID"])
    features = features.rename(columns={col: f"{modality}::{col}" for col in features.columns})
    features.insert(0, "Patient_ID", cleaned.index.to_numpy())

    modality_df = patient_info.merge(features, on="Patient_ID", how="left")
    modality_df = _remove_non_patient_samples(modality_df)
    if not include_normal:
        modality_df = modality_df[modality_df["Tumor_Present"]].copy()
    return modality_df


def _get_abundance(study, study_name: str, modality: str) -> pd.DataFrame:
    source = _choose_source(study, modality, study_name)
    return getattr(study, f"get_{modality}")(source=source)


def combine_modalities(
    proteomics: pd.DataFrame,
    phosphoproteomics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build patient_info and a wide proteomics + phosphoproteomics matrix.
    """
    patient_info = (
        pd.concat([proteomics[META_COLS], phosphoproteomics[META_COLS]], ignore_index=True)
        .drop_duplicates(subset=["Patient_ID"], keep="first")
        .sort_values(["Base_Patient_ID", "Tumor_Present"], ascending=[True, False])
        .reset_index(drop=True)
    )

    prot_features = proteomics.drop(columns=[col for col in META_COLS if col != "Patient_ID"])
    phos_features = phosphoproteomics.drop(columns=[col for col in META_COLS if col != "Patient_ID"])

    combined = (
        patient_info
        .merge(prot_features, on="Patient_ID", how="left")
        .merge(phos_features, on="Patient_ID", how="left")
    )
    return patient_info, combined


def process_cancer(
    study_name: str,
    recurrence_column: str | None = None,
    include_normal: bool = False,
) -> MultiOmicsOutput:
    """
    Pull proteomics and phosphoproteomics for one cancer and return split + combined tables.
    """
    study_name = study_name.lower()
    if study_name not in ALL_STUDIES:
        raise ValueError(f"Study must be one of: {ALL_STUDIES}")

    study = _get_study(study_name)
    clinical = _get_clinical_data(study, study_name)
    recurrence_col = _find_recurrence_column(clinical, recurrence_column)

    modality_frames = {}
    for modality in MODALITIES:
        abundance = _get_abundance(study, study_name, modality)
        modality_frames[modality] = clean_abundance_data(
            abundance=abundance,
            modality=modality,
            study_name=study_name,
            clinical=clinical,
            recurrence_col=recurrence_col,
            include_normal=include_normal,
        )

    patient_info, combined = combine_modalities(
        modality_frames["proteomics"],
        modality_frames["phosphoproteomics"],
    )

    return MultiOmicsOutput(
        patient_info=patient_info,
        proteomics=modality_frames["proteomics"],
        phosphoproteomics=modality_frames["phosphoproteomics"],
        combined=combined,
    )


def _save_output(
    output: MultiOmicsOutput,
    output_dir: Path,
    study_name: str,
    save_modalities: bool = False,
) -> Path:
    study_dir = output_dir / "Recurrence" / study_name.upper()
    study_dir.mkdir(parents=True, exist_ok=True)

    combined_path = study_dir / f"{study_name}_multiomics_recurrence.csv"
    output["combined"].to_csv(combined_path, index=False)

    if save_modalities:
        output["patient_info"].to_csv(study_dir / f"{study_name}_patient_info.csv", index=False)
        output["proteomics"].to_csv(study_dir / f"{study_name}_proteomics.csv", index=False)
        output["phosphoproteomics"].to_csv(
            study_dir / f"{study_name}_phosphoproteomics.csv",
            index=False,
        )

    return combined_path


def main():
    parser = argparse.ArgumentParser(
        description="Process one CPTAC cancer into a proteomics + phosphoproteomics recurrence table."
    )
    parser.add_argument(
        "-s",
        "--study",
        choices=ALL_STUDIES,
        required=True,
        help="Single CPTAC study to process.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=Path,
        required=True,
        help="Base directory to save output files. Created if it does not exist.",
    )
    parser.add_argument(
        "--recurrence-column",
        default=None,
        help="Exact clinical column to use as the recurrence label. If omitted, inferred.",
    )
    parser.add_argument(
        "--save-modalities",
        action="store_true",
        help="Also save patient_info, proteomics-only, and phosphoproteomics-only CSVs.",
    )
    parser.add_argument(
        "--include-normal",
        action="store_true",
        help=(
            "Keep normal .N samples. By default recurrence outputs are tumor-only "
            "because recurrence is a patient-level outcome label."
        ),
    )

    args = parser.parse_args()
    output = process_cancer(
        study_name=args.study,
        recurrence_column=args.recurrence_column,
        include_normal=args.include_normal,
    )
    combined_path = _save_output(
        output=output,
        output_dir=args.output_dir,
        study_name=args.study,
        save_modalities=args.save_modalities,
    )

    print(f"Saved combined data to {combined_path}")
    print(f"Combined data shape: {output['combined'].shape}")
    print(f"Proteomics-only shape: {output['proteomics'].shape}")
    print(f"Phosphoproteomics-only shape: {output['phosphoproteomics'].shape}")


if __name__ == "__main__":
    main()