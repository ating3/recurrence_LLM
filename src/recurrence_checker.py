import re
import pandas as pd
import cptac

ALL_STUDIES = [
    "luad", "ccrcc", "gbm", "ucec", "pdac", "ov", "brca",
    "hnscc", "lscc", "coad",
]

EVENT_COL = "Recurrence status (1, yes; 0, no)"
TIME_COL = "Recurrence-free survival from collection, days"

PATTERN = re.compile(
    r"recur|recurrence|relapse|progress|progression|disease.?free|rfs|free survival",
    re.IGNORECASE,
)

def as_list(x):
    if isinstance(x, (list, tuple, set, pd.Index)):
        return list(x)
    return [x]

def get_study(name):
    return getattr(cptac, name.capitalize())()

def summarize_values(series):
    s = series.dropna()
    if s.empty:
        return "all missing"

    if pd.api.types.is_numeric_dtype(s):
        return f"nonmissing={len(s)}, min={s.min()}, max={s.max()}, unique={s.nunique()}"

    counts = s.astype(str).value_counts().head(8)
    return "nonmissing=" + str(len(s)) + ", values=" + "; ".join(
        f"{idx}: {val}" for idx, val in counts.items()
    )

for study_name in ALL_STUDIES:
    print("\n" + "=" * 90)
    print(f"{study_name.upper()}")

    try:
        study = get_study(study_name)
        sources_df = study.list_data_sources()
        clinical_sources = sources_df.loc[
            sources_df["Data type"] == "clinical",
            "Available sources"
        ].values

        if len(clinical_sources) == 0:
            print("No clinical source found.")
            continue

        for source in as_list(clinical_sources[0]):
            print(f"\nClinical source: {source}")

            try:
                clinical = study.get_clinical(source=source)
            except Exception as e:
                print(f"Could not load clinical source {source}: {e}")
                continue

            print(f"Clinical shape: {clinical.shape}")

            matching_cols = [
                col for col in clinical.columns
                if PATTERN.search(str(col))
            ]

            if not matching_cols:
                print("No recurrence/relapse/progression-like columns found.")
            else:
                print("Matching columns:")
                for col in matching_cols:
                    print(f"  - {col}")
                    print(f"    {summarize_values(clinical[col])}")

            if EVENT_COL in clinical.columns and TIME_COL in clinical.columns:
                event = clinical[EVENT_COL]
                time = pd.to_numeric(clinical[TIME_COL], errors="coerce")
                usable = clinical[event.notna() & time.notna() & (time > 0)]

                event_usable = usable[EVENT_COL].astype(float)
                n_recurred = int((event_usable == 1).sum())
                n_nonrecurred = int((event_usable == 0).sum())

                print("\nCurrent model columns usable count:")
                print(f"  n usable: {len(usable)}")
                print(f"  recurred/events: {n_recurred}")
                print(f"  censored/non-recurred: {n_nonrecurred}")
            else:
                print("\nCurrent model columns usable count:")
                print(f"  Missing EVENT_COL? {EVENT_COL not in clinical.columns}")
                print(f"  Missing TIME_COL? {TIME_COL not in clinical.columns}")

    except Exception as e:
        print(f"FAILED {study_name.upper()}: {e}")