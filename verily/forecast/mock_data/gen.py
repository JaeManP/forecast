"""Mock AoU tabular event dataset generator for local testing."""

import random
from pathlib import Path

import pandas as pd
from verily.forecast import config

random.seed(777)
BASE_PATH = Path(__file__).parent
COLUMNS = ["subject_id", "time", "code", "numeric_value", "visit_id", "visit_type"]
VISIT_TYPES = [
    "Inpatient Visit",
    "Emergency Room Visit",
    "Outpatient Visit",
    "Telehealth visit",
]


def build_rows(subject_id: int, base_ts: int):
    visit_type = random.choice(VISIT_TYPES)
    visit_base = subject_id * 1000

    def make_row(code, offset_days, numeric_value=None, visit_id=None, visit_type_value=""):
        return {
            "subject_id": subject_id,
            "time": base_ts + offset_days * 24 * 3600,
            "code": code,
            "numeric_value": numeric_value,
            "visit_id": visit_id,
            "visit_type": visit_type_value,
        }

    rows = []
    rows.append(make_row(f"CDzz{50000 + subject_id}", 0, None, visit_base + 1, visit_type))
    rows.append(make_row(f"PRzz{30000 + subject_id}", 1, None, visit_base + 1, visit_type))
    rows.append(
        make_row(
            f"MSzz{20000 + subject_id}",
            2,
            round(random.uniform(60, 140), 1),
            visit_base + 2,
            visit_type,
        )
    )
    rows.append(
        make_row(
            f"DRzz{70000 + subject_id}",
            3,
            round(random.uniform(1, 4), 1),
            visit_base + 2,
            visit_type,
        )
    )
    rows.append(make_row(f"VSzz{9000 + subject_id % 20}", 4, None, None, ""))
    rows.append(make_row(f"GENzz{8500 + subject_id % 4}", -200, None, None, ""))
    rows.append(make_row(f"SEXzz{8600 + subject_id % 3}", -199, None, None, ""))
    rows.append(make_row(f"RACzz{8700 + subject_id % 5}", -198, None, None, ""))
    rows.append(make_row("MEDSzzBIRTH", -365 * 30, None, None, ""))
    return rows


def build_split(subject_ids, epoch_start_days):
    base_epoch = 1_577_836_800  # Jan 2020
    rows = []
    for idx, subject_id in enumerate(subject_ids):
        base_ts = base_epoch + (epoch_start_days + idx * 15) * 24 * 3600
        rows.extend(build_rows(subject_id, base_ts))
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["visit_id"] = pd.Series(df["visit_id"], dtype="Int64")
    df = df.sort_values(["subject_id", "time", "code"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    train_subjects = list(range(1001, 1009))
    test_subjects = list(range(2001, 2003))

    train_df = build_split(train_subjects, 0)
    test_df = build_split(test_subjects, 120)

    outputs = {
        "train": train_df,
        "test": test_df,
    }

    for split, df in outputs.items():
        output_path = BASE_PATH / split / config.RAW_DATA_FILENAME
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Wrote {len(df)} rows to {output_path}")
