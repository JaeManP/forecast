import argparse
import os
import subprocess

import numpy as np
import pandas as pd
import pandas_gbq
from google.cloud import bigquery

from verily.forecast import config
from verily.forecast.constants import LOCAL_DIR

CDR = os.environ.get("WORKSPACE_CDR")
BUCKET = os.environ.get("WORKSPACE_BUCKET")
if not BUCKET:
    # AoU Workbench-specific: auto-detect the GCS bucket via `wb` CLI
    try:
        BUCKET = "gs://" + subprocess.check_output(
            """/usr/bin/wb resource describe --id $(/usr/bin/wb resource list | awk '$2 == "GCS_BUCKET" {print $1}') | awk -F': ' '/GCS bucket name/ {print $2}'""",
            shell=True,
            text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Warning: Could not determine bucket from workspace resources")
DESTINATION_DIR = os.path.join(BUCKET, "data")
DESTINATION_FILE_NAME = config.RAW_DATA_FILENAME
DESTINATION_DICT_FILE_NAME = "v1.csv"

RANDOM_SEED = 866  # For train/test split
P_TEST = 0.2


def query_data(query):
    return pandas_gbq.read_gbq(
        query, dialect="standard", progress_bar_type=None, use_bqstorage_api=True
    )


def query_data_chunked(query, chunk_size=100000):
    """Stream query results in chunks to avoid OOM for large results."""
    client = bigquery.Client()
    query_job = client.query(query)
    return query_job.result().to_dataframe_iterable(bqstorage_client=None, max_queue_size=chunk_size)


def show_n_rows():
    query = f"""
    SELECT 'condition' source, COUNT(*) n_row FROM `{CDR}.ds_condition_occurrence`
    UNION ALL
    SELECT 'procedure' source, COUNT(*) n_row FROM `{CDR}.ds_procedure_occurrence`
    UNION ALL
    SELECT 'measurement' source, COUNT(*) n_row FROM `{CDR}.ds_measurement`
    UNION ALL
    SELECT 'drug' source, COUNT(*) n_row FROM `{CDR}.ds_drug_exposure`
    UNION ALL
    SELECT 'visit' source, COUNT(*) n_row FROM `{CDR}.ds_visit_occurrence` WHERE standard_concept_name IN ('Inpatient Visit', 'Emergency Room Visit')
    UNION ALL
    SELECT 'demographics' source, COUNT(*)*3 n_row FROM `{CDR}.person`
    """
    df = query_data(query)
    df = pd.concat(
        [df, pd.DataFrame({"source": ["total"], "n_row": [df["n_row"].sum()]})],
        ignore_index=True,
    )
    print(df)


def read_data(limit: int | None = None):
    query = """
    (
        SELECT
            person_id subject_id,
            UNIX_SECONDS(condition_start_datetime) time,
            CONCAT('CDzz', condition_concept_id) code,
            NULL numeric_value,
            VISIT_OCCURRENCE_ID visit_id,
            VISIT_OCCURRENCE_CONCEPT_NAME visit_type,
        FROM `{CDR}.ds_condition_occurrence`
        {LIMIT}
    )
    UNION ALL
    (
        SELECT
            person_id subject_id,
            UNIX_SECONDS(procedure_datetime) time,
            CONCAT('PRzz', procedure_concept_id) code,
            NULL numeric_value,
            VISIT_OCCURRENCE_ID visit_id,
            VISIT_OCCURRENCE_CONCEPT_NAME visit_type,
        FROM `{CDR}.ds_procedure_occurrence`
        {LIMIT}
    )
    UNION ALL
    (
        SELECT
            person_id subject_id,
            UNIX_SECONDS(measurement_datetime) time,
            CONCAT('MSzz', measurement_concept_id) code,
            VALUE_AS_NUMBER numeric_value,
            VISIT_OCCURRENCE_ID visit_id,
            VISIT_OCCURRENCE_CONCEPT_NAME visit_type,
        FROM `{CDR}.ds_measurement`
        {LIMIT}
    )
    UNION ALL
    (
        SELECT
            person_id subject_id,
            UNIX_SECONDS(drug_exposure_start_datetime) time,
            CONCAT('DRzz', drug_concept_id) code,
            QUANTITY numeric_value,
            VISIT_OCCURRENCE_ID visit_id,
            VISIT_OCCURRENCE_CONCEPT_NAME visit_type,
        FROM `{CDR}.ds_drug_exposure`
        {LIMIT}
    )
    UNION ALL
    (
        SELECT
            person_id subject_id,
            UNIX_SECONDS(visit_start_datetime) time,
            CONCAT('VSzz', visit_concept_id) code,
            NULL numeric_value,
            NULL visit_id,
            '' visit_type,
        FROM `{CDR}.ds_visit_occurrence`
        WHERE standard_concept_name IN ('Inpatient Visit', 'Emergency Room Visit')
        {LIMIT}
    )
    UNION ALL
    (
        SELECT
            person_id subject_id,
            NULL time,
            CONCAT('GENzz', gender_concept_id) code,
            NULL numeric_value,
            NULL visit_id,
            '' visit_type,
        FROM `{CDR}.person`
        {LIMIT}
    )
    UNION ALL
    (
        SELECT
            person_id subject_id,
            NULL time,
            CONCAT('SEXzz', sex_at_birth_concept_id) code,
            NULL numeric_value,
            NULL visit_id,
            '' visit_type,
        FROM `{CDR}.person`
        {LIMIT}
    )
    UNION ALL
    (
        SELECT
            person_id subject_id,
            NULL time,
            CONCAT('RACzz', race_concept_id) code,
            NULL numeric_value,
            NULL visit_id,
            '' visit_type,
        FROM `{CDR}.person`
        {LIMIT}
    )
    UNION ALL
    (
        SELECT
            person_id subject_id,
            UNIX_SECONDS(birth_datetime) time,
            'MEDSzzBIRTH' code,
            NULL numeric_value,
            NULL visit_id,
            '' visit_type,
        FROM `{CDR}.person`
        {LIMIT}
    )
    ORDER BY subject_id, time """
    if limit is None:
        limit_str = ""
    else:
        limit_str = f"LIMIT {limit}"
    query = query.format_map({"CDR": CDR, "LIMIT": limit_str})
    df = query_data(query)
    return df


def get_subject_ids() -> np.ndarray:
    """Gets all unique subject IDs from the person table."""
    query = f"SELECT DISTINCT person_id as subject_id FROM `{CDR}.person`"
    df = query_data(query)
    return np.array(df["subject_id"])


def split_train_test_subjects() -> tuple[set, set]:
    """Splits subject IDs into train and test sets."""
    subjects = get_subject_ids()
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(subjects)
    i_split = int((1 - P_TEST) * subjects.size)
    subjects_train = set(subjects[:i_split])
    subjects_test = set(subjects[i_split:])
    return subjects_train, subjects_test


def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits data into train and test subsets."""
    subjects = np.array(df["subject_id"].unique())
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(subjects)
    i_split = int((1 - P_TEST) * subjects.size)
    subjects_train = subjects[:i_split]
    subjects_test = subjects[i_split:]
    df_train = df[df["subject_id"].isin(subjects_train)].copy()
    df_test = df[df["subject_id"].isin(subjects_test)].copy()
    return df_train, df_test


def create_data_dictionary() -> pd.DataFrame:
    query = f"""
    (
        SELECT
            CONCAT('CDzz', condition_concept_id) code,
            standard_concept_name concept_name,
        FROM `{CDR}.ds_condition_occurrence`
        GROUP BY condition_concept_id, standard_concept_name
    )
    UNION ALL
    (
        SELECT
            CONCAT('PRzz', procedure_concept_id) code,
            standard_concept_name concept_name,
        FROM `{CDR}.ds_procedure_occurrence`
        GROUP BY procedure_concept_id, standard_concept_name
    )
    UNION ALL
    (
        SELECT
            CONCAT('MSzz', measurement_concept_id) code,
            standard_concept_name concept_name,
        FROM `{CDR}.ds_measurement`
        GROUP BY measurement_concept_id, standard_concept_name
    )
    UNION ALL
    (
        SELECT
            CONCAT('DRzz', drug_concept_id) code,
            standard_concept_name concept_name,
        FROM `{CDR}.ds_drug_exposure`
        GROUP BY drug_concept_id, standard_concept_name
    )
    UNION ALL
    (
        SELECT
            CONCAT('VSzz', visit_concept_id) code,
            standard_concept_name concept_name,
        FROM `{CDR}.ds_visit_occurrence`
        WHERE standard_concept_name IN ('Inpatient Visit', 'Emergency Room Visit')
        GROUP BY visit_concept_id, standard_concept_name
    )
    UNION ALL
    (
        SELECT
            CONCAT('GENzz', a.gender_concept_id) code,
            b.concept_name,
        FROM (
            SELECT
                DISTINCT gender_concept_id,
            FROM `{CDR}.person`
        ) a
        LEFT JOIN `{CDR}.concept` b
        ON a.gender_concept_id=b.concept_id
    )
    UNION ALL
    (
        SELECT
            CONCAT('SEXzz', a.sex_at_birth_concept_id) code,
            b.concept_name,
        FROM (
            SELECT
                DISTINCT sex_at_birth_concept_id,
            FROM `{CDR}.person`
        ) a
        LEFT JOIN `{CDR}.concept` b
        ON a.sex_at_birth_concept_id=b.concept_id
    )
    UNION ALL
    (
        SELECT
            CONCAT('RACzz', a.race_concept_id) code,
            b.concept_name,
        FROM (
            SELECT
                DISTINCT race_concept_id,
            FROM `{CDR}.person`
        ) a
        LEFT JOIN `{CDR}.concept` b
        ON a.race_concept_id=b.concept_id
    )
    UNION ALL
    (
        SELECT
            'MEDSzzBIRTH' code,
            'Birth date' concept_name 
    )
    ORDER BY code
    """
    df = query_data(query)
    return df


def write_to_bucket(df, local_path, destination_dir):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    df.to_csv(local_path, index=False)
    args = ["gsutil", "cp", local_path, destination_dir]
    output = subprocess.run(args, capture_output=True)
    if output.returncode != 0:
        print(f"gsutil error: {output.stderr}")


def export_data_chunked():
    """Export data in chunks to avoid OOM for large datasets."""
    # First, get the train/test split on subject IDs (small query)
    print("Getting subject IDs for train/test split...")
    subjects_train, subjects_test = split_train_test_subjects()
    print(f"Train subjects: {len(subjects_train)}, Test subjects: {len(subjects_test)}")

    # Prepare local file paths
    train_local_path = os.path.join(LOCAL_DIR, "train", DESTINATION_FILE_NAME)
    test_local_path = os.path.join(LOCAL_DIR, "test", DESTINATION_FILE_NAME)
    os.makedirs(os.path.dirname(train_local_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_local_path), exist_ok=True)

    # Remove existing files to avoid appending to stale data
    for path in [train_local_path, test_local_path]:
        if os.path.exists(path):
            os.remove(path)

    # Build the query (same as read_data but without LIMIT)
    query = f"""
    (
        SELECT
            person_id subject_id,
            UNIX_SECONDS(condition_start_datetime) time,
            CONCAT('CDzz', condition_concept_id) code,
            NULL numeric_value,
            VISIT_OCCURRENCE_ID visit_id,
            VISIT_OCCURRENCE_CONCEPT_NAME visit_type,
        FROM `{CDR}.ds_condition_occurrence`
    )
    UNION ALL
    (
        SELECT
            person_id subject_id,
            UNIX_SECONDS(procedure_datetime) time,
            CONCAT('PRzz', procedure_concept_id) code,
            NULL numeric_value,
            VISIT_OCCURRENCE_ID visit_id,
            VISIT_OCCURRENCE_CONCEPT_NAME visit_type,
        FROM `{CDR}.ds_procedure_occurrence`
    )
    UNION ALL
    (
        SELECT
            person_id subject_id,
            UNIX_SECONDS(measurement_datetime) time,
            CONCAT('MSzz', measurement_concept_id) code,
            VALUE_AS_NUMBER numeric_value,
            VISIT_OCCURRENCE_ID visit_id,
            VISIT_OCCURRENCE_CONCEPT_NAME visit_type,
        FROM `{CDR}.ds_measurement`
    )
    UNION ALL
    (
        SELECT
            person_id subject_id,
            UNIX_SECONDS(drug_exposure_start_datetime) time,
            CONCAT('DRzz', drug_concept_id) code,
            QUANTITY numeric_value,
            VISIT_OCCURRENCE_ID visit_id,
            VISIT_OCCURRENCE_CONCEPT_NAME visit_type,
        FROM `{CDR}.ds_drug_exposure`
    )
    UNION ALL
    (
        SELECT
            person_id subject_id,
            UNIX_SECONDS(visit_start_datetime) time,
            CONCAT('VSzz', visit_concept_id) code,
            NULL numeric_value,
            NULL visit_id,
            '' visit_type,
        FROM `{CDR}.ds_visit_occurrence`
        WHERE standard_concept_name IN ('Inpatient Visit', 'Emergency Room Visit')
    )
    UNION ALL
    (
        SELECT
            person_id subject_id,
            NULL time,
            CONCAT('GENzz', gender_concept_id) code,
            NULL numeric_value,
            NULL visit_id,
            '' visit_type,
        FROM `{CDR}.person`
    )
    UNION ALL
    (
        SELECT
            person_id subject_id,
            NULL time,
            CONCAT('SEXzz', sex_at_birth_concept_id) code,
            NULL numeric_value,
            NULL visit_id,
            '' visit_type,
        FROM `{CDR}.person`
    )
    UNION ALL
    (
        SELECT
            person_id subject_id,
            NULL time,
            CONCAT('RACzz', race_concept_id) code,
            NULL numeric_value,
            NULL visit_id,
            '' visit_type,
        FROM `{CDR}.person`
    )
    UNION ALL
    (
        SELECT
            person_id subject_id,
            UNIX_SECONDS(birth_datetime) time,
            'MEDSzzBIRTH' code,
            NULL numeric_value,
            NULL visit_id,
            '' visit_type,
        FROM `{CDR}.person`
    )
    ORDER BY subject_id, time """

    # Stream data in chunks and write to train/test CSVs
    print("Streaming data and writing to CSVs...")
    train_header_written = False
    test_header_written = False
    chunk_count = 0

    for chunk_df in query_data_chunked(query):
        chunk_count += 1
        # Split chunk into train and test
        train_mask = chunk_df["subject_id"].isin(subjects_train)
        train_chunk = chunk_df[train_mask]
        test_chunk = chunk_df[~train_mask]

        # Append to train CSV
        if not train_chunk.empty:
            train_chunk.to_csv(
                train_local_path,
                mode="a",
                header=not train_header_written,
                index=False,
            )
            train_header_written = True

        # Append to test CSV
        if not test_chunk.empty:
            test_chunk.to_csv(
                test_local_path,
                mode="a",
                header=not test_header_written,
                index=False,
            )
            test_header_written = True

        if chunk_count % 10 == 0:
            print(f"Processed {chunk_count} chunks...")

    print(f"Finished processing {chunk_count} chunks.")

    # Upload to bucket
    train_destination = os.path.join(f"{BUCKET}/data/train", DESTINATION_FILE_NAME)
    test_destination = os.path.join(f"{BUCKET}/data/test", DESTINATION_FILE_NAME)

    print(f"Uploading train data to {train_destination}...")
    args = ["gsutil", "cp", train_local_path, train_destination]
    subprocess.run(args, capture_output=True)
    print(f"Successfully wrote train data to {train_destination}")

    print(f"Uploading test data to {test_destination}...")
    args = ["gsutil", "cp", test_local_path, test_destination]
    subprocess.run(args, capture_output=True)
    print(f"Successfully wrote test data to {test_destination}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        required=True,
        choices=["show_num_rows", "show_sample", "export", "create_dict"],
        help=(
            "Specify the mode of operation. 'show_num_rows' shows the total number of rows for each category of data, "
            "'show_sample' prints a few rows from each data category, 'export' downloads and writes all specified "
            "data to a CSV file, and 'create_dict' creates and saves the data dictionary to a CSV file."
        ),
    )

    parser.add_argument(
        "-n",
        "--sample-size",
        type=int,
        default=None,
        help=(
            "Optional sample size to export. When provided with mode 'export', "
            "the value is passed to read_data(limit=n)."
        ),
    )

    args = parser.parse_args()
    os.makedirs(LOCAL_DIR, exist_ok=True)

    if args.mode == "show_num_rows":
        show_n_rows()
    elif args.mode == "show_sample":
        df = read_data(limit=3)
        print(df)
    elif args.mode == "export":
        user_approved = False
        while not user_approved:
            user_input = (
                input("This will overwrite existing csv files. Are you sure (y/n)? ")
                .lower()
                .strip()
            )
            if user_input in ["y", "yes"]:
                user_approved = True
            elif user_input in ["n", "no"]:
                return
            else:
                print("Invalid input. Please enter 'y' for yes or 'n' for no.")

        if args.sample_size is not None:
            # Small sample - use original in-memory approach
            df = read_data(limit=args.sample_size)
            df_train, df_test = split_train_test(df)
            # Write train
            local_path = os.path.join(LOCAL_DIR, os.path.join("train", DESTINATION_FILE_NAME))
            destination_path = os.path.join(f"{BUCKET}/data/train", DESTINATION_FILE_NAME)
            write_to_bucket(df_train, local_path, destination_path)
            print(f"Successfully wrote train data to {destination_path}")
            # Write test
            local_path = os.path.join(LOCAL_DIR, os.path.join("test", DESTINATION_FILE_NAME))
            destination_path = os.path.join(f"{BUCKET}/data/test", DESTINATION_FILE_NAME)
            write_to_bucket(df_test, local_path, destination_path)
            print(f"Successfully wrote test data to {destination_path}")
        else:
            # Full export - use chunked streaming to avoid OOM
            export_data_chunked()
    elif args.mode == "create_dict":
        df = create_data_dictionary()
        local_path = os.path.join(LOCAL_DIR, os.path.join("dictionary", DESTINATION_DICT_FILE_NAME))
        destination_path = os.path.join(f"{BUCKET}/data/dictionary", DESTINATION_DICT_FILE_NAME)
        write_to_bucket(df, local_path, destination_path)
        print(f"Successfully wrote data dictionary to {destination_path}")


if __name__ == "__main__":
    main()
