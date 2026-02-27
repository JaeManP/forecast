import argparse
import copy
import os
import random
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
import polars as pl
from datasets import Dataset
from tqdm.auto import tqdm

from verily.forecast import config, model_util
from verily.forecast.constants import CPU_COUNT, DUMMY_TOKEN, LOCAL_DIR

MODULE_DIR = Path(__file__).resolve().parent
MOCK_DATA_DIR = MODULE_DIR / "mock_data"
MOCK_DATA_TRAIN_PATH = MOCK_DATA_DIR / config.RAW_DATA_PATH


def ensure_parent_dir(path) -> None:
    """Ensures the parent directory for a path exists before writing."""
    path_str = os.fspath(path)
    directory = os.path.dirname(path_str)
    if directory:
        os.makedirs(directory, exist_ok=True)


def count_csv_rows(csv_path: str) -> int:
    """Counts the number of data rows (excluding header) in a CSV file."""
    with open(csv_path, encoding="utf-8") as f:
        next(f, None)  # skip header
        return sum(1 for _ in f)


class VisitMapper:
    def __init__(self):
        self.grouped_visits = {
            "VIOP": [
                "Outpatient Visit",
                "Office Visit",
                "Outpatient Hospital",
                "Ambulatory Surgical Center",
                "Ambulatory Clinic / Center",
                "Ambulatory Rehabilitation Visit",
                "Health examination",
                "Case Management Visit",
                "Comprehensive Outpatient Rehabilitation Facility",
            ],
            "VIIP": [
                "Inpatient Visit",
                "Inpatient Hospital",
                "Hospital",
                "Rehabilitation Hospital",
                "Inpatient Psychiatric Facility",
                "Comprehensive Inpatient Rehabilitation Facility",
                "Psychiatric Hospital",
                "Intensive Care",
                "Inpatient Hospice",
            ],
            "VIER": [
                "Emergency Room and Inpatient Visit",
                "Emergency Room Visit",
                "Emergency Room - Hospital",
                "Urgent Care Facility",
            ],
            "VITele": ["Telehealth"],
            "VIPharm": ["Pharmacy visit", "Pharmacy"],
            "VIAmb": [
                "Observation Room",
                "Laboratory Visit",
                "Ambulatory Radiology Clinic / Center",
                "Ambulatory Infusion Therapy Clinic / Center",
                "Ambulatory Endoscopy Cinic / Center",
                "Ambulatory Oncology Clinic / Center",
                "Ambulatory Oncological Radiation Clinic / Center",
                "Ambulatory Magnetic Resonance Imaging (MRI) Clinic / Center",
                "Ambulatory Mammography Clinic / Center",
                "Ambulatory Pain Clinic / Center",
                "Ambulatory Occupational Medicine Clinic / Center",
                "Ambulatory Physical Therapy Clinic / Center",
                "Ambulatory Ophthalmologic Surgery Clinic / Center",
                "Ambulatory Research Clinic / Center",
                "Ambulatory Podiatric Clinic / CenterRadiation Therapy Center",
            ],
            "VIHome": [
                "Home Visit",
                "Nursing Facility",
                "Hospice",
                "Behavioral Disturbances Assisted Living Facility",
            ],
            "VIOther": [
                "Ambulance Visit",
                "Ambulance - Land",
                "Place of Employment-Worksite",
                "Other Place of Service",
                "Unknown Value (but present in data)",
                "Mass Immunization Center",
            ],
            "VIEnd": [
                "VIEnd",
            ],
        }
        self.visit_type_to_group = flatten_grouped_dict(self.grouped_visits)


def flatten_grouped_dict(grouped_dict):
    """
    Transposes a grouped dictionary {group: [terms]} into {term: group}.
    """
    flat_dict = {}
    for group, terms in grouped_dict.items():
        for term in terms:
            flat_dict[term] = group
    return flat_dict


class TimeMapper:
    """
    Maps time chunks to string tokens.
    """

    def __init__(self):
        self.time_tokens = (
            [f"TIME_10Y_{i}" for i in range(1, 9)]
            + [f"TIME_Y_{i}" for i in range(1, 10)]
            + [f"TIME_M_{i}" for i in range(1, 12)]
            + [f"TIME_W_{i}" for i in range(1, 5)]
            + [f"TIME_D_{i}" for i in range(1, 7)]
            + [DUMMY_TOKEN]
        )
        self.day_cuts = {
            3650: "TIME_10Y",
            365: "TIME_Y",
            30: "TIME_M",
            7: "TIME_W",
            1: "TIME_D",
        }
        self.dummy_token = DUMMY_TOKEN
        self.init_all_day_cuts()

    def time_chunk_to_token(self, day_diff):
        for cut, token in self.day_cuts.items():
            if day_diff >= cut:
                return f"{token}_{day_diff // cut}"
        return self.dummy_token

    def init_all_day_cuts(self):
        """
        Initializes all day cuts for the time mapper.
        This method is called to ensure that the time mapper has all necessary day cuts.
        """
        all_day_cuts = {}
        for i in range(1, 7):
            all_day_cuts[f"TIME_D_{i}"] = i * 1
        for i in range(1, 5):
            all_day_cuts[f"TIME_W_{i}"] = i * 7
        for i in range(1, 12):
            all_day_cuts[f"TIME_M_{i}"] = i * 30
        for i in range(1, 10):
            all_day_cuts[f"TIME_Y_{i}"] = i * 365
        for i in range(1, 9):
            all_day_cuts[f"TIME_10Y_{i}"] = i * 3650
        self.all_day_cuts = all_day_cuts


# -------------------------------------------------------------------------


class BaseSequenceCurator(ABC):
    def __init__(
        self,
        filter_low_codes=True,
        filter_low_codes_only=False,
        code_counts=80,
        subject_counts=15,
        max_subject_counts=10000,
        truncate_sequences_length=None,
        collect_engine="auto",  # "auto"/"gpu" for Polars, not relevant for Pandas
    ):
        self.dummy_token = DUMMY_TOKEN
        self.time_mapper = TimeMapper()
        self.visit_mapper = VisitMapper()

        # Initialize time tokens, quantile tokens, visit tokens, and code tokens
        self.time_tokens = self.time_mapper.time_tokens
        self.quantile_tokens = [f"Q{i}" for i in range(0, 10)]
        self.visit_tokens = list(self.visit_mapper.grouped_visits.keys())
        self.code_tokens = []

        self.filter_low_codes = filter_low_codes
        self.filter_low_codes_only = filter_low_codes_only
        self.code_counts = code_counts
        self.subject_counts = subject_counts
        self.max_subject_counts = max_subject_counts
        self.truncate_sequences_length = truncate_sequences_length

        self.collect_engine = collect_engine
        if collect_engine not in ["auto", "gpu"]:
            raise ValueError("collect_engine must be 'auto' or 'gpu' for Polars. ")

    @property
    def all_tokens(self):
        return (
            sorted(self.time_tokens)
            + sorted(self.quantile_tokens)
            + sorted(self.visit_tokens)
            + sorted(self.code_tokens)
        )

    @abstractmethod
    def create_text_sequences(self, df):
        pass


# -------------------------------------------------------------------------


class PolarsSequenceCurator(BaseSequenceCurator):
    def create_text_sequences(self, df: pl.DataFrame | pl.LazyFrame):
        if self.filter_low_codes:
            df = self._filter_low_codes_and_subjects(df)
        df = self._preprocess(df)
        self._curate_unique_tokens(df)
        df, sequences_txt = self._group_events_to_sequence(df)
        return df, sequences_txt

    def _group_events_to_sequence(self, df: pl.DataFrame | pl.LazyFrame):
        df = df.sort(["subject_id", "timestamp", "visit_id"], nulls_last=False).with_columns(
            [
                pl.concat_str(
                    [
                        pl.col("time_token"),
                        pl.col("visit_token"),
                        pl.col("code"),
                        pl.col("quantile_token"),
                        pl.col("visit_end_token"),
                    ],
                    separator=" ",
                    ignore_nulls=True,
                )
                .str.strip_chars()
                .alias("all_tokens")
            ]
        )
        grouped = df.group_by("subject_id", maintain_order=True).agg(
            pl.col("all_tokens").alias("sequence_list")
        )
        grouped = grouped.with_columns(
            pl.col("sequence_list").drop_nulls().list.join(" ").alias("sequence_str")
        )
        if self.truncate_sequences_length is not None:
            grouped = grouped.with_columns(
                pl.col("sequence_str")
                .str.split(" ")
                .list.slice(0, self.truncate_sequences_length)
                .list.join(" ")
                .alias("sequence_str")
            )
        grouped = self._collect_if_lazy(grouped.select(["subject_id", "sequence_str"]))
        sequence_dict = dict(zip(grouped["subject_id"], grouped["sequence_str"]))
        return df, sequence_dict

    def _filter_low_codes_and_subjects(
        self, df: pl.DataFrame | pl.LazyFrame, num_rounds=2, pbar=None
    ):
        if not self.filter_low_codes:
            return df

        def filter_low_codes(df):
            df = (
                df.group_by("code")
                .agg(pl.col("subject_id").n_unique().alias("subject_count_per_code"))
                .filter(pl.col("subject_count_per_code") >= self.code_counts)
                .join(df, on="code", how="inner")
            )
            return df

        def filter_low_subjects(df):
            df = (
                df.group_by("subject_id")
                .agg(pl.col("code").count().alias("code_count_per_subject"))
                .filter(pl.col("code_count_per_subject") >= self.subject_counts)
                .filter(pl.col("code_count_per_subject") <= self.max_subject_counts)
                .join(df, on="subject_id", how="inner")
            )
            return df

        if self.filter_low_codes_only:
            return filter_low_codes(df)

        for i in range(num_rounds):
            if pbar:
                pbar.set_postfix(step=f"Filtering low codes and subjects - round{i + 1}")
            # Filter low codes, process twice to ensure both code and subject counts
            df = filter_low_codes(df)
            df = filter_low_subjects(df)
        return df

    def _preprocess(self, df: pl.DataFrame | pl.LazyFrame, pbar=None):
        time_dtype = df.collect_schema()["time"]
        if pbar:
            pbar.set_postfix(step="Preprocessing time column")
        if time_dtype == pl.Datetime:
            df = df.with_columns(
                [
                    pl.col("time").dt.timestamp("s").alias("timestamp"),
                    pl.col("time").alias("datetime"),
                ]
            )
        elif time_dtype in (pl.Int64, pl.Float64):
            df = df.with_columns(
                [
                    (pl.col("time") * 1e6).cast(pl.Float64).alias("timestamp"),
                    (pl.col("time") * 1e6).cast(pl.Datetime).alias("datetime"),
                ]
            )
        else:
            raise ValueError(f"Unsupported time dtype: {time_dtype}")

        # Time is now a float column (seconds since epoch), sort time non first
        if pbar:
            pbar.set_postfix(step="Sorting by subject_id, timestamp, visit_id")
        df = df.sort(["subject_id", "timestamp", "visit_id"], nulls_last=False)

        # Compute time deltas relative to previous event
        if pbar:
            pbar.set_postfix(step="Computing time tokens")
        df = df.with_columns(
            [
                pl.col("datetime")
                .diff()
                .over("subject_id")
                .dt.total_days()
                .fill_null(0)
                .alias("day_delta")
            ]
        )
        # Build time token using day deltas and map to string tokens
        df = df.with_columns(
            pl.col("day_delta")
            .cut(
                breaks=list(self.time_mapper.all_day_cuts.values()),
                labels=[self.dummy_token] + list(self.time_mapper.all_day_cuts.keys()),
                left_closed=True,
            )
            .cast(pl.String)
            .replace(self.dummy_token, None)
            .alias("time_token")
        ).drop("day_delta")

        # Create quantile tokens as Q...
        # USING QCUT FUNCTION
        if pbar:
            pbar.set_postfix(step="Creating quantile tokens")
        df = df.with_columns(
            pl.col("numeric_value")
            .qcut(10, labels=[f"Q{i}" for i in range(10)], allow_duplicates=True)
            .over("code")
            .cast(pl.String)
            .alias("quantile_token")
        )

        # Create visit tokens if visit id differs row to row within subject
        if pbar:
            pbar.set_postfix(step="Creating visit tokens")
        df = df.with_columns(
            pl.col("visit_id")
            .cast(pl.Int64)
            .fill_null(0)
            .diff()
            .over("subject_id")
            .alias("visit_diff")
        ).with_columns(
            pl.when(pl.col("visit_diff") != 0)
            .then(
                pl.col("visit_type").map_elements(
                    lambda x: self.visit_mapper.visit_type_to_group.get(x, "Other"),
                    return_dtype=pl.String,
                )
            )
            .otherwise(None)
            .alias("visit_token")
        )

        # Create end of visit token if visit id differs row to row within subject,
        df = (
            df.with_columns(
                pl.col("visit_diff").shift(-1).over("subject_id").alias("visit_end_diff")
            )
            .with_columns(
                pl.when((pl.col("visit_end_diff") != 0) & (pl.col("visit_id").is_not_null()))
                .then(pl.lit(config.ENDVISIT_TOKEN))
                .otherwise(None)
                .alias("visit_end_token")
            )
            .drop("visit_end_diff")
            .drop("visit_diff")
        )

        return df

    def _curate_unique_tokens(self, df: pl.DataFrame | pl.LazyFrame):
        """Collects unique tokens from the DataFrame and updates the class attributes."""

        def collect_col(col):
            return self._collect_if_lazy(df.select(col).unique()).to_series().to_list()

        self.code_tokens = unique(self.code_tokens + [t for t in collect_col("code") if t])
        print(f"Collected {len(self.all_tokens)} unique tokens.")

    def _collect_if_lazy(self, df):
        return df.collect(engine=self.collect_engine) if isinstance(df, pl.LazyFrame) else df


# --- Data augmentation ---
class DataAugmentor:
    def __init__(
        self,
        dataset,
        target_len=None,
        num_sequences=1,
        additional_visits_to_remove=0,
        augment_prs=False,
    ):
        self.dataset = dataset
        self.dataset_features = self.dataset.features

        self.target_len = target_len
        self.num_sequences = num_sequences
        self.additional_visits_to_remove = additional_visits_to_remove

        self.augment_prs = augment_prs
        self.augment_data = (target_len is not None) or (additional_visits_to_remove > 0)
        if (not self.augment_data) and (self.num_sequences > 1):
            raise ValueError("Data augmentation is not enabled.")

    def remove_visits_augment(self, text, target_len, num_seq):
        xx = text.split(" ")

        # From the sequence, Identify the visits which start with VIOP/VIAmb... and end with VIEnd
        visits = []
        i = 0
        while i < len(xx):
            token = xx[i]
            if token.startswith("VI") and (token != "VIEnd"):
                try:
                    end_idx = xx.index("VIEnd", i)
                except ValueError:
                    end_idx = len(xx) - 1
                visits.append((i, end_idx))
                i = end_idx
            i += 1

        out = []
        for _ in range(num_seq):
            to_remove = []
            to_remove_len = 0
            visits_y = copy.deepcopy(visits)
            additional_visits_removed = 0
            additional_visits_to_remove = random.randint(0, self.additional_visits_to_remove + 1)

            while len(visits_y) > 0:
                if additional_visits_to_remove == 0:
                    if len(xx) - to_remove_len <= target_len:
                        break
                else:
                    if additional_visits_removed >= additional_visits_to_remove:
                        break
                    if len(xx) - to_remove_len <= target_len:
                        additional_visits_removed += 1

                # Iteratively drop a random visit from y
                visit_to_drop = random.choice(visits_y)
                to_remove.append(visit_to_drop)
                to_remove_len += visit_to_drop[1] - visit_to_drop[0] + 1
                visits_y.remove(visit_to_drop)

            # Sort visits to drop by their start index and remove from original visits
            to_remove.sort(key=lambda x: x[0])
            y = copy.deepcopy(xx)
            for start, end in reversed(to_remove):
                del y[start : end + 1]
            out += [y]

        return out

    # Defined dataset augmentation for the extend function above
    def augment_by_extending(self, examples):
        outputs = {c: [] for c in self.dataset_features}
        n = len(examples["subject_id"])

        for i in range(n):
            extended_texts = self.remove_visits_augment(
                examples["text"][i],
                target_len=self.target_len,
                num_seq=self.num_sequences,
            )
            for new_text in extended_texts:
                for c in self.dataset_features:
                    outputs[c].append(examples[c][i])
                outputs["text"][-1] = " ".join(new_text)
        return outputs

    def augment_prs_data(self, examples):
        outputs = {c: [] for c in self.dataset_features}
        n = len(examples["subject_id"])

        for i in range(n):
            for c in self.dataset_features:
                outputs[c].append(examples[c][i])

            prs = examples["prs"][i]

            # Check for any non-zero element in the nested lists
            flat_prs = [item for sublist in prs for item in sublist]
            if any(x != 0 for x in flat_prs):
                # Create a zeros-like structure
                zero_prs = [[0 for _ in sublist] for sublist in prs]
                for c in self.dataset_features:
                    outputs[c].append(examples[c][i])
                outputs["prs"][-1] = zero_prs
        return outputs

    def augment_map(self):
        dataset = self.dataset
        if self.augment_data:
            print("Augmenting dataset by extending visits...")
            dataset = dataset.map(
                self.augment_by_extending,
                batched=True,
                num_proc=CPU_COUNT // 2,
            )
        if self.augment_prs:
            print("Augmenting dataset by duplicating PRS data...")
            dataset = dataset.map(
                self.augment_prs_data,
                batched=True,
                num_proc=CPU_COUNT // 2,
            )
        return dataset


# --- Utility Functions ---
def unique(lst):
    """Return a sorted list of unique elements from the input list."""
    return sorted(set(lst)) if lst else []


def save_sequences_to_csv(sequences_txt, sequences_path):
    sequences_path_csv = sequences_path.replace(".txt", ".csv")
    print(f"Saving sequences df to {sequences_path_csv}")

    sequences_df = (
        pd.DataFrame.from_dict(sequences_txt, orient="index", columns=["sequence"])
        .reset_index()
        .rename(columns={"index": "subject_id"})
    )
    sequences_df.to_csv(sequences_path_csv, index=False)
    return sequences_df


# --- Tokenization ---
def augment_and_tokenize_data(
    dataset,
    dataset_path: str | None = None,
    vocab_path: str | None = None,
    augment_prs=False,
):
    if dataset_path is None:
        dataset_path = config.paths.dataset_path
    if vocab_path is None:
        vocab_path = config.paths.vocab_path
    if isinstance(dataset, Dataset):
        print("Splitting train/test")
        # Split train/test if not IterableDataset
        dataset = dataset.train_test_split(test_size=0.1, seed=42)

    print("Augmenting train data")
    data_augmentor = DataAugmentor(
        dataset["train"],
        target_len=config.MODEL_WINDOW_SIZE,
        num_sequences=config.NUM_SEQUENCES_AUGMENTATION,
        additional_visits_to_remove=config.ADDITIONAL_VISITS_TO_REMOVE,
        augment_prs=augment_prs,
    )
    dataset["train"] = data_augmentor.augment_map()

    print("Tokenizing sequences...")
    tokenizer = model_util.get_tokenizer(config.MODEL_NAME, vocab_path=vocab_path)
    dataset = encode_dataset_with_tokenizer(
        dataset,
        tokenizer,
        iterable_dataset=False,
        remove_text=False,
        add_eos_token=False,
        truncation=False,
    )

    # Save to disk
    dataset.save_to_disk(dataset_path)
    print(f"Dataset saved to {dataset_path}")
    return dataset


def encode_dataset_with_tokenizer(
    dataset: Dataset,
    tokenizer,
    text_column_name="text",
    max_len=config.MODEL_WINDOW_SIZE,
    remove_text=True,
    add_eos_token=True,
    padding_side="left",
    truncation=True,
    num_proc=CPU_COUNT // 2,
    **kwargs,
):
    if (not truncation) and (padding_side is not None):
        print("Warning: padding_side is ignored when truncation/padding is False.")

    def encode(d):
        # Return encoded sequences with eos token at end
        if add_eos_token:
            texts_with_eos = [text + " " + config.EOS_TOKEN for text in d[text_column_name]]
        else:
            texts_with_eos = d[text_column_name]
        encodings = tokenizer(
            texts_with_eos,
            truncation=truncation,
            padding="max_length" if truncation else False,
            max_length=max_len,
            return_tensors="pt" if truncation else None,
            padding_side=padding_side,
        )
        return encodings

    dataset = dataset.map(
        lambda batch: encode(batch),
        remove_columns=[text_column_name, "subject_id"] if remove_text else None,
        batched=True,
        batch_size=1000,
        num_proc=None,  # Disable multiprocessing for stability
        desc="Tokenizing",
    )
    return dataset


# --- Main ---
def main(
    num_rows=config.NUM_DATA_ROWS,  # 624_550_551  # int(1e9)
    batch_size=config.BATCH_SIZE_PREPROCESS,  # 5000000
    data_path=os.path.join(LOCAL_DIR, config.RAW_DATA_PATH),
    sequences_path: str | None = None,
    vocab_path: str | None = None,
    dataset_path: str | None = None,
    truncate_sequences_length=config.MODEL_WINDOW_SIZE,
    use_mock_data: bool = False,
    skip_filtering: bool = False,
):
    if sequences_path is None:
        sequences_path = config.paths.sequences_path
    if vocab_path is None:
        vocab_path = config.paths.vocab_path
    if dataset_path is None:
        dataset_path = config.paths.dataset_path
    ensure_parent_dir(sequences_path)
    ensure_parent_dir(vocab_path)
    ensure_parent_dir(dataset_path)
    ensure_parent_dir(dataset_path + "_unsplit")

    effective_batch_size = max(1, batch_size)
    ratio = max(1, num_rows // effective_batch_size)
    i = 0
    sequence_curator = PolarsSequenceCurator(
        code_counts=max(1, config.MIN_CODES_PER_SUBJECT // ratio),
        subject_counts=config.MIN_SUBJECTS_PER_CODE,
        max_subject_counts=config.MAX_CODES_PER_SUBJECT,
        truncate_sequences_length=truncate_sequences_length,
        filter_low_codes=not (use_mock_data or skip_filtering),
    )
    sequences_txt = {}

    pbar = tqdm(range(num_rows // batch_size + 1))
    for ii in pbar:
        if i >= num_rows:
            break
        print(f"Batch {ii}: {i}")
        pbar.set_postfix(step="Scanning")
        try:
            df = pl.scan_csv(data_path, n_rows=batch_size, skip_rows_after_header=i).collect()
        except pl.exceptions.NoDataError:
            print(
                "Polars reported insufficient rows to skip; reached end of file earlier "
                "than expected."
            )
            break
        if df.height == 0:
            print("No more rows returned by scan_csv; stopping early.")
            break
        pbar.set_postfix(step="Loaded")

        if i + batch_size < num_rows:
            # Remove the last rows corresponding to the last incomplete batch by subject_id
            last_subject_id = df.select(["subject_id"]).tail(1).to_series().item()
            n_remove = (
                df.filter(pl.col("subject_id") == last_subject_id).select("subject_id").shape[0]
            )

            df = df.filter(pl.col("subject_id") != last_subject_id)
            i += batch_size - n_remove
        else:
            i += batch_size

        df = sequence_curator._filter_low_codes_and_subjects(df, pbar=pbar)
        pbar.set_postfix(step="Filtered")
        df = sequence_curator._preprocess(df, pbar=pbar)
        pbar.set_postfix(step="Preprocessed")

        df2, seq_batch = sequence_curator._group_events_to_sequence(df)
        sequences_txt.update(seq_batch)
        pbar.set_postfix(step="Sentences done")

        if len(seq_batch) == 0:
            print(f"No sequences found in batch starting at row {i}. Skipping.")
            break

        with open(sequences_path, "a") as f:
            for seq in seq_batch.values():
                f.write(seq + "\n")
        print("Curated #sentences:", len(seq_batch), "Total:", len(sequences_txt))
        sequence_curator._curate_unique_tokens(df)

    if not sequences_txt:
        raise ValueError(
            "No sequences were generated. Try lowering filtering thresholds or "
            "running with --use-mock-data."
        )

    print("=" * 80, "\nProcessing done.\n", "=" * 80)

    # Save vocabulary to file
    print(f"Saving vocabulary to {vocab_path}")
    model_util.create_vocab_file(
        sequence_curator.all_tokens, vocab_path=vocab_path, special_tokens=[]
    )

    # Save sequences_txt to csv
    print(f"Saving sequences CSV to {sequences_path}")
    sequences_df = save_sequences_to_csv(sequences_txt, sequences_path)

    # Save sequences to dataset
    print(f"Saving unsplit dataset {dataset_path}_unsplit")
    dataset = Dataset.from_dict(
        {
            "subject_id": sequences_df["subject_id"].tolist(),
            "text": sequences_df["sequence"].tolist(),
        }
    )
    dataset.save_to_disk(dataset_path + "_unsplit")

    # Tokenize the sequences and save to dataset
    augment_and_tokenize_data(
        dataset,
        dataset_path=dataset_path,
        vocab_path=vocab_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create AoU FM training datasets.")
    parser.add_argument(
        "--use-mock-data",
        action="store_true",
        help=(
            "Use the repo-local mock_data/train/ehr_v1.csv file instead of "
            "reading from WORKSPACE storage. Useful for quick testing."
        ),
    )
    parser.add_argument(
        "--skip-filtering",
        action="store_true",
        help=(
            "Disable code/subject filtering. Helpful when using sampled datasets "
            "that do not meet the default frequency thresholds."
        ),
    )
    parser.add_argument(
        "--post",
        type=str,
        default=None,
        help=(
            "Override the POST identifier used in output file paths. "
            "If not provided, defaults to 'v0_<git_commit_hash>'."
        ),
    )
    args = parser.parse_args()

    if args.post:
        config.paths.set_post(args.post)

    if args.use_mock_data:
        if not MOCK_DATA_TRAIN_PATH.exists():
            raise FileNotFoundError(f"Mock data file not found at {MOCK_DATA_TRAIN_PATH}")
        data_path = os.fspath(MOCK_DATA_TRAIN_PATH)
        sequences_path = os.fspath(MOCK_DATA_DIR / config.paths.sequences_path)
        vocab_path = os.fspath(MOCK_DATA_DIR / config.paths.vocab_path)
        dataset_path = os.fspath(MOCK_DATA_DIR / config.paths.dataset_path)
        num_rows = count_csv_rows(data_path)
        if num_rows == 0:
            raise ValueError(f"Mock data file at {data_path} is empty.")
        batch_size = max(1, num_rows)
    else:
        os.makedirs(LOCAL_DIR, exist_ok=True)
        data_path = os.path.join(LOCAL_DIR, config.RAW_DATA_PATH)
        sequences_path = os.path.join(LOCAL_DIR, config.paths.sequences_path)
        vocab_path = os.path.join(LOCAL_DIR, config.paths.vocab_path)
        dataset_path = os.path.join(LOCAL_DIR, config.paths.dataset_path)
        num_rows = config.NUM_DATA_ROWS
        batch_size = config.BATCH_SIZE_PREPROCESS

    main(
        num_rows=num_rows,
        batch_size=batch_size,
        data_path=data_path,
        sequences_path=sequences_path,
        vocab_path=vocab_path,
        dataset_path=dataset_path,
        use_mock_data=args.use_mock_data,
        skip_filtering=args.skip_filtering,
    )
