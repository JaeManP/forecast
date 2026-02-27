import os
import subprocess


def get_git_commit_hash():
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
            )
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"

# Get github commit
GIT_COMMIT_HASH = get_git_commit_hash()

# Data preprocessing params
NUM_DATA_ROWS = 635_439_704
BATCH_SIZE_PREPROCESS = 50_000_000
MIN_SUBJECTS_PER_CODE = 200
MIN_CODES_PER_SUBJECT = 100
MAX_CODES_PER_SUBJECT = 2000

# Data augmentation params
NUM_SEQUENCES_AUGMENTATION = 5
ADDITIONAL_VISITS_TO_REMOVE = 3
DATALOADER_NUM_CPUS = 16

# Model genomics params
MODEL_GENOMICS_INPUT_1 = 3481  # 445
MODEL_GENOMICS_INPUT_2 = 1
MODEL_GENOMICS_HIDDEN_DIM = 4096  # 2048
MODEL_GENOMICS_TOKENS = 10
MODEL_GENOMICS_ACTIVATION = "gelu"
MODEL_GENOMICS_CLIP = 1000

# Training params
BATCH_SIZE = 2
EPOCHS = 1
LEARNING_RATE = 1e-4
WANDB_PROJECT = "forecast"

# Inference params
MAX_NEW_TOKENS = 128
NUM_RETURN_SEQUENCES = 1

# Paths
RAW_DATA_FILENAME = "ehr_v1.csv"
RAW_DATA_PATH = f"train/{RAW_DATA_FILENAME}"

# Visits of interest
ER_TOKEN = "VIER"
ENDVISIT_TOKEN = "VIEnd"
EOS_TOKEN = "</s>"


class PathConfig:
    """Configuration for POST-dependent paths."""

    def __init__(self, post: str):
        self.post = post

    @property
    def sequences_path(self) -> str:
        return f"datasets/seqs_{self.post}.txt"

    @property
    def vocab_path(self) -> str:
        return f"vocab/vocab_{self.post}.txt"

    @property
    def dataset_path(self) -> str:
        return f"datasets/cnd_{self.post}"

    @property
    def model_name_path(self) -> str:
        return f"model/gpt_model_{self.post}"

    @property
    def tokenizer_path(self) -> str:
        return f"{self.model_name_path}_tokenizer"

    @property
    def inference_output_csv_path(self) -> str:
        return f"results/explo_{self.post}.csv"

    def set_post(self, post_value: str) -> None:
        """Override POST and update all dependent paths."""
        self.post = post_value
        print(f"POST overridden to: {post_value}")


# Module-level instance
if "WORKSPACE_CDR" in os.environ:
    paths = PathConfig(post="v0_" + GIT_COMMIT_HASH[:7])
else:
    paths = PathConfig(post="unset")

# ------------------ Shared params ------------------
# Model params
MODEL_NAME = "gpt"
MODEL_WINDOW_SIZE = 2048
USE_SLIDING_WINDOW = False

MODEL_EMBEDDING_SIZE = 960
MODEL_LAYERS = 12
MODEL_HEADS = 8

# Qwen-specific
ROPE_THETA = 10000.0
