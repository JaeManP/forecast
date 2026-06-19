#!/usr/bin/env python3
"""
Inference script for downstream tasks.
Performs batch inference on evaluation dataset and stores results in CSV format.
"""

import argparse
import os
from collections.abc import Callable
from pathlib import Path

from accelerate import Accelerator

from verily.forecast import analysis, config, guardrails, model_util
from verily.forecast.analysis import TASK_CODES
from verily.forecast.constants import LOCAL_DIR


def validate_paths(args: argparse.Namespace) -> None:
    """Validate that required paths exist."""
    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"Model path does not exist: {args.model_path}")

    if not Path(args.dataset_path).exists():
        raise FileNotFoundError(f"Dataset path does not exist: {args.dataset_path}")

    tokenizer_path = args.tokenizer_path or f"{args.model_path}_tokenizer"
    if not Path(tokenizer_path).exists():
        vocab_path = args.vocab_path
        if vocab_path and Path(vocab_path).exists():
            print(
                "Tokenizer path not found. Building tokenizer from vocab and saving to "
                f"{tokenizer_path}"
            )
            tokenizer = model_util.get_tokenizer(
                args.model_name, vocab_path=vocab_path
            )
            tokenizer.save_pretrained(tokenizer_path)
        else:
            raise FileNotFoundError(
                "Tokenizer path does not exist: "
                f"{tokenizer_path}. Provide --tokenizer-path or --vocab-path."
            )
    args.tokenizer_path = tokenizer_path

    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)


def create_output_path(args: argparse.Namespace) -> str:
    """Create output CSV path based on model and task."""
    model_tag = Path(args.model_path).name.replace("-", "_")
    output_filename = f"eval_{model_tag}_{args.task}.csv"
    return str(Path(args.output_dir) / output_filename)


def batch_inference_wrapper(args: argparse.Namespace) -> Callable:
    """Create a batch inference function configured with the given arguments."""

    def batch_inference():
        # Initialize accelerator
        log_with = "wandb" if args.enable_wandb else None
        accelerator = Accelerator(
            mixed_precision=args.mixed_precision, log_with=log_with
        )

        output_csv_path = create_output_path(args)
        print(f"Output will be saved to: {output_csv_path}")

        # Initialize tracking if not disabled
        if args.enable_wandb:
            accelerator.init_trackers(
                "eval_inference",
                config={
                    "dataset_path": args.dataset_path,
                    "model_name_path": args.model_path,
                    "output_csv_path": output_csv_path,
                    "task": args.task,
                    "dataloader_batch_size": args.batch_size,
                    "n_sequences": args.n_sequences,
                    "max_new_tokens": args.max_new_tokens,
                    "sample_size": args.sample_size,
                },
                init_kwargs={"wandb": {"name": Path(output_csv_path).stem}},
            )
        # Get target codes for the task
        target_codes = TASK_CODES[args.task]
        print(f"Found {len(target_codes)} target codes")

        # Initialize ERPredictor
        er_predictor_args = {
            "dataset_path": None,
            "model_path": args.model_path,
            "tokenizer_path": args.tokenizer_path,
            "force_gpt2_config": args.force_gpt2_config,
            "nemo_config_path": args.nemo_config_path,
            "label_time_range": args.label_time_range,
            "model_name": args.model_name,
            "target_codes": target_codes,
        }

        er_predictor = analysis.ERPredictor(
            dataloader_batch_size=args.batch_size, **er_predictor_args
        )

        # Run batch inference
        analysis.batch_decode(
            accelerator,
            er_predictor,
            args.dataset_path,
            args.model_path,
            output_csv_path,
            n_sequences=args.n_sequences,
            max_new_tokens=args.max_new_tokens,
            unwrap_model=False,
            sample_size=args.sample_size,
            compute_loss=args.compute_loss,
        )

        print(f"Inference completed. Results saved to: {output_csv_path}")

    return batch_inference


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        "--model-path", "-m", type=str, required=True, help="Path to the trained model"
    )

    parser.add_argument(
        "--model-name", "-mn", type=str, required=True, help="Model architecture name"
    )

    parser.add_argument(
        "--dataset-path",
        "-d",
        type=str,
        required=True,
        help="Path to the evaluation dataset",
    )

    parser.add_argument(
        "--task",
        "-t",
        type=str,
        required=True,
        choices=list(TASK_CODES.keys()),
        help="Task type",
    )

    # Optional arguments
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="results",
        help="Output directory for results",
    )

    parser.add_argument(
        "--tokenizer-path",
        "-tk",
        type=str,
        default=None,
        help="Path to the tokenizer (defaults to model_path + '_tokenizer')",
    )
    parser.add_argument(
        "--vocab-path",
        "-vp",
        type=str,
        default=os.path.join(LOCAL_DIR, config.paths.vocab_path),
        help="Path to vocab file used to build tokenizer if missing",
    )

    parser.add_argument(
        "--batch-size", "-bs", type=int, default=2, help="Batch size for inference"
    )

    parser.add_argument(
        "--n-sequences",
        "-ns",
        type=int,
        default=16,
        help="Number of sequences to generate",
    )

    parser.add_argument(
        "--max-new-tokens",
        "-mnt",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate",
    )

    parser.add_argument(
        "--sample-size",
        "-ss",
        type=int,
        default=None,
        help="Sample size for evaluation (None for full dataset)",
    )

    parser.add_argument(
        "--label-time-range",
        "-ltr",
        type=int,
        default=730,  # 2 years
        help="Label time range in days",
    )

    parser.add_argument(
        "--mixed-precision",
        "-mp",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision training mode",
    )
    parser.add_argument(
        "--force-gpt2-config",
        "-fg",
        action="store_true",
        help="Force GPT-2 config when loading checkpoints without model_type",
    )
    parser.add_argument(
        "--nemo-config-path",
        "-nc",
        type=str,
        default=None,
        help="Path to NeMo config.yaml to build GPT-2 config when missing",
    )

    parser.add_argument(
        "--enable-wandb",
        "-ew",
        action="store_true",
        help="Enable Weights & Biases logging (requires wandb login)",
    )

    parser.add_argument(
        "--compute-loss",
        "-cl",
        action="store_true",
        help="Compute loss during inference",
    )
    args = parser.parse_args()
    guardrails.validate_external_logging_disabled(args.enable_wandb)

    print("=" * 50)
    print("Inference Configuration:")
    print("=" * 50)
    print(f"Model Path: {args.model_path}")
    print(f"Dataset Path: {args.dataset_path}")
    print(f"Task: {args.task}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Sequences: {args.n_sequences}")
    print(f"Max New Tokens: {args.max_new_tokens}")
    print(f"Sample Size: {args.sample_size}")
    print("=" * 50)

    validate_paths(args)
    print("Path validation completed successfully.")

    print("Starting batch inference...")
    batch_inference_func = batch_inference_wrapper(args)
    batch_inference_func()


if __name__ == "__main__":
    main()
