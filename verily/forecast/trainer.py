import argparse
import gc
import os
import shutil
from pathlib import Path

import datasets
import pandas as pd
import torch
from accelerate import Accelerator
from datasets import DatasetDict
from tqdm.auto import tqdm
from transformers import (
    Trainer,
    TrainingArguments,
    get_scheduler,
)

from verily.forecast import config, model_util
from verily.forecast.constants import LOCAL_DIR

os.environ["TOKENIZERS_PARALLELISM"] = "false"

precision_map = model_util.precision_map

MODULE_DIR = Path(__file__).resolve().parent
MOCK_DATA_DIR = MODULE_DIR / "mock_data"
MOCK_DATASET_PATH = MOCK_DATA_DIR / config.paths.dataset_path / "train"
MOCK_VOCAB_PATH = MOCK_DATA_DIR / config.paths.vocab_path
MOCK_MODEL_PATH = MOCK_DATA_DIR / config.paths.model_name_path


def accelerator_print(message, accelerator=None):
    if not accelerator:
        print(message)
    else:
        accelerator.print(message)


def get_optimizer_and_scheduler(
    model, dataloader, num_epochs=1, accelerator=None, lr=config.LEARNING_RATE
):
    """
    Set up the optimizer and learning rate scheduler for the model.
    """
    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    try:
        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=len(dataloader) * num_epochs,
        )
        accelerator_print(
            f"Using linear learning rate scheduler with {len(dataloader)} steps per epoch.",
            accelerator=accelerator,
        )

    except TypeError:
        # Dataloader is an IterableDataset, so we cannot get its length
        lr_scheduler = get_scheduler(
            name="constant",
            optimizer=optimizer,
        )
        accelerator_print(
            "Using constant learning rate scheduler since dataloader is an IterableDataset.",
            accelerator=accelerator,
        )

    return optimizer, lr_scheduler


def process_batch(
    model,
    batch,
    accelerator=None,
    training=True,
    optimizer=None,
    lr_scheduler=None,
    return_full_loss=False,
    train_on_full_loss=False,
):
    full_loss = None
    if training:
        outputs = model(**batch)
    else:
        with torch.no_grad():
            outputs = model(**batch)
    loss = outputs.loss

    if return_full_loss or train_on_full_loss:
        # Manually compute full loss per example in batch
        logits = outputs.logits
        labels = batch["labels"]

        # Shift labels and logits to align for cross-entropy
        shift_logits = logits[..., -labels.size(-1) : -1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        full_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        full_loss = full_loss.view(shift_labels.size())
        # mean loss per example in batch, removing padding zeros
        mask = shift_labels.ne(-100)
        full_loss = (full_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    if training:
        if train_on_full_loss and full_loss is not None:
            train_loss = full_loss.mean()
        else:
            train_loss = loss
        optimizer.zero_grad()
        accelerator.backward(train_loss)
        optimizer.step()
        lr_scheduler.step()

    return loss, full_loss


def run_training(
    model,
    dataloader,
    optimizer,
    lr_scheduler,
    accelerator,
    num_epochs=1,
    global_step=0,
    model_filepath="exploration_bert/model",
    wandb_project_name=config.WANDB_PROJECT,
    train_on_full_loss=True,
    skip_checkpoints=False,
    **kwargs,
):
    """
    Run the training loop for the model.
    """
    accelerator.init_trackers(
        wandb_project_name,
        config={
            "num_epochs": num_epochs,
            "batch_size": dataloader.batch_size,
            "model_filepath": model_filepath,
        },
        init_kwargs={"wandb": {"name": os.path.basename(model_filepath)}},
    )
    model.train()

    accelerator.print(f"Starting training for {num_epochs} epochs...")
    for epoch in tqdm(range(num_epochs), disable=not accelerator.is_local_main_process):
        epoch_loss = 0.0
        step_count = 0
        pbar = tqdm(
            dataloader,
            disable=not accelerator.is_local_main_process,
            desc=f"Epoch {epoch+1}",
        )

        for batch in pbar:
            with accelerator.autocast():
                if global_step == 0:
                    model_util.print_model_info(accelerator, model)
                    model_util.print_batch_dtype(accelerator, batch)

                # Process batch using shared logic
                loss, full_loss = process_batch(
                    model,
                    batch,
                    accelerator,
                    training=True,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    return_full_loss=True,
                    train_on_full_loss=train_on_full_loss,
                )

                # Logging
                epoch_loss += loss.item()
                step_count += 1
                global_step += 1
                accelerator.log(
                    {
                        "loss": loss.item(),
                        "full_loss": full_loss.mean().item(),
                        "loss_diff": (loss - full_loss.mean()).item(),
                    },
                    step=global_step,
                )
                if global_step % 100 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
                pbar.set_postfix(loss=loss.item(), full_loss=full_loss.mean().item())

        avg_epoch_loss = epoch_loss / step_count
        accelerator.print(f"Epoch {epoch+1} completed. Avg Loss: {avg_epoch_loss:.4f}")
        accelerator.log({"epoch_loss": avg_epoch_loss, "epoch": epoch + 1})

        if not skip_checkpoints:
            save_model(
                f"checkpoints/{model_filepath}_ep{epoch + 1}",
                model,
                tokenizer=None,
                accelerator=accelerator,
            )


def compute_test_loss(
    model,
    test_dataset,
    tokenizer,
    accelerator,
    batch_size=1,
    num_cpus=4,
    model_name="gpt",
    output_filepath="results/test_loss/test_loss.txt",
    save_full_loss=False,
):
    """
    Compute loss on the test dataset.
    """
    if test_dataset is None or len(test_dataset) == 0:
        accelerator.print("No test dataset available for evaluation.")
        return None

    accelerator.print(f"Computing test loss on {len(test_dataset)} samples...")
    if batch_size > 1:
        accelerator.print(
            f"Warning: Test batch size > 1 ({batch_size}), all results may break."
        )

    # Create test dataloader
    test_dataloader = get_data_loader(
        model_name,
        tokenizer,
        test_dataset,
        batch_size=batch_size,
        num_cpus=num_cpus,
    )

    # Prepare dataloader
    model, test_dataloader = accelerator.prepare(model, test_dataloader)

    model.eval()
    total_loss = []
    all_losses = {}
    num_batches = 0

    pbar = tqdm(
        test_dataloader,
        disable=not accelerator.is_local_main_process,
        desc="Computing test loss",
    )

    for batch in pbar:
        # Process batch using shared logic
        with accelerator.autocast():
            loss, full_loss = process_batch(
                model,
                batch,
                accelerator,
                training=False,
                return_full_loss=save_full_loss,
            )
            total_loss += [loss.item()]
            num_batches += 1
        # Cast to numpy for saving later
        if save_full_loss and full_loss is not None:
            subjects = batch.get("subject_id")
            batch_losses = full_loss.cpu().numpy().flatten().tolist()
            for i, subject in enumerate(subjects):
                all_losses[subject] = batch_losses[i]
        pbar.set_postfix(loss=loss.item(), full_loss=full_loss.mean().item())

    # Gather losses from all processes
    total_loss_tensor = torch.tensor(total_loss, device=accelerator.device)
    num_batches_tensor = torch.tensor(num_batches, device=accelerator.device)

    gathered_loss = accelerator.gather_for_metrics(total_loss_tensor).sum().item()
    gathered_batches = accelerator.gather_for_metrics(num_batches_tensor).sum().item()

    avg_test_loss = gathered_loss / gathered_batches
    accelerator.print(f"Test Loss: {avg_test_loss}")

    if save_full_loss:
        # all_losses is a dict: {subject_id: loss}
        # Convert to two lists for gathering: subject_ids and losses
        subject_ids = list(all_losses.keys())
        losses = list(all_losses.values())

        # Convert to tensors
        subject_ids_tensor = torch.tensor(subject_ids, device=accelerator.device)
        losses_tensor = torch.tensor(losses, device=accelerator.device)

        # Gather across processes
        gathered_subject_ids = accelerator.gather_for_metrics(subject_ids_tensor)
        gathered_losses = accelerator.gather_for_metrics(losses_tensor)

        # Optionally, print mean loss
        accelerator.print(f"Test loss from all: {gathered_losses.mean().item()}")

        # For saving, combine subject_ids and losses into a dict
        gathered_all_losses = dict(zip(gathered_subject_ids.cpu().tolist(), gathered_losses.cpu().tolist()))

    # Save all test losses to a file
    if accelerator.is_main_process and save_full_loss:
        parent_path = os.path.dirname(output_filepath)
        if parent_path and not os.path.exists(parent_path):
            os.makedirs(parent_path, exist_ok=True)
        # Delete existing file if exists
        if os.path.exists(output_filepath):
            os.remove(output_filepath)
        d = pd.DataFrame.from_dict(
            gathered_all_losses, orient="index", columns=["test_loss"]
        ).reset_index()
        d.rename(columns={"index": "subject_id"}, inplace=True)
        d.to_csv(output_filepath, header=True)
        accelerator.print(f"Saved individual test losses to {output_filepath}")
    accelerator.wait_for_everyone()

    return avg_test_loss


def get_data_loader(
    model_name,
    tokenizer,
    dataset,
    batch_size=32,
    num_cpus=config.DATALOADER_NUM_CPUS,
    kept_columns=["subject_id"],
    padding_side="right",
):
    """
    Create a DataLoader for the dataset.
    """
    return model_util.load_dataloader(
        dataset,
        tokenizer,
        kept_columns=kept_columns,
        num_workers=num_cpus,
        shuffle=True,
        dataloader_batch_size=batch_size,
        model_name=model_name,
        padding_side=padding_side,
    )


def create_dataset_and_vocab(**kwargs):
    raise NotImplementedError(
        "Preprocessing no longer supported, assumes dataset available in dataset_path. "
        "Handled directly in aou_data_loader.py"
    )


def save_model(model_filepath, model, tokenizer, accelerator=None):
    model_save_path = model_filepath
    os.makedirs(model_save_path, exist_ok=True)
    unwrapped_model = accelerator.unwrap_model(model)
    if accelerator.is_main_process:
        unwrapped_model.save_pretrained(model_save_path)
        if tokenizer is not None:
            tokenizer.save_pretrained(f"{model_filepath}_tokenizer")


def _resolve_local_path(path: str | os.PathLike[str]) -> str:
    """Return an absolute path string; leave remote URIs untouched."""
    path_str = os.fspath(path)
    if "://" in path_str:
        return path_str
    return str(Path(path_str).expanduser().resolve())


def main(
    accelerator=None,
    model_name="gpt",
    batch_size=32,
    model_filepath="exploration_bert/gpt-custom",
    num_epochs=1,
    num_cpus=4,
    native_trainer=False,
    dataset_path="exploration_bert/dataset",
    preprocess_data=False,
    vocab_path="data/vocab_codes.txt",
    wandb_project_name=config.WANDB_PROJECT,
    cast_model_dtype=False,
    lr=config.LEARNING_RATE,
    padding_side="right",
    train_on_full_loss=True,
    skip_checkpoints=False,
):
    """
    Main function to run the training pipeline using accelerate and transformers.
    This function handles the following:
    - Data loading
        - UNSUPPORTED: Preprocessing, tokenization and dataset creation
    - Model and tokenizer initialization
    - Training the model using either native Trainer or custom training loop
    - Saving the trained model and tokenizer

    Args:
        model_name (str): Name of the model to use.
        sample_size (int): Number of rows to sample from the dataset. UNSUPPORTED.
        batch_size (int): Batch size for training.
        model_filepath (str): Path to save the trained model.
        num_epochs (int): Number of epochs to train for.
        num_cpus (int): Number of CPU cores to use for data loading.
        native_trainer (bool): Whether to use the native Trainer class for training.
        dummy_data (bool): Whether to use dummy data for training. # UNSUPPORTED.
        create_vocab (bool): Whether to create a vocabulary file. # UNSUPPORTED.
        iterable_dataset (bool): Whether to use an IterableDataset. # UNSUPPORTED.
        dataset_path (str): Path to save the dataset.
        data_path (str): Path to the raw data file. # UNSUPPORTED.
        preprocess_data (bool): Whether to preprocess the data.
        vocab_path (str): Path to the vocabulary file.
        sequences_path (str): Path to the sequences file. # UNSUPPORTED
        wandb_project_name (str): Name of the Weights & Biases project.
        cast_model_dtype (bool): Whether to cast the model to a specific dtype.
            Should be unnecessary with accelerate -- defaults to False
    """
    accelerator.print(
        "=" * 80 + f"\nSetting up the Accelerator - Using device: {accelerator.device}"
    )

    # Assert no model is already saved at the filepath
    if os.path.exists(model_filepath):
        raise ValueError(
            f"Model already exists at {model_filepath}. Please choose a different path."
        )

    # 0. Data processing to sequences then tokenization dataset
    if accelerator.is_main_process and preprocess_data:
        # create_dataset_and_vocab(***)
        raise NotImplementedError(
            "Data preprocessing is no longer supported, assumes dataset available under dataset_path."
        )
    else:
        accelerator.print(
            f"Skipping data preprocessing, using existing dataset {dataset_path}"
        )
    accelerator.wait_for_everyone()
    dataset = datasets.load_from_disk(dataset_path)
    if isinstance(dataset, DatasetDict):
        for split in dataset.keys():
            dataset[split].set_format(type=None)
    else:
        dataset.set_format(type=None)

    train_dataset = dataset["train"] if isinstance(dataset, DatasetDict) else dataset
    train_dataset = train_dataset.with_format(type="torch")
    if "text" in train_dataset.column_names:
        train_dataset = train_dataset.remove_columns("text")

    try:
        accelerator.print(f"Dataset features: {train_dataset.features}")
        accelerator.print(f"Dataset size: {len(train_dataset)}")
    except TypeError:
        accelerator.print("Dataset is an IterableDataset, size/sample not available.")

    # 1. Load custom tokenizer
    tokenizer = model_util.get_tokenizer(model_name, vocab_path=vocab_path)
    accelerator.print(f"Using tokenizer: {tokenizer.__class__.__name__}")
    accelerator.print(f"Tokenizer vocab size: {len(tokenizer)}")

    # 2. Configure and create model
    model = model_util.get_model(model_name, tokenizer)
    accelerator.print(f"Model loaded: {model_name}: {model.__class__.__name__}")
    total_params = model_util.print_number_trainable_params(model, accelerator)
    if cast_model_dtype:
        model = model_util.cast_model_to_dtype(model, dtype_or_accelerator=accelerator)

    # 3. Training setup
    if native_trainer:  # Use the Trainer training
        accelerator.print("Using native Trainer for training...")
        training_args = TrainingArguments(
            output_dir=model_filepath,
            overwrite_output_dir=True,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            save_steps=5,
            save_total_limit=2,
            logging_steps=5,
            logging_dir=model_filepath + "/logs",
        )

        data_collator = model_util.get_data_collator(model_name, tokenizer)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
        trainer.train()

    else:
        accelerator.print("Using GPU & pytorch code for training...")
        dataloader = get_data_loader(
            model_name,
            tokenizer,
            train_dataset,
            batch_size=batch_size,
            num_cpus=num_cpus,
            padding_side=padding_side,
        )

        # Prepare everything with Accelerate
        optimizer, lr_scheduler = get_optimizer_and_scheduler(
            model, dataloader, num_epochs, accelerator=accelerator, lr=lr
        )
        model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, dataloader, lr_scheduler
        )
        accelerator.print("Model prepared.")

        # Run training
        model_filepath = model_filepath + f"_{total_params//1_000_000}M"
        run_training(
            model,
            dataloader,
            optimizer,
            lr_scheduler,
            accelerator,
            num_epochs=num_epochs,
            model_filepath=model_filepath,
            wandb_project_name=wandb_project_name,
            train_on_full_loss=train_on_full_loss,
            skip_checkpoints=skip_checkpoints,
        )

    # Unwrap and save the model
    accelerator.print(f"Saving the model to {model_filepath}...")
    save_model(model_filepath, model, tokenizer, accelerator)
    accelerator.print("Done saving the model.")
    accelerator.wait_for_everyone()

    # Compute test loss if test dataset is available
    if isinstance(dataset, DatasetDict) and "test" in dataset:
        test_dataset = dataset["test"].with_format(type="torch")
        if "text" in test_dataset.column_names:
            test_dataset = test_dataset.remove_columns("text")

        accelerator.print("Computing test loss...")
        test_loss = compute_test_loss(
            model,
            test_dataset,
            tokenizer,
            accelerator,
            batch_size=batch_size,
            num_cpus=num_cpus,
            model_name=model_name,
            save_full_loss=True,
            output_filepath=f"{model_filepath}_test_loss.txt",
        )
        accelerator.print(f"Test loss: {test_loss}")

    accelerator.end_training()
    accelerator.print("Saved and Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AoU FM models.")
    parser.add_argument(
        "--use-mock-data",
        action="store_true",
        help="Use dataset/vocab artifacts from the local mock_data directory.",
    )
    parser.add_argument(
        "--enable-wandb",
        action="store_true",
        help="Enable Weights & Biases logging (requires wandb login).",
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

    cli_args = parser.parse_args()
    if cli_args.post:
        config.paths.set_post(cli_args.post)

    os.makedirs(LOCAL_DIR, exist_ok=True)
    vocab_path = os.path.join(LOCAL_DIR, config.paths.vocab_path)
    model_path = os.path.join(LOCAL_DIR, config.paths.model_name_path)

    args = {
        "model_name": "gpt",
        "model_filepath": model_path,
        "batch_size": config.BATCH_SIZE,
        "num_epochs": config.EPOCHS,
        "num_cpus": 16,
        "native_trainer": False,
        "dataset_path": os.path.join(LOCAL_DIR, config.paths.dataset_path),
        "preprocess_data": False,
        "vocab_path": vocab_path,
        "wandb_project_name": config.WANDB_PROJECT,
        "cast_model_dtype": False,  # Should be unnecessary with accelerate
        "lr": config.LEARNING_RATE,
        "padding_side": "left",
    }

    if cli_args.use_mock_data:
        dataset_path = os.fspath(MOCK_DATASET_PATH)
        vocab_path = os.fspath(MOCK_VOCAB_PATH)
        model_path = os.fspath(MOCK_MODEL_PATH)
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(
                f"Mock dataset not found at {dataset_path}. Run aou_data_loader.py --use-mock-data first."
            )
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(
                f"Mock vocab not found at {vocab_path}. Run aou_data_loader.py --use-mock-data first."
            )
        args["dataset_path"] = dataset_path
        args["vocab_path"] = vocab_path
        args["model_filepath"] = model_path
        args["skip_checkpoints"] = True

    log_with = "wandb" if cli_args.enable_wandb else None
    accelerator = Accelerator(log_with=log_with, mixed_precision="fp16")
    accelerator.print(args)

    # remove local model path if exists
    if (os.path.exists(args["model_filepath"])) and ("dummy" in args["model_filepath"]):
        accelerator.print(f"Removing existing model at {args['model_filepath']}")
        shutil.rmtree(args["model_filepath"], ignore_errors=True)

    main(accelerator=accelerator, **args)
