"""
AoU Dataset loader for NeMo Automodel pretraining.

This module provides a Dataset class that loads HuggingFace datasets
produced by aou_data_loader.py and makes them compatible with
NeMo Automodel's training pipeline.
"""

from __future__ import annotations

import torch
from datasets import load_from_disk
from torch.utils.data import Dataset


class AoUDataset(Dataset):
    """
    Dataset that loads HuggingFace datasets saved by aou_data_loader.py.

    This dataset wraps a HuggingFace Dataset saved with `save_to_disk()` and
    returns samples compatible with NeMo Automodel's causal LM training loop.

    Returns input_ids as lists (not tensors) for compatibility with NeMo's
    default_collater which handles padding and batching.

    Args:
        dataset_path: Path to the HuggingFace dataset directory (saved via save_to_disk).
        split: Which split to use ('train' or 'test'). Default: 'train'.
        seq_len: Maximum sequence length. Sequences are truncated to this length.
        shuffle: Whether to shuffle the dataset on load.
    """

    def __init__(
        self,
        dataset_path: str,
        split: str = "train",
        seq_len: int = 1024,
        shuffle: bool = False,
    ):
        self.dataset_path = dataset_path
        self.split = split
        self.seq_len = seq_len

        # Load the HuggingFace dataset from disk
        dataset = load_from_disk(dataset_path)

        # Handle DatasetDict vs Dataset
        if hasattr(dataset, "keys"):
            # It's a DatasetDict with train/test splits
            if split not in dataset:
                available = list(dataset.keys())
                raise ValueError(
                    f"Split '{split}' not found. Available splits: {available}"
                )
            self.dataset = dataset[split]
        else:
            # It's a single Dataset
            self.dataset = dataset

        # Reset format to ensure we get Python lists, not tensors/numpy arrays
        self.dataset = self.dataset.with_format(None)

        if shuffle:
            self.dataset = self.dataset.shuffle()

        print(f"Loaded AoUDataset from {dataset_path}")
        print(f"  Split: {split}, Size: {len(self.dataset)}, seq_len: {seq_len}")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single sample.

        Returns:
            dict with keys:
                - input_ids: list of ints
            Note: DataCollatorForLanguageModeling creates labels automatically.
        """
        sample = self.dataset[idx]

        # Get input_ids and ensure it's a plain Python list
        input_ids = sample["input_ids"]

        # Convert to list regardless of input type
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()
        elif hasattr(input_ids, "tolist"):  # numpy array
            input_ids = input_ids.tolist()

        # Ensure it's a list (handle any remaining edge cases)
        if not isinstance(input_ids, list):
            input_ids = list(input_ids)

        # Ensure all elements are Python ints (not numpy.int64, etc.)
        input_ids = [int(x) for x in input_ids]

        # Truncate to seq_len (collater will handle padding)
        if len(input_ids) > self.seq_len:
            input_ids = input_ids[: self.seq_len]

        # Return only input_ids - DataCollatorForLanguageModeling creates labels
        return {"input_ids": input_ids}


def build_aou_dataset(
    dataset_path: str,
    split: str = "train",
    seq_len: int = 1024,
    shuffle: bool = False,
    **kwargs,
) -> AoUDataset:
    """
    Factory function to build an AoUDataset.

    This function can be used as a _target_ in NeMo Automodel YAML configs.

    Args:
        dataset_path: Path to the HuggingFace dataset directory.
        split: Which split to use ('train' or 'test').
        seq_len: Maximum sequence length.
        shuffle: Whether to shuffle the dataset.
        **kwargs: Additional arguments (ignored for compatibility).

    Returns:
        AoUDataset instance.
    """
    return AoUDataset(
        dataset_path=dataset_path,
        split=split,
        seq_len=seq_len,
        shuffle=shuffle,
    )


def build_hf_gpt2_model(
    vocab_size: int = 50257,
    n_positions: int = 2048,
    n_embd: int = 768,
    n_layer: int = 12,
    n_head: int = 12,
    bos_token_id: int = 50256,
    eos_token_id: int = 50256,
    **kwargs,
):
    """
    Build a HuggingFace GPT2LMHeadModel for use with NeMo Automodel.

    This wrapper ignores extra NeMo-specific kwargs (tp_size, cp_size, etc.)
    that the HuggingFace model doesn't accept.

    The HuggingFace GPT2LMHeadModel computes loss internally when labels are
    provided, with proper label shifting for causal LM training.

    Args:
        vocab_size: Size of the vocabulary.
        n_positions: Maximum sequence length (context window).
        n_embd: Embedding dimension / hidden size.
        n_layer: Number of transformer layers.
        n_head: Number of attention heads.
        bos_token_id: Beginning of sequence token ID.
        eos_token_id: End of sequence token ID.
        **kwargs: Extra arguments (ignored for NeMo compatibility).

    Returns:
        GPT2LMHeadModel instance.
    """
    from transformers import GPT2Config, GPT2LMHeadModel

    if kwargs:
        ignored = ", ".join(kwargs.keys())
        print(f"[build_hf_gpt2_model] Ignoring extra kwargs: {ignored}")

    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=n_positions,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
    )
    return GPT2LMHeadModel(config)


def aou_collate_fn(batch: list[dict], pad_token_id: int = 0) -> dict:
    """
    Collate function for causal LM training with an external loss function.

    Labels are pre-shifted: labels[i] = input_ids[i+1]. This is required because
    NeMo's MaskedCrossEntropy computes CE(logits[i], labels[i]) directly without
    shifting. Padding positions use -100 for labels (ignored in loss computation).

    Args:
        batch: List of dicts with 'input_ids' key.
        pad_token_id: Token ID to use for padding input_ids.

    Returns:
        Dict with 'input_ids' and 'labels' tensors.
    """
    # Find max length in batch
    max_len = max(len(sample["input_ids"]) for sample in batch)

    input_ids_batch = []
    labels_batch = []

    for sample in batch:
        input_ids = sample["input_ids"]
        seq_len = len(input_ids)
        pad_len = max_len - seq_len

        # Pad input_ids with pad_token_id
        input_ids_padded = input_ids + [pad_token_id] * pad_len

        # Pre-shift labels for next-token prediction: labels[i] = input_ids[i+1]
        # The last real position gets -100 (no target beyond the sequence)
        # Pad with -100 (ignored in loss)
        labels_padded = input_ids[1:] + [-100] + [-100] * pad_len

        input_ids_batch.append(input_ids_padded)
        labels_batch.append(labels_padded)

    return {
        "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
        "labels": torch.tensor(labels_batch, dtype=torch.long),
    }

