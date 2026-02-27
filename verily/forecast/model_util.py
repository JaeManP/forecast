import json
import os

import numpy as np
import torch
import torch.nn as nn
import yaml
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    BertConfig,
    BertForMaskedLM,
    BertTokenizerFast,
    DataCollatorForLanguageModeling,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    PreTrainedTokenizerFast,
    Qwen3Config,
    Qwen3ForCausalLM,
)

from verily.forecast import config

# Paths for data and vocab
vocab_path = config.paths.vocab_path


def print_number_trainable_params(model, accelerator=None):
    # print number of model parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if accelerator is not None:
        accelerator.print(f"Total trainable parameters: {total_params:,}")
    else:
        print(f"Total trainable parameters: {total_params:,}")
    return total_params


# ---------------- Dtype and Model Casting ----------------
precision_map = {
    "no": None,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def get_precision_dtype(accelerator) -> torch.dtype:
    """
    Get the precision dtype based on the accelerator's mixed precision setting.
    """
    if hasattr(accelerator, "mixed_precision"):
        return precision_map[accelerator.mixed_precision]
    else:
        return torch.float32  # Default to float32 if no mixed precision is set


def cast_batch_to_dtype(batch, dtype):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor) and v.dtype.is_floating_point:
            try:
                batch[k] = v.to(dtype)
            except Exception:
                batch[k] = v.to(precision_map[dtype])
    return batch


def cast_model_to_dtype(model, dtype_or_accelerator):
    # Accept either a dtype string or an accelerator object
    if hasattr(dtype_or_accelerator, "mixed_precision"):
        dtype = dtype_or_accelerator.mixed_precision
        accelerator = dtype_or_accelerator
        accelerator.print(f"Using mixed precision: {dtype}. Casting model...")
    else:
        dtype = dtype_or_accelerator
        accelerator = None

    if (dtype == "fp16") or (dtype == torch.float16):
        model = model.half()
    elif (dtype == "bf16") or (dtype == torch.bfloat16):
        model = model.bfloat16()

    if accelerator is not None:
        print_model_info(accelerator, model)
    return model


def print_model_info(accelerator, model):
    accelerator.print(f"Model device: {model.device}")
    try:
        accelerator.print(f"Model precision: {model.dtype}")
        accelerator.print(f"Model param precision: {next(model.parameters()).dtype}")
    except AttributeError:
        accelerator.print(f"Model param precision: {next(model.parameters()).dtype}")
    return


def print_batch_dtype(accelerator, batch):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            accelerator.print(f"Batch {key} dtype: {value.dtype}")
        else:
            accelerator.print(f"Batch {key} type: {type(value)}")


def create_vocab_file(codes, vocab_path=vocab_path, special_tokens=[]):
    os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
    try:
        os.remove(vocab_path)
        print(f"Deleted existing vocab file: {vocab_path}")
    except FileNotFoundError:
        pass

    print(f"Creating vocab file: {vocab_path}")
    print(f"Vocab size: {len(codes) + len(special_tokens)}")
    with open(vocab_path, "w") as f:
        for line in special_tokens + codes:
            f.write(line + "\n")


# ---------------- Get model/tokenizer ----------------
def get_tokenizer(model_name, vocab_path=vocab_path):
    """
    Get the tokenizer for the specified model.
    """
    if model_name == "gpt2-naive":
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    elif "gpt" in model_name or "qwen" in model_name:
        # Load your vocabulary
        with open(vocab_path, encoding="utf-8") as f:
            vocab_list = [line.strip() for line in f]

        # Create a vocab dictionary: {word: index}
        vocab = {word: idx for idx, word in enumerate(vocab_list)}
        vocab["<unk>"] = len(vocab)  # Add unknown token

        # Initialize the tokenizer
        tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
        tokenizer.pre_tokenizer = Whitespace()

        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            unk_token="<unk>",
            pad_token="<pad>",  # Optional
            bos_token="<s>",  # Optional
            eos_token="</s>",  # Optional
        )

    elif model_name == "bert":
        tokenizer = BertTokenizerFast(
            vocab_file=vocab_path,
            do_lower_case=False,
        )
        tokenizer.backend_tokenizer.pre_tokenizer = Whitespace()
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return tokenizer


def get_model(model_name, tokenizer=None, add_cross_attention=False, **kwargs):
    if model_name == "gpt":
        n_positions = kwargs.get("n_positions", config.MODEL_WINDOW_SIZE)
        config_gpt = GPT2Config(
            vocab_size=len(tokenizer),
            n_positions=n_positions,
            n_embd=config.MODEL_EMBEDDING_SIZE,
            n_layer=config.MODEL_LAYERS,
            n_head=config.MODEL_HEADS,
            pad_token_id=tokenizer.pad_token_id,
            add_cross_attention=add_cross_attention,
        )
        model = AutoModelForCausalLM.from_config(config_gpt)
    elif model_name == "qwen":
        config_qwen = Qwen3Config(
            vocab_size=len(tokenizer),
            hidden_size=config.MODEL_EMBEDDING_SIZE,
            num_hidden_layers=config.MODEL_LAYERS,
            num_attention_heads=config.MODEL_HEADS,
            num_key_value_heads=config.MODEL_HEADS,
            max_position_embeddings=config.MODEL_WINDOW_SIZE,
            intermediate_size=config.MODEL_EMBEDDING_SIZE * 4,  # Standard transformer ratio
            use_sliding_window=config.USE_SLIDING_WINDOW,
            rope_theta=config.ROPE_THETA,
            pad_token_id=(tokenizer.pad_token_id if hasattr(tokenizer, "pad_token_id") else None),
        )
        model = Qwen3ForCausalLM(config_qwen)
    elif model_name == "bert":
        config_bert = BertConfig(
            vocab_size=len(tokenizer),
            hidden_size=config.MODEL_EMBEDDING_SIZE,
            num_hidden_layers=config.MODEL_LAYERS,
            num_attention_heads=config.MODEL_HEADS,
            intermediate_size=config.MODEL_EMBEDDING_SIZE * 4,
            max_position_embeddings=config.MODEL_WINDOW_SIZE,
            pad_token_id=tokenizer.pad_token_id,
        )
        model = BertForMaskedLM(config_bert)
    elif model_name == "prs":
        n_out = kwargs.get("n_out", config.MODEL_EMBEDDING_SIZE)
        model_genomics = ModelGenomics(
            n_input_1=config.MODEL_GENOMICS_INPUT_1,
            n_input_2=config.MODEL_GENOMICS_INPUT_2,
            hidden_dim=config.MODEL_GENOMICS_HIDDEN_DIM,
            n_out=n_out,  # Must be equal to the GPT embedding size
            n_tokens=config.MODEL_GENOMICS_TOKENS,
            activation=config.MODEL_GENOMICS_ACTIVATION,
            **kwargs,
        )
        return model_genomics
    elif model_name == "gpt-prs":
        model_gpt = get_model("gpt", tokenizer)
        model_genomics = get_model("prs")
        model = ModelPRSSoft(model_genomics, model_gpt, truncate_first=False, padding_side="left")
    elif model_name == "qwen-prs":
        model_qwen = get_model("qwen", tokenizer)
        model_genomics = get_model("prs")
        model = ModelPRSSoft(model_genomics, model_qwen, truncate_first=False, padding_side="left")
    elif model_name == "gpt-prs-trunc":
        model_gpt = get_model(
            "gpt",
            tokenizer,
            n_positions=config.MODEL_WINDOW_SIZE + config.MODEL_GENOMICS_TOKENS,
        )
        model_genomics = get_model("prs")  # , zero_mask=True)
        model = ModelPRSSoft(model_genomics, model_gpt, truncate_first=True, padding_side="left")
    elif model_name == "gpt-prs-cross":
        model_gpt = get_model("gpt", tokenizer, add_cross_attention=True)
        model_genomics = get_model("prs")
        model = ModelPRSCross(model_genomics, model_gpt, zero_cross_attention=True)
    elif model_name.startswith("gpt-prs-pretrained-gpt/"):
        pretrained_model_name = model_name[len("gpt-prs-pretrained-gpt/") :]
        model_gpt = get_trained_model(pretrained_model_name, "gpt")
        model_genomics = get_model("prs", n_out=model_gpt.config.hidden_size)
        model = ModelPRSSoft(model_genomics, model_gpt)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model


def get_trained_model(
    model_path, model_name, force_gpt2_config=False, nemo_config_path=None, **kwargs
):
    """
    Load a trained model from the specified path.
    """
    if model_name == "gpt":
        if force_gpt2_config:
            config_path = os.path.join(model_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path) as f:
                    config_data = json.load(f)
            elif nemo_config_path and os.path.exists(nemo_config_path):
                with open(nemo_config_path) as f:
                    nemo_cfg = yaml.safe_load(f)
                model_cfg = nemo_cfg.get("model", {})
                config_data = {
                    "vocab_size": model_cfg.get("vocab_size", 50258),
                    "n_positions": model_cfg.get("n_positions", 2048),
                    "n_embd": model_cfg.get("n_embd", 768),
                    "n_layer": model_cfg.get("n_layer", 12),
                    "n_head": model_cfg.get("n_head", 12),
                    "bos_token_id": model_cfg.get("bos_token_id", 50256),
                    "eos_token_id": model_cfg.get("eos_token_id", 50256),
                }
            else:
                raise FileNotFoundError(
                    "Missing config.json for checkpoint. Provide --nemo-config "
                    "or save a HuggingFace config.json alongside the model."
                )
            config_data.setdefault("model_type", "gpt2")
            config_data.setdefault("architectures", ["GPT2LMHeadModel"])
            config = GPT2Config(**config_data)
            model = GPT2LMHeadModel.from_pretrained(
                model_path, config=config, **kwargs
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    elif model_name == "qwen":
        model = Qwen3ForCausalLM.from_pretrained(model_path, **kwargs)
    elif model_name == "bert":
        model = BertForMaskedLM.from_pretrained(model_path, **kwargs)
    elif model_name == "prs":
        model = ModelGenomics.from_pretrained(model_path, **kwargs)
    elif model_name in ("gpt-prs", "gpt-prs-trunc", "qwen-prs"):
        model = ModelPRSSoft.from_pretrained(model_path, model_name, **kwargs)
    elif model_name == "gpt-prs-cross":
        model = ModelPRSCross.from_pretrained(model_path, **kwargs)
    elif model_name == "gpt-prs-prefix":
        raise NotImplementedError("gpt-prs-prefix model is not implemented")
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model


# ---------------- Genomics models ----------------
class ModelGenomics(nn.Module):
    def __init__(
        self,
        n_input_1,
        n_input_2,
        hidden_dim,
        n_out,
        n_tokens,
        activation="relu",
        zero_mask=False,
        clip_value=config.MODEL_GENOMICS_CLIP,
    ):
        super().__init__()
        self.n_input_1 = n_input_1
        self.n_input_2 = n_input_2
        self.input_dim = n_input_1 * n_input_2
        self.n_out = n_out
        self.n_tokens = n_tokens
        self.activation_str = activation
        self.hidden_dim = hidden_dim
        self.zero_mask = zero_mask
        self.clip_value = clip_value

        if activation == "gelu":
            act_layer = nn.GELU()
        else:
            act_layer = nn.ReLU()

        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            act_layer,
            nn.Linear(hidden_dim, n_out * n_tokens),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Clip values from outliers
        clip_value = x.new_tensor(self.clip_value)
        x = x.clip(clip_value * x.new_tensor(-1), clip_value) / clip_value

        # Zero mask for later
        zero_mask = x.view(x.size(0), -1).any(dim=1)

        # Forward pass
        batch_size = x.size(0)  # (batch_size, n_input_1, n_input_2)
        x = x.view(batch_size, -1)  # (batch_size, n_input_1 * n_input_2)
        x = self.mlp(x)  # (batch_size, n_tokens * n_out)
        x = (x - x.new_tensor(0.5)) * x.new_tensor(2)
        x = x.view(batch_size, self.n_tokens, self.n_out)  # (batch_size, n_tokens, n_out)

        if self.zero_mask:
            # Zero out x[i] if x[i] all zero
            x[~zero_mask] = 0
        return x

    @property
    def device(self):
        return next(self.parameters()).device

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)

        # Save weights
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))

        # Save config
        config = {
            "n_input_1": self.n_input_1,
            "n_input_2": self.n_input_2,
            "hidden_dim": self.hidden_dim,
            "n_out": self.n_out,
            "n_tokens": self.n_tokens,
            "activation": self.activation_str,
            "zero_mask": self.zero_mask,
            "clip_value": self.clip_value,
        }
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config, f)

    @classmethod
    def from_pretrained(cls, load_directory, map_location=None, torch_dtype=None, **kwargs):
        # Load config
        with open(os.path.join(load_directory, "config.json")) as f:
            config = json.load(f)

        # Instantiate model
        model = cls(**config)

        # Load weights
        state_dict = torch.load(
            os.path.join(load_directory, "pytorch_model.bin"), map_location=map_location
        )
        model.load_state_dict(state_dict)

        # Cast model to dtype if specified
        if torch_dtype is not None:
            model = cast_model_to_dtype(model, torch_dtype)

        return model


class ModelPRSSoft(nn.Module):
    def __init__(
        self,
        model_genomics,
        model_gpt: GPT2LMHeadModel | Qwen3ForCausalLM,
        padding_side="left",
        truncate_first=False,
        **kwargs,
    ):
        super().__init__()
        self.model_genomics = model_genomics
        self.model_gpt = model_gpt
        self.n_positions = (
            model_gpt.config.n_positions
            if isinstance(model_gpt, GPT2LMHeadModel)
            else model_gpt.config.max_position_embeddings
        )
        self.n_soft_tokens = model_genomics.n_tokens
        self.zero_cross_attention = None
        self.padding_side = padding_side
        self.truncate_first = truncate_first

        # Config setup for Deepspeed2 compatibility
        self.hidden_size = model_gpt.config.hidden_size
        self.config = type("Cfg", (), {"hidden_size": self.hidden_size})()

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def joint_config(self):
        joint_config = {
            "n_positions": self.n_positions,
            "n_soft_tokens": self.n_soft_tokens,
            "zero_cross_attention": self.zero_cross_attention,
            "padding_side": self.padding_side,
            "truncate_first": self.truncate_first,
        }
        return joint_config

    def forward(self, **kwargs):
        if self.padding_side == "right":
            inputs_embeds, attention_mask, labels = self.prep_batch_for_gpt_right(**kwargs)
        elif self.padding_side == "left":
            inputs_embeds, attention_mask, labels = self.prep_batch_for_gpt_left(**kwargs)

        outputs = self.model_gpt(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )
        return outputs

    def prep_batch_for_gpt_right(self, with_labels=True, **kwargs):
        prs = kwargs["prs"]
        input_ids = kwargs["input_ids"]
        attention_mask = kwargs["attention_mask"]
        if with_labels:
            labels = kwargs["labels"]

        # Calculate genomics embeddings and concatenate with GPT input embeddings
        genomics_embed = self.model_genomics(prs)
        if hasattr(self.model_gpt, "transformer"):
            # GPT model
            real_inputs_embeds = self.model_gpt.transformer.wte(input_ids)
        else:
            # Qwen model
            real_inputs_embeds = self.model_gpt.model.embed_tokens(input_ids)
        inputs_embeds = torch.cat([genomics_embed, real_inputs_embeds], dim=1)
        attention_mask = self.extend_attention_mask(
            attention_mask, genomics_embed, pad_side="right"
        )

        # Label genomics with -100 to ignore them in loss computation
        if with_labels:
            genomics_labels = torch.ones_like(genomics_embed[:, :, 0]) * genomics_embed.new_tensor(
                -100
            )
            labels = torch.cat([genomics_labels, labels], dim=1).long()

        # Ensure inputs_embeds, attention_mask, and labels are within model's n_positions
        inputs_embeds = inputs_embeds[:, : self.n_positions, :]
        attention_mask = attention_mask[:, : self.n_positions]
        if with_labels:
            labels = labels[:, : self.n_positions]
            return inputs_embeds, attention_mask, labels
        else:
            return inputs_embeds, attention_mask, None

    def prep_batch_for_gpt_left(self, with_labels=True, **kwargs):
        attention_mask = kwargs.get("attention_mask")
        input_ids = kwargs.get("input_ids")
        prs = kwargs.get("prs")
        labels = kwargs["labels"] if with_labels else None

        # Initial embeddings
        genomics_embed = self.model_genomics(prs)  # B * T * D
        if hasattr(self.model_gpt, "transformer"):
            # GPT model
            inputs_embeds = self.model_gpt.transformer.wte(input_ids)  # B * L * D
        else:
            # Qwen model
            inputs_embeds = self.model_gpt.model.embed_tokens(input_ids)  # B * L * D
        text_lengths = attention_mask.sum(dim=1)  # B

        if not self.truncate_first:
            # Append soft tokens right before the text
            # extend left input_embeds and attention mask
            attention_mask = torch.cat(
                [torch.zeros_like(genomics_embed[:, :, 0]), attention_mask], dim=1
            )
            inputs_embeds = torch.cat([torch.zeros_like(genomics_embed), inputs_embeds], dim=1)
            if with_labels:
                labels = torch.cat(
                    [torch.ones_like(genomics_embed[:, :, 0]) * -100, labels], dim=1
                ).long()

            # Insert genomic embeddings right before the attention mask
            for i, t_len in enumerate(text_lengths):
                start = -t_len - self.n_soft_tokens
                end = -t_len
                inputs_embeds[i, start:end, :] = genomics_embed[i, :, :]
                attention_mask[i, start:end] = 1

            # Truncate the extended inputs_embeds and attention_mask on the left
            inputs_embeds = inputs_embeds[:, -self.n_positions :, :]
            attention_mask = attention_mask[:, -self.n_positions :]
            if with_labels:
                labels = labels[:, -self.n_positions :]

        else:
            # Truncate first
            # Soft tokens appended way before (with empty tokens following)
            n_preprocess = self.n_positions - self.n_soft_tokens
            inputs_embeds = inputs_embeds[:, -n_preprocess:, :]  # B * L' * D
            attention_mask = attention_mask[:, -n_preprocess:]  # B * L'
            if with_labels:
                labels = labels[:, -n_preprocess:]

            # Append the prs data
            attention_mask = torch.cat(
                [
                    torch.ones_like(genomics_embed[:, :, 0])
                    * torch.any(prs != 0, dim=[1, 2]).long().unsqueeze(-1),
                    attention_mask,
                ],
                dim=1,
            )  # B * (T + L')
            inputs_embeds = torch.cat([genomics_embed, inputs_embeds], dim=1)  # B * (T + L') * D
            if with_labels:
                labels = torch.cat(
                    [torch.ones_like(genomics_embed[:, :, 0]) * -100, labels], dim=1
                ).long()  # B * (T + L')

        assert inputs_embeds.shape[1] <= self.n_positions
        assert inputs_embeds.shape[:-1] == attention_mask.shape
        if with_labels:
            assert labels.shape == attention_mask.shape
        return inputs_embeds, attention_mask, labels

    def extend_attention_mask(self, attention_mask, genomics_embed, pad_side="right"):
        # Create attention mask
        if pad_side == "right":
            # Add ones to left
            attention_mask = torch.cat(
                [torch.ones_like(genomics_embed[:, :, 0]), attention_mask], dim=1
            )
        elif pad_side == "left":
            # Add ones to right
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(genomics_embed[:, :, 0])], dim=1
            )
        else:
            raise ValueError("Invalid padding side")
        return attention_mask

    def generate(self, **kwargs):
        inputs_embeds, attention_mask, _ = self.prep_batch_for_gpt_left(with_labels=False, **kwargs)

        # Removed used keys from kwargs:
        for key in ["labels", "prs", "input_ids", "attention_mask"]:
            kwargs.pop(key, None)

        # Generate sequences
        outputs = self.model_gpt.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs,
        )
        return outputs

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, **kwargs
    ):
        """
        This is called internally by `generate`. We only feed text tokens here —
        soft tokens are only used at the *first step*.
        """
        if past_key_values is not None:
            # only the last token for fast autoregressive generation
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
        }

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)

        # Save submodels
        self.model_genomics.save_pretrained(os.path.join(save_directory, "genomics"))
        self.model_gpt.save_pretrained(os.path.join(save_directory, "gpt"))

        # Save joint model config (just metadata here)
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(self.joint_config, f)

    @classmethod
    def from_pretrained(cls, load_directory, model_name="gpt", **kwargs):
        # Load genomics model
        model_genomics = ModelGenomics.from_pretrained(
            os.path.join(load_directory, "genomics"), **kwargs
        )

        # Load gpt or qwen model
        model_for_causal_lm = AutoModelForCausalLM if "gpt" in model_name else Qwen3ForCausalLM
        model_gpt = model_for_causal_lm.from_pretrained(
            os.path.join(load_directory, "gpt"),
            **kwargs,
        )

        # Optionally load joint metadata config
        config_path = os.path.join(load_directory, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config_dict = json.load(f)

        return cls(model_genomics, model_gpt, **config_dict)


class ModelPRSCross(ModelPRSSoft):
    def __init__(self, model_genomics, model_gpt, zero_cross_attention=False, **kwargs):
        super().__init__(model_genomics, model_gpt)
        self.zero_cross_attention = zero_cross_attention

    def get_genomics_embed(self, prs):
        # Get genomics embeddings
        genomics_embed = self.model_genomics(prs)
        attention_mask_cross = torch.ones(
            genomics_embed.size(0), genomics_embed.size(1), device=genomics_embed.device
        )

        # Set attention mask to zero for rows with all-zero prs
        if self.zero_cross_attention:
            batch_zero_mask = (prs.view(prs.size(0), -1).any(dim=1)).long()
            for i, mask in enumerate(batch_zero_mask):
                if mask == 0:
                    attention_mask_cross[i, :] = 0
        return genomics_embed, attention_mask_cross

    def forward(self, **kwargs):
        genomics_embed, attention_mask_cross = self.get_genomics_embed(kwargs.get("prs"))

        # Forward through GPT with cross attention
        outputs = self.model_gpt(
            input_ids=kwargs.get("input_ids"),
            attention_mask=kwargs.get("attention_mask"),
            labels=kwargs.get("labels"),
            encoder_hidden_states=genomics_embed,
            encoder_attention_mask=attention_mask_cross,
            use_cache=False,  # important for training/fine-tuning
        )
        return outputs

    def generate(self, prs=None, input_ids=None, attention_mask=None, **kwargs):
        # prs, input_ids, and attention_mask are expected to be provided
        if prs is None or input_ids is None or attention_mask is None:
            raise ValueError("Must provide prs, input_ids, and attention_mask")

        genomics_embed, attention_mask_cross = self.get_genomics_embed(prs)
        # Remove already used keys from kwargs
        for key in ["labels"]:
            kwargs.pop(key, None)

        # Generate sequences with cross attention
        outputs = self.model_gpt.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=genomics_embed,
            encoder_attention_mask=attention_mask_cross,
            **kwargs,
        )
        return outputs

    @classmethod
    def from_pretrained(cls, load_directory, **kwargs):
        model_gpt = AutoModelForCausalLM.from_pretrained(
            os.path.join(load_directory, "gpt"),
            # add_cross_attention=True, # redundant if model was saved with it
            **kwargs,
        )
        model_genomics = ModelGenomics.from_pretrained(
            os.path.join(load_directory, "genomics"), **kwargs
        )

        config_path = os.path.join(load_directory, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config_dict = json.load(f)
        else:
            config_dict = {}

        return cls(
            model_genomics,
            model_gpt,
            **config_dict,
        )


# ---------------- Generation of sequences using Decoder models ----------------
def generate_sequences(model, batch, max_new_tokens=50, n_sequences=1, compute_loss=False):
    """Generate sequences using the model.
    Args:
        model: The model to use for generation.
        batch: A dictionary containing the input data.
        max_new_tokens: The maximum number of new tokens to generate.
        n_sequences: The number of sequences to return.
        compute_loss: Whether to compute the loss for the batch.
    Returns:
        outputs: The generated sequences as a GenerateOutput object (sequences, scores)
        input_ids: The input IDs used for generation.
        attention_mask: The attention mask used for generation.
        subject_id: The subject IDs from the batch.
        label_token: The label tokens from the batch.
        loss: The computed loss (if compute_loss=True), otherwise None."""

    def list_tensors_to_list(tensors):
        return [tensor.item() for tensor in tensors]

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    subject_id = list_tensors_to_list(batch["subject_id"])
    label_token = list_tensors_to_list(batch["label_token"])
    input_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

    # if model is ModelPRSSoft or ModelPRSCross, add prs
    if hasattr(model, "model_genomics") and ("prs" in batch):
        prs = batch["prs"]
        input_kwargs["prs"] = prs
    elif hasattr(model, "model_genomics") and ("prs" not in batch):
        raise ValueError("Batch must contain 'prs' for ModelPRSSoft or ModelPRSCross.")

    # Generation (adjust parameters as needed)
    def get_outputs(n):
        with torch.no_grad():
            outputs = model.generate(
                **input_kwargs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=1.0,
                num_return_sequences=n,
                output_scores=True,  # <-- collect logits
                return_dict_in_generate=True,  # <-- returns a GenerateOutput
            )
        return outputs

    outputs = None
    n = n_sequences
    while (outputs is None) and n > 1:
        try:
            outputs = get_outputs(n)
        except torch.OutOfMemoryError:
            n = n // 2
    if (n == 1) and (outputs is None):
        try:
            outputs = get_outputs(n)
        except Exception:
            pass

    # Compute loss if requested
    loss = None
    if compute_loss and "labels" in batch:
        with torch.no_grad():
            # Create a copy of input_kwargs for loss computation
            loss_kwargs = input_kwargs.copy()
            loss_kwargs["labels"] = batch["labels"]
            loss_outputs = model(**loss_kwargs)
            loss = loss_outputs.loss.item() if hasattr(loss_outputs, "loss") else 47

    return outputs, input_ids, attention_mask, subject_id, label_token, loss


def load_dataloader(
    dataset,
    tokenizer,
    kept_columns=["subject_id"],
    num_workers=config.DATALOADER_NUM_CPUS,
    shuffle=False,
    dataloader_batch_size=1,
    model_name="gpt",
    padding_side="left",
):
    # Convert to right input type
    base_cols = ["input_ids", "attention_mask"]
    if "prs" in dataset.column_names:
        base_cols.append("prs")
    dataset = dataset.with_format(type="np", columns=kept_columns + base_cols)

    # Configure padding side
    tokenizer.padding_side = padding_side
    tokenizer.truncation_side = "left" if padding_side == "right" else "right"

    # Get flexible collator
    custom_collate = get_data_collator(
        model_name=model_name,
        tokenizer=tokenizer,
        base_cols=base_cols,
        kept_columns=kept_columns,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=dataloader_batch_size,
        collate_fn=custom_collate,
        num_workers=num_workers,
        shuffle=shuffle,
    )

    return dataloader


def get_data_collator(
    model_name,
    tokenizer,
    base_cols=["input_ids", "attention_mask"],
    kept_columns=["subject_id"],
):
    if model_name == "gpt":
        # GPT does not use MLM, so we use a simple collator
        return DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    if model_name == "bert":
        # BERT uses MLM, so we use the MLM collator
        return DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    elif model_name in ("gpt", "gpt-prs", "gpt-prs-cross", "qwen", "qwen-prs") or (
        "gpt-prs" in model_name
    ):
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        def custom_collate(batch):
            tokenized_inputs = [{k: v for k, v in item.items() if k in base_cols} for item in batch]
            held_columns = {
                k: np.array([item[k] for item in batch]) for k in kept_columns if k in batch[0]
            }

            # Apply the data collator to tokenized inputs
            model_inputs = data_collator(tokenized_inputs)

            # Add the subject_id back into the batch
            for k, v in held_columns.items():
                model_inputs[k] = v
            return model_inputs

        return custom_collate
    else:
        raise ValueError(f"Unsupported model for data collator: {model_name}")
