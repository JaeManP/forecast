# Verily Forecast: EHR Foundation Model on _All of Us_

This repository provides the training and evaluation framework for an Electronic Health Record (EHR)
Foundation Model. Developed within Verily Workbench using the _All of Us_ Research Program dataset,
this model transforms complex longitudinal medical histories into actionable insights for:

- **Disease Forecasting**: Predicting future diagnoses based on historical clinical markers.
- **Risk Stratification**: Identifying high-risk patient cohorts for clinical intervention.

For full details on the approach and results, see our paper:

> **Integrating Genomics into Multimodal EHR Foundation Models**
> [arXiv:2510.23639](https://arxiv.org/abs/2510.23639)

## Overview

The pipeline covers the full workflow from raw EHR data to evaluation:

1. **Data export** -- Extract structured clinical data from _All of Us_ BigQuery tables
2. **Tokenization** -- Transform records into token sequences suitable for autoregressive modeling
3. **Pre-training** -- Train a GPT-style foundation model on the tokenized sequences (supports
   single- and multi-GPU setups via HuggingFace Accelerate or NeMo)
4. **Task evaluation** -- Generate labeled evaluation datasets for downstream clinical prediction
   tasks (e.g., Type 2 Diabetes onset)
5. **Inference & scoring** -- Run predictions and compute evaluation metrics

A mock dataset is included so you can verify the pipeline end-to-end without access to _All of Us_
data.

## Getting Started

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (Python package manager)
- NVIDIA Tesla V100 or higher

### Installation

1. Install `uv` and create a virtual environment:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   uv venv
   ```

   If running inside the AoU Researcher Workbench, you may need to clear pre-installed environments
   from your path:

   ```bash
   export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v -E 'workbench|conda' | tr '\n' ':' | sed 's/:$//')
   export PYTHONPATH=$(echo "${PYTHONPATH:-}" | tr ':' '\n' | grep -v 'workbench' | tr '\n' ':' | sed 's/:$//')
   ```

2. Install dependencies:

   ```bash
   uv sync
   ```

### Access the Data

To run this model on the _All of Us_ dataset, you must be a registered researcher with the _All of
Us_ program.

1. **Register**: Sign up at the
   [_All of Us_ Research Hub](https://www.researchallofus.org/register/).
2. **Environment**: Once granted access, we recommend executing the model via the new Researcher
   Workbench on [Verily Workbench](https://workbench.verily.com/).

### No Access? Use Mock Data

If you do not yet have data access, we’ve included a mock dataset within the /verily/forecast/mock_data
directory. This allows you to test the pipeline architecture and training scripts immediately.

```bash
# Tokenize the mock dataset
uv run verily/forecast/aou_data_loader.py --use-mock-data

# Train a small model
uv run verily/forecast/trainer.py --use-mock-data

# Run inference
uv run verily/forecast/inference.py -m <path-to-saved-model> -mn gpt -d <path-to-eval-dataset> -t T2D
```

### Using _All of Us_ Data

1. **Export data** from the _All of Us_ BigQuery tables. Use `-n` to export a smaller sample for
   faster iteration:

   ```bash
   uv run verily/forecast/export_data.py -m export -n 10000
   ```

   This step requires access to the AoU CDR BigQuery dataset. Set the `WORKSPACE_CDR` environment
   variable to point to your CDR, e.g. for the Registered Tier dataset:

   ```bash
   export WORKSPACE_CDR="wb-affable-acorn-7941.R2024Q3R8"
   ```

2. **Tokenize** the exported data into model-ready sequences. If you sampled with `-n` in the
   previous step, add `--skip-filtering`:

   ```bash
   uv run verily/forecast/aou_data_loader.py
   ```

3. **Train** the model. On a multi-GPU machine, use `accelerate`:

   ```bash
   # Single GPU
   uv run verily/forecast/trainer.py

   # Multi-GPU
   uv run accelerate launch verily/forecast/trainer.py

   # With Weights & Biases logging (see "Weights & Biases" section below)
   uv run verily/forecast/trainer.py --enable-wandb
   ```

   Alternatively, train with NeMo by pointing the YAML config at your dataset:

   ```bash
   source .venv/bin/activate
   cd verily/forecast/nemo && python pretrain.py --config aou_gpt_pretrain.yaml

   # Multi-GPU with NeMo
   torchrun --nproc-per-node 8 pretrain.py --config aou_gpt_pretrain.yaml
   ```

4. **Generate evaluation data** for a downstream task (e.g., Type 2 Diabetes):

   ```bash
   uv run verily/forecast/analysis.py --task T2D
   ```

5. **Run inference** on the evaluation set:

   ```bash
   uv run verily/forecast/inference.py -m <path-to-saved-model> -mn gpt -d <path-to-eval-dataset> -t T2D
   ```

   For models trained with NeMo, add the `-fg` and `-nc` flags:

   ```bash
   uv run verily/forecast/inference.py -m <path-to-saved-model> -mn gpt -d <path-to-eval-dataset> -t T2D -fg -nc <path-to-nemo-yaml-config>
   ```

6. **(Optional) Evaluate** a batch of labeled predictions:

   ```bash
   uv run verily/forecast/eval.py --inference-path <path-to-inference>
   ```

See [`single_subject_inference.ipynb`](verily/forecast/single_subject_inference.ipynb) for an end-to-end
walkthrough of running the model for a single patient.

## Weights & Biases

[Weights & Biases](https://wandb.ai/) (W&B) integration is available for experiment tracking during
training and inference. It is **disabled by default** and can be enabled with the `--enable-wandb`
flag.

Do not use `--enable-wandb` with All of Us participant-level data. When `WORKSPACE_CDR` is set,
Forecast fails closed if W&B logging is requested so participant-level runs stay inside the
Researcher Workbench without external telemetry.

### Setup

1. **Create an account** at [wandb.ai](https://wandb.ai/) (free for personal and academic use).

2. **Log in** from the command line:

   ```bash
   uv run wandb login
   ```

   This will prompt you for an API key, which you can find at
   [wandb.ai/authorize](https://wandb.ai/authorize). The key is saved to `~/.netrc` so you only need
   to do this once per machine.

3. **Enable logging** by passing `--enable-wandb` to the training or inference script:

   ```bash
   # Training with W&B
   uv run verily/forecast/trainer.py --enable-wandb

   # Inference with W&B
   uv run verily/forecast/inference.py --enable-wandb -m <model-path> -mn gpt -d <dataset-path> -t T2D
   ```

Runs will appear under the `forecast` project (training) or `eval_inference` project (inference) in
your W&B dashboard. Logged metrics include training loss, epoch loss, and inference configuration.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{amar2025integratinggenomicsmultimodalehr,
      title={Integrating Genomics into Multimodal EHR Foundation Models},
      author={Jonathan Amar and Edward Liu and Alessandra Breschi and Liangliang Zhang and Pouya Kheradpour and Sylvia Li and Lisa Soleymani Lehmann and Alessandro Giulianelli and Matt Edwards and Yugang Jia and David Nola and Raghav Mani and Pankaj Vats and Jesse Tetreault and T. J. Chen and Cory Y. McLean},
      year={2025},
      eprint={2510.23639},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.23639},
}
```

## License

This project is provided for research purposes. See [LICENSE](LICENSE) for details.
