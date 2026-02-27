import argparse
import os
import random
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from datasets import Dataset, DatasetDict
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from verily.forecast import config, model_util
from verily.forecast.aou_data_loader import TimeMapper
from verily.forecast.constants import LOCAL_DIR

MODULE_DIR = Path(__file__).resolve().parent
MOCK_DATA_DIR = MODULE_DIR / "mock_data"
MOCK_DATASET_PATH = MOCK_DATA_DIR / config.paths.dataset_path / "test"

# CONDITION_CONCEPT_ID
_TASK_CODES = {
    "AD": [378419, 37117145, 43530664, 4218017, 44784643, 4043379, 4278830, 4220313, 44782432, 4277444],
    "T2D": [376065, 45757363, 45757499, 37018912, 4304377, 43530656, 45769906, 4230254, 443732, 443734, 201826, 609104, 45769875, 45770831, 4228443, 43531577, 609105, 45770880, 45771064, 45770883, 43531578, 4321756, 43531010, 4196141, 4226121, 443731, 609103, 37016354, 43531564, 609117, 609106, 4177050, 37016768, 43530690, 40485020, 443733, 43530685, 45757277, 45770881, 45757449, 43531653, 45769888, 4130162, 43530689, 609112, 4193704, 36714116, 45757508, 37312202, 37016349, 609101, 443729, 43531563, 35626070, 609116, 45769905, 4063043, 37018728, 609099, 4221495, 45773064, 36712687, 4099216, 43531597, 4130164, 37312205, 45757278, 4221487, 602345, 43531651, 43531566, 45757280, 4099651, 36712686, 45757446, 37017432, 45770830, 201530, 46274058, 43531608, 45757445, 45757435, 43531559, 43531588, 43531616, 45757474, 4140466, 4129519, 4215719, 4222415, 4200875, 43531562, 609119, 609109, 4222876],
    "DEPR":	[377527, 379784, 432285, 432883, 433751, 433991, 434911, 435220, 435520, 438406, 438727, 438998, 439259, 440383, 440698, 441534, 443864, 607540, 607543, 762504, 3656234, 4025677, 4031328, 4049623, 4077577, 4094358, 4098302, 4103126, 4103574, 4114950, 4131545, 4141292, 4141454, 4144233, 4148630, 4149320, 4149321, 4151170, 4152280, 4154309, 4154391, 4154805, 4161569, 4174987, 4176002, 4191716, 4195572, 4205471, 4223090, 4224940, 4226155, 4228802, 4242733, 4250023, 4263748, 4269493, 4282096, 4282316, 4298317, 4304140, 4307111, 4314692, 4323418, 4327337, 4328217, 4336957, 4338031, 35615152, 35615153, 35615154, 35615155, 36713698, 36714389, 36714998, 36715000, 36717092, 37016718, 37018656, 37110429, 37111697, 40481798, 42538590, 42872411, 42872722, 43021839, 43531624, 44782943, 45757196],
    "GLAUC": [376688, 432311, 432312, 432626, 432908, 433473, 433767, 433768, 434030, 434928, 435262, 435543, 435809, 436108, 436110, 436398, 436687, 436972, 436975, 437269, 437273, 437276, 437541, 437553, 438151, 438155, 440396, 441005, 441284, 441556, 441561, 604779, 760888, 760891, 761148, 761205, 761286, 761327, 761575, 761579, 761580, 761609, 761610, 761621, 765051, 765264, 765904, 3662317, 4035651, 4041191, 4065195, 4072218, 4078543, 4102183, 4109420, 4152558, 4191001, 4194237, 4195502, 4213414, 4231284, 4244668, 4246656, 4316071, 4318691, 4334259, 4336013, 36684635, 36684647, 36684668, 36684699, 36684712, 36684734, 36684765, 36684774, 36684794, 36713123, 36713124, 36713152, 36713858, 36716344, 37018559, 37108932, 37204003, 37204004, 37207851, 37207911, 37207972, 37208211, 37311952, 40403168, 40481141, 42537442],
    "CAD": [134057, 314666, 315286, 315296, 316427, 317576, 318443, 438168, 443563, 764123, 764149, 4092936, 4110961, 4111393, 4124682, 4124683, 4155007, 4178622, 4185932, 4242670, 4317287, 36712779, 36712928, 36712982, 36712983, 36714444, 37115756, 37312532, 40481132, 40481919, 40482638, 40482655, 40483833, 43021857],
    "RA": [72705, 72714, 75621, 75622, 80809, 81097, 256197, 4013263, 4030424, 4035611, 4083556, 4102493, 4107913, 4114439, 4114441, 4114444, 4115050, 4115161, 4116150, 4116151, 4116439, 4116440, 4116441, 4116445, 4116446, 4117686, 4117687, 4142899, 4179378, 4200987, 4253901, 4269880, 4270869, 4271003, 4306357, 35609009, 35609010, 36684997, 36684998, 36685018, 36685019, 36685020, 36685023, 36685024, 36687002, 36687005, 36687006, 37108590, 37108591, 37108714, 37207809, 37207815, 37209321, 37209322, 37209323, 42534834, 42534835, 42534836, 42534837],
    "COPD":	[255573, 257004, 4046986, 4110056, 4115044, 4193588, 4196712, 4209097, 43530693, 44782563, 46269701, 46270376, 46274062],
    "IBD":	[4177488, 37116713, 46269884, 46269877, 4212992, 40479837, 46269878, 4246693, 4116142, 46269885, 46269887, 46269888, 46269951, 4242392, 195585, 46269891, 46274082, 46273477, 36715915, 40481367, 46269886, 46269876, 195575, 46269889, 201606, 4055020, 4301738, 4297644, 46269880, 4210469, 46269882, 46269890, 4302002, 4244235, 4029372, 40482865, 46274073, 4187900, 4264850, 194684, 46269875, 46269879, 4116143, 81893, 46269952, 46269881, 46273478, 4142544, 46269883, 46269874, 4340812, 36716986],
    "STROKE":	[4110192, 313226, 45772786, 381316, 377254, 321887, 45767658, 4213731, 4131383, 4045745, 443239, 4046360, 4110185, 4110186, 4110196, 4299377, 36716581, 765568, 4211509, 42535227, 4045748, 42535112, 42535111, 42539262, 762933, 762351, 4045740, 42535512, 4090122, 762344, 42535461, 43530674, 4319328, 4144154, 762340, 36684840, 4176892, 4046362, 4111714, 36716999, 43531605, 43530683, 761793, 42872427, 4222582, 4129534, 44782773, 40492969, 443883, 4145897, 763015, 4301259, 4189462, 761795, 432795, 42539472, 42535149, 764721, 42535465, 602590, 4112018, 376713, 40479572, 4153352, 4111710, 4249574, 42535114, 4338227, 761794, 42535459, 4077086, 609313, 4326561, 604192, 761797, 4138327, 442615, 42539166, 761792, 4112022, 4099974, 4108356, 43530727, 4049659, 4043731, 4111711, 4274969, 603326, 4045738, 46270031, 4311124, 42535425, 42535113, 609263, 37109288, 4218781, 443790, 379778, 4142739, 4045734, 42539195, 44782730, 618688, 4031045, 42535511, 42539269, 37396293, 40480491, 4273526, 443454, 4110189, 4110190, 4219010, 4159140, 4132091, 4310996, 761798, 46273649, 443864, 443752, 4006294, 761790, 42538998, 761789, 4328027, 761796, 4045737, 619933, 444091, 43531607, 4146185, 4046090, 4238315, 37395575, 375557, 42536243, 42535148],
    "SCZ":	[4008566, 433443, 4310121, 432598, 436071, 444396, 436944, 4100366, 433734, 435235, 436067, 441538, 432299, 432597, 439274, 435218, 4105330, 441835, 437243, 440368, 434321, 433996, 440686, 441828, 436673, 439275, 435782, 433742, 433450, 435236, 436384, 432300, 438724, 433442, 434332, 435783, 436385, 434901, 440373, 432865, 439004, 435219, 435217]
}

TASK_CODES = {
    task: [f"CDzz{code}" for code in codes]
    for task, codes in _TASK_CODES.items()
}

def safe_str_code_to_list(seq) -> list[str]:
    if isinstance(seq, str):
        seq = seq.split(" ")
    return seq


class ERPredictor:
    def __init__(
        self,
        dataset_path=None,
        model_path=None,
        tokenizer_path=None,
        force_gpt2_config=False,
        nemo_config_path=None,
        dataloader_batch_size=8,
        sample_size=None,
        exclude_visits=1,
        target_codes=None,
        followup_target_codes=None,
        label_time_range=30,
        eos_token=config.EOS_TOKEN,
        model_name="gpt",
        stopping_time_function_name="end_visit",
        min_ehr_time=None,
    ):
        """Initializes the ERPredictor with dataset and model configurations.
        Args:
            dataset_path (str): Path to the dataset -- for preprocessing the dataset
            model_path (str): Path to the trained model -- for inference
            tokenizer_path (str): Path to the tokenizer -- for inference
            dataloader_batch_size (int, optional): Batch size for the dataloader. Defaults to 8. -- for inference
            sample_size (int, optional): Number of samples to use from the dataset. Defaults to None (use all). -- for preprocessing
            exclude_visits (int, optional): Number of end visits to exclude when determining stopping time. Defaults to 1. -- for preprocessing
            target_codes (list, optional): List of target codes for prediction. Defaults to [config.ER_TOKEN]. -- for preprocessing and inference
            followup_target_codes (list, optional): List of followup target codes for prediction. Defaults to None. -- for preprocessing and inference
            label_time_range (int, optional): Time range in days for labeling. Defaults to 30. -- for preprocessing and inference
            eos_token (int, optional): End-of-sequence token ID. Defaults to config.EOS_TOKEN. -- for inference
            model_name (str, optional): Model architecture name ('gpt' or 'aou'). Defaults to 'gpt'. -- for inference
            stopping_time_function_name (str, optional): Name of the function to determine stopping time. Defaults to None (uses 'end_visit'). -- for preprocessing
                values: [None, 'end_visit', 'demo_age', 'random']
        """
        self.time_mapper = TimeMapper()
        self.all_day_cuts = self.time_mapper.all_day_cuts
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.force_gpt2_config = force_gpt2_config
        self.nemo_config_path = nemo_config_path
        self.dataloader_batch_size = dataloader_batch_size
        self.sample_size = sample_size
        self.eos_token = eos_token
        self.model_name = model_name
        self.stopping_time_function_name = stopping_time_function_name
        self.min_ehr_time = min_ehr_time

        self.model = None
        self.tokenizer = None
        self.time_codes_ids = {}

        # Targets and their IDs
        self.target_codes = target_codes if target_codes is not None else [config.ER_TOKEN]
        self.followup_target_codes = followup_target_codes or []
        self.target_codes_ids = []

        self.exclude_visits = exclude_visits
        if exclude_visits < 1:
            raise ValueError("exclude_visits should be >= 1")
        self.label_time_range = label_time_range

    def stopping_time_funct(self, seq, **kwargs) -> int:
        funct_dict = {
            "end_visit": self.stopping_time_end_visit,
            "demo_age": self.stopping_time_demo_age,
            "random": self.stopping_time_random,
            "first_target": self.stopping_time_first_target,
        }
        if self.stopping_time_function_name not in funct_dict:
            raise ValueError(
                f"Unknown stopping time function: {self.stopping_time_function_name}"
            )
        return funct_dict[self.stopping_time_function_name](seq, **kwargs)

    def stopping_time_first_target(self, seq, first_target=None) -> int:
        if first_target is None:
            # If target is not found the observation will be filtered in the
            # next step where stopping_time > 0
            return -1
        else:
            # + 1 so the first target is included in the input sequence
            return first_target + 1

    def stopping_time_random(self, seq, **kwargs) -> int:
        seq = safe_str_code_to_list(seq)
        x = len(seq)
        # Randomly choose a stopping time between 1 and the length of the sequence
        return random.randint(x // 4, x // 4 * 3)

    def stopping_time_demo_age(self, seq, **kwargs) -> int:
        seq = safe_str_code_to_list(seq)
        for i, code in enumerate(seq):
            if "TIME" in code:
                return i + 1  # +1 to include the time token itself
        return -1  # No time token found

    def stopping_time_end_visit(self, seq, exclude_visits=1, first_target=None) -> int:
        seq = safe_str_code_to_list(seq)
        if first_target is not None:
            # Remove all tokens after the first target
            seq = seq[:first_target]

        ind = np.where(np.array(seq) == config.ENDVISIT_TOKEN)[0]
        if len(ind) <= 1:
            return -1
        else:
            ind = ind[:-exclude_visits]
            return random.choice(ind) + 1  # +1 to include the end visit token itself

    def label_token(self, seq, from_index, on_token=False) -> int:
        if on_token:
            raise NotImplementedError(
                "on_token=True is not implemented. Use on_token=False for now with sentences."
            )
        else:
            target_codes = self.followup_target_codes or self.target_codes
            seq = safe_str_code_to_list(seq)
            curr_time = 0
            for code in seq[from_index:]:
                if code.startswith("TIME_"):
                    curr_time += self.all_day_cuts[code]
                    if curr_time > self.label_time_range:
                        return 0
                elif any([target in code for target in target_codes]):
                    return 1
            return 0

    def score_outputs(self, scores, sequences) -> list[float]:
        # scores is a tuple[max_length] of tensors of shape (batch_size * n_sequences, vocab_size)
        # We want to score the probability of the target tokens
        # keeping a bayesian score of probability
        all_probabilities = []
        for batch_idx in range(len(scores[0])):
            neg_prob = 1.0
            seq_probs = []
            curr_time = 0
            for step_idx, step_logits in enumerate(scores):
                # step_logits: (batch_size * n_sequences, vocab_size)
                logits = step_logits[batch_idx]
                # Get the probability of all target tokens
                prob = torch.softmax(logits, dim=-1)
                target_probs = prob[self.target_codes_ids].sum()
                seq_probs.append(target_probs.item() * neg_prob)

                # Update the negative probability
                neg_prob *= 1 - target_probs.item()

                # Keep track of the time
                curr_time += self.all_day_cuts.get(
                    self.time_codes_ids.get(sequences[batch_idx][step_idx]), 0
                )
                if curr_time > self.label_time_range:
                    break
            # Store the final probability for this sequence
            all_probabilities.append(sum(seq_probs))
        return all_probabilities

    def extract_age(self, seq, start_token=0) -> float:
        seq = safe_str_code_to_list(seq)
        seq = seq[start_token:]
        seq = [code for code in seq if "TIME" in code]
        age = sum(self.all_day_cuts.get(code, 0) for code in seq) / 365
        return age

    def extract_list_ages(self, seq, start_token=0) -> list[float]:
        seq = safe_str_code_to_list(seq)
        seq = seq[start_token:]
        # seq = [code for code in seq if "TIME" in code]
        ages = [self.all_day_cuts.get(code, 0) / 365 for code in seq]
        # Cumulative ages
        for i in range(1, len(ages)):
            ages[i] += ages[i - 1]
        return ages

    def find_first_target(self, seq) -> int:
        seq = safe_str_code_to_list(seq)
        first_target = -1
        for target in self.target_codes:
            ind = seq.index(target) if target in seq else -1
            if ind != -1 and (first_target == -1 or ind < first_target):
                first_target = ind
        return first_target

    def find_n_y_token(self, time_ehr_all: list[float], birth_token: int, n=3):
        # Returns index of first token above n Y
        # birth token is index of BIRTH token
        try:
            birth_age = time_ehr_all[birth_token + 1]

            cur_time = birth_age
            i = birth_token + 1
            while cur_time < birth_age + n:
                i += 1
                cur_time = time_ehr_all[i]

            return i
        except IndexError:
            return -1

    def load_dataset(self, split="test", **kwargs) -> Dataset:
        dataset_path = self.dataset_path
        try:
            dataset = DatasetDict.load_from_disk(dataset_path)
        except (FileNotFoundError, TypeError, ValueError):
            dataset = datasets.load_from_disk(dataset_path)

        def subsample_dataset(dataset):
            return dataset.shuffle(seed=42).select(range(self.sample_size))

        if isinstance(dataset, DatasetDict):
            if split != "all":
                dataset = dataset[split]
            if self.sample_size:
                for s in dataset:
                    dataset[s] = subsample_dataset(dataset[s])
        elif isinstance(dataset, Dataset):
            if self.sample_size:
                dataset = subsample_dataset(dataset)

        return dataset

    def preprocess_dataset(
        self, dataset: Dataset, num_proc=10, stop_before_stopping_time=False
    ) -> Dataset:
        print("Preprocessing dataset...")
        # Length of sequence and birth token
        dataset = dataset.filter(
            lambda x: "MEDSzzBIRTH" in x["text"], num_proc=num_proc, desc="Filter birth"
        )
        dataset = dataset.map(
            lambda x: {
                "len": len(safe_str_code_to_list(x["text"])),
                "birth_token": safe_str_code_to_list(x["text"]).index("MEDSzzBIRTH"),
            },
            num_proc=num_proc,
            desc="Length and birth token",
        )

        # Calculate amount of EHR time in years and filter by min_ehr_time if provided
        dataset = dataset.map(
            lambda x: {
                "time_ehr": self.extract_age(
                    x["text"], start_token=x["birth_token"] + 2
                )
            },
            num_proc=num_proc,
            desc="Extract EHR time",
        )
        if self.min_ehr_time is not None:
            dataset = dataset.filter(
                lambda x: x["time_ehr"] > self.min_ehr_time,
                num_proc=num_proc,
                desc=f"Filter ehr_time > {self.min_ehr_time}",
            )

        dataset = dataset.map(
            lambda x: {"first_target": self.find_first_target(x["text"])},
            num_proc=num_proc,
            desc="Finding first target code",
        )
        if stop_before_stopping_time:
            print("Stopping before stopping time processing")
            return dataset

        dataset = dataset.map(
            lambda x: {
                "stopping_time": self.stopping_time_funct(
                    x["text"], first_target=x["first_target"]
                )
            },
            num_proc=num_proc,
            desc=f"Determining stopping time: {self.stopping_time_function_name}",
        )

        dataset = dataset.filter(
            lambda x: x["stopping_time"] > 0 and x["stopping_time"] < x["len"],
            num_proc=num_proc,
            desc="Filter valid stopping time",
        )

        # Extracting labels based on stopping time and target codes
        dataset = dataset.map(
            lambda x: {"label_token": self.label_token(x["text"], x["stopping_time"])},
            num_proc=num_proc,
            desc="Finding target label after stopping time",
        )

        # Truncate sequences at stopping time
        dataset = dataset.map(
            lambda x: {
                "input_ids": x["input_ids"][: x["stopping_time"]],
                "attention_mask": x["attention_mask"][: x["stopping_time"]],
            },
            num_proc=num_proc,
            desc="Truncating sequences at stopping time",
        )

        print("Ensuring no EOS token at the end of the sequence")
        filtered_dataset = dataset.filter(
            lambda x: all([i == self.eos_token for i in x["input_ids"][-5:]]),
            num_proc=num_proc,
        )
        if isinstance(filtered_dataset, DatasetDict):
            assert all(
                [len(filtered_dataset[split]) == 0 for split in filtered_dataset]
            )
        elif isinstance(filtered_dataset, Dataset):
            assert len(filtered_dataset) == 0
        return dataset

    def load_model_tokenizer(self, dtype=None):
        if self.tokenizer is None:
            tokenizer = self.load_tokenizer()
        if self.model is None:
            model = self.load_model(dtype=dtype)
        return model, tokenizer

    def load_model(self, dtype=None):
        model = model_util.get_trained_model(
            self.model_path,
            self.model_name,
            torch_dtype=dtype,
            force_gpt2_config=self.force_gpt2_config,
            nemo_config_path=self.nemo_config_path,
        )
        if self.model_name in ["gpt", "qwen"]:
            model.config.pad_token_id = self.tokenizer.pad_token_id
        else:
            model.model_gpt.config.pad_token_id = self.tokenizer.pad_token_id
        self.model = model
        return model

    def load_tokenizer(self) -> AutoTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"
        tokenizer.pad_token = tokenizer.eos_token

        self.tokenizer = tokenizer
        self.eos_token = tokenizer.eos_token_id

        self.target_codes_ids = [
            self.tokenizer.vocab.get(code)
            for code in self.target_codes
            if code in self.tokenizer.vocab
        ]
        self.time_codes_ids = {
            self.tokenizer.vocab[code]: code
            for code in self.time_mapper.all_day_cuts.keys()
        }
        return tokenizer

    def load_dataloader(
        self,
        dataset,
        tokenizer,
        kept_columns=["subject_id"],
        num_workers=config.DATALOADER_NUM_CPUS,
        shuffle=False,
    ):
        return model_util.load_dataloader(
            dataset,
            tokenizer,
            kept_columns=kept_columns,
            num_workers=num_workers,
            shuffle=shuffle,
            dataloader_batch_size=self.dataloader_batch_size,
            model_name=self.model_name,
            padding_side="left",
        )


def reformat_list_to_list_of_list(
    lst: list[float], n_sequences: int
) -> list[list[float]]:
    """
    Reformat a flat list into a list of lists, each containing n_sequences elements.
    """
    return [
        lst[i * n_sequences : (i + 1) * n_sequences] for i in range(len(lst) // n_sequences)
    ]


def process_dataset_for_prediction(
    er_predictor: ERPredictor,
    dataset_path,
    max_new_tokens=50,
    max_position_embeddings=config.MODEL_WINDOW_SIZE,
    postfix="_eval",
):
    dataset = er_predictor.load_dataset()
    dataset = er_predictor.preprocess_dataset(dataset)
    dataset = dataset.filter(
        lambda x: x["stopping_time"] < max_position_embeddings - max_new_tokens
    )
    output_path = dataset_path + postfix
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    dataset.save_to_disk(output_path)


def save_results(
    accelerator, d, output_csv_path, n_sequences, max_new_tokens, model_name_path
):
    if accelerator.is_main_process:
        df = pd.DataFrame(d)
        df["num_predictions"] = n_sequences
        df["max_new_tokens"] = max_new_tokens
        df["model_name"] = model_name_path

        # Initialize output_csv_path if its directory does not exist
        if not os.path.exists(os.path.dirname(output_csv_path)):
            os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

        df.to_csv(output_csv_path, index=False)
        print(f"\nSaved to {output_csv_path}")
    accelerator.wait_for_everyone()


def batch_decode(
    accelerator: Accelerator,
    er_predictor: ERPredictor,
    dataset_path,
    model_name_path,
    output_csv_path,
    max_new_tokens=50,
    n_sequences=10,
    sample_size=None,
    unwrap_model=True,
    compute_loss=False,
):
    accelerator.print("Starting - model used:", model_name_path)

    # Load preprocessed dataset
    accelerator.print("Loading preprocessed dataset", dataset_path)
    dataset = datasets.load_from_disk(dataset_path)
    if sample_size:
        dataset = dataset.shuffle(seed=42).select(range(sample_size))
    accelerator.print(f"Evaluation dataset size: {len(dataset)}")

    # Load models and prepare dataloader
    accelerator.print(
        f"Loading model and tokenizer -- precision: {accelerator.mixed_precision}"
    )
    model, tokenizer = er_predictor.load_model_tokenizer(
        dtype=model_util.get_precision_dtype(accelerator),
    )
    model_util.print_number_trainable_params(model, accelerator)

    accelerator.print("Loading dataloader")
    dataloader = er_predictor.load_dataloader(
        dataset, tokenizer, kept_columns=["subject_id", "label_token"]
    )

    accelerator.print("Preparing model and dataloader")
    dataloader = accelerator.prepare(dataloader)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    if unwrap_model:
        model = accelerator.prepare_model(model)
        accelerator.print("Unwrapping model")
        model = accelerator.unwrap_model(model)
    else:
        model.to(accelerator.device)

    d = {}
    if compute_loss:
        model.config.loss_type = "ForCausalLMLoss"
    losses = []

    def extend_gather(d, key, value, accelerator=accelerator):
        if key not in d:
            d[key] = []
        d[key].extend(accelerator.gather_for_metrics(value))

    accelerator.print(f"Inferring, dataset columns: {dataloader.dataset.column_names}")
    global_inference_step = 0
    for batch in tqdm(dataloader, disable=not accelerator.is_local_main_process):
        with accelerator.autocast():
            (
                outputs,
                input_ids,
                attention_mask,
                subject_id,
                label_token,
                valid_loss,
            ) = model_util.generate_sequences(
                model,
                batch,
                n_sequences=n_sequences,
                max_new_tokens=max_new_tokens,
                compute_loss=compute_loss,
            )

        n_subjects = len(batch["subject_id"])
        decoded_i = tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        if outputs is not None:
            sequences = outputs.sequences
            scores = outputs.scores
            n_outputs = len(sequences) // n_subjects

            # Decode and gather predictions
            decoded = tokenizer.batch_decode(
                sequences[:, -max_new_tokens:], skip_special_tokens=True
            )
            predicted = [
                er_predictor.label_token(d, 0, on_token=False)
                for d in decoded
            ]

            # Score probabilities
            predicted_scores = er_predictor.score_outputs(scores, sequences)

            # Ensure decoded and predicted are lists of lists
            decoded = reformat_list_to_list_of_list(decoded, n_outputs)
            predicted = reformat_list_to_list_of_list(predicted, n_outputs)
            predicted_scores = reformat_list_to_list_of_list(
                predicted_scores, n_outputs
            )
            probability = list(np.mean(predicted, 1))

        else:
            decoded = [[] for _ in range(n_subjects)]
            predicted = [[] for _ in range(n_subjects)]
            predicted_scores = [[] for _ in range(n_subjects)]
            probability = [[] for _ in range(n_subjects)]

        # Gather results
        extend_gather(d, "original_text", decoded_i)
        extend_gather(d, "generated_text", decoded)
        extend_gather(d, "pred_labels", predicted)
        extend_gather(d, "subject_id", subject_id)
        extend_gather(d, "label_token", label_token)
        extend_gather(d, "probability", probability)
        extend_gather(d, "probability_scores", predicted_scores)

        # Log metrics
        global_inference_step += 1
        loss_value = valid_loss if valid_loss is not None else float("nan")
        if not np.isnan(loss_value):
            losses.append(loss_value)

        accelerator.log({"valid_loss": loss_value}, step=global_inference_step)
        del outputs

        # Save every 100
        if global_inference_step % 100 == 0:
            accelerator.print(f"Saving interim results, step {global_inference_step}")
            save_results(
                accelerator,
                d,
                output_csv_path,
                n_sequences,
                max_new_tokens,
                model_name_path,
            )

    accelerator.print("Done")

    # Calculate and log average loss
    if losses:
        losses_tensor = torch.tensor(losses, device=accelerator.device)
        gathered_losses = accelerator.gather_for_metrics(losses_tensor)
        avg_loss = torch.mean(gathered_losses).item()
        accelerator.log({"avg_valid_loss": avg_loss})
        accelerator.print(f"Average validation loss across all batches: {avg_loss:.4f}")
    else:
        accelerator.print("No valid losses computed - unable to calculate average")

    accelerator.wait_for_everyone()
    save_results(
        accelerator, d, output_csv_path, n_sequences, max_new_tokens, model_name_path
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process an AoU dataset for diagnosis prediction.")
    parser.add_argument("--dataset-path", type=str, default=os.path.join(LOCAL_DIR, config.paths.dataset_path), help="Path to input dataset")
    parser.add_argument(
        "--task", type=str, default="T2D", 
        help="Diagnosis code to predict"
    )
    parser.add_argument(
        "--label-time-range", type=int, default=365*2,
        help="Label time range (days)"
    )
    parser.add_argument(
        "--use-mock-data",
        action="store_true",
        help="Use the test split from mock_data/ instead of real data.",
    )
    args = parser.parse_args()
    if args.use_mock_data:
        args.dataset_path = str(MOCK_DATASET_PATH)
        if not MOCK_DATASET_PATH.exists():
            parser.error(
                f"Mock dataset not found at {MOCK_DATASET_PATH}. "
                "Run: python aou_data_loader.py --use-mock-data"
            )
    er_predictor = ERPredictor(
        dataset_path=args.dataset_path,
        target_codes=TASK_CODES[args.task],
        label_time_range=args.label_time_range,
        stopping_time_function_name="end_visit",
    )
    process_dataset_for_prediction(
        er_predictor,
        args.dataset_path,
        max_new_tokens=config.MAX_NEW_TOKENS,
        max_position_embeddings=config.MODEL_WINDOW_SIZE,
        postfix=f"_{args.task}_{args.label_time_range}"
    )