from dataclasses import dataclass
from typing import Dict


@dataclass
class ModelConfig:
    name: str
    model_name_or_path: str
    max_seq_length: int


@dataclass
class TrainConfig:
    neg_ratio: int
    batch_size: int
    learning_rate: float
    eval_batch_size: int
    epoch: int
    warmup_steps: float
    use_amp: bool


@dataclass
class DataConfig:
    data_path: str
    train_split_file: str
    test_split_files: Dict[str, str]


@dataclass
class HydraRunConfig:
    dir: str


@dataclass
class HydraJobConfig:
    chdir: bool


@dataclass
class HydraConfig:
    run: HydraRunConfig
    job: HydraJobConfig
    verbose: str


@dataclass
class Config:
    model: ModelConfig
    train: TrainConfig
    seed: int
    data: DataConfig
    hydra: HydraConfig
