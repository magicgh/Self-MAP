from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainConfig:
    neg_ratio: float
    num_candidates: int
    max_context_len: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    learning_rate: float
    epoch: int
    num_gpus: int
    bf16: bool
    tf32: bool
    lora: bool
    optim: str
    gradient_accumulation_steps: int
    fsdp_policy: str
    fsdp: bool
    resume_from_checkpoint: bool


@dataclass
class SelfMapConfig:
    generation: bool
    memory_simplification: bool
    memory_refinement: bool
    multifaceted_matching: bool


@dataclass
class KnnConfig:
    top_k: int
    sort_by: str


@dataclass
class OpenAIConfig:
    rate_limit: int
    model: str
    display_cost: bool


@dataclass
class TestDataConfig:
    test_task: str
    test_website: str
    test_subdomain: str


@dataclass
class DataConfig:
    data_path: str
    train_split_file: str
    test_split_files: TestDataConfig
    score_file: str


@dataclass
class ModelConfig:
    tokenizer_name: str
    arch: str
    name: str
    model_name_or_path: str
    max_seq_length: int
    fsdp_transformer_layer_cls_to_wrap: Optional[str]


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
    train: TrainConfig
    self_map: SelfMapConfig
    knn: KnnConfig
    seed: int
    openai: OpenAIConfig
    data: DataConfig
    run_id: str
    model: ModelConfig
    hydra: HydraConfig


@dataclass
class MemoryConfig:
    knn: KnnConfig
    openai: OpenAIConfig
    hydra: HydraConfig
