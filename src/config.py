from dataclasses import dataclass
from typing import Optional
import yaml
from pathlib import Path

@dataclass
class TrainingConfig:
    n_split: int
    num_epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    hidden_weights_pruning: bool
    hidden_weights_pruning_ratio: float
    early_stop: int

@dataclass
class ModelConfig:
    embedding_size: int
    embedding_size_last: int
    num_blocks: int
    n_head: int
    using_cancer_type: bool
    scale_y: bool
    using_tf_gene_matrix: bool
    trim_ratio: float

@dataclass
class DataConfig:
    data_dir: str
    embed_pretrain_gene2vec: bool
    permutation_sga_tissue: bool

@dataclass
class Config:
    training: TrainingConfig
    model: ModelConfig
    data: DataConfig
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'Config':
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
            return cls(
                training=TrainingConfig(**config_dict['training']),
                model=ModelConfig(**config_dict['model']),
                data=DataConfig(**config_dict['data'])
            ) 