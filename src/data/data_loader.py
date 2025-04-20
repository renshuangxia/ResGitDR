import logging
from pathlib import Path
from typing import Tuple, Dict
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class GeneExpressionDataset(Dataset):
    def __init__(self, sga_data: pd.DataFrame, rnaseq_data: pd.DataFrame, cancer_data: pd.DataFrame):
        self.sga_data = torch.FloatTensor(sga_data.values)
        self.rnaseq_data = torch.FloatTensor(rnaseq_data.values)
        self.cancer_data = torch.LongTensor(cancer_data.values)
        
    def __len__(self):
        return len(self.sga_data)
    
    def __getitem__(self, idx):
        return {
            'sga': self.sga_data[idx],
            'cancer_type': self.cancer_data[idx],
            'target': self.rnaseq_data[idx]
        }

class DataManager:
    def __init__(self, config: DataConfig):
        self.config = config
        self.data_dir = Path(config.data_dir)
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def load_tcga_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load and preprocess TCGA dataset."""
        try:
            sga_df = pd.read_csv(self.data_dir / "TCGA_sga_data.csv", index_col=0)
            rnaseq_df = pd.read_csv(self.data_dir / "TCGA_RNAseq_data.csv", index_col=0)
            cancer_df = pd.read_csv(self.data_dir / "TCGA_xena_map_cancertype.csv", index_col=0)
            
            # Preprocessing
            rnaseq_df = rnaseq_df.clip(lower=0)
            
            # Align indices
            rnaseq_df = rnaseq_df.loc[sga_df.index, :]
            cancer_df = cancer_df.loc[sga_df.index, ["cancer_type"]]
            
            self.logger.info(f"Loaded TCGA data: {len(sga_df)} samples")
            return sga_df, rnaseq_df, cancer_df
            
        except Exception as e:
            self.logger.error(f"Error loading TCGA data: {e}")
            raise 