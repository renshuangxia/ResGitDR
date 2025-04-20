import logging
from pathlib import Path
from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from ..config import TrainingConfig

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.model.to(self.device)
        
    def train_epoch(self, train_loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
        self.model.train()
        total_loss = 0
        
        with tqdm(train_loader, desc="Training") as pbar:
            for batch in pbar:
                # Move data to device
                sga = batch['sga'].to(self.device)
                cancer_type = batch['cancer_type'].to(self.device)
                target = batch['target'].to(self.device)
                
                optimizer.zero_grad()
                output = self.model(sga, cancer_type)
                loss = self.criterion(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                sga = batch['sga'].to(self.device)
                cancer_type = batch['cancer_type'].to(self.device)
                target = batch['target'].to(self.device)
                
                output = self.model(sga, cancer_type)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                
        return total_loss / len(val_loader) 