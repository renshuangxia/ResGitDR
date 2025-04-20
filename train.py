import logging
import torch
from pathlib import Path
from src.config import Config
from src.data.data_loader import DataManager, GeneExpressionDataset
from src.training.trainer import Trainer
from models.ResGit import ResGit  # Your existing model

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = Config.from_yaml('config/config.yaml')
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load data
    data_manager = DataManager(config.data)
    sga_df, rnaseq_df, cancer_df = data_manager.load_tcga_data()
    
    # Create datasets
    dataset = GeneExpressionDataset(sga_df, rnaseq_df, cancer_df)
    
    # Create model
    model = ResGit(
        embedding_size=config.model.embedding_size,
        num_blocks=config.model.num_blocks,
        n_head=config.model.n_head,
        # ... other parameters
    )
    
    # Create trainer
    trainer = Trainer(model, config.training, device)
    
    # Train model
    trainer.train(dataset)

if __name__ == "__main__":
    main() 