import logging
from pathlib import Path
from typing import Dict, Any
import torch
from ai_engine.core.dataset_handler import CyberSecurityDataHandler
from ai_engine.donquixote_service import load_default_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_dataset():
    try:
        # 1. Setup device (GPU/CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # 2. Load configuration
        config = load_default_config()
        
        # 3. Initialize handler with config and device
        handler = CyberSecurityDataHandler(config, device)
        
        # 4. Validate dataset directory
        dataset_dir = Path("E:/AthalaSIEM/AthalaSIEM/backend/ai_engine/dataset")
        if not dataset_dir.exists():
            logger.error(f"Dataset directory not found: {dataset_dir}")
            return False
            
        stats = handler.validate_dataset_directory(str(dataset_dir))
        logger.info(f"Dataset statistics: {stats}")
        
        # 5. Process all files
        logger.info("Processing dataset files...")
        combined_df = handler.process_dataset_files(str(dataset_dir))
        logger.info(f"Combined DataFrame shape: {combined_df.shape}")
        
        # 6. Validate processed data
        logger.info("Validating processed data...")
        validated_df = handler._validate_data(combined_df)
        logger.info(f"Validated DataFrame shape: {validated_df.shape}")
        
        # 7. Process and clean data
        logger.info("Processing and cleaning data...")
        processed_df = handler._process_data(validated_df)
        logger.info(f"Final processed DataFrame shape: {processed_df.shape}")
        
        # 8. Save processed dataset
        output_path = dataset_dir / "processed" / "cyber_threat_intelligence.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        processed_df.to_csv(output_path, index=False)
        
        logger.info(f"Dataset setup completed. Processed data saved to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Dataset setup failed: {e}")
        return False

if __name__ == "__main__":
    setup_dataset()