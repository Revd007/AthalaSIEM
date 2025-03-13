import logging
import logging.config
from pathlib import Path
import yaml

def setup_logging():
    """Setup logging configuration"""
    try:
        log_config_path = Path("backend/config/logging.yaml")
        if log_config_path.exists():
            with open(log_config_path) as f:
                config = yaml.safe_load(f)
                logging.config.dictConfig(config)
        else:
            # Default logging configuration
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
        # Create logger
        logger = logging.getLogger(__name__)
        logger.info("Logging setup completed")
        
    except Exception as e:
        logging.basicConfig(level=logging.INFO)
        logging.error(f"Error setting up logging: {e}")