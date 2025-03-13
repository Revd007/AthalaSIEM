import asyncio
import logging
from pathlib import Path
import yaml
import torch
import gc
from tqdm import tqdm
from typing import Dict, Any
import json
import argparse

from ..core.dataset_handler import CyberSecurityDataHandler
from ..training.training_manager import TrainingManager
from ..processors.data_cleaning import DataCleaner
from ..processors.data_normalization import DataNormalizer
from ..models.unified_threat_detector import UnifiedThreatDetector
from ..core.evaluator import ModelEvaluator
from ..utils.metrics import MetricsTracker
from ..evaluation.metrics.custom_metrics import CustomMetrics
from ..evaluation.metrics.accuracy_metrics import AccuracyMetrics
from ..core.model_manager import ModelManager

async def train_security_model(config_path: str = None):
    try:
        # Mengatur path konfigurasi
        if config_path is None:
            current_dir = Path(__file__).parent
            config_path = current_dir.parent / "config" / "ai_settings.yaml"

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        # Mengatur perangkat ke GPU jika tersedia
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Menggunakan perangkat: {device}")

        # Memuat konfigurasi
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Inisialisasi ModelManager
        model_manager = ModelManager(config)
        model = model_manager.get_model('unified_threat_detector')
        model.to(device)  # Memindahkan model ke GPU

        # Initialize data handler
        data_handler = CyberSecurityDataHandler(config)
        
        # Load datasets automatically without manual configuration
        df = data_handler.load_datasets()  # Will use automatic discovery
        
        # Process data
        train_loader, test_loader = data_handler.process_data(df)

        # Inisialisasi TrainingManager
        training_manager = TrainingManager(
            model=model,
            config=config,
            model_manager=model_manager,
            experiment_name="security_model_training"
        )

        # Memulai pelatihan
        training_results = await training_manager.train(
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=config['training']['epochs'],
            learning_rate=config['training']['learning_rate']
        )

        # Evaluasi model
        evaluation_results = await evaluate_trained_model(
            model=model,
            test_loader=test_loader,
            config=config
        )

        # Menyimpan hasil
        results = {
            'training_results': training_results,
            'evaluation_results': evaluation_results
        }

        return results

    except Exception as e:
        logger.error(f"Training/evaluation error: {e}")
        raise

async def evaluate_trained_model(
    model: UnifiedThreatDetector,
    test_loader,
    config: Dict[str, Any]
):
    # Initialize evaluator
    evaluator = ModelEvaluator(model_manager=None, config=config)
    metrics_tracker = MetricsTracker(save_dir="metrics")

    try:
        # Evaluate model using the correct method name
        evaluation_results = await evaluator.evaluate_model(
            model=model,
            test_loader=test_loader,
            metrics={
                'basic': AccuracyMetrics.calculate_basic_metrics,
                'custom': CustomMetrics.calculate_prediction_confidence
            }
        )

        # Track and save metrics
        metrics_tracker.update_metrics(
            metrics=evaluation_results,
            step=0,
            model_name="unified_threat_detector"
        )

        # Generate visualization plots
        evaluator.plot_metrics(save_path=Path("evaluation_results"))

        return evaluation_results

    except Exception as e:
        logging.error(f"Evaluation error: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train security model')
    parser.add_argument('--config', type=str, help='Path to config file')
    args = parser.parse_args()
    
    asyncio.run(train_security_model(config_path=args.config))