import logging
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import torch
from pathlib import Path
from typing import Dict, Any
import yaml
import asyncio

from ai_engine.core.dataset_handler import CyberSecurityDataHandler
from ai_engine.core.model_manager import ModelManager
from ai_engine.training.training_manager import TrainingManager
from ai_engine.core.evaluator import ModelEvaluator

async def train_and_evaluate():
    """Run complete training and evaluation pipeline"""
    try:
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        
        # Load config
        config_path = Path(__file__).parents[2] / "ai_engine" / "config" / "ai_settings.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)
            
        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Initialize components
        model_manager = ModelManager(config)
        data_handler = CyberSecurityDataHandler(config, device)
        
        # Load and prepare dataset
        dataset_path = Path(__file__).parents[2] / "ai_engine" / "dataset" / "processed" / "cyber_threat_intelligence.csv"
        logger.info("Loading processed dataset...")
        
        # Prepare data loaders with default values
        train_loader, val_loader, test_loader = data_handler.prepare_data_loaders(
            dataset_path,
            batch_size=config.get('training', {}).get('batch_size', 32),
            val_split=config.get('training', {}).get('validation_split', 0.2),
            test_split=config.get('training', {}).get('test_split', 0.1)
        )
        
        # Initialize models
        unified_threat_detector = model_manager.get_model('unified_threat_detector')
        unified_anomaly_detector = model_manager.get_model('unified_anomaly_detector')
        
        # Save results
        results_dir = Path(__file__).parents[2] / "ai_engine" / "results"
        results_dir.mkdir(exist_ok=True)
        
        # Update config with checkpoint directory
        config['training']['checkpoint_dir'] = str(results_dir / "checkpoints")
        
        # Initialize training manager with updated config
        training_manager = TrainingManager(
            model=unified_threat_detector,
            config=config,
            model_manager=model_manager,
            experiment_name="threat_detection_training"
        )
        
        # Train models
        logger.info("Starting model training...")
        training_results = await training_manager.train(
            train_loader=train_loader,
            test_loader=val_loader,
            epochs=config['training']['epochs'],
            learning_rate=config['training']['learning_rate']
        )
        
        # Evaluate models
        logger.info("Starting model evaluation...")
        evaluator = ModelEvaluator(model_manager, config)
        evaluation_results = await evaluator.evaluate_model(
            model=unified_threat_detector,
            test_loader=test_loader,
            metrics={
                'accuracy': lambda y_true, y_pred: accuracy_score(y_true, y_pred),
                'precision_recall': lambda y_true, y_pred: precision_recall_fscore_support(
                    y_true, y_pred, average='weighted', zero_division=0
                )
            }
        )
        
        # Save results
        results_dir = Path(__file__).parents[2] / "ai_engine" / "results"
        results_dir.mkdir(exist_ok=True)
        
        # Save model checkpoints
        checkpoint_dir = results_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        torch.save({
            'model_state_dict': unified_threat_detector.state_dict(),
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'config': config
        }, checkpoint_dir / "unified_threat_detector.pt")
        
        logger.info("Training and evaluation completed successfully")
        logger.info(f"Training results: {training_results}")
        logger.info(f"Evaluation results: {evaluation_results}")
        
        return {
            'training_results': training_results,
            'evaluation_results': evaluation_results
        }
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(train_and_evaluate()) 