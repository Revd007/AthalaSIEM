import pytest
import torch
from ai_engine.models import ModelFactory
from ai_engine.training import Trainer
from ai_engine.evaluation import Evaluator
from ai_engine.data import DatasetManager
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class TestFullPipeline:
    @pytest.fixture
    def setup_pipeline(self):
        """Setup test pipeline"""
        self.config = {
            'model_version': 'latest',
            'batch_size': 32,
            'learning_rate': 1e-4,
            'num_epochs': 5
        }
        
        # Initialize components
        self.model = ModelFactory.create_model(self.config)
        self.dataset_manager = DatasetManager(self.config)
        self.trainer = Trainer(self.model, self.config)
        self.evaluator = Evaluator(self.model, self.config)
        
        return self.model, self.dataset_manager, self.trainer, self.evaluator

    @pytest.mark.asyncio
    async def test_training_pipeline(self, setup_pipeline):
        """Test full training pipeline"""
        model, dataset_manager, trainer, evaluator = setup_pipeline
        
        # Prepare data
        train_loader, val_loader, test_loader = dataset_manager.prepare_datasets()
        
        # Train model
        training_results = await trainer.train(train_loader, val_loader)
        
        # Assertions for training
        assert training_results['final_loss'] < training_results['initial_loss']
        assert all(val_loss > 0 for val_loss in training_results['val_losses'])
        
        # Evaluate model
        eval_results = await evaluator.evaluate(test_loader)
        
        # Assertions for evaluation
        assert eval_results['accuracy'] > 0.7  # Minimum accuracy threshold
        assert eval_results['f1_score'] > 0.7  # Minimum F1 score threshold
        
        # Test model predictions
        test_batch = next(iter(test_loader))
        with torch.no_grad():
            predictions = model(test_batch['input_ids'])
            
        assert predictions.shape == test_batch['labels'].shape
        
        # Additional assertions for model behavior
        assert torch.all(predictions >= 0) and torch.all(predictions <= 1), "Predictions should be probabilities between 0 and 1"
        assert not torch.isnan(predictions).any(), "Predictions should not contain NaN values"
        assert not torch.isinf(predictions).any(), "Predictions should not contain infinity values"
        
        # Test batch processing
        batch_size = test_batch['input_ids'].size(0)
        assert predictions.size(0) == batch_size, f"Expected predictions batch size {batch_size}, got {predictions.size(0)}"
        
        # Verify prediction classes
        pred_classes = torch.argmax(predictions, dim=-1)
        num_classes = predictions.size(-1)
        assert torch.all(pred_classes < num_classes), "Predicted classes should be valid indices"
        self._test_model_behavior(model)
        
    def _test_model_behavior(self, model):
        """Test specific model behaviors"""
        # Test attention mechanism
        test_input = torch.randint(0, 1000, (1, 50))  # Example input
        attention_weights = model.get_attention_weights(test_input)
        
        assert attention_weights.sum(dim=-1).allclose(torch.ones_like(attention_weights.sum(dim=-1)))
        
        # Test embedding layer
        embeddings = model.get_embeddings(test_input)
        assert not torch.isnan(embeddings).any()
        
        # Test gradient flow
        test_output = model(test_input)
        loss = test_output.mean()
        loss.backward()
        
        for param in model.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()