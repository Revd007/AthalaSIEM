from typing import Dict, Any, List, Optional
import torch
import numpy as np
from datetime import datetime
import logging
from ..core.model_manager import ModelManager
from ..training.training_manager import TrainingManager

class FeedbackLoop:
    def __init__(self, model_manager: ModelManager, training_manager: TrainingManager):
        self.model_manager = model_manager
        self.training_manager = training_manager
        self.logger = logging.getLogger(__name__)
        self.feedback_buffer: List[Dict[str, Any]] = []
        self.retraining_threshold = 100  # Number of feedback items before retraining

    async def process_feedback(self, feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Validate feedback
            if not self._validate_feedback(feedback_data):
                return {"status": "error", "message": "Invalid feedback format"}

            # Store feedback
            self.feedback_buffer.append({
                **feedback_data,
                "timestamp": datetime.utcnow().isoformat()
            })

            # Check if retraining is needed
            if len(self.feedback_buffer) >= self.retraining_threshold:
                await self._trigger_model_update()

            return {
                "status": "success",
                "message": "Feedback processed successfully",
                "feedback_buffer_size": len(self.feedback_buffer)
            }

        except Exception as e:
            self.logger.error(f"Error processing feedback: {e}")
            return {"status": "error", "message": str(e)}

    async def _trigger_model_update(self) -> None:
        try:
            # Prepare feedback data for training
            training_data = self._prepare_training_data()
            
            # Update model with new feedback
            model_updates = await self.training_manager.train(
                train_loader=training_data['train_loader'],
                val_loader=training_data['val_loader'],
                num_epochs=5  # Adjust based on needs
            )

            # Clear feedback buffer after successful update
            self.feedback_buffer.clear()
            
            self.logger.info("Model successfully updated with feedback data")
            
        except Exception as e:
            self.logger.error(f"Error updating model with feedback: {e}")

    def _validate_feedback(self, feedback: Dict[str, Any]) -> bool:
        required_fields = ['model_name', 'prediction_id', 'correct_label', 'confidence']
        return all(field in feedback for field in required_fields)

    def _prepare_training_data(self) -> Dict[str, Any]:
        # Transform feedback data into training format
        features = []
        labels = []
        
        for feedback in self.feedback_buffer:
            features.append(feedback.get('features', []))
            labels.append(feedback.get('correct_label'))

        # Convert to tensors
        feature_tensor = torch.tensor(features, dtype=torch.float32)
        label_tensor = torch.tensor(labels, dtype=torch.long)

        return {
            'train_loader': torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(feature_tensor, label_tensor),
                batch_size=32,
                shuffle=True
            ),
            'val_loader': None  # Could split data for validation if needed
        }