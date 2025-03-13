from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import json
from pathlib import Path
import numpy as np
from ..core.knowledge_graph import KnowledgeGraph

class FeedbackManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.knowledge_graph = KnowledgeGraph()
        self.feedback_history = []
        self.feedback_file = Path("data/feedback_history.jsonl")
        self.feedback_file.parent.mkdir(parents=True, exist_ok=True)
        
    async def process_feedback(
        self,
        event_id: str,
        feedback: Dict[str, Any],
        analyst_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process analyst feedback for AI improvement"""
        try:
            # Record feedback
            feedback_entry = self._create_feedback_entry(
                event_id, feedback, analyst_id
            )
            
            # Update knowledge graph
            self.knowledge_graph.update_from_feedback(feedback)
            
            # Save feedback history
            self._save_feedback(feedback_entry)
            
            # Trigger model updates if needed
            if self._should_update_models(feedback):
                await self._trigger_model_update(feedback)
                
            return {
                'status': 'success',
                'feedback_id': feedback_entry['id'],
                'timestamp': feedback_entry['timestamp']
            }
            
        except Exception as e:
            self.logger.error(f"Error processing feedback: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }