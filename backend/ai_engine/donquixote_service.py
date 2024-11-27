from typing import *
import torch
from .models.anomaly_detector import AnomalyDetector, VariationalAutoencoder
from .models.threat_detector import ThreatDetector
from .core.model_manager import ModelManager
from .core.dataset_handler import CyberSecurityDataHandler
from .processors.data_cleaning import DataCleaner
from .processors.data_normalization import DataNormalizer
from .processors.feature_engineering import FeatureEngineer
from .training.training_manager import TrainingManager
from .training.adaptive_learner import AdaptiveLearner
from .core.evaluator import ModelEvaluator
from .core.knowledge_graph import KnowledgeGraph
from .ensemble.ensemble_manager import AIEnsembleManager
from .feedback.feedback_manager import FeedbackManager
import logging

class DonquixoteService:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Default config if none provided
        default_config = {
            'model_version': '1.0.0',
            'batch_size': 32,
            'learning_rate': 1e-4,
            'num_epochs': 10,
            'model_dir': 'models/',
            'log_dir': 'logs/',
            'data_dir': 'data/',
            'enable_ensemble': True,
            'enable_adaptive_learning': True
        }
        self.config = {**default_config, **(config or {})}
        
        try:
            # Initialize core components
            self.model_manager = ModelManager(self.config)
            self.dataset_handler = CyberSecurityDataHandler(self.config)
            self.data_cleaner = DataCleaner()
            self.data_normalizer = DataNormalizer()
            self.feature_engineer = FeatureEngineer()
            
            # Initialize training and evaluation components
            self.training_manager = TrainingManager(
                model=self.model_manager.get_model(),
                config=self.config,
                experiment_name="donquixote_training"
            )
            self.adaptive_learner = AdaptiveLearner(
                model=self.model_manager.get_model(),
                config=self.config
            )
            self.evaluator = ModelEvaluator(
                model=self.model_manager.get_model(),
                config=self.config
            )
            
            # Initialize advanced components
            self.knowledge_graph = KnowledgeGraph()
            self.ensemble_manager = AIEnsembleManager(self.model_manager)
            self.feedback_manager = FeedbackManager()
            
            self.logger.info("Models initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing models: {e}")
            raise

    async def train_model(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """Train the model with new data"""
        try:
            # Preprocess training data
            cleaned_data = self.data_cleaner.clean_data(training_data)
            normalized_data = self.data_normalizer.normalize_data(cleaned_data)
            
            # Create data loaders
            train_loader, val_loader = self.dataset_handler.prepare_data(normalized_data)
            
            # Train model
            training_results = await self.training_manager.train(
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=self.config['num_epochs']
            )
            
            # Update knowledge graph
            self.knowledge_graph.update(training_results.get('patterns', []))
            
            return training_results
        except Exception as e:
            self.logger.error(f"Training error: {e}")
            return {'error': str(e)}

    async def analyze_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single event"""
        try:
            # Preprocess event data
            cleaned_event = self.data_cleaner.clean_event(event_data)
            normalized_event = self.data_normalizer.normalize_event(cleaned_event)
            features = self.feature_engineer.extract_features(normalized_event)
            input_tensor = self._preprocess_data(features)
            
            # Get predictions from ensemble
            if self.config['enable_ensemble']:
                predictions = await self.ensemble_manager.get_ensemble_predictions(input_tensor)
            else:
                predictions = await self.model_manager.predict(input_tensor)
            
            # Analyze threats and anomalies
            threat_analysis = await self._analyze_threats(predictions)
            anomaly_analysis = await self._analyze_anomalies(predictions)
            
            # Update knowledge graph with new patterns
            self.knowledge_graph.update([{
                'event': normalized_event,
                'predictions': predictions,
                'timestamp': event_data.get('timestamp')
            }])
            
            return {
                'threat_analysis': threat_analysis,
                'anomaly_analysis': anomaly_analysis,
                'risk_score': self._calculate_risk_score(threat_analysis, anomaly_analysis),
                'confidence': predictions.get('confidence', 0.0),
                'recommendations': self._generate_recommendations(
                    threat_analysis, 
                    anomaly_analysis
                )
            }
        except Exception as e:
            self.logger.error(f"Analysis error: {e}")
            return {'error': str(e)}

    async def evaluate_model(self) -> Dict[str, Any]:
        """Evaluate model performance"""
        try:
            test_loader = self.dataset_handler.get_test_loader()
            evaluation_results = await self.evaluator.evaluate(test_loader)
            return evaluation_results
        except Exception as e:
            self.logger.error(f"Evaluation error: {e}")
            return {'error': str(e)}

    async def process_feedback(self, feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process feedback and update models"""
        try:
            feedback_results = await self.feedback_manager.process_feedback(feedback_data)
            
            if self.config['enable_adaptive_learning']:
                await self.adaptive_learner.adapt(feedback_data)
            
            return feedback_results
        except Exception as e:
            self.logger.error(f"Feedback processing error: {e}")
            return {'error': str(e)}

    def _preprocess_data(self, features: Dict[str, Any]) -> torch.Tensor:
        """Convert features to tensor format"""
        try:
            feature_vector = self.feature_engineer.combine_features(features)
            return torch.FloatTensor(feature_vector).unsqueeze(0).to(self.device)
        except Exception as e:
            self.logger.error(f"Preprocessing error: {e}")
            return torch.randn(1, 512).to(self.device)

    async def _analyze_threats(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze threat predictions"""
        return {
            'is_threat': predictions.get('threat_score', 0.0) > 0.5,
            'threat_score': predictions.get('threat_score', 0.0),
            'threat_type': predictions.get('threat_type', 'unknown'),
            'confidence': predictions.get('threat_confidence', 0.0)
        }

    async def _analyze_anomalies(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze anomaly predictions"""
        return {
            'is_anomaly': predictions.get('anomaly_score', 0.0) > 0.5,
            'anomaly_score': predictions.get('anomaly_score', 0.0),
            'anomaly_type': predictions.get('anomaly_type', 'unknown'),
            'confidence': predictions.get('anomaly_confidence', 0.0)
        }

    def _calculate_risk_score(self, threat_analysis: Dict[str, Any], 
                            anomaly_analysis: Dict[str, Any]) -> float:
        """Calculate overall risk score"""
        threat_score = threat_analysis.get('threat_score', 0.0)
        anomaly_score = anomaly_analysis.get('anomaly_score', 0.0)
        return (threat_score * 0.6 + anomaly_score * 0.4) * 100

    def _generate_recommendations(self, threat_analysis: Dict[str, Any],
                                anomaly_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        if threat_analysis.get('is_threat', False):
            recommendations.append(f"Detected threat of type {threat_analysis['threat_type']}")
            
        if anomaly_analysis.get('is_anomaly', False):
            recommendations.append(f"Detected anomaly of type {anomaly_analysis['anomaly_type']}")
            
        return recommendations