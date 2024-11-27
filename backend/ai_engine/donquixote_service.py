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
from .prediction.prediction_service import PredictionService
import logging
import psutil
import numpy as np
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

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
        
        # Initialize models directly
        self.anomaly_detector = None
        self.variational_autoencoder = None
        self.threat_detector = None
        
        # Add advanced ML Components from PredictionService
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        
        # Enhanced Temporal Analysis
        self.temporal_patterns = {
            'hourly': {},
            'daily': {},
            'weekly': {}
        }
        self.behavior_sequences = []
        self.sequence_length = 10
        
        # Advanced Behavioral Indicators
        self.behavior_profiles = {}
        self.anomaly_patterns = set()
        self.threat_chains = []
        
        # Dynamic Thresholds
        self.threshold_history = []
        self.adaptive_threshold = self.config.get('prediction_threshold', 0.75)
        
        # Enhanced prediction tracking
        self.pattern_memory = {}
        self.behavioral_patterns = []
        self.threat_signatures = set()
        
        try:
            # Initialize core components with system checks
            self.system_specs = self._check_system_specs()
            self._validate_system_requirements()
            
            # Initialize models based on system capabilities
            self._initialize_base_models()
            
            # Initialize existing components
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
            
            # Initialize prediction service with enhanced capabilities
            self.prediction_service = PredictionService(self.model_manager, self.config)
            
            self.logger.info("Models initialized successfully")
            self.logger.info("Enhanced prediction capabilities initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing models: {e}")
            raise

    def _check_system_specs(self) -> Dict[str, Any]:
        """Check system specifications"""
        return {
            'cpu_count': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'cuda_available': torch.cuda.is_available(),
            'cuda_devices': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'cuda_memory_gb': (torch.cuda.get_device_properties(0).total_memory / (1024**3)
                             if torch.cuda.is_available() else 0)
        }

    def _validate_system_requirements(self):
        """Validate system meets minimum requirements"""
        minimum_requirements = {
            'cpu_count': 2,
            'memory_gb': 4.0
        }
        
        if self.system_specs['cpu_count'] < minimum_requirements['cpu_count']:
            self.logger.warning("System below minimum CPU requirement. Some features may be disabled.")
        if self.system_specs['memory_gb'] < minimum_requirements['memory_gb']:
            self.logger.warning("System below minimum memory requirement. Some features may be disabled.")

    def _initialize_models(self):
        """Initialize base models with fallback options"""
        try:
            # Initialize models based on system capabilities
            if self.system_specs['memory_gb'] >= 4.0:  # Minimum memory requirement
                # Initialize Anomaly Detector
                self.anomaly_detector = AnomalyDetector(
                    input_dim=512,
                    hidden_dim=256
                ).to(self.device)
                
                # Initialize VAE if more memory available
                if self.system_specs['memory_gb'] >= 8.0:
                    self.variational_autoencoder = VariationalAutoencoder(
                        input_dim=512,
                        hidden_dim=256,
                        latent_dim=64
                    ).to(self.device)

                self.anomaly_detector = AnomalyDetector(
                    input_dim=512,
                    hidden_dim=256
                ).to(self.device)
                self.logger.info("Anomaly Detector initialized")

                self.variational_autoencoder = VariationalAutoencoder(
                    input_dim=512,
                    hidden_dim=256,
                    latent_dim=64
                ).to(self.device)
                self.logger.info("VAE initialized")
                
                # Initialize Threat Detector
                self.threat_detector = ThreatDetector(
                    input_dim=512,
                    hidden_dim=256,
                    num_classes=2
                ).to(self.device)
                
                self.logger.info("All models initialized successfully")
            else:
                self.logger.warning("Limited memory available. Running in minimal mode.")
                # Initialize only essential model
                self.anomaly_detector = AnomalyDetector(
                    input_dim=256,  # Reduced dimensions
                    hidden_dim=128
                ).to(self.device)
                
        except Exception as e:
            self.logger.error(f"Error initializing base models: {e}")
            # Set failed models to None but don't stop initialization
            if not hasattr(self, 'anomaly_detector'):
                self.anomaly_detector = None
            if not hasattr(self, 'variational_autoencoder'):
                self.variational_autoencoder = None
            if not hasattr(self, 'threat_detector'):
                self.threat_detector = None
            raise

    async def get_service_status(self) -> Dict[str, Any]:
        """Get the status of the service"""
        return {
            'status': 'activate',
            'device': str(self.device),
            'system_specs': self._check_system_specs(),
            'active_models': self._get_active_models(),
            'config': {
                'model_version': self.config['model_version'],
                'enable_ensemble': self.config['enable_ensemble'],
                'enable_adaptive_learning': self.config['enable_adaptive_learning']
            }
        }

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
        """Enhanced event analysis with advanced prediction capabilities"""
        try:
            # Get base prediction
            prediction = await self.prediction_service.predict_threat(event_data)
            
            # Extract advanced features
            temporal_features = self._extract_temporal_features(event_data)
            behavioral_features = self._extract_behavioral_features(event_data)
            
            # Combine with existing analysis
            analysis_result = await super().analyze_event(event_data)
            
            # Enhanced analysis with advanced predictions
            enhanced_result = {
                **analysis_result,
                'prediction': prediction,
                'temporal_analysis': {
                    'features': temporal_features,
                    'patterns': self.temporal_patterns
                },
                'behavioral_analysis': {
                    'features': behavioral_features,
                    'profiles': self.behavior_profiles
                },
                'threat_chain': self._identify_threat_chain(event_data),
                'combined_risk_score': self._calculate_enhanced_risk(
                    analysis_result.get('risk_score', 0),
                    prediction.get('threat_score', 0),
                    temporal_features,
                    behavioral_features
                )
            }

            # Update pattern memory and behavioral analysis
            self._update_pattern_memory(event_data, enhanced_result)
            
            return enhanced_result

        except Exception as e:
            self.logger.error(f"Error in enhanced event analysis: {e}")
            return {
                'error': str(e),
                'status': 'failed'
            }

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

    def _get_active_models(self) -> List[str]:
        """Get list of currently active models"""
        active_models = []
        if self.anomaly_detector is not None:
            active_models.append('anomaly_detector')
        if self.variational_autoencoder is not None:
            active_models.append('variational_autoencoder')
        if self.threat_detector is not None:
            active_models.append('threat_detector')
        return active_models

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

    def _calculate_combined_risk(self, analysis_score: float, prediction_score: float) -> float:
        """Calculate combined risk score from analysis and prediction"""
        # Weight prediction more heavily for real-time response
        return (analysis_score * 0.4) + (prediction_score * 0.6)

    def _calculate_enhanced_risk(
        self,
        analysis_score: float,
        prediction_score: float,
        temporal_features: np.ndarray,
        behavioral_features: np.ndarray
    ) -> float:
        """Calculate enhanced risk score with multiple factors"""
        # Base risk calculation
        base_risk = (analysis_score * 0.4) + (prediction_score * 0.6)
        
        # Add temporal risk factor
        temporal_risk = np.mean(temporal_features) * 0.2
        
        # Add behavioral risk factor
        behavioral_risk = np.mean(behavioral_features) * 0.2
        
        # Combine all risk factors
        total_risk = (base_risk * 0.6) + (temporal_risk * 0.2) + (behavioral_risk * 0.2)
        
        return min(total_risk * 100, 100.0)

    def _extract_temporal_features(self, event_data: Dict[str, Any]) -> np.ndarray:
        """Extract temporal features from event data"""
        timestamp = event_data.get('timestamp', datetime.now())
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
            
        features = [
            timestamp.hour / 24.0,
            timestamp.weekday() / 7.0,
            timestamp.day / 31.0,
            self._get_temporal_density(timestamp),
            self._get_periodic_score(timestamp),
            self._calculate_temporal_risk(timestamp)
        ]
        
        return np.array(features)

    def _extract_behavioral_features(self, event_data: Dict[str, Any]) -> np.ndarray:
        """Extract behavioral features from event data"""
        user_id = event_data.get('user_id', 'unknown')
        
        features = [
            self._get_user_risk_score(user_id),
            self._get_behavior_deviation_score(event_data),
            self._get_sequence_similarity(event_data),
            self._calculate_threat_chain_probability(event_data),
            self._get_anomaly_correlation_score(event_data)
        ]
        
        return np.array(features)

    def _identify_threat_chain(self, event_data: Dict[str, Any]) -> str:
        """Identify threat chain in event"""
        threat_chain = []
        
        # Check various threat indicators
        if event_data.get('ip_address') and self._check_ip_reputation(event_data['ip_address']) > 0.7:
            threat_chain.append('initial_access')
            
        # Add other threat chain checks from PredictionService
        
        return ' -> '.join(threat_chain) if threat_chain else 'unknown'