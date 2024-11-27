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
import psutil

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
            
            self.logger.info("Models initialized successfully")
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
        """Analyze a single event using available models"""
        try:
            input_tensor = self._preprocess_data(event_data)
            results = {}
            
            # Use Anomaly Detector if available
            if self.anomaly_detector is not None:
                anomaly_score = self.anomaly_detector(input_tensor)
                results['anomaly_score'] = float(anomaly_score.mean().item())
            
            # Use VAE if available
            if self.variational_autoencoder is not None:
                recon, mu, log_var = self.variational_autoencoder(input_tensor)
                vae_score = self.variational_autoencoder.loss_function(
                    recon, input_tensor, mu, log_var
                )
                results['vae_score'] = float(vae_score.mean().item())
            
            # Use Threat Detector if available
            if self.threat_detector is not None:
                threat_output = self.threat_detector(input_tensor)
                results['threat_score'] = float(threat_output.mean().item())
            
            return {
                'status': 'success',
                'results': results,
                'system_info': {
                    'active_models': self._get_active_models(),
                    'system_specs': self.system_specs
                }
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