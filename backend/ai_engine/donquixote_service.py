from pkgutil import get_data
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
from .collectors.windows_collector import WindowsEventCollector
from .collectors.linux_collector import LinuxLogCollector
from .collectors.network_collector import NetworkCollector
from .collectors.cloud_collector import CloudCollector
from .collectors.macos_collector import MacOSCollector
import os
from pathlib import Path
import yaml

def load_default_config() -> Dict[str, Any]:
    """Load default configuration from yaml file"""
    try:
        config_dir = Path(__file__).parent.parent / 'config'
        default_config_path = config_dir / 'default_config.yaml'
        
        if default_config_path.exists():
            with open(default_config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}
    except Exception as e:
        logging.error(f"Error loading default config: {e}")
        return {}

class DonquixoteService:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load default config first
        self.default_config = load_default_config()
        
        # Tentukan path absolut ke direktori config
        config_dir = Path(__file__).parent.parent / 'config'
        default_config_path = config_dir / 'default_config.yaml'
        ai_config_path = config_dir / 'ai_settings.yaml'

        # Periksa apakah file ada
        if not ai_config_path.exists():
            self.logger.warning(f"AI config file not found at {ai_config_path}, using default settings")
            # Gunakan default_config atau lakukan tindakan lain
            self.ai_config = self.default_config  # Gunakan default_config yang sudah didefinisikan
        else:
            # Muat konfigurasi dari file
            with open(ai_config_path, 'r') as f:
                self.ai_config = yaml.safe_load(f)
        
        # Default config if none provided
        default_config = {
            'experiment_name': 'default_experiment',
            'model_version': '1.0.0',
            'batch_size': 32,
            'learning_rate': 1e-4,
            'num_epochs': 10,
            'model_dir': 'models/',
            'log_dir': 'logs/',
            'data_dir': 'data/',
            'enable_ensemble': True,
            'enable_adaptive_learning': True,
            'anomaly_detector': {
                'input_dim': 512,
                'hidden_dim': 256
            },
            'vae': {
                'input_dim': 512,
                'hidden_dims': [256, 128],
                'latent_dim': 64
            },
            'threat_detector': {
                'input_dim': 512,
                'hidden_dim': 256,
                'num_classes': 2
            }
        }
        self.config = {**default_config, **self.ai_config, **(config or {})}
        
        try:
            # Initialize model manager first
            self.model_manager = ModelManager(self.config)
            
            # Get models from model manager
            self.anomaly_detector = self.model_manager.get_model('anomaly_detector')
            self.variational_autoencoder = self.model_manager.get_model('vae')
            self.threat_detector = self.model_manager.get_model('threat_detector')
            
            if not all([self.anomaly_detector, self.variational_autoencoder, self.threat_detector]):
                self.logger.warning("Some models failed to initialize")
            
            # Initialize other components
            self.evaluator = ModelEvaluator(self.model_manager, self.config)
            self.knowledge_graph = KnowledgeGraph()
            self.ensemble_manager = AIEnsembleManager(self.model_manager)
            
            # Initialize data processing components
            self.data_handler = CyberSecurityDataHandler(self.config)
            self.data_cleaner = DataCleaner()
            self.data_normalizer = DataNormalizer()
            self.feature_engineer = FeatureEngineer()
            
            # Initialize training and feedback components
            self.training_manager = TrainingManager(
                config=self.config,
                experiment_name=self.config.get('experiment_name', 'default_experiment')
            )
            self.adaptive_learner = AdaptiveLearner(
                models={
                    'anomaly_detector': self.anomaly_detector,
                    'threat_detector': self.threat_detector
                },
                config=self.config
            )
            self.feedback_manager = FeedbackManager(self.config)
            self.prediction_service = PredictionService(self.model_manager, self.config)
            
            # Initialize ML components
            self.isolation_forest = IsolationForest()
            self.standard_scaler = StandardScaler()
            
            # Initialize collectors
            self.collectors = {
                'windows': WindowsEventCollector(self.config),
                'linux': LinuxLogCollector(self.config),
                'network': NetworkCollector(self.config),
                'cloud': CloudCollector(self.config),
                'macos': MacOSCollector()
            }
            self.collection_status = {collector: False for collector in self.collectors}

             # Initialize model manager with config
            self.model_manager = ModelManager(self.default_config)
    
            # Initialize models
            self.anomaly_detector = self.model_manager.get_model('anomaly_detector')
            self.variational_autoencoder = self.model_manager.get_model('vae')
            self.threat_detector = self.model_manager.get_model('threat_detector')
    
            # Verify model initialization
            if not all([self.anomaly_detector, self.variational_autoencoder, self.threat_detector]):
                self.logger.warning("Some models failed to initialize")
            else:
                self.logger.info("All models initialized successfully")
        
            # Move models to appropriate device
            if self.anomaly_detector:
                self.anomaly_detector.to(self.device)
            if self.variational_autoencoder:
                self.variational_autoencoder.to(self.device)
            if self.threat_detector:
                self.threat_detector.to(self.device)
            
            # Setup wandb tracking
            self.training_manager.setup_experiment_tracking()
            
            self.logger.info("All components initialized successfully")
        except Exception as e:
            self.logger.error(f"Error in initialization: {e}")
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
            self.model_manager = ModelManager(self.config)
            self.anomaly_detector = self.model_manager.get_model('anomaly_detector')
            self.variational_autoencoder = self.model_manager.get_model('vae')
            self.threat_detector = self.model_manager.get_model('threat_detector')
            
            if not all([self.anomaly_detector, self.variational_autoencoder, self.threat_detector]):
                self.logger.warning("Some models failed to initialize")
            else:
                self.logger.info("All models initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing models: {e}")
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
            # Process feedback through feedback manager
            feedback_results = await self.feedback_manager.process_feedback(feedback_data)
            
            # Update knowledge graph with feedback
            self.knowledge_graph.update(
                patterns=[],  # Empty patterns list since we're only incorporating feedback
                feedback=feedback_data
            )
            
            # Apply adaptive learning if enabled
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
        
        # Add temporal risk factors
        # Calculate time-based risk components
        hour_risk = self._calculate_hour_risk(temporal_features[0])  # Higher risk during off-hours
        day_risk = self._calculate_day_risk(temporal_features[1])    # Higher risk on weekends
        density_risk = temporal_features[3]  # Event density risk from temporal features
        
        # Calculate velocity and acceleration of events
        event_velocity = self._calculate_event_velocity(temporal_features)
        event_acceleration = self._calculate_event_acceleration(temporal_features)
        
        # Analyze periodic patterns
        periodic_deviation = self._analyze_periodic_deviation(temporal_features)
        seasonal_factor = self._calculate_seasonal_factor(temporal_features)
        
        # Combine temporal risk components with weights
        temporal_risk = (
            hour_risk * 0.25 +
            day_risk * 0.15 +
            density_risk * 0.2 +
            event_velocity * 0.15 +
            event_acceleration * 0.1 +
            periodic_deviation * 0.1 +
            seasonal_factor * 0.05
        )
        temporal_risk = np.mean(temporal_features) * 0.2
        
        # Add behavioral risk factors
        # Calculate user behavior risk
        user_risk = self._get_user_risk_score(get_data.get('user_id', 'unknown'))
        behavior_deviation = self._get_behavior_deviation_score(get_data)
        sequence_similarity = self._get_sequence_similarity(get_data)
        
        # Calculate threat chain probabilities
        threat_chain_prob = self._calculate_threat_chain_probability(get_data)
        anomaly_correlation = self._get_anomaly_correlation_score(get_data)
        
        # Analyze access patterns
        access_pattern_risk = self._analyze_access_patterns(behavioral_features)
        privilege_escalation_risk = self._detect_privilege_escalation(behavioral_features)
        lateral_movement_risk = self._detect_lateral_movement(behavioral_features)
        
        # Combine behavioral risk components with weights
        behavioral_risk = (
            user_risk * 0.25 +
            behavior_deviation * 0.2 + 
            sequence_similarity * 0.15 +
            threat_chain_prob * 0.15 +
            anomaly_correlation * 0.1 +
            access_pattern_risk * 0.05 +
            privilege_escalation_risk * 0.05 +
            lateral_movement_risk * 0.05
        )
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
            
        # Check for credential access attempts
        if event_data.get('event_type') in ['failed_login', 'brute_force', 'password_spray']:
            threat_chain.append('credential_access')
            
        # Check for lateral movement indicators
        if (event_data.get('source_ip') != event_data.get('destination_ip') and
            self._get_network_movement_score(event_data) > 0.6):
            threat_chain.append('lateral_movement')
            
        # Check for privilege escalation
        if event_data.get('privilege_level', '').lower() in ['admin', 'system', 'root']:
            if self._get_privilege_anomaly_score(event_data) > 0.7:
                threat_chain.append('privilege_escalation')
                
        # Check for data exfiltration patterns
        if (event_data.get('bytes_transferred', 0) > self.config.get('exfil_threshold', 1000000) and
            self._get_data_transfer_anomaly(event_data) > 0.8):
            threat_chain.append('data_exfiltration')
            
        # Check for persistence mechanisms
        if event_data.get('event_type') in ['scheduled_task', 'registry_mod', 'startup_mod']:
            if self._get_persistence_score(event_data) > 0.65:
                threat_chain.append('persistence')
                
        # Check for defense evasion
        if event_data.get('event_type') in ['log_clear', 'av_disable', 'fw_disable']:
            threat_chain.append('defense_evasion')
            
        # Check for command and control activity
        if (self._check_c2_patterns(event_data) > 0.75 and
            self._get_connection_anomaly_score(event_data) > 0.7):
            threat_chain.append('command_and_control')
        
        return ' -> '.join(threat_chain) if threat_chain else 'unknown'

    def _update_knowledge_graph(self, event_data: Dict[str, Any], analysis_result: Dict[str, Any]):
        """Update knowledge graph with new event patterns"""
        try:
            # Extract patterns from event analysis
            patterns = [
                {
                    'centroid': torch.tensor(self._extract_pattern_features(event_data), dtype=torch.float32),
                    'frequency': 1,
                    'event_type': event_data.get('event_type', 'unknown'),
                    'severity': analysis_result.get('risk_score', 0),
                    'timestamp': datetime.now().isoformat()
                }
            ]
            
            # Update knowledge graph with new patterns
            self.knowledge_graph.update(patterns)
            
        except Exception as e:
            self.logger.error(f"Error updating knowledge graph: {e}")

    def _extract_pattern_features(self, event_data: Dict[str, Any]) -> np.ndarray:
        """Extract feature vector for knowledge graph patterns"""
        features = []
        
        # Add temporal features
        temporal_features = self._extract_temporal_features(event_data)
        features.extend(temporal_features)
        
        # Add behavioral features
        behavioral_features = self._extract_behavioral_features(event_data)
        features.extend(behavioral_features)
        
        # Add event-specific features
        event_features = [
            float(event_data.get('severity', 0)),
            float(self._calculate_event_risk(event_data)),
            float(event_data.get('priority', 0))
        ]
        features.extend(event_features)
        
        return np.array(features, dtype=np.float32)

    def _calculate_event_risk(self, event_data: Dict[str, Any]) -> float:
        """Calculate basic risk score for an event"""
        base_risk = event_data.get('severity', 0) * 0.6
        priority_factor = event_data.get('priority', 0) * 0.4
        return min(base_risk + priority_factor, 1.0)

    async def start_collectors(self, collector_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """Start specified collectors or all collectors if none specified"""
        try:
            if not collector_types:
                collector_types = list(self.collectors.keys())

            for collector_type in collector_types:
                if collector_type in self.collectors:
                    self.collection_status[collector_type] = True
                    await self.collectors[collector_type].start_collection()

            return {
                'status': 'success',
                'active_collectors': [
                    collector for collector, status in self.collection_status.items()
                    if status
                ]
            }
        except Exception as e:
            self.logger.error(f"Error starting collectors: {e}")
            return {'error': str(e)}