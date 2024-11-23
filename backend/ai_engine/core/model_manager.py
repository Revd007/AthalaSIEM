from typing import Dict, Any, Optional, List, Tuple
import torch
import torch.nn as nn
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json
import numpy as np
from ..models.threat_detections import ThreatDetector
from ..models.anomaly_detector import AnomalyDetector, VariationalAutoencoder
from ..training.adaptive_learner import AdaptiveLearner
import asyncio
import psutil
from torch.utils.data import DataLoader
import yaml

class AdaptiveLearner:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.learning_rate = config.get('training', {}).get('learning_rate', 2e-5)
        self.batch_size = config.get('training', {}).get('batch_size', 32)
        self.adaptive_threshold = config.get('inference', {}).get('threshold_confidence', 0.85)
        
    async def update_model(self, model: nn.Module, new_data: Dict[str, torch.Tensor]) -> None:
        """Update model with new data using adaptive learning"""
        try:
            # Validate input data
            if not isinstance(new_data, dict) or not new_data:
                raise ValueError("Invalid training data format")
                
            # Prepare optimizer
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.learning_rate
            )
            
            # Update model in training mode and enable gradient computation
            model.train()
            torch.set_grad_enabled(True)
            model.train()
            
            # Compute loss and update
            loss = model(new_data)
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                self.config.get('training', {}).get('gradient_clip', 1.0)
            )
            
            optimizer.step()
            optimizer.zero_grad()
            
            self.logger.info(f"Model updated successfully with loss: {loss.item():.4f}")
            
        except Exception as e:
            self.logger.error(f"Error in adaptive learning update: {e}")
            raise
            
    async def evaluate_performance(self, model: nn.Module, eval_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Evaluate model performance after update"""
        try:
            model.eval()
            with torch.no_grad():
                metrics = {
                    'accuracy': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0
                }
                
                # Compute evaluation metrics
                outputs = model(eval_data)
                
                # Update metrics based on outputs
                # Calculate accuracy
                predictions = (outputs['threat_score'] > 0.5).float()
                correct = (predictions == eval_data['labels']).float()
                metrics['accuracy'] = correct.mean().item()
                
                # Calculate precision, recall, and F1 score
                true_positives = (predictions * eval_data['labels']).sum()
                predicted_positives = predictions.sum()
                actual_positives = eval_data['labels'].sum()
                
                metrics['precision'] = (true_positives / predicted_positives).item() if predicted_positives > 0 else 0.0
                metrics['recall'] = (true_positives / actual_positives).item() if actual_positives > 0 else 0.0
                
                if metrics['precision'] + metrics['recall'] > 0:
                    metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
                # Calculate additional model-specific metrics
                if isinstance(model, ThreatDetector):
                    # For threat detector, calculate pattern and behavior accuracies
                    pattern_acc = (outputs['patterns'].argmax(-1) == eval_data['pattern_labels']).float().mean()
                    behavior_acc = (outputs['behaviors'].argmax(-1) == eval_data['behavior_labels']).float().mean()
                    metrics['pattern_accuracy'] = pattern_acc.item()
                    metrics['behavior_accuracy'] = behavior_acc.item()
                
                return metrics
                
        except Exception as e:
            self.logger.error(f"Error in performance evaluation: {e}")
            return {}



class ModelManager:
    def __init__(self, device: str = None):
        self.logger = logging.getLogger(__name__)
        
        # Set device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load config
        try:
            config_path = Path(__file__).parent.parent / "config" / "default_config.yaml"
            if not config_path.exists():
                self.logger.warning("Config file not found, creating default config")
                self._create_default_config(config_path)
                
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
                
        except Exception as e:
            self.logger.error(f"Error loading config: {str(e)}")
            raise
            
        # Initialize models
        self._initialize_models()

    async def process_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Process security event through AI models"""
        try:
            # Preprocess event data
            features = self._extract_features(event)
            
            # Run threat detection
            threat_result = await self._detect_threats(features)
            
            # Run anomaly detection
            anomaly_result = await self._detect_anomalies(features)
            
            # Combine and analyze results
            analysis = self._combine_analysis(threat_result, anomaly_result)
            
            # Update adaptive learning
            await self.adaptive_learner.learn_from_event(event, analysis)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error processing event: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    async def _detect_threats(self, features: torch.Tensor) -> Dict[str, Any]:
        """Detect threats using threat detection model"""
        try:
            if 'threat_detector' not in self.models:
                return {
                    'threat_score': 0.0,
                    'patterns': [],
                    'behaviors': [],
                    'confidence': 0.0
                }
            
            with torch.no_grad():
                model = self.models['threat_detector']
                outputs = model(features)
                
                return {
                    'threat_score': float(outputs['threat_score'].mean()),
                    'patterns': outputs['patterns'].tolist(),
                    'behaviors': outputs['behaviors'].tolist(),
                    'confidence': float(outputs['threat_score'].max())
                }
        except Exception as e:
            self.logger.error(f"Error in threat detection: {e}")
            return {
                'threat_score': 0.0,
                'patterns': [],
                'behaviors': [],
                'confidence': 0.0
            }

    async def _detect_anomalies(self, features: torch.Tensor) -> Dict[str, Any]:
        """Detect anomalies using VAE model"""
        with torch.no_grad():
            anomaly_model = self.models['anomaly']
            reconstruction, mu, logvar = anomaly_model(features)
            
            anomaly_score = anomaly_model.compute_anomaly_score(
                features, reconstruction, mu, logvar
            )
            
            return {
                'is_anomaly': anomaly_score > anomaly_model.threshold,
                'anomaly_score': float(anomaly_score),
                'reconstruction_error': float(torch.mean((features - reconstruction)**2))
            }

    async def detect_anomalies(self, features: torch.Tensor) -> Dict[str, Any]:
        """Detect anomalies using anomaly detection model"""
        try:
            if 'anomaly_detector' not in self.models:
                return {'is_anomaly': False, 'anomaly_score': 0.0, 'confidence': 0.0}
                
            with torch.no_grad():
                model = self.models['anomaly_detector']
                outputs = model(features)
                reconstruction_error = torch.mean((features - outputs['reconstruction'])**2)
                
                return {
                    'is_anomaly': reconstruction_error > model.threshold,
                    'anomaly_score': float(reconstruction_error),
                    'confidence': float(outputs.get('confidence', 0.8))
                }
        except Exception as e:
            self.logger.error(f"Error in anomaly detection: {e}")
            return {'is_anomaly': False, 'anomaly_score': 0.0, 'confidence': 0.0}

    async def analyze_threats(self, features: torch.Tensor) -> Dict[str, Any]:
        """Analyze threats using threat detection model"""
        try:
            if 'threat_detector' not in self.models:
                return {'is_threat': False, 'threat_score': 0.0, 'confidence': 0.0}
                
            with torch.no_grad():
                model = self.models['threat_detector']
                outputs = model(features)
                threat_score = outputs.get('threat_score', 0.0)
                
                return {
                    'is_threat': threat_score > model.threshold,
                    'threat_score': float(threat_score),
                    'indicators': outputs.get('indicators', []),
                    'confidence': float(outputs.get('confidence', 0.8))
                }
        except Exception as e:
            self.logger.error(f"Error in threat analysis: {e}")
            return {'is_threat': False, 'threat_score': 0.0, 'confidence': 0.0}

    async def auto_train_and_evaluate(self, data_interval: int = 3600) -> None:
        """
        Auto training dan evaluasi yang berjalan secara periodik
        data_interval: interval dalam detik (default 1 jam)
        """
        self.logger.info("Starting automatic training and evaluation cycle")
        
        while True:
            try:
                # Dapatkan data baru
                new_data = await self._get_new_training_data()
                if not new_data:
                    self.logger.info("No new data available, waiting...")
                    await asyncio.sleep(data_interval)
                    continue

                # Bagi data untuk training dan evaluasi
                train_loader, eval_loader = self._prepare_data_loaders(new_data)

                # Training
                self.logger.info("Starting training cycle")
                for model_name, model in self.models.items():
                    try:
                        # Training phase
                        metrics = await model.train_epoch(train_loader, self.device)
                        self.logger.info(f"Training metrics for {model_name}: {metrics}")

                        # Evaluation phase
                        eval_metrics = await model.evaluate(eval_loader, self.device)
                        self.logger.info(f"Evaluation metrics for {model_name}: {eval_metrics}")

                        # Save checkpoint jika performa meningkat
                        if self._should_save_checkpoint(eval_metrics):
                            checkpoint_path = f"checkpoints/{model_name}_{datetime.now().strftime('%Y%m%d_%H%M')}.pt"
                            await model.save_checkpoint(
                                checkpoint_path, 
                                metrics=metrics, 
                                eval_metrics=eval_metrics
                            )
                            self.logger.info(f"Saved checkpoint to {checkpoint_path}")

                        # Analisis attention untuk monitoring
                        sample_input = next(iter(eval_loader))[0]
                        attention_maps = await model.get_attention_maps(sample_input)
                        await self._analyze_attention_patterns(attention_maps, model_name)

                        # Update model health metrics
                        await self._update_health_metrics(model_name, eval_metrics)

                    except Exception as e:
                        self.logger.error(f"Error in training cycle for {model_name}: {e}")
                        continue

                # Tunggu sampai interval berikutnya
                await asyncio.sleep(data_interval)

            except Exception as e:
                self.logger.error(f"Error in auto training cycle: {e}")
                await asyncio.sleep(60)  # Tunggu 1 menit jika ada error

    async def _get_new_training_data(self) -> Optional[Dict[str, torch.Tensor]]:
        """Dapatkan data training baru dari sistem"""
        try:
            # Cek apakah ada data baru yang tersedia
            if not self.event_collector.has_new_events():
                self.logger.info("No new training data available")
                return None
                
            # Ambil data dari event collector dengan batasan waktu
            cutoff_time = datetime.now() - timedelta(hours=24)
            raw_data = await self.event_collector.get_events_since(cutoff_time)
            
            # Validasi data yang diterima
            if not raw_data or len(raw_data) < self.config.min_samples:
                self.logger.warning(f"Insufficient data samples: {len(raw_data) if raw_data else 0}")
                return None
                
            # Pre-process data mentah
            processed_data = self.data_preprocessor.transform(raw_data)
            
            # Validasi hasil preprocessing
            if not self._validate_processed_data(processed_data):
                self.logger.error("Data validation failed after preprocessing")
                return None
            # Contoh: mengambil dari database atau sistem logging
            new_events = await self.event_collector.get_recent_events()
            if not new_events:
                return None

            # Proses data mentah menjadi tensor
            processed_data = self.feature_engineer.process_events(new_events)
            return processed_data

        except Exception as e:
            self.logger.error(f"Error getting new training data: {e}")
            return None

    def _prepare_data_loaders(self, data: Dict[str, torch.Tensor]) -> Tuple[DataLoader, DataLoader]:
        """Siapkan data loaders untuk training dan evaluasi"""
        try:
            # Bagi data 80/20
            split_idx = int(len(data) * 0.8)
            
            train_data = {k: v[:split_idx] for k, v in data.items()}
            eval_data = {k: v[split_idx:] for k, v in data.items()}

            train_loader = DataLoader(
                train_data, 
                batch_size=self.config.batch_size,
                shuffle=True
            )
            eval_loader = DataLoader(
                eval_data,
                batch_size=self.config.batch_size,
                shuffle=False
            )

            return train_loader, eval_loader

        except Exception as e:
            self.logger.error(f"Error preparing data loaders: {e}")
            raise

    def _should_save_checkpoint(self, eval_metrics: Dict[str, float]) -> bool:
        """Tentukan apakah perlu menyimpan checkpoint baru"""
        if not hasattr(self, 'best_metrics'):
            self.best_metrics = eval_metrics
            return True

        # Bandingkan dengan metrics terbaik sebelumnya
        improvement = (
            eval_metrics['accuracy'] > self.best_metrics['accuracy'] or
            eval_metrics['f1_score'] > self.best_metrics['f1_score']
        )

        if improvement:
            self.best_metrics = eval_metrics
            return True

        return False

    async def _analyze_attention_patterns(
        self, 
        attention_maps: List[torch.Tensor], 
        model_name: str
    ) -> None:
        """Analisis pola attention untuk monitoring"""
        try:
            # Hitung statistik attention
            attention_stats = {
                'mean_attention': torch.mean(attention_maps[-1]).item(),
                'max_attention': torch.max(attention_maps[-1]).item(),
                'attention_entropy': self._calculate_attention_entropy(attention_maps[-1])
            }

            # Log statistik
            self.logger.info(f"Attention statistics for {model_name}: {attention_stats}")

            # Simpan untuk monitoring
            await self._save_attention_stats(model_name, attention_stats)

        except Exception as e:
            self.logger.error(f"Error analyzing attention patterns: {e}")

    async def _update_health_metrics(
        self, 
        model_name: str, 
        metrics: Dict[str, float]
    ) -> None:
        """Update metrics kesehatan model"""
        try:
            health_metrics = {
                'model_name': model_name,
                'timestamp': datetime.now().isoformat(),
                'performance_metrics': metrics,
                'resource_usage': {
                    'memory': psutil.Process().memory_info().rss / 1024 / 1024,  # MB
                    'cpu': psutil.Process().cpu_percent()
                }
            }

            # Simpan metrics
            await self._save_health_metrics(health_metrics)

        except Exception as e:
            self.logger.error(f"Error updating health metrics: {e}")

    def _init_models(self):
        """Initialize all required models"""
        try:
            # Initialize anomaly detector
            anomaly_config = self.config.get('models', {}).get('anomaly_detector', {})
            self.models['anomaly_detector'] = VariationalAutoencoder(
                input_dim=anomaly_config.get('input_dim', 256),
                hidden_dims=anomaly_config.get('hidden_dims', [128, 64]),
                latent_dim=anomaly_config.get('latent_dim', 32)
            )
            
            # Initialize threat detector with proper config access
            threat_config = self.config.get('models', {}).get('threat_detector', {})
            threat_detector_config = {
                'embedding_dim': threat_config.get('embedding_dim', 768),
                'num_heads': threat_config.get('num_heads', 8),
                'num_layers': threat_config.get('num_layers', 2),
                'num_patterns': threat_config.get('num_patterns', 10),
                'num_behaviors': threat_config.get('num_behaviors', 5)
            }
            self.models['threat_detector'] = ThreatDetector(
                type('Config', (), threat_detector_config)  # Convert dict to object
            )
            
            # Move models to appropriate device
            device = torch.device('cuda' if torch.cuda.is_available() and 
                                self.config.get('model_settings', {}).get('use_gpu', True) 
                                else 'cpu')
            
            for model_name, model in self.models.items():
                self.models[model_name] = model.to(device)
            
            self.logger.info(f"Models initialized successfully on device: {device}")
            
        except Exception as e:
            self.logger.error(f"Error initializing models: {e}")
            raise

    def _initialize_models(self):
        try:
            # Load config
            config_path = Path(__file__).parent.parent / "config" / "default_config.yaml"
            with open(config_path) as f:
                config = yaml.safe_load(f)
                
            # Initialize models with config
            self.threat_detector = ThreatDetector(config['model_settings']['threat_detector'])
            self.anomaly_detector = AnomalyDetector(config['model_settings']['anomaly_detector'])
            
            # Move models to device
            self.threat_detector.to(self.device)
            self.anomaly_detector.to(self.device)
            
        except Exception as e:
            self.logger.error(f"Error initializing models: {str(e)}")
            raise

    def _create_default_config(self, config_path: Path) -> None:
        """Create and save default configuration for model manager
    
        Args:
        config_path (Path): Path where config file should be saved
        """
        default_config = {
            "model_settings": {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 10,
            "validation_split": 0.2,
            "early_stopping_patience": 5,
            "threat_detector": {
                "embedding_dim": 768,
                "num_heads": 8,
                "num_layers": 2,
                "num_patterns": 10,
                "num_behaviors": 5
            },
            "anomaly_detector": {
                "input_dim": 256,
                "hidden_dims": [128, 64],
                "latent_dim": 32
            }
        },
        "training": {
            "enable_checkpoints": True,
            "checkpoint_frequency": 5,
            "enable_early_stopping": True,
            "max_epochs_without_improvement": 10
        },
        "optimization": {
            "enable_mixed_precision": True,
            "gradient_clipping": True,
            "max_gradient_norm": 1.0
        },
        "evaluation": {
            "metrics": ["accuracy", "precision", "recall", "f1"],
            "threshold": 0.5
        },
        "paths": {
            "model_dir": "models/",
                "checkpoint_dir": "checkpoints/",
                "log_dir": "logs/"
            }
        }
    
        # Create parent directories if they don't exist
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
    
        self.config = default_config