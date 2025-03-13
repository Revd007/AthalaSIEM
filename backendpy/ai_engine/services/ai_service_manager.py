import psutil
import torch
from typing import Dict, Any, Optional
import logging
from ..core.dataset_handler import CyberSecurityDataHandler
from ..core.model_manager import ModelManager
from ..core.evaluator import ModelEvaluator

class AIServiceManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.is_running = False
        self.evaluator = ModelEvaluator(None, config)
        
        # Set default AI settings if not provided
        if 'ai_settings' not in self.config:
            self.config['ai_settings'] = {
                'enabled': False,  # Default to disabled
                'subscription_type': 'development',  # or 'free', 'premium', etc.
                'features': {
                    'threat_detection': True,
                    'anomaly_detection': True,
                    'adaptive_learning': True
                }
            }
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize AI components based on config"""
        if self.config.get('ai_settings', {}).get('enabled', False):
            try:
                self._check_system_requirements()
                self.dataset_handler = CyberSecurityDataHandler(self.config)
                self.model_manager = ModelManager(self.config)
                self.evaluator = ModelEvaluator(self.model_manager, self.config)
                self.is_running = True
                self.logger.info("AI Service initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize AI components: {e}")
                self.is_running = False
        else:
            self.logger.info("AI Service is disabled")
            self.is_running = False
    
    def _check_system_requirements(self) -> bool:
        """Check if system meets requirements"""
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent()
        
        min_memory = self.config.get('resource_settings', {}).get('min_memory_mb', 2048)
        max_cpu = self.config.get('resource_settings', {}).get('max_cpu_usage', 80)
        
        if memory.available < min_memory * 1024 * 1024:
            raise Exception("Insufficient memory")
            
        if cpu_percent > max_cpu:
            raise Exception("CPU usage too high")
            
        return True
    
    async def enable_ai(self) -> Dict[str, Any]:
        """Enable AI functionality"""
        try:
            self.config['ai_settings']['enabled'] = True
            self._initialize_components()
            self.logger.info("AI system enabled successfully")
            return {
                'status': 'success',
                'message': 'AI system enabled successfully',
                'is_running': self.is_running
            }
        except Exception as e:
            self.logger.error(f"Failed to enable AI: {e}")
            return {
                'status': 'error',
                'message': f'Failed to enable AI: {str(e)}',
                'is_running': self.is_running
            }
    
    async def disable_ai(self) -> Dict[str, Any]:
        """Disable AI functionality"""
        try:
            self.config['ai_settings']['enabled'] = False
            self.is_running = False
            # Clean up resources
            self.dataset_handler = None
            self.model_manager = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.logger.info("AI system disabled successfully")
            return {
                'status': 'success',
                'message': 'AI system disabled successfully',
                'is_running': self.is_running
            }
        except Exception as e:
            self.logger.error(f"Failed to disable AI: {e}")
            return {
                'status': 'error',
                'message': f'Failed to disable AI: {str(e)}',
                'is_running': self.is_running
            }
    
    async def get_ai_status(self) -> Dict[str, Any]:
        """Get current AI system status"""
        memory = psutil.virtual_memory()
        return {
            'enabled': self.config.get('ai_settings', {}).get('enabled', False),
            'is_running': self.is_running,
            'subscription': {
                'type': self.config.get('ai_settings', {}).get('subscription_type', 'development'),
                'features': self.config.get('ai_settings', {}).get('features', {})
            },
            'resource_usage': {
                'memory_used': memory.percent,
                'cpu_used': psutil.cpu_percent(),
                'gpu_used': self._get_gpu_usage() if torch.cuda.is_available() else None
            }
        }
    
    def _get_gpu_usage(self) -> Optional[float]:
        """Get GPU usage if available"""
        if torch.cuda.is_available():
            try:
                return torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
            except:
                return None
        return None