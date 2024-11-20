import psutil
import torch
from typing import Dict, Any, Optional
import logging
from ..core.dataset_handler import CyberSecurityDataHandler
from ..core.model_manager import ModelManager
from ..core.evaluator import Evaluator

class AIServiceManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.is_running = False
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize AI components based on config"""
        if self.config['ai_enabled']:
            try:
                self._check_system_requirements()
                self.dataset_handler = CyberSecurityDataHandler(self.config)
                self.model_manager = ModelManager(self.config)
                self.evaluator = Evaluator(self.config)
                self.is_running = True
            except Exception as e:
                self.logger.error(f"Failed to initialize AI components: {e}")
                self.is_running = False
    
    def _check_system_requirements(self) -> bool:
        """Check if system meets requirements"""
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent()
        
        if memory.available < self.config['resource_settings']['max_memory_usage'] * 1024 * 1024:
            raise Exception("Insufficient memory")
            
        if cpu_percent > self.config['resource_settings']['max_cpu_usage']:
            raise Exception("CPU usage too high")
            
        return True
    
    async def enable_ai(self):
        """Enable AI functionality"""
        try:
            self.config['ai_enabled'] = True
            self._initialize_components()
            self.logger.info("AI system enabled successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to enable AI: {e}")
            return False
    
    async def disable_ai(self):
        """Disable AI functionality"""
        try:
            self.config['ai_enabled'] = False
            self.is_running = False
            # Clean up resources
            self.dataset_handler = None
            self.model_manager = None
            self.evaluator = None
            torch.cuda.empty_cache()  # Clear GPU memory if used
            self.logger.info("AI system disabled successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to disable AI: {e}")
            return False
    
    async def get_ai_status(self) -> Dict[str, Any]:
        """Get current AI system status"""
        memory = psutil.virtual_memory()
        return {
            'enabled': self.config['ai_enabled'],
            'is_running': self.is_running,
            'resource_usage': {
                'memory_used': memory.percent,
                'cpu_used': psutil.cpu_percent(),
                'gpu_used': self._get_gpu_usage() if torch.cuda.is_available() else None
            },
            'feature_toggles': self.config['feature_toggles']
        }
    
    def _get_gpu_usage(self) -> Optional[float]:
        """Get GPU usage if available"""
        if torch.cuda.is_available():
            try:
                return torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
            except:
                return None
        return None