from typing import *
from .core.logger_config import setup_logger
from .core.model_manager import ModelManager
from .threat_intelligence import ThreatIntelligence
import yaml

class AIEngine:
    def __init__(self, config_path: str = "config/ai_settings.yaml"):
        # Setup logging
        self.logger = setup_logger()
        self.logger.info("Initializing AI Engine")
        
        # Load configuration
        with open(config_path) as f:
            config = yaml.safe_load(f)
            
        # Initialize components
        self.model_manager = ModelManager(config)
        self.threat_intelligence = ThreatIntelligence(self.model_manager)
        
    async def analyze_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Main entry point for event analysis"""
        try:
            self.logger.info("Received event for analysis")
            return await self.threat_intelligence.analyze_threats(event)
        except Exception as e:
            self.logger.error(f"Error in event analysis: {e}")
            return {
                'error': str(e),
                'status': 'failed'
            }