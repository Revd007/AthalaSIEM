from typing import Dict, Any, Optional
import logging
from .models.threat_detections import ThreatDetector
from .core.model_manager import ModelManager
from .processors.feature_engineering import FeatureEngineer

class ThreatIntelligence:
    def __init__(self, model_manager: ModelManager):
        self.logger = logging.getLogger(__name__)
        self.model_manager = model_manager
        self.feature_engineer = FeatureEngineer()

    async def analyze_threats(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze threats in Windows events"""
        try:
            self.logger.info("Starting threat analysis for event")
            
            # Validate event data
            if not event or not isinstance(event, dict):
                self.logger.warning("Invalid event data received")
                return {
                    'is_threat': False,
                    'threat_score': 0.0,
                    'indicators': [],
                    'confidence': 0.0,
                    'error': 'Invalid event data'
                }

            # Extract features
            features = self.feature_engineer.extract_features(event)
            
            # Get threat analysis results
            results = await self.model_manager.analyze_threats(features)
            
            self.logger.info(f"Threat analysis completed: {results}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in threat analysis: {e}")
            return {
                'is_threat': False,
                'threat_score': 0.0,
                'indicators': [],
                'confidence': 0.0,
                'error': str(e)
            }

    async def analyze_linux_threats(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze threats in Linux log events"""
        try:
            # Extract features from Linux log event
            features = self.feature_engineer.extract_features(event)
            
            # Get threat analysis results
            results = await self.model_manager.analyze_threats(features)
            
            # Add Linux-specific threat indicators
            if 'message' in event:
                # Check for common Linux attack patterns
                message = event['message'].lower()
                if any(pattern in message for pattern in [
                    'failed password', 'authentication failure', 
                    'permission denied', 'sudo:', 'invalid user'
                ]):
                    results['indicators'].append('suspicious_auth_activity')
                
                if any(pattern in message for pattern in [
                    'port scan', 'connection refused',
                    'possible break-in attempt'
                ]):
                    results['indicators'].append('potential_intrusion_attempt')
            
            return {
                'is_threat': results['is_threat'],
                'threat_score': results['threat_score'],
                'indicators': results['indicators'],
                'confidence': results['confidence']
            }
            
        except Exception as e:
            self.logger.error(f"Linux threat analysis error: {e}")
            return {
                'is_threat': False,
                'threat_score': 0.0,
                'indicators': [],
                'confidence': 0.0,
                'error': str(e)
            }
        
    async def analyze_firewall_threats(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze threats in firewall log events"""
        try:
            # Extract features from firewall event
            features = self.feature_engineer.extract_features(event)
            
            # Get threat analysis results
            results = await self.model_manager.analyze_threats(features)
            
            # Add firewall-specific threat indicators
            if 'message' in event:
                message = event['message'].lower()
                
                # Check for common firewall attack patterns
                if any(pattern in message for pattern in [
                    'denied', 'blocked', 'dropped',
                    'reject', 'blacklist'
                ]):
                    results['indicators'].append('access_denied')
                    
                if any(pattern in message for pattern in [
                    'port scan', 'probe', 'recon',
                    'multiple connection attempts'
                ]):
                    results['indicators'].append('reconnaissance_activity')
                    
                if any(pattern in message for pattern in [
                    'ddos', 'flood', 'dos attack',
                    'high traffic', 'rate limit exceeded'
                ]):
                    results['indicators'].append('denial_of_service')
                    
                if any(pattern in message for pattern in [
                    'malware', 'virus', 'trojan',
                    'exploit', 'backdoor', 'shellcode'
                ]):
                    results['indicators'].append('malware_detection')
            
            return {
                'is_threat': results['is_threat'],
                'threat_score': results['threat_score'],
                'indicators': results['indicators'],
                'confidence': results['confidence']
            }
            
        except Exception as e:
            self.logger.error(f"Firewall threat analysis error: {e}")
            return {
                'is_threat': False,
                'threat_score': 0.0,
                'indicators': [],
                'confidence': 0.0,
                'error': str(e)
            }
