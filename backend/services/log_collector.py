import win32evtlog
import win32con
import win32evtlogutil
import win32security
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from ..ai_engine.services.ai_service_manager import AIServiceManager
from ..config.ai_config import AIConfig

class LogCollector:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.ai_config = AIConfig()
        self.ai_manager = AIServiceManager(self.ai_config.current_config)
        
        # Initialize collectors
        self.collectors = {
            'windows': self._collect_windows_logs,
            'syslog': self._collect_syslog,
            'application': self._collect_application_logs
        }
        
    async def start_collection(self):
        """Start log collection process"""
        try:
            collected_logs = []
            
            # Collect from different sources
            for source, collector in self.collectors.items():
                if self.config.get(f'collect_{source}', True):
                    logs = await collector()
                    collected_logs.extend(logs)
            
            # Process logs with AI if enabled
            if self.ai_config.current_config['ai_enabled']:
                processed_logs = await self._process_with_ai(collected_logs)
            else:
                processed_logs = await self._basic_processing(collected_logs)
            
            return processed_logs
            
        except Exception as e:
            self.logger.error(f"Error in log collection: {e}")
            return []
    
    async def _collect_windows_logs(self) -> List[Dict[str, Any]]:
        """Collect Windows event logs"""
        logs = []
        log_types = ['System', 'Application', 'Security']
        
        for log_type in log_types:
            handle = win32evtlog.OpenEventLog(None, log_type)
            flags = win32evtlog.EVENTLOG_BACKWARDS_READ | win32evtlog.EVENTLOG_SEQUENTIAL_READ
            
            try:
                events = win32evtlog.ReadEventLog(
                    handle,
                    flags,
                    0
                )
                
                for event in events:
                    logs.append({
                        'source': event.SourceName,
                        'event_id': event.EventID,
                        'event_type': event.EventType,
                        'time_generated': event.TimeGenerated.strftime('%Y-%m-%d %H:%M:%S'),
                        'message': win32evtlogutil.SafeFormatMessage(event, log_type)
                    })
                    
            except Exception as e:
                self.logger.error(f"Error reading {log_type} logs: {e}")
                
            finally:
                win32evtlog.CloseEventLog(handle)
        
        return logs
    
    async def _collect_syslog(self) -> List[Dict[str, Any]]:
        """Collect Syslog entries"""
        logs = []
        syslog_path = '/var/log/syslog' if os.path.exists('/var/log/syslog') else None
        
        if syslog_path:
            try:
                with open(syslog_path, 'r') as f:
                    for line in f:
                        logs.append({
                            'source': 'syslog',
                            'message': line.strip(),
                            'time_generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        })
            except Exception as e:
                self.logger.error(f"Error reading syslog: {e}")
        
        return logs
    
    async def _collect_application_logs(self) -> List[Dict[str, Any]]:
        """Collect application-specific logs"""
        logs = []
        app_log_path = self.config.get('app_log_path')
        
        if app_log_path and os.path.exists(app_log_path):
            try:
                with open(app_log_path, 'r') as f:
                    for line in f:
                        logs.append({
                            'source': 'application',
                            'message': line.strip(),
                            'time_generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        })
            except Exception as e:
                self.logger.error(f"Error reading application logs: {e}")
        
        return logs
    
    async def _process_with_ai(self, logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process logs using AI capabilities"""
        try:
            # Get AI model manager
            model_manager = self.ai_manager.model_manager
            
            for log in logs:
                # Extract features
                features = self._extract_features(log)
                
                # Analyze with AI model
                if model_manager and features:
                    # Threat detection
                    threat_score = await model_manager.predict_threat(features)
                    log['threat_score'] = threat_score
                    
                    # Anomaly detection
                    is_anomaly = await model_manager.detect_anomaly(features)
                    log['is_anomaly'] = is_anomaly
                    
                    # Add AI insights
                    log['ai_insights'] = await model_manager.generate_insights(features)
                
            return logs
            
        except Exception as e:
            self.logger.error(f"Error in AI processing: {e}")
            return logs
    
    async def _basic_processing(self, logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Basic log processing without AI"""
        for log in logs:
            # Add basic severity classification
            log['severity'] = self._classify_severity(log)
            
            # Add basic categorization
            log['category'] = self._categorize_log(log)
            
        return logs
    
    def _extract_features(self, log: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract features for AI processing"""
        try:
            return {
                'message_length': len(log.get('message', '')),
                'source': log.get('source'),
                'event_type': log.get('event_type'),
                'hour': datetime.strptime(log['time_generated'], '%Y-%m-%d %H:%M:%S').hour,
                'is_error': 'error' in log.get('message', '').lower(),
                'is_warning': 'warning' in log.get('message', '').lower()
            }
        except Exception as e:
            self.logger.error(f"Error extracting features: {e}")
            return None
    
    def _classify_severity(self, log: Dict[str, Any]) -> str:
        """Basic severity classification"""
        message = log.get('message', '').lower()
        
        if 'error' in message or 'critical' in message:
            return 'high'
        elif 'warning' in message:
            return 'medium'
        else:
            return 'low'
    
    def _categorize_log(self, log: Dict[str, Any]) -> str:
        """Basic log categorization"""
        message = log.get('message', '').lower()
        
        if 'authentication' in message or 'login' in message:
            return 'authentication'
        elif 'firewall' in message:
            return 'network'
        elif 'permission' in message or 'access' in message:
            return 'access_control'
        else:
            return 'system'