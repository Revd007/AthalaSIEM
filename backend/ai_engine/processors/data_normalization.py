import pandas as pd
import numpy as np
from typing import Dict, List, Union, Any
from datetime import datetime
import re
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import logging

class DataNormalizer:
    def __init__(self):
        self.standard_scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        self.robust_scaler = RobustScaler()
        
        # Define normalization schemas
        self.schemas = {
            'windows': self._normalize_windows_event,
            'linux': self._normalize_linux_event,
            'network': self._normalize_network_event,
            'application': self._normalize_application_event,
            'security': self._normalize_security_event
        }
        
        # Initialize field mappings
        self.field_mappings = self._initialize_field_mappings()
        
    def _initialize_field_mappings(self) -> Dict[str, str]:
        """Initialize standard field mappings for different sources"""
        return {
            # Windows Event Log mappings
            'EventID': 'event_id',
            'Message': 'message',
            'Level': 'severity',
            'TimeCreated': 'timestamp',
            'Provider': 'source',
            
            # Linux Syslog mappings
            'facility': 'source',
            'priority': 'severity',
            'msg': 'message',
            'time': 'timestamp',
            
            # Network device mappings
            'log_time': 'timestamp',
            'device_type': 'source',
            'alert_level': 'severity',
            'alert_msg': 'message'
        }
    
    def normalize_event(self, event: Dict[str, Any], source_type: str) -> Dict[str, Any]:
        """Normalize event based on source type"""
        try:
            # Apply source-specific normalization
            if source_type in self.schemas:
                normalized_event = self.schemas[source_type](event)
            else:
                normalized_event = self._normalize_generic_event(event)
            
            # Apply common normalization steps
            normalized_event = self._normalize_common_fields(normalized_event)
            
            # Validate normalized event
            if self._validate_normalized_event(normalized_event):
                return normalized_event
            else:
                raise ValueError("Event failed validation after normalization")
                
        except Exception as e:
            logging.error(f"Error normalizing event: {e}")
            return None
    
    def _normalize_windows_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize Windows event log format"""
        normalized = {}
        
        # Map standard fields
        for win_field, std_field in self.field_mappings.items():
            if win_field in event:
                normalized[std_field] = event[win_field]
        
        # Normalize severity
        if 'Level' in event:
            normalized['severity'] = self._normalize_windows_severity(event['Level'])
        
        # Extract additional metadata
        normalized['metadata'] = {
            'computer_name': event.get('Computer'),
            'channel': event.get('Channel'),
            'task_category': event.get('TaskCategory'),
            'keywords': event.get('Keywords', [])
        }
        
        return normalized
    
    def _normalize_linux_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize Linux syslog format"""
        normalized = {}
        
        # Map standard fields
        for syslog_field, std_field in self.field_mappings.items():
            if syslog_field in event:
                normalized[std_field] = event[syslog_field]
        
        # Normalize severity
        if 'priority' in event:
            normalized['severity'] = self._normalize_syslog_severity(event['priority'])
        
        # Extract process information
        if 'process' in event:
            normalized['metadata'] = {
                'process_name': event['process'].get('name'),
                'process_id': event['process'].get('pid'),
                'command': event['process'].get('cmd')
            }
        
        return normalized
    
    def _normalize_network_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize network device logs"""
        normalized = {}
        
        # Map standard fields
        for net_field, std_field in self.field_mappings.items():
            if net_field in event:
                normalized[std_field] = event[net_field]
        
        # Normalize network-specific fields
        if 'ip_address' in event:
            normalized['source_ip'] = self._normalize_ip_address(event['ip_address'])
        
        if 'port' in event:
            normalized['source_port'] = int(event['port'])
        
        # Extract protocol information
        normalized['metadata'] = {
            'protocol': event.get('protocol'),
            'interface': event.get('interface'),
            'direction': event.get('direction')
        }
        
        return normalized
    
    def _normalize_application_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize application logs"""
        normalized = {}
        
        # Map standard fields
        normalized.update({
            'timestamp': event.get('timestamp', datetime.utcnow()),
            'source': event.get('application', 'unknown'),
            'severity': self._normalize_application_severity(event.get('level')),
            'message': event.get('message', '')
        })
        
        # Extract application-specific metadata
        normalized['metadata'] = {
            'version': event.get('version'),
            'environment': event.get('env'),
            'component': event.get('component')
        }
        
        return normalized
    
    def _normalize_security_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize security-related events"""
        normalized = {}
        
        # Map standard fields
        normalized.update({
            'timestamp': event.get('timestamp', datetime.utcnow()),
            'source': event.get('security_tool', 'unknown'),
            'severity': self._normalize_security_severity(event.get('severity')),
            'message': event.get('alert_message', '')
        })
        
        # Extract security-specific metadata
        normalized['metadata'] = {
            'alert_type': event.get('alert_type'),
            'category': event.get('category'),
            'mitre_tactic': event.get('mitre_tactic'),
            'mitre_technique': event.get('mitre_technique')
        }
        
        return normalized
    
    def _normalize_generic_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize events with unknown source type"""
        normalized = {}
        
        # Apply basic normalization
        normalized.update({
            'timestamp': self._normalize_timestamp(event.get('timestamp')),
            'source': str(event.get('source', 'unknown')).lower(),
            'severity': self._normalize_generic_severity(event.get('severity')),
            'message': str(event.get('message', '')),
            'metadata': {}
        })
        
        # Extract any additional fields as metadata
        for key, value in event.items():
            if key not in ['timestamp', 'source', 'severity', 'message']:
                normalized['metadata'][key] = value
        
        return normalized
    
    def _normalize_common_fields(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Apply common normalization steps to all events"""
        # Ensure timestamp is UTC
        if 'timestamp' in event:
            event['timestamp'] = self._normalize_timestamp(event['timestamp'])
        
        # Normalize message format
        if 'message' in event:
            event['message'] = self._normalize_message(event['message'])
        
        # Ensure severity is 1-5 scale
        if 'severity' in event:
            event['severity'] = self._normalize_severity_scale(event['severity'])
        
        return event
    
    # Helper methods for specific normalizations
    def _normalize_timestamp(self, timestamp: Union[str, datetime, Any]) -> datetime:
        """Normalize timestamp to UTC datetime"""
        if isinstance(timestamp, str):
            try:
                return datetime.fromisoformat(timestamp)
            except ValueError:
                return datetime.utcnow()
        elif isinstance(timestamp, datetime):
            return timestamp
        elif hasattr(timestamp, 'timestamp'):  # Handle pywintypes.datetime
            try:
                return datetime.fromtimestamp(timestamp.timestamp())
            except:
                return datetime.utcnow()
        else:
            return datetime.utcnow()
    
    def _normalize_message(self, message: str) -> str:
        """Normalize message format"""
        if not message:
            return ""
        
        # Remove extra whitespace
        message = ' '.join(message.split())
        
        # Remove control characters
        message = ''.join(char for char in message if ord(char) >= 32)
        
        return message
    
    def _normalize_severity_scale(self, severity: Union[str, int]) -> int:
        """Normalize severity to 1-5 scale"""
        if isinstance(severity, str):
            severity = severity.lower()
            if severity in ['critical', 'emergency', 'alert']:
                return 5
            elif severity in ['error', 'err']:
                return 4
            elif severity in ['warning', 'warn']:
                return 3
            elif severity in ['notice', 'info']:
                return 2
            elif severity in ['debug']:
                return 1
        elif isinstance(severity, int):
            return max(1, min(severity, 5))
        return 3  # Default to medium severity
    
    def _normalize_windows_severity(self, level: Union[str, int]) -> int:
        """Normalize Windows event severity"""
        mapping = {
            1: 5,  # Critical
            2: 4,  # Error
            3: 3,  # Warning
            4: 2,  # Information
            5: 1   # Debug
        }
        try:
            level = int(level)
            return mapping.get(level, 3)
        except (ValueError, TypeError):
            return 3
    
    def _normalize_syslog_severity(self, priority: Union[str, int]) -> int:
        """Normalize syslog severity"""
        try:
            priority = int(priority)
            if priority <= 1:
                return 5
            elif priority <= 3:
                return 4
            elif priority <= 5:
                return 3
            elif priority <= 6:
                return 2
            else:
                return 1
        except (ValueError, TypeError):
            return 3
    
    def _normalize_ip_address(self, ip: str) -> str:
        """Normalize IP address format"""
        ip_pattern = re.compile(r'^(\d{1,3}\.){3}\d{1,3}$')
        if ip_pattern.match(ip):
            return ip
        return '0.0.0.0'
    
    def _validate_normalized_event(self, event: Dict[str, Any]) -> bool:
        """Validate normalized event structure"""
        required_fields = ['timestamp', 'source', 'severity', 'message']
        
        # Check required fields
        if not all(field in event for field in required_fields):
            return False
        
        # Validate timestamp
        if not isinstance(event['timestamp'], datetime):
            return False
        
        # Validate severity
        if not isinstance(event['severity'], int) or not 1 <= event['severity'] <= 5:
            return False
        
        # Validate message
        if not isinstance(event['message'], str):
            return False
        
        return True