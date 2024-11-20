import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional
from datetime import datetime
import re
import json
from collections import defaultdict

from backend.api.middleware import logging

class DataCleaner:
    def __init__(self):
        self.missing_value_strategies = {
            'timestamp': self._handle_missing_timestamp,
            'severity': self._handle_missing_severity,
            'source': self._handle_missing_source,
            'message': self._handle_missing_message,
            'ip_address': self._handle_missing_ip,
            'user': self._handle_missing_user
        }
        
        self.field_validators = {
            'timestamp': self._validate_timestamp,
            'severity': self._validate_severity,
            'ip_address': self._validate_ip,
            'message': self._validate_message
        }
        
        self.normalizers = {
            'timestamp': self._normalize_timestamp,
            'severity': self._normalize_severity,
            'message': self._normalize_message,
            'source': self._normalize_source
        }
        
    def clean_event(self, event: Dict) -> Optional[Dict]:
        """Clean and validate a single event"""
        try:
            # Create a copy to avoid modifying original
            cleaned_event = event.copy()
            
            # Handle missing values
            cleaned_event = self._handle_missing_values(cleaned_event)
            
            # Validate fields
            if not self._validate_event(cleaned_event):
                return None
            
            # Normalize fields
            cleaned_event = self._normalize_event(cleaned_event)
            
            # Remove duplicated or redundant information
            cleaned_event = self._remove_redundant_info(cleaned_event)
            
            return cleaned_event
            
        except Exception as e:
            logging.error(f"Error cleaning event: {e}")
            return None
    
    def _handle_missing_values(self, event: Dict) -> Dict:
        """Handle missing values in the event"""
        for field, strategy in self.missing_value_strategies.items():
            if field not in event or event[field] is None or event[field] == '':
                event[field] = strategy(event)
        return event
    
    def _validate_event(self, event: Dict) -> bool:
        """Validate all fields in the event"""
        return all(
            validator(event.get(field))
            for field, validator in self.field_validators.items()
        )
    
    def _normalize_event(self, event: Dict) -> Dict:
        """Normalize all fields in the event"""
        for field, normalizer in self.normalizers.items():
            if field in event:
                event[field] = normalizer(event[field])
        return event
    
    # Missing value handlers
    def _handle_missing_timestamp(self, event: Dict) -> datetime:
        """Handle missing timestamp"""
        return datetime.utcnow()
    
    def _handle_missing_severity(self, event: Dict) -> int:
        """Handle missing severity"""
        # Default to medium severity (3)
        return 3
    
    def _handle_missing_source(self, event: Dict) -> str:
        """Handle missing source"""
        return 'unknown'
    
    def _handle_missing_message(self, event: Dict) -> str:
        """Handle missing message"""
        return json.dumps({k: v for k, v in event.items() if k != 'message'})
    
    def _handle_missing_ip(self, event: Dict) -> str:
        """Handle missing IP address"""
        return '0.0.0.0'
    
    def _handle_missing_user(self, event: Dict) -> str:
        """Handle missing user"""
        return 'unknown'
    
    # Validators
    def _validate_timestamp(self, timestamp: Union[str, datetime]) -> bool:
        """Validate timestamp"""
        if isinstance(timestamp, datetime):
            return True
        try:
            datetime.fromisoformat(timestamp)
            return True
        except (ValueError, TypeError):
            return False
    
    def _validate_severity(self, severity: Union[str, int]) -> bool:
        """Validate severity"""
        try:
            severity = int(severity)
            return 1 <= severity <= 5
        except (ValueError, TypeError):
            return False
    
    def _validate_ip(self, ip: str) -> bool:
        """Validate IP address"""
        ip_pattern = re.compile(r'^(\d{1,3}\.){3}\d{1,3}$')
        if not ip_pattern.match(ip):
            return False
        return all(0 <= int(i) <= 255 for i in ip.split('.'))
    
    def _validate_message(self, message: str) -> bool:
        """Validate message"""
        return isinstance(message, str) and len(message.strip()) > 0
    
    # Normalizers
    def _normalize_timestamp(self, timestamp: Union[str, datetime]) -> datetime:
        """Normalize timestamp to UTC datetime"""
        if isinstance(timestamp, str):
            return datetime.fromisoformat(timestamp)
        return timestamp
    
    def _normalize_severity(self, severity: Union[str, int]) -> int:
        """Normalize severity to integer 1-5"""
        try:
            severity = int(severity)
            return max(1, min(severity, 5))
        except (ValueError, TypeError):
            return 3
    
    def _normalize_message(self, message: str) -> str:
        """Normalize message format"""
        # Remove extra whitespace
        message = ' '.join(message.split())
        # Convert to lowercase
        message = message.lower()
        # Remove special characters
        message = re.sub(r'[^\w\s]', '', message)
        return message
    
    def _normalize_source(self, source: str) -> str:
        """Normalize source field"""
        source = source.lower()
        source = re.sub(r'[^\w\s-]', '', source)
        return source.strip()
    
    def _remove_redundant_info(self, event: Dict) -> Dict:
        """Remove redundant or duplicated information"""
        # Remove empty fields
        event = {k: v for k, v in event.items() if v is not None and v != ''}
        
        # Remove duplicate information in message
        if 'message' in event:
            for field, value in event.items():
                if field != 'message' and str(value) in event['message']:
                    event['message'] = event['message'].replace(str(value), f'{{{field}}}')
        
        return event