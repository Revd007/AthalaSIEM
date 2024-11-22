import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Any
from datetime import datetime
import ipaddress
import re
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

class FeatureEngineer:
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.tfidf = TfidfVectorizer(max_features=100)
        self.ip_patterns = self._compile_ip_patterns()
        self.feature_columns = [
            'severity', 'source_type', 'event_type', 'time_of_day',
            'is_weekend', 'user_risk_score'
        ]
        
    def _compile_ip_patterns(self) -> Dict[str, re.Pattern]:
        return {
            'private': re.compile(r'^(10\.|172\.(1[6-9]|2[0-9]|3[01])\.|192\.168\.)'),
            'loopback': re.compile(r'^127\.'),
            'multicast': re.compile(r'^(22[4-9]|23[0-9])\.')
        }
    
    def extract_features(self, event: Dict) -> Dict[str, np.ndarray]:
        """Extract all features from an event"""
        features = {}
        
        # Temporal features
        features.update(self.extract_temporal_features(event))
        
        # Network features
        if 'ip_address' in event:
            features.update(self.extract_network_features(event))
        
        # Text features
        if 'message' in event:
            features.update(self.extract_text_features(event['message']))
        
        # Categorical features
        features.update(self.extract_categorical_features(event))
        
        # Numerical features
        features.update(self.extract_numerical_features(event))
        
        return features
    
    def extract_temporal_features(self, event: Dict) -> Dict[str, float]:
        """Extract time-based features"""
        timestamp = event.get('timestamp')
        if not timestamp:
            return {}
            
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
            
        return {
            'hour': timestamp.hour,
            'day_of_week': timestamp.weekday(),
            'is_weekend': int(timestamp.weekday() >= 5),
            'is_business_hours': int(8 <= timestamp.hour <= 18),
            'minute_of_day': timestamp.hour * 60 + timestamp.minute
        }
    
    def extract_network_features(self, event: Dict) -> Dict[str, Union[float, int]]:
        """Extract network-related features"""
        ip = event.get('ip_address', '')
        features = {}
        
        try:
            ip_obj = ipaddress.ip_address(ip)
            features.update({
                'is_ipv6': int(isinstance(ip_obj, ipaddress.IPv6Address)),
                'is_private': int(ip_obj.is_private),
                'is_global': int(ip_obj.is_global),
                'is_multicast': int(ip_obj.is_multicast),
                'is_loopback': int(ip_obj.is_loopback)
            })
            
            # Add IP octets as features
            if isinstance(ip_obj, ipaddress.IPv4Address):
                octets = str(ip_obj).split('.')
                for i, octet in enumerate(octets):
                    features[f'ip_octet_{i+1}'] = int(octet)
                    
        except ValueError:
            features.update({
                'is_ipv6': 0,
                'is_private': 0,
                'is_global': 0,
                'is_multicast': 0,
                'is_loopback': 0
            })
            
        return features
    
    def extract_text_features(self, text: str) -> Dict[str, np.ndarray]:
        """Extract features from text content"""
        # TF-IDF features
        tfidf_features = self.tfidf.fit_transform([text]).toarray()
        
        # Basic text statistics
        text_stats = {
            'text_length': len(text),
            'word_count': len(text.split()),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            'digit_ratio': sum(1 for c in text if c.isdigit()) / len(text) if text else 0,
            'special_char_ratio': sum(1 for c in text if not c.isalnum()) / len(text) if text else 0
        }
        
        return {
            'tfidf_features': tfidf_features,
            **text_stats
        }
    
    def extract_categorical_features(self, event: Dict) -> Dict[str, np.ndarray]:
        """Extract and encode categorical features"""
        categorical_fields = ['source', 'event_type', 'user', 'status']
        features = {}
        
        for field in categorical_fields:
            if field in event:
                if field not in self.encoders:
                    self.encoders[field] = OneHotEncoder(sparse=False, handle_unknown='ignore')
                    
                value = event[field]
                if isinstance(value, (list, tuple)):
                    value = ','.join(map(str, value))
                else:
                    value = str(value)
                    
                encoded = self.encoders[field].fit_transform([[value]])
                features[f'{field}_encoded'] = encoded
                
        return features
    
    def extract_numerical_features(self, event: Dict) -> Dict[str, float]:
        """Extract and scale numerical features"""
        numerical_fields = ['severity', 'duration', 'count', 'size']
        features = {}
        
        for field in numerical_fields:
            if field in event:
                if field not in self.scalers:
                    self.scalers[field] = StandardScaler()
                    
                value = float(event[field])
                scaled = self.scalers[field].fit_transform([[value]])[0][0]
                features[f'{field}_scaled'] = scaled
                
        return features
    
    def combine_features(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine all features into a single array"""
        feature_arrays = []
        
        # Combine numerical features
        numerical = np.array([
            features[key] for key in sorted(features.keys())
            if isinstance(features[key], (int, float))
        ])
        feature_arrays.append(numerical)
        
        # Combine categorical features
        categorical = np.concatenate([
            features[key] for key in sorted(features.keys())
            if key.endswith('_encoded')
        ])
        feature_arrays.append(categorical)
        
        # Add TF-IDF features if present
        if 'tfidf_features' in features:
            feature_arrays.append(features['tfidf_features'].flatten())
        
        return np.concatenate(feature_arrays)
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names"""
        feature_names = []
        
        # Temporal features
        feature_names.extend([
            'hour', 'day_of_week', 'is_weekend', 
            'is_business_hours', 'minute_of_day'
        ])
        
        # Network features
        feature_names.extend([
            'is_ipv6', 'is_private', 'is_global', 
            'is_multicast', 'is_loopback'
        ])
        feature_names.extend([f'ip_octet_{i}' for i in range(1, 5)])
        
        # Text statistics
        feature_names.extend([
            'text_length', 'word_count', 'uppercase_ratio',
            'digit_ratio', 'special_char_ratio'
        ])
        
        # Add encoded categorical features
        for encoder_name, encoder in self.encoders.items():
            feature_names.extend([
                f'{encoder_name}_{cat}' for cat in encoder.categories_[0]
            ])
        
        # Add scaled numerical features
        for scaler_name in self.scalers.keys():
            feature_names.append(f'{scaler_name}_scaled')
        
        # Add TF-IDF feature names
        feature_names.extend([
            f'tfidf_{i}' for i in range(self.tfidf.max_features)
        ])
        
        return feature_names
    
    def process_features(self, event: Dict[str, Any]) -> np.ndarray:
        """Convert raw event data into feature vector"""
        try:
            features = []
            
            # Severity (normalized)
            severity = float(event.get('severity', 0)) / 5.0
            features.append(severity)
            
            # Source type (one-hot encoded)
            source_type = self._encode_source(event.get('source', ''))
            features.extend(source_type)
            
            # Event type (one-hot encoded)
            event_type = self._encode_event_type(event.get('event_type', ''))
            features.extend(event_type)
            
            # Time features
            timestamp = event.get('timestamp')
            if timestamp:
                time_features = self._extract_time_features(timestamp)
                features.extend(time_features)
            else:
                features.extend([0, 0])  # Default time features
            
            # User risk score (if available)
            user_risk = float(event.get('user_risk_score', 0))
            features.append(user_risk)
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logging.error(f"Feature engineering error: {e}")
            # Return zero vector if processing fails
            return np.zeros(len(self.feature_columns), dtype=np.float32)

    def _encode_source(self, source: str) -> list:
        """Simple one-hot encoding for source types"""
        source_types = ['System', 'Security', 'Application', 'Other']
        encoding = [1 if source in s else 0 for s in source_types]
        return encoding

    def _encode_event_type(self, event_type: str) -> list:
        """Simple one-hot encoding for event types"""
        event_types = ['login', 'logout', 'error', 'warning', 'info']
        encoding = [1 if event_type in e else 0 for e in event_types]
        return encoding

    def _extract_time_features(self, timestamp) -> list:
        """Extract time-based features"""
        hour = timestamp.hour / 24.0  # Normalize hour
        is_weekend = 1.0 if timestamp.weekday() >= 5 else 0.0
        return [hour, is_weekend]