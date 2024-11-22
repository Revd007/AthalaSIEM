from collections import defaultdict
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
from sklearn.preprocessing import StandardScaler
from ai_engine.models.anomaly_detector import AnomalyDetector
from ai_engine.models.threat_detections import ThreatDetector

class CorrelationEngine:
    def __init__(self, 
                 time_window: int = 300,
                 anomaly_detector: Optional[AnomalyDetector] = None,
                 threat_detector: Optional[ThreatDetector] = None):
        self.time_window = time_window
        self.event_buffer = defaultdict(list)
        self.correlation_rules = self.load_correlation_rules()
        self.anomaly_detector = anomaly_detector
        self.threat_detector = threat_detector
        self.scaler = StandardScaler()
        
        # Feature extraction settings
        self.numerical_features = ['severity', 'duration', 'count']
        self.categorical_features = ['source', 'event_type', 'user', 'ip_address']
        
        # Initialize pattern storage
        self.known_patterns = defaultdict(list)
        self.pattern_weights = defaultdict(float)
        
    def load_correlation_rules(self) -> List[Dict]:
        """Load correlation rules from configuration"""
        # Implementation for loading rules
        pass
        
    async def process_event(self, event: Dict) -> Dict:
        """Process a single event through the correlation pipeline"""
        # 1. Clean and normalize event data
        cleaned_event = self.clean_event(event)
        
        # 2. Extract features
        features = self.extract_features(cleaned_event)
        
        # 3. Check for anomalies if detector is available
        # Scale features before anomaly detection
        scaled_features = self.scaler.fit_transform([features])[0] if features else None
        anomaly_result = None
        if self.anomaly_detector:
            anomaly_result = self.anomaly_detector.predict(features)
        
        # 4. Check for threats
        threat_result = None
        if self.threat_detector:
            threat_result = self.threat_detector.predict(str(cleaned_event))
        
        # 5. Add to event buffer
        self.event_buffer[cleaned_event['source']].append({
            **cleaned_event,
            'features': features,
            'anomaly_result': anomaly_result,
            'threat_result': threat_result
        })
        
        # 6. Clean old events
        self.clean_old_events()
        
        # 7. Find correlations
        correlations = await self.find_correlations(cleaned_event)
        
        # 8. Update patterns
        self.update_patterns(cleaned_event, correlations)
        
        # 9. Generate alerts if necessary
        if correlations or (anomaly_result and anomaly_result['is_anomaly']):
            await self.trigger_alert(cleaned_event, correlations, anomaly_result, threat_result)
        
        return {
            'event': cleaned_event,
            'correlations': correlations,
            'anomaly_result': anomaly_result,
            'threat_result': threat_result
        }
    
    def clean_event(self, event: Dict) -> Dict:
        """Clean and normalize event data"""
        cleaned = event.copy()
        
        # Ensure timestamp is datetime
        if isinstance(cleaned.get('timestamp'), str):
            cleaned['timestamp'] = datetime.fromisoformat(cleaned['timestamp'])
        
        # Normalize severity
        if 'severity' in cleaned:
            cleaned['severity'] = int(cleaned['severity'])
        
        # Normalize IP addresses
        if 'ip_address' in cleaned:
            cleaned['ip_address'] = self.normalize_ip(cleaned['ip_address'])
        
        return cleaned
    
    def extract_features(self, event: Dict) -> np.ndarray:
        """Extract numerical and categorical features from event"""
        features = []
        
        # Extract numerical features
        for feature in self.numerical_features:
            features.append(float(event.get(feature, 0)))
        
        # Extract categorical features (one-hot encoding)
        for feature in self.categorical_features:
            value = event.get(feature, '')
            if value in self.known_patterns[feature]:
                features.extend(self._one_hot_encode(value, self.known_patterns[feature]))
        
        return np.array(features)
    
    async def find_correlations(self, trigger_event: Dict) -> List[Dict]:
        """Find correlations between events"""
        correlations = []
        
        # Time-based correlation
        time_corr = self.find_time_correlations(trigger_event)
        if time_corr:
            correlations.extend(time_corr)
        
        # Pattern-based correlation
        pattern_corr = self.find_pattern_correlations(trigger_event)
        if pattern_corr:
            correlations.extend(pattern_corr)
        
        # Rule-based correlation
        rule_corr = self.find_rule_correlations(trigger_event)
        if rule_corr:
            correlations.extend(rule_corr)
        
        return correlations
    
    def find_time_correlations(self, event: Dict) -> List[Dict]:
        """Find time-based correlations"""
        correlations = []
        event_time = event['timestamp']
        
        for source, events in self.event_buffer.items():
            if source != event['source']:
                for other_event in events:
                    if abs((event_time - other_event['timestamp']).total_seconds()) <= self.time_window:
                        if self.are_events_related(event, other_event):
                            correlations.append({
                                'type': 'time_correlation',
                                'event': event,
                                'related_event': other_event,
                                'time_diff': abs((event_time - other_event['timestamp']).total_seconds())
                            })
        
        return correlations
    
    def find_pattern_correlations(self, event: Dict) -> List[Dict]:
        """Find pattern-based correlations"""
        correlations = []
        event_features = self.extract_features(event)
        
        for pattern_name, pattern in self.known_patterns.items():
            similarity = self.calculate_pattern_similarity(event_features, pattern)
            if similarity > 0.8:  # Threshold for pattern matching
                correlations.append({
                    'type': 'pattern_correlation',
                    'event': event,
                    'pattern_name': pattern_name,
                    'similarity': similarity
                })
        
        return correlations
    
    def find_rule_correlations(self, event: Dict) -> List[Dict]:
        """Find rule-based correlations"""
        correlations = []
        
        for rule in self.correlation_rules:
            if self.matches_rule_pattern(event, rule):
                related_events = self.find_related_events(event, rule)
                if related_events:
                    correlations.append({
                        'type': 'rule_correlation',
                        'event': event,
                        'rule': rule,
                        'related_events': related_events
                    })
        
        return correlations
    
    def update_patterns(self, event: Dict, correlations: List[Dict]):
        """Update known patterns based on new events and correlations"""
        features = self.extract_features(event)
        
        # Update pattern weights
        for correlation in correlations:
            if correlation['type'] == 'pattern_correlation':
                pattern_name = correlation['pattern_name']
                self.pattern_weights[pattern_name] *= 0.95  # Decay factor
                self.pattern_weights[pattern_name] += correlation['similarity']
        
        # Add new patterns if necessary
        if not correlations:
            pattern_name = f"pattern_{len(self.known_patterns)}"
            self.known_patterns[pattern_name] = features
            self.pattern_weights[pattern_name] = 1.0
        
        # Remove old patterns with low weights
        self._cleanup_patterns()
    
    def _cleanup_patterns(self):
        """Remove patterns with low weights"""
        threshold = 0.1
        to_remove = []
        
        for pattern_name, weight in self.pattern_weights.items():
            if weight < threshold:
                to_remove.append(pattern_name)
        
        for pattern_name in to_remove:
            del self.known_patterns[pattern_name]
            del self.pattern_weights[pattern_name]
    
    async def trigger_alert(self, event: Dict, correlations: List[Dict],
                          anomaly_result: Optional[Dict], threat_result: Optional[Dict]):
        """Trigger alert based on correlations and detection results"""
        alert = {
            'timestamp': datetime.utcnow(),
            'event': event,
            'correlations': correlations,
            'anomaly_result': anomaly_result,
            'threat_result': threat_result,
            'severity': self.calculate_alert_severity(event, correlations, anomaly_result, threat_result)
        }
        
        # Add alert to database
        await self.db.alerts.insert_one(alert)
        
        # Notify relevant systems
        if alert['severity'] >= 4:
            # High severity - notify immediately
            await self.notification_service.send_urgent_alert(alert)
        else:
            # Lower severity - add to notification queue
            await self.notification_service.queue_alert(alert)
        
        # Log alert
        self.logger.info(
            f"Alert triggered - Severity: {alert['severity']}, "
            f"Event ID: {event.get('id', 'unknown')}"
        )
        # Update alert metrics
        await self.metrics_tracker.update_metrics({
            'alert_severity': alert['severity'],
            'correlation_count': len(correlations),
            'has_anomaly': bool(anomaly_result and anomaly_result.get('is_anomaly')),
            'has_threat': bool(threat_result and threat_result.get('threat_probs', [0])[0] > 0.8)
        }, step=self.alert_counter)
        
        # Store alert metadata for pattern analysis
        alert_metadata = {
            'id': str(alert['_id']),
            'severity': alert['severity'],
            'timestamp': alert['timestamp'],
            'correlation_patterns': [c['pattern'] for c in correlations] if correlations else []
        }
        self.alert_history.append(alert_metadata)
        pass
    
    def calculate_alert_severity(self, event: Dict, correlations: List[Dict],
                               anomaly_result: Optional[Dict], threat_result: Optional[Dict]) -> int:
        """Calculate alert severity based on all available information"""
        base_severity = event.get('severity', 3)
        
        # Adjust based on correlations
        if correlations:
            base_severity = min(base_severity + len(correlations), 5)
        
        # Adjust based on anomaly detection
        if anomaly_result and anomaly_result.get('is_anomaly'):
            base_severity = min(base_severity + 1, 5)
        
        # Adjust based on threat detection
        if threat_result and threat_result.get('threat_probs', [0])[0] > 0.8:
            base_severity = min(base_severity + 2, 5)
        
        return base_severity