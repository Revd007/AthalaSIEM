from attr import s
import torch
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
from ..models.threat_detector import ThreatDetector
from ..models.anomaly_detector import VariationalAutoencoder
from ..processors.feature_engineering import FeatureEngineer
from ..core.knowledge_graph import KnowledgeGraph
from ..training.adaptive_learner import AdaptiveLearner
from ..core.evaluator import ModelEvaluator
from ..core.model_manager import ModelManager
import logging

class PredictionService:
    def __init__(self, model_manager: ModelManager, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        self.model_manager = model_manager
        self.config = config
        self.feature_engineer = FeatureEngineer()
        self.knowledge_graph = KnowledgeGraph()
        self.prediction_threshold = config.get('prediction_threshold', 0.75)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Prediction confidence tracking
        self.prediction_history = []
        self.max_history_size = 1000

        # Add advanced components
        self.adaptive_learner = AdaptiveLearner(
            models=model_manager.models,  # Pass the models dictionary
            config=config
        )
        self.evaluator = ModelEvaluator(model_manager, config)
        
        # Enhanced prediction tracking
        self.pattern_memory = {}
        self.behavioral_patterns = []
        self.threat_signatures = set()
        
        # Confidence thresholds
        self.high_confidence_threshold = 0.85
        self.medium_confidence_threshold = 0.65

        # Advanced ML Components
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        
        # Enhanced Temporal Analysis
        self.temporal_patterns = {
            'hourly': {},
            'daily': {},
            'weekly': {}
        }
        self.behavior_sequences = []
        self.sequence_length = 10
        
        # Advanced Behavioral Indicators
        self.behavior_profiles = {}
        self.anomaly_patterns = set()
        self.threat_chains = []
        
        # Dynamic Thresholds
        self.threshold_history = []
        self.adaptive_threshold = self.prediction_threshold
        
    async def predict_threat(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict potential threats from event data"""
        try:
            # Enhanced feature extraction
            base_features = self.feature_engineer.process_features(event_data)
            temporal_features = self._extract_temporal_features(event_data)
            behavioral_features = self._extract_behavioral_features(event_data)
            
            # Combine all features
            enhanced_features = np.concatenate([
                base_features,
                temporal_features,
                behavioral_features
            ])
            
            # Advanced anomaly detection
            isolation_forest_score = self._get_isolation_forest_score(enhanced_features)
            
            # Get contextual predictions
            threat_predictions = await self._get_contextual_threat_prediction(
                torch.FloatTensor(enhanced_features).to(self.device),
                event_data
            )
            
            # Enhanced behavioral analysis
            behavior_analysis = self._analyze_advanced_behavior_patterns(
                event_data,
                enhanced_features
            )
            
            # Temporal pattern matching
            temporal_matches = self._match_temporal_patterns(event_data)
            
            # Combine predictions with advanced weighting
            combined_prediction = self._combine_advanced_predictions(
                threat_predictions,
                isolation_forest_score,
                behavior_analysis,
                temporal_matches,
                event_data
            )
            
            # Update models and patterns
            self._update_prediction_history(combined_prediction)
            self._update_temporal_patterns(event_data, combined_prediction)
            self._update_behavior_profiles(event_data, combined_prediction)
            
            return combined_prediction
            
        except Exception as e:
            self.logger.error(f"Advanced prediction error: {e}")
            return {'error': str(e), 'status': 'failed'}

    async def _get_contextual_threat_prediction(
        self, 
        features: torch.Tensor,
        event_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get threat predictions with contextual awareness"""
        # Get base prediction
        base_prediction = self.model_manager.threat_detector(features.unsqueeze(0))
        
        # Enhance with historical context
        historical_context = self._get_historical_context(event_data)
        
        # Adjust prediction based on context
        adjusted_prediction = self._adjust_prediction_with_context(
            base_prediction,
            historical_context
        )
        
        # Add confidence metrics
        confidence_metrics = self._calculate_confidence_metrics(
            adjusted_prediction,
            event_data
        )
        
        # Combine predictions
        combined_prediction = {
            **adjusted_prediction,
            **confidence_metrics
        }

        return combined_prediction

    def _combine_predictions(self, 
                           threat_pred: Dict[str, torch.Tensor],
                           anomaly_pred: Dict[str, torch.Tensor],
                           behavioral_analysis: Dict[str, Any],
                           event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Combine predictions from different models using knowledge graph"""
        
        # Get threat scores
        threat_score = float(threat_pred['threat_score'].cpu().item())
        anomaly_score = float(anomaly_pred.get('anomaly_score', torch.tensor([0.0])).cpu().item())

        # Get pattern matches from knowledge graph
        pattern_matches = self.knowledge_graph.find_matching_patterns(event_data)
        
        # Calculate confidence based on multiple factors
        confidence = self._calculate_confidence(
            threat_score,
            anomaly_score,
            pattern_matches
        )

        # Generate detailed prediction
        prediction = {
            'threat_detected': threat_score > self.prediction_threshold,
            'threat_score': threat_score,
            'anomaly_score': anomaly_score,
            'confidence': confidence,
            'threat_type': self._determine_threat_type(threat_pred),
            'risk_factors': self._analyze_risk_factors(event_data),
            'recommended_actions': self._generate_recommendations(threat_score, anomaly_score),
            'pattern_matches': pattern_matches,
            'timestamp': event_data.get('timestamp', None)
        }

        return prediction

    def _calculate_confidence(self, 
                            threat_score: float,
                            anomaly_score: float,
                            pattern_matches: List[Dict[str, Any]]) -> float:
        """Calculate prediction confidence score"""
        # Base confidence from threat score
        confidence = threat_score * 0.6

        # Add anomaly detection confidence
        confidence += anomaly_score * 0.2

        # Add pattern matching confidence
        if pattern_matches:
            pattern_confidence = sum(p.get('confidence', 0) for p in pattern_matches) / len(pattern_matches)
            confidence += pattern_confidence * 0.2

        return min(confidence, 1.0)

    def _determine_threat_type(self, threat_pred: Dict[str, torch.Tensor]) -> str:
        """Determine specific threat type from predictions"""
        threat_patterns = threat_pred.get('patterns', torch.zeros(1))
        pattern_idx = torch.argmax(threat_patterns).item()
        
        # Map pattern index to threat type
        threat_types = self.config.get('threat_types', [])
        if threat_types and pattern_idx < len(threat_types):
            return threat_types[pattern_idx]
        return 'unknown'

    def _analyze_risk_factors(self, event_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze specific risk factors in the event"""
        risk_factors = []
        
        # Check various risk indicators
        if 'ip_address' in event_data:
            risk_factors.append({
                'type': 'network',
                'factor': 'ip_reputation',
                'score': self._check_ip_reputation(event_data['ip_address'])
            })

        if 'user' in event_data:
            risk_factors.append({
                'type': 'user',
                'factor': 'user_behavior',
                'score': self._analyze_user_behavior(event_data['user'])
            })

        return risk_factors

    def _generate_recommendations(self, 
                                threat_score: float,
                                anomaly_score: float) -> List[str]:
        """Generate action recommendations based on predictions"""
        recommendations = []
        
        if threat_score > self.prediction_threshold:
            recommendations.append("Immediate investigation recommended")
            if threat_score > 0.9:
                recommendations.append("Consider blocking suspicious activity")
                recommendations.append("Escalate to security team")

        if anomaly_score > self.prediction_threshold:
            recommendations.append("Review system logs for unusual patterns")
            recommendations.append("Monitor affected systems closely")

        return recommendations

    def _update_prediction_history(self, prediction: Dict[str, Any]):
        """Update prediction history for trend analysis"""
        self.prediction_history.append(prediction)
        if len(self.prediction_history) > self.max_history_size:
            self.prediction_history.pop(0)

    def analyze_prediction_trends(self) -> Dict[str, Any]:
        """Analyze trends in recent predictions"""
        if not self.prediction_history:
            return {}

        recent_predictions = self.prediction_history[-100:]
        
        return {
            'average_threat_score': np.mean([p['threat_score'] for p in recent_predictions]),
            'average_confidence': np.mean([p['confidence'] for p in recent_predictions]),
            'threat_types_distribution': self._get_threat_distribution(recent_predictions),
            'detection_rate': self._calculate_detection_rate(recent_predictions)
        }

    def _get_threat_distribution(self, predictions: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate distribution of threat types"""
        distribution = {}
        for pred in predictions:
            threat_type = pred['threat_type']
            distribution[threat_type] = distribution.get(threat_type, 0) + 1
        return distribution

    def _calculate_detection_rate(self, predictions: List[Dict[str, Any]]) -> float:
        """Calculate threat detection rate"""
        if not predictions:
            return 0.0
        detected = sum(1 for p in predictions if p['threat_detected'])
        return detected / len(predictions)

    def _extract_temporal_features(self, event_data: Dict[str, Any]) -> np.ndarray:
        """Extract advanced temporal features"""
        timestamp = event_data.get('timestamp', datetime.now())
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
            
        features = [
            timestamp.hour / 24.0,  # Hour of day
            timestamp.weekday() / 7.0,  # Day of week
            timestamp.day / 31.0,  # Day of month
            self._get_temporal_density(timestamp),  # Event density
            self._get_periodic_score(timestamp),  # Periodicity score
            self._calculate_temporal_risk(timestamp)  # Time-based risk
        ]
        
        return np.array(features)

    def _extract_behavioral_features(self, event_data: Dict[str, Any]) -> np.ndarray:
        """Extract advanced behavioral features"""
        user_id = event_data.get('user_id', 'unknown')
        
        features = [
            self._get_user_risk_score(user_id),
            self._get_behavior_deviation_score(event_data),
            self._get_sequence_similarity(event_data),
            self._calculate_threat_chain_probability(event_data),
            self._get_anomaly_correlation_score(event_data)
        ]
        
        return np.array(features)

    def _combine_advanced_predictions(
        self,
        threat_pred: Dict[str, Any],
        isolation_score: float,
        behavior_analysis: Dict[str, Any],
        temporal_matches: Dict[str, Any],
        event_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Combine predictions using advanced weighting"""
        # Calculate base scores
        threat_score = float(threat_pred['threat_score'])
        behavior_score = behavior_analysis['risk_score']
        temporal_score = temporal_matches['match_score']
        
        # Dynamic weight adjustment based on confidence
        weights = self._calculate_dynamic_weights(
            threat_pred['confidence'],
            behavior_analysis['confidence'],
            temporal_matches['confidence']
        )
        
        # Combined score with weighted average
        combined_score = (
            threat_score * weights['threat'] +
            isolation_score * weights['anomaly'] +
            behavior_score * weights['behavior'] +
            temporal_score * weights['temporal']
        )
        
        # Adjust threshold based on historical performance
        self._update_adaptive_threshold(combined_score)
        
        return {
            'threat_detected': combined_score > self.adaptive_threshold,
            'threat_score': combined_score,
            'confidence': self._calculate_advanced_confidence(weights, combined_score),
            'risk_factors': self._analyze_advanced_risk_factors(event_data),
            'temporal_patterns': temporal_matches['patterns'],
            'behavior_indicators': behavior_analysis['indicators'],
            'threat_chain': self._identify_threat_chain(event_data),
            'recommendations': self._generate_advanced_recommendations(combined_score)
        }

    def _calculate_advanced_confidence(self, weights: Dict[str, float], combined_score: float) -> float:
        """Calculate advanced confidence score"""
        # Base confidence from threat score
        confidence = combined_score * 0.6

        # Add anomaly detection confidence
        confidence += weights['anomaly'] * 0.2

        # Add behavioral analysis confidence
        confidence += weights['behavior'] * 0.1

        # Add temporal pattern matching confidence
        confidence += weights['temporal'] * 0.1

        return min(confidence, 1.0)

    def _analyze_advanced_risk_factors(self, event_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze advanced risk factors in the event"""
        risk_factors = []
        
        # Check various advanced risk indicators
        if 'ip_address' in event_data:
            risk_factors.append({
                'type': 'network',
                'factor': 'ip_reputation',
                'score': self._check_ip_reputation(event_data['ip_address'])
            })

        if 'user' in event_data:
            risk_factors.append({
                'type': 'user',
                'factor': 'user_behavior',
                'score': self._analyze_user_behavior(event_data['user'])
            })

        return risk_factors

    def _identify_threat_chain(self, event_data: Dict[str, Any]) -> str:
        """Identify threat chain in the event"""
        # Initialize threat chain components
        threat_chain = []
        
        # Check for initial access indicators
        if event_data.get('ip_address') and self._check_ip_reputation(event_data['ip_address']) > 0.7:
            threat_chain.append('initial_access')
            
        # Check for execution indicators
        if event_data.get('process_name') or event_data.get('command_line'):
            threat_chain.append('execution')
            
        # Check for persistence indicators
        if event_data.get('registry_changes') or event_data.get('scheduled_tasks'):
            threat_chain.append('persistence')
            
        # Check for privilege escalation
        if event_data.get('admin_rights') or event_data.get('sudo_commands'):
            threat_chain.append('privilege_escalation')
            
        # Check for lateral movement
        if event_data.get('network_connections') or event_data.get('remote_access'):
            threat_chain.append('lateral_movement')
            
        # Check for data exfiltration
        if event_data.get('data_transfer') or event_data.get('unusual_outbound_traffic'):
            threat_chain.append('data_exfiltration')
            
        # Return identified threat chain stages or 'unknown' if none found
        return ' -> '.join(threat_chain) if threat_chain else 'unknown'
        return 'unknown'

    def _generate_advanced_recommendations(self, combined_score: float) -> List[str]:
        """Generate advanced action recommendations based on predictions"""
        recommendations = []
        
        if combined_score > self.prediction_threshold:
            recommendations.append("Immediate investigation recommended")
            if combined_score > 0.9:
                recommendations.append("Consider blocking suspicious activity")
                recommendations.append("Escalate to security team")

        if combined_score > 0.5:
            recommendations.append("Review system logs for unusual patterns")
            recommendations.append("Monitor affected systems closely")

        return recommendations

    def _get_isolation_forest_score(self, features: np.ndarray) -> float:
        """Get isolation forest score for anomaly detection"""
        # Scale features
        scaled_features = self.scaler.fit_transform(features.reshape(1, -1))
        
        # Get isolation forest score
        isolation_forest_score = self.isolation_forest.score(scaled_features)
        
        return isolation_forest_score

    def _get_temporal_density(self, timestamp: datetime) -> float:
        """Calculate event density in the temporal patterns"""
        # Get time window (e.g., last hour)
        time_window = timedelta(hours=1)
        window_start = timestamp - time_window
        
        # Count events in time window from event history
        events_in_window = sum(1 for event_time in self.event_history 
                             if window_start <= event_time <= timestamp)
        
        # Calculate density as events per minute
        minutes = time_window.total_seconds() / 60
        density = events_in_window / minutes if minutes > 0 else 0
        
        # Normalize density to 0-1 range based on historical patterns
        max_density = max(1.0, self.max_historical_density)
        normalized_density = min(1.0, density / max_density)
        
        return normalized_density
        return 0.0

    def _get_periodic_score(self, timestamp: datetime) -> float:
        """Calculate periodicity score in the temporal patterns"""
        # Convert timestamp to hour of day (0-23)
        hour = timestamp.hour
        
        # Define typical active hours (e.g., 9 AM - 5 PM)
        active_hours_start = 9
        active_hours_end = 17
        
        # Calculate base periodicity score
        if active_hours_start <= hour < active_hours_end:
            # During typical active hours
            base_score = 0.3
        elif hour < 6 or hour >= 22:
            # Very unusual hours (midnight to 6 AM, or after 10 PM)
            base_score = 0.9
        else:
            # Moderately unusual hours
            base_score = 0.6
            
        # Adjust score based on day of week
        if timestamp.weekday() >= 5:  # Weekend
            base_score = min(1.0, base_score + 0.3)
            
        # Adjust score based on historical patterns at this time
        historical_events = [evt for evt in self.event_history 
                           if evt.hour == hour]
        if historical_events:
            frequency = len(historical_events) / len(self.event_history)
            # Increase score if this is an unusual time based on history
            if frequency < 0.1:  # Less than 10% of events occur at this hour
                base_score = min(1.0, base_score + 0.2)
                
        return base_score
        return 0.0

    def _calculate_temporal_risk(self, timestamp: datetime) -> float:
        """Calculate time-based risk in the temporal patterns"""
        # Calculate density and periodic scores
        density_score = self._get_density_score(timestamp)
        periodic_score = self._get_periodic_score(timestamp)
        
        # Combine scores with weighted average
        # Give more weight to density as it's based on recent activity
        density_weight = 0.7
        periodic_weight = 0.3
        
        temporal_risk = (density_score * density_weight + 
                        periodic_score * periodic_weight)
        
        # Ensure final score is between 0 and 1
        temporal_risk = max(0.0, min(1.0, temporal_risk))
        
        return temporal_risk
        return 0.0

    def _get_user_risk_score(self, user_id: str) -> float:
        """Calculate user risk score"""
        # Get user behavior history
        user_history = self.behavior_profiles.get(user_id, {})
        
        if not user_history:
            # New user with no history - assign moderate risk
            return 0.5
            
        # Calculate base risk from behavioral factors
        risk_factors = []
        
        # Check login patterns
        login_times = user_history.get('login_times', [])
        if login_times:
            unusual_hours = sum(1 for t in login_times if t.hour < 6 or t.hour > 22)
            login_risk = unusual_hours / len(login_times) * 0.3
            risk_factors.append(login_risk)
            
        # Check failed login attempts
        failed_logins = user_history.get('failed_logins', 0) 
        if failed_logins > 0:
            login_fail_risk = min(failed_logins / 10, 1.0) * 0.3
            risk_factors.append(login_fail_risk)
            
        # Check access patterns
        access_patterns = user_history.get('access_patterns', {})
        if access_patterns:
            unusual_access = len(access_patterns.get('unusual', []))
            access_risk = min(unusual_access / 5, 1.0) * 0.2
            risk_factors.append(access_risk)
            
        # Check privilege level
        privilege_level = user_history.get('privilege_level', 'low')
        privilege_risk = {
            'low': 0.1,
            'medium': 0.2, 
            'high': 0.3,
            'admin': 0.4
        }.get(privilege_level, 0.1)
        risk_factors.append(privilege_risk)
        
        # Calculate final risk score
        if risk_factors:
            final_risk = sum(risk_factors) / len(risk_factors)
            return min(final_risk, 1.0)
            
        return 0.3  # Default moderate-low risk
        return 0.0

    def _get_behavior_deviation_score(self, event_data: Dict[str, Any]) -> float:
        """Calculate behavior deviation score"""
        # Extract behavioral features
        current_behavior = {
            'timestamp': event_data.get('timestamp', datetime.now()),
            'action': event_data.get('action', ''),
            'resource': event_data.get('resource', ''),
            'location': event_data.get('location', ''),
            'device': event_data.get('device', '')
        }

        # Get historical behavior patterns
        user_id = event_data.get('user_id')
        historical_patterns = self.behavior_profiles.get(user_id, {}).get('patterns', [])

        if not historical_patterns:
            return 0.5  # Moderate score for new patterns

        # Calculate temporal deviation
        hour = current_behavior['timestamp'].hour
        typical_hours = [p['timestamp'].hour for p in historical_patterns]
        temporal_deviation = abs(hour - np.mean(typical_hours)) / 12.0

        # Calculate resource access deviation
        typical_resources = [p['resource'] for p in historical_patterns]
        resource_freq = {}
        for r in typical_resources:
            resource_freq[r] = resource_freq.get(r, 0) + 1
        resource_deviation = 1.0
        if current_behavior['resource'] in resource_freq:
            resource_deviation = 1.0 - (resource_freq[current_behavior['resource']] / len(historical_patterns))

        # Calculate location/device deviation
        location_freq = {}
        device_freq = {}
        for p in historical_patterns:
            location_freq[p['location']] = location_freq.get(p['location'], 0) + 1
            device_freq[p['device']] = device_freq.get(p['device'], 0) + 1

        location_deviation = 1.0
        if current_behavior['location'] in location_freq:
            location_deviation = 1.0 - (location_freq[current_behavior['location']] / len(historical_patterns))

        device_deviation = 1.0
        if current_behavior['device'] in device_freq:
            device_deviation = 1.0 - (device_freq[current_behavior['device']] / len(historical_patterns))

        # Combine deviations with weights
        deviation_score = (
            temporal_deviation * 0.3 +
            resource_deviation * 0.3 +
            location_deviation * 0.2 +
            device_deviation * 0.2
        )

        return min(deviation_score, 1.0)
        return 0.0

    def _get_sequence_similarity(self, event_data: Dict[str, Any]) -> float:
        """Calculate sequence similarity score"""
        # Extract current sequence
        current_sequence = [event_data.get('action_type'), 
                          event_data.get('resource'),
                          event_data.get('location')]

        # Get recent behavior sequences
        recent_sequences = self.behavior_sequences[-self.sequence_length:]
        if not recent_sequences:
            return 0.0

        # Calculate similarity scores using dynamic time warping
        similarity_scores = []
        for sequence in recent_sequences:
            # Convert sequence elements to comparable format
            seq = [s.get('action_type', ''), 
                  s.get('resource', ''),
                  s.get('location', '')]
            
            # Calculate element-wise similarity
            element_scores = []
            for curr, hist in zip(current_sequence, seq):
                if not curr or not hist:
                    element_scores.append(0.0)
                    continue
                    
                # String similarity using Levenshtein distance
                distance = float(sum(1 for a, b in zip(curr, hist) if a != b))
                similarity = 1.0 - (distance / max(len(curr), len(hist)))
                element_scores.append(similarity)
            
            # Average element similarities for sequence score
            if element_scores:
                similarity_scores.append(sum(element_scores) / len(element_scores))

        # Return weighted average of similarities, more weight to recent sequences
        weights = np.linspace(0.5, 1.0, len(similarity_scores))
        weighted_scores = np.multiply(similarity_scores, weights)
        return float(np.mean(weighted_scores))
        return 0.0

    def _calculate_threat_chain_probability(self, event_data: Dict[str, Any]) -> float:
        """Calculate threat chain probability"""
        # Extract key event attributes
        event_type = event_data.get('event_type', '')
        source_ip = event_data.get('source_ip', '')
        target_resource = event_data.get('target_resource', '')
        timestamp = event_data.get('timestamp', datetime.now())

        # Create event signature
        event_signature = f"{event_type}:{source_ip}:{target_resource}"

        # Add to threat chains tracking
        self.threat_chains.append({
            'signature': event_signature,
            'timestamp': timestamp
        })

        # Keep only recent chains (last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.threat_chains = [
            chain for chain in self.threat_chains 
            if chain['timestamp'] > cutoff_time
        ]

        # Analyze chain patterns
        chain_probability = 0.0
        if len(self.threat_chains) > 1:
            # Look for similar events in chain
            similar_events = [
                chain for chain in self.threat_chains
                if self._calculate_signature_similarity(
                    chain['signature'], 
                    event_signature
                ) > 0.7
            ]

            # Calculate probability based on:
            # 1. Frequency of similar events
            frequency_score = len(similar_events) / len(self.threat_chains)
            
            # 2. Time proximity of events
            time_diffs = [
                (timestamp - event['timestamp']).total_seconds() / 3600
                for event in similar_events
            ]
            time_score = 1.0 - (min(time_diffs) / 24.0 if time_diffs else 1.0)
            
            # 3. Sequential pattern matching
            sequence_score = self._get_sequence_similarity(event_data)
            
            # Combine scores with weights
            chain_probability = (
                frequency_score * 0.4 +
                time_score * 0.3 +
                sequence_score * 0.3
            )

        return min(chain_probability, 1.0)
        return 0.0

    def _get_anomaly_correlation_score(self, event_data: Dict[str, Any]) -> float:
        """Calculate anomaly correlation score"""
        # Get historical anomaly scores for comparison
        historical_scores = [
            event['anomaly_score'] 
            for event in self.event_history[-100:]  # Look at last 100 events
            if 'anomaly_score' in event
        ]

        if not historical_scores:
            return 0.0

        # Calculate baseline statistics
        mean_score = sum(historical_scores) / len(historical_scores)
        variance = sum((x - mean_score) ** 2 for x in historical_scores) / len(historical_scores)
        std_dev = variance ** 0.5

        # Calculate z-score for current event
        current_score = event_data.get('anomaly_score', 0.0)
        if std_dev == 0:
            z_score = 0.0
        else:
            z_score = abs(current_score - mean_score) / std_dev

        # Look for temporal correlations
        time_window = 3600  # 1 hour in seconds
        recent_events = [
            event for event in self.event_history[-20:]  # Last 20 events
            if (event_data['timestamp'] - event['timestamp']).total_seconds() <= time_window
        ]

        # Calculate temporal density
        temporal_density = len(recent_events) / 20.0  # Normalize by window size

        # Look for attribute correlations
        attribute_matches = 0
        key_attributes = ['source_ip', 'destination_ip', 'event_type', 'severity']
        
        for event in recent_events:
            matches = sum(
                1 for attr in key_attributes 
                if attr in event and attr in event_data 
                and event[attr] == event_data[attr]
            )
            attribute_matches += matches / len(key_attributes)

        attribute_correlation = attribute_matches / len(recent_events) if recent_events else 0.0

        # Combine scores with weights
        correlation_score = (
            0.4 * min(z_score / 3.0, 1.0) +  # Cap z-score contribution
            0.3 * temporal_density +
            0.3 * attribute_correlation
        )

        return min(correlation_score, 1.0)
        return 0.0

    def _update_adaptive_threshold(self, combined_score: float):
        """Update adaptive threshold based on historical performance"""
        # Calculate statistics from recent predictions
        recent_scores = [p['threat_score'] for p in self.prediction_history[-100:]]
        if not recent_scores:
            return

        # Calculate baseline statistics
        mean_score = np.mean(recent_scores)
        std_dev = np.std(recent_scores)

        # Calculate false positive and true positive rates
        true_positives = sum(1 for p in self.prediction_history[-100:] 
                           if p['threat_detected'] and p.get('verified_threat', False))
        false_positives = sum(1 for p in self.prediction_history[-100:]
                            if p['threat_detected'] and not p.get('verified_threat', False))
        
        tp_rate = true_positives / max(1, len(recent_scores))
        fp_rate = false_positives / max(1, len(recent_scores))

        # Adjust threshold based on performance metrics
        if fp_rate > 0.1:  # Too many false positives
            self.prediction_threshold = min(0.95, self.prediction_threshold + 0.05)
        elif tp_rate < 0.8 and fp_rate < 0.05:  # Room for more sensitivity
            self.prediction_threshold = max(0.5, self.prediction_threshold - 0.05)

        # Factor in the current combined score
        score_diff = abs(combined_score - mean_score)
        if score_diff > 2 * std_dev:  # Significant deviation
            # Adjust threshold more aggressively
            adjustment = 0.1 if combined_score > mean_score else -0.1
            self.prediction_threshold = max(0.5, min(0.95, 
                self.prediction_threshold + adjustment))

        # Apply temporal decay to threshold adjustments
        decay_factor = 0.95
        self.prediction_threshold = (
            decay_factor * self.prediction_threshold + 
            (1 - decay_factor) * self.config.get('prediction_threshold', 0.75)
        )