from typing import Dict, List, Any
from datetime import datetime, timedelta
import ipaddress
import re

class DefaultThreatAnalytics:
    def __init__(self):
        self.threat_indicators = {
            'suspicious_ips': set(),
            'malicious_patterns': [
                r'(?i)(?:shell|cmd|powershell).exe',
                r'(?i)select.*from.*where',
                r'(?i)union.*select',
                r'(?i)exec.*xp_'
            ],
            'attack_signatures': {
                'brute_force': r'failed login attempt',
                'sql_injection': r'(?i)(\%27)|(\')|(\-\-)|(\%23)|(#)',
                'xss': r'(?i)(<script|javascript:)',
                'directory_traversal': r'(?i)\.\./'
            }
        }

    def analyze_threats(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze threats in events"""
        analysis = {
            'threat_summary': self._generate_threat_summary(events),
            'detected_attacks': self._detect_attacks(events),
            'risk_assessment': self._assess_risks(events),
            'recommendations': self._generate_recommendations(events)
        }
        return analysis

    def _generate_threat_summary(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary of threats"""
        return {
            'total_events': len(events),
            'high_severity': len([e for e in events if e['severity'] == 'high']),
            'medium_severity': len([e for e in events if e['severity'] == 'medium']),
            'low_severity': len([e for e in events if e['severity'] == 'low']),
            'threat_categories': self._categorize_threats(events)
        }

    def _detect_attacks(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect potential attacks"""
        attacks = []
        
        for event in events:
            detected_attacks = []
            
            # Check against attack signatures
            for attack_type, pattern in self.threat_indicators['attack_signatures'].items():
                if re.search(pattern, event.get('description', '')):
                    detected_attacks.append(attack_type)
            
            if detected_attacks:
                attacks.append({
                    'timestamp': event['timestamp'],
                    'source': event['source'],
                    'attack_types': detected_attacks,
                    'severity': event['severity'],
                    'details': event
                })
                
        return attacks

    def _assess_risks(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess security risks"""
        return {
            'risk_level': self._calculate_risk_level(events),
            'vulnerable_assets': self._identify_vulnerable_assets(events),
            'attack_vectors': self._identify_attack_vectors(events)
        }

    def _generate_recommendations(self, events: List[Dict[str, Any]]) -> List[str]:
        """Generate security recommendations"""
        recommendations = []
        
        # Analyze patterns and generate recommendations
        if self._has_authentication_failures(events):
            recommendations.append("Implement multi-factor authentication")
            
        if self._has_suspicious_network_activity(events):
            recommendations.append("Review firewall rules and network segmentation")
            
        if self._has_malware_indicators(events):
            recommendations.append("Update antivirus and conduct system scan")
            
        return recommendations