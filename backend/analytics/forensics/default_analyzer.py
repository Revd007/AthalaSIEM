from typing import Dict, List, Any
from datetime import datetime, timedelta
import re
import logging

class DefaultForensicAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.patterns = {
            'ip_address': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'timestamp': r'\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}',
            'username': r'user[:\s]+([^\s]+)',
            'process_id': r'pid[:\s]+(\d+)',
        }

    def analyze_event_sequence(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze sequence of events for patterns and anomalies"""
        analysis = {
            'timeline': self._create_timeline(events),
            'key_entities': self._extract_key_entities(events),
            'suspicious_patterns': self._identify_suspicious_patterns(events),
            'impact_assessment': self._assess_impact(events)
        }
        return analysis

    def _create_timeline(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create chronological timeline of events"""
        timeline = []
        for event in sorted(events, key=lambda x: x['timestamp']):
            timeline.append({
                'timestamp': event['timestamp'],
                'event_type': event['type'],
                'description': event['description'],
                'severity': event['severity'],
                'source': event['source']
            })
        return timeline

    def _extract_key_entities(self, events: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Extract key entities from events"""
        entities = {
            'ip_addresses': set(),
            'usernames': set(),
            'processes': set(),
            'files': set()
        }

        for event in events:
            description = event.get('description', '')
            
            # Extract IPs
            ips = re.findall(self.patterns['ip_address'], description)
            entities['ip_addresses'].update(ips)
            
            # Extract usernames
            usernames = re.findall(self.patterns['username'], description)
            entities['usernames'].update(usernames)
            
            # Extract processes
            if 'process' in event:
                entities['processes'].add(event['process'])
                
            # Extract files
            if 'file' in event:
                entities['files'].add(event['file'])

        return {k: list(v) for k, v in entities.items()}

    def _identify_suspicious_patterns(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify suspicious patterns in events"""
        patterns = []
        
        # Check for failed login attempts
        login_attempts = self._analyze_login_attempts(events)
        if login_attempts['suspicious']:
            patterns.append(login_attempts)
            
        # Check for file system anomalies
        fs_anomalies = self._analyze_filesystem_activity(events)
        if fs_anomalies['suspicious']:
            patterns.append(fs_anomalies)
            
        # Check for network anomalies
        net_anomalies = self._analyze_network_activity(events)
        if net_anomalies['suspicious']:
            patterns.append(net_anomalies)

        return patterns

    def _assess_impact(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess potential impact of events"""
        return {
            'affected_systems': self._identify_affected_systems(events),
            'severity_level': self._calculate_severity_level(events),
            'potential_risks': self._identify_potential_risks(events)
        }