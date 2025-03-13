import json
from typing import List, Dict, Any
import logging
from datetime import datetime, timedelta
from pathlib import Path

from ai_engine.collectors.base_collector import BaseCollector
from ai_engine.collectors.windows_collector import WindowsCollector
from ai_engine.collectors.linux_collector import LinuxCollector
from ai_engine.collectors.network_collector import NetworkCollector
from ai_engine.collectors.cloud_collector import CloudCollector
from services.log_collector import LogCollector
from analytics.forensics.evidence_collector import EvidenceCollector

class EventAggregator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Default cloud config
        cloud_config = {
            'aws': {'enabled': False},
            'azure': {'enabled': False},
            'gcp': {'enabled': False}
        }
        
        # Initialize all collectors
        self.collectors = {
            'windows': WindowsCollector(),
            'linux': LinuxCollector(),
            'network': NetworkCollector(),
            'cloud': CloudCollector(config=cloud_config),
            'log': LogCollector(config={}),
            'evidence': EvidenceCollector()
        }
        
        # Track active collectors
        self.active_collectors = set()
        
    async def initialize_collectors(self):
        """Initialize and check which collectors are available"""
        for name, collector in self.collectors.items():
            try:
                if await collector.check_availability():
                    self.active_collectors.add(name)
                    self.logger.info(f"Collector {name} is available")
            except Exception as e:
                self.logger.error(f"Error initializing {name} collector: {e}")

    async def get_recent_events(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Aggregate recent events from all active collectors"""
        try:
            all_events = []
            
            # Collect from each active collector
            for collector_name in self.active_collectors:
                collector = self.collectors[collector_name]
                try:
                    # Get events from collector
                    events = await collector.collect_events()
                    
                    # Format events to standard structure
                    formatted_events = [
                        {
                            "id": f"{collector_name.upper()}-{event.get('id', hash(str(event)))}",
                            "type": event.get('type', collector_name),
                            "severity": self._determine_severity(event),
                            "source": event.get('source', collector_name),
                            "timestamp": event.get('timestamp', datetime.utcnow().isoformat()),
                            "description": event.get('message', event.get('description', 'No description')),
                            "status": event.get('status', 'active'),
                            "details": {
                                "source_ip": event.get('source_ip', 'N/A'),
                                "destination_ip": event.get('destination_ip', 'N/A'),
                                "protocol": event.get('protocol', 'N/A'),
                                "port": event.get('port', 0),
                                "user": event.get('user', 'system'),
                                "action_taken": event.get('action', 'logged'),
                                "collector_specific": event.get('details', {})
                            },
                            "metadata": {
                                "collector": collector_name,
                                "collection_time": datetime.utcnow().isoformat(),
                                "raw_event": event  # Store original event for AI analysis
                            }
                        }
                        for event in events
                    ]
                    
                    all_events.extend(formatted_events)
                    
                except Exception as e:
                    self.logger.error(f"Error collecting from {collector_name}: {e}")
                    continue

            # Sort by timestamp and limit
            all_events.sort(key=lambda x: x['timestamp'], reverse=True)
            recent_events = all_events[:limit]

            # Store events for AI analysis
            await self._store_for_analysis(recent_events)
            
            return recent_events

        except Exception as e:
            self.logger.error(f"Error aggregating events: {e}")
            return []

    async def _store_for_analysis(self, events: List[Dict[str, Any]]):
        """Store events for AI analysis"""
        try:
            # Save to dataset for AI training
            dataset_path = Path("backend/ai_engine/dataset/processed/events.json")
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Load existing events
            existing_events = []
            if dataset_path.exists():
                with open(dataset_path) as f:
                    existing_events = json.load(f)
            
            # Add new events
            existing_events.extend(events)
            
            # Keep only last 10000 events
            if len(existing_events) > 10000:
                existing_events = existing_events[-10000:]
            
            # Save updated dataset
            with open(dataset_path, 'w') as f:
                json.dump(existing_events, f)
                
        except Exception as e:
            self.logger.error(f"Error storing events for analysis: {e}")

    def _determine_severity(self, event: Dict[str, Any]) -> str:
        """Determine event severity using multiple factors"""
        # Check if severity is already set
        if 'severity' in event:
            return event['severity']
            
        # Check message content
        message = str(event.get('message', '')).lower()
        description = str(event.get('description', '')).lower()
        
        # Combine message and description for analysis
        content = f"{message} {description}"
        
        # Critical indicators
        if any(word in content for word in [
            'critical', 'attack', 'breach', 'compromise', 'malware',
            'ransomware', 'exploit', 'remote code execution'
        ]):
            return "critical"
            
        # High severity indicators
        if any(word in content for word in [
            'error', 'fail', 'denied', 'violation', 'suspicious',
            'unauthorized', 'blocked', 'invalid'
        ]):
            return "high"
            
        # Medium severity indicators
        if any(word in content for word in [
            'warning', 'warn', 'retry', 'unusual', 'notice'
        ]):
            return "medium"
            
        return "low" 