from typing import List, Dict, Any
import logging
from datetime import datetime, timedelta
from .base_collector import BaseCollector
import boto3
from azure.monitor.ingestion import LogsIngestionClient
from google.cloud import logging as google_logging

class CloudCollector(BaseCollector):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.config = config or {
            'aws': {
                'enabled': False,
                'region': 'us-east-1',
                'access_key_id': None,
                'secret_access_key': None
            },
            'azure': {
                'enabled': False,
                'subscription_id': None,
                'tenant_id': None,
                'client_id': None,
                'client_secret': None
            },
            'gcp': {
                'enabled': False,
                'project_id': None,
                'credentials_path': None
            }
        }
        self.providers = []
        self._initialize_providers()

    def _initialize_providers(self):
        """Initialize enabled cloud providers"""
        if self.config.get('aws', {}).get('enabled'):
            self.providers.append('aws')
        if self.config.get('azure', {}).get('enabled'):
            self.providers.append('azure')
        if self.config.get('gcp', {}).get('enabled'):
            self.providers.append('gcp')

    async def check_availability(self) -> bool:
        """Check if cloud services are accessible"""
        try:
            return len(self.providers) > 0
        except Exception as e:
            self.logger.error(f"Error checking cloud availability: {e}")
            return False

    async def collect_events(self, hours: int = 1) -> List[Dict[str, Any]]:
        """Collect events from cloud services"""
        events = []
        try:
            for provider in self.providers:
                if provider == 'aws':
                    events.extend(await self._collect_aws_events(hours))
                elif provider == 'azure':
                    events.extend(await self._collect_azure_events(hours))
                elif provider == 'gcp':
                    events.extend(await self._collect_gcp_events(hours))
            return events
        except Exception as e:
            self.logger.error(f"Error collecting cloud events: {e}")
            return []

    async def get_collector_status(self) -> Dict[str, Any]:
        """Get collector status and metrics"""
        try:
            is_available = await self.check_availability()
            recent_events = await self.collect_events(hours=1)
            
            return {
                'status': 'active' if is_available else 'inactive',
                'collector_type': 'cloud',
                'metrics': {
                    'events_collected': len(recent_events),
                    'providers_enabled': len(self.providers),
                    'providers': self.providers,
                    'last_collection': datetime.utcnow().isoformat()
                },
                'errors': [],
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error getting collector status: {e}")
            return {
                'status': 'error',
                'collector_type': 'cloud',
                'errors': [str(e)],
                'timestamp': datetime.utcnow().isoformat()
            }

    async def _collect_aws_events(self, hours: int) -> List[Dict[str, Any]]:
        """Collect AWS CloudTrail events"""
        events = []
        try:
            if 'aws' not in self.providers:
                return []

            # Initialize AWS CloudTrail client
            cloudtrail = boto3.client(
                'cloudtrail',
                region_name=self.config['aws']['region'],
                aws_access_key_id=self.config['aws']['access_key_id'],
                aws_secret_access_key=self.config['aws']['secret_access_key']
            )

            # Calculate time range
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=hours)

            # Get events from CloudTrail
            response = cloudtrail.lookup_events(
                StartTime=start_time,
                EndTime=end_time,
                MaxResults=1000  # Adjust as needed
            )

            for event in response.get('Events', []):
                events.append({
                    'id': event.get('EventId'),
                    'type': 'aws_cloudtrail',
                    'source': event.get('EventSource'),
                    'timestamp': event.get('EventTime').isoformat(),
                    'severity': self._determine_aws_severity(event),
                    'description': f"AWS CloudTrail: {event.get('EventName')}",
                    'status': 'active',
                    'details': {
                        'event_name': event.get('EventName'),
                        'username': event.get('Username'),
                        'resources': event.get('Resources', []),
                        'aws_region': event.get('AwsRegion'),
                        'source_ip': event.get('SourceIPAddress'),
                        'user_agent': event.get('UserAgent'),
                        'error_code': event.get('ErrorCode'),
                        'error_message': event.get('ErrorMessage')
                    }
                })

            return events

        except Exception as e:
            self.logger.error(f"Error collecting AWS events: {e}")
            return []

    async def _collect_azure_events(self, hours: int) -> List[Dict[str, Any]]:
        """Collect Azure Activity Log events"""
        events = []
        try:
            if 'azure' not in self.providers:
                return []

            # Initialize Azure Monitor client
            client = LogsIngestionClient(
                endpoint=f"https://management.azure.com/subscriptions/{self.config['azure']['subscription_id']}",
                credential=self.config['azure']['client_secret']
            )

            # Calculate time range
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=hours)

            # Query Activity Log
            filter_query = f"eventTimestamp ge '{start_time.isoformat()}' and eventTimestamp le '{end_time.isoformat()}'"
            
            activity_logs = client.activity_logs.list(
                filter=filter_query,
                select="eventTimestamp,eventName,resourceId,caller,level,operationName,status"
            )

            for log in activity_logs:
                events.append({
                    'id': log.correlation_id,
                    'type': 'azure_activity',
                    'source': 'azure_monitor',
                    'timestamp': log.event_timestamp.isoformat(),
                    'severity': log.level.lower(),
                    'description': f"Azure: {log.operation_name}",
                    'status': log.status.value,
                    'details': {
                        'operation': log.operation_name,
                        'resource_id': log.resource_id,
                        'caller': log.caller,
                        'category': log.category,
                        'subscription_id': log.subscription_id,
                        'tenant_id': log.tenant_id,
                        'properties': log.properties
                    }
                })

            return events

        except Exception as e:
            self.logger.error(f"Error collecting Azure events: {e}")
            return []

    async def _collect_gcp_events(self, hours: int) -> List[Dict[str, Any]]:
        """Collect Google Cloud Audit Log events"""
        events = []
        try:
            if 'gcp' not in self.providers:
                return []

            # Initialize GCP Logging client
            logging_client = google_logging.Client.from_service_account_json(
                self.config['gcp']['credentials_path']
            )

            # Calculate time range
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=hours)

            # Create filter for audit logs
            filter_str = (
                f'resource.type="audited_resource" AND '
                f'timestamp >= "{start_time.isoformat()}Z" AND '
                f'timestamp <= "{end_time.isoformat()}Z"'
            )

            # Get audit log entries
            try:
                entries = logging_client.list_entries(
                    project_ids=[self.config['gcp']['project_id']],
                    filter_=filter_str,
                    order_by="timestamp desc"
                )

                for entry in entries:
                    # Extract audit log data
                    audit_data = entry.to_api_repr()
                    
                    events.append({
                        'id': entry.insert_id,
                        'type': 'gcp_audit',
                        'source': audit_data.get('resource', {}).get('type'),
                        'timestamp': entry.timestamp.isoformat(),
                        'severity': entry.severity.lower() if entry.severity else 'default',
                        'description': f"GCP: {audit_data.get('protoPayload', {}).get('methodName', 'unknown')}",
                        'status': 'active',
                        'details': {
                            'method': audit_data.get('protoPayload', {}).get('methodName'),
                            'resource_name': audit_data.get('resource', {}).get('labels', {}).get('name'),
                            'service_name': audit_data.get('protoPayload', {}).get('serviceName'),
                            'principal_email': audit_data.get('protoPayload', {}).get('authenticationInfo', {}).get('principalEmail'),
                            'project_id': self.config['gcp']['project_id'],
                            'labels': dict(entry.labels or {}),
                            'metadata': dict(entry.metadata or {})
                        }
                    })

            except Exception as e:
                self.logger.error(f"Error processing GCP audit log entry: {e}")

            return events

        except Exception as e:
            self.logger.error(f"Error collecting GCP events: {e}")
            return []

    def _determine_aws_severity(self, event: Dict[str, Any]) -> str:
        """Determine severity of AWS events"""
        # Error events are high severity
        if event.get('ErrorCode'):
            return 'high'
            
        # Check event name for critical operations
        event_name = event.get('EventName', '').lower()
        if any(word in event_name for word in [
            'delete', 'remove', 'modify', 'update', 'create'
        ]):
            return 'medium'
            
        # Check for security-related events
        if any(word in event.get('EventSource', '').lower() for word in [
            'iam', 'security', 'guard', 'kms'
        ]):
            return 'medium'
            
        return 'low'