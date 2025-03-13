from typing import Dict, Any, Optional
import CloudFlare  # pip install cloudflare
import boto3  # pip install boto3
import logging
from config.security_config import DomainSettings

class DNSService:
    def __init__(self, settings: DomainSettings):
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        self._init_provider()
        
    def _init_provider(self):
        """Initialize DNS provider client"""
        if self.settings.DNS_PROVIDER == "cloudflare":
            self.client = CloudFlare.CloudFlare(token=self.settings.DNS_API_TOKEN)
        elif self.settings.DNS_PROVIDER == "route53":
            self.client = boto3.client('route53')
        else:
            raise ValueError(f"Unsupported DNS provider: {self.settings.DNS_PROVIDER}")
            
    async def add_record(
        self,
        name: str,
        type: str,
        content: str,
        ttl: int = 1,  # 1 = Auto
        proxied: bool = True
    ):
        """Add DNS record"""
        try:
            if self.settings.DNS_PROVIDER == "cloudflare":
                record = {
                    'name': name,
                    'type': type,
                    'content': content,
                    'ttl': ttl,
                    'proxied': proxied
                }
                self.client.zones.dns_records.post(
                    self.settings.DNS_ZONE_ID, 
                    data=record
                )
            elif self.settings.DNS_PROVIDER == "route53":
                self.client.change_resource_record_sets(
                    HostedZoneId=self.settings.DNS_ZONE_ID,
                    ChangeBatch={
                        'Changes': [{
                            'Action': 'CREATE',
                            'ResourceRecordSet': {
                                'Name': name,
                                'Type': type,
                                'TTL': ttl,
                                'ResourceRecords': [{'Value': content}]
                            }
                        }]
                    }
                )
                
            self.logger.info(f"Added DNS record: {name} {type} {content}")
            
        except Exception as e:
            self.logger.error(f"Failed to add DNS record: {str(e)}")
            raise
            
    async def update_ssl_records(self, validation_records: Dict[str, Any]):
        """Update DNS records for SSL validation"""
        for record in validation_records:
            await self.add_record(
                name=record['name'],
                type=record['type'],
                content=record['content'],
                ttl=1,
                proxied=False
            ) 