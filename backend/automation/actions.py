import subprocess
import requests
from typing import Dict, Any
import logging
from datetime import datetime

class SecurityActions:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.firewall_api = config.get('firewall_api')
        self.email_config = config.get('email_config')

    async def block_ip(self, ip_address: str) -> Dict[str, Any]:
        try:
            # Add IP to firewall blocklist
            response = requests.post(
                f"{self.firewall_api}/block",
                json={"ip": ip_address}
            )
            
            return {
                'status': 'success',
                'message': f'IP {ip_address} blocked successfully',
                'timestamp': datetime.utcnow()
            }
        except Exception as e:
            logging.error(f"Failed to block IP {ip_address}: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'timestamp': datetime.utcnow()
            }

    async def quarantine_host(self, hostname: str) -> Dict[str, Any]:
        try:
            # Add host to quarantine VLAN
            response = requests.post(
                f"{self.firewall_api}/quarantine",
                json={"hostname": hostname}
            )
            
            # Block all outbound traffic except essential services
            response = requests.post(
                f"{self.firewall_api}/restrict",
                json={
                    "hostname": hostname,
                    "allowed_ports": [53, 80, 443]  # DNS and web traffic only
                }
            )
            
            # Log quarantine action
            logging.info(f"Host {hostname} placed in quarantine VLAN with restricted access")
            # Apply VLAN changes
            response = requests.post(
                f"{self.firewall_api}/vlan",
                json={
                    "hostname": hostname,
                    "vlan": "quarantine"
                }
            )
            
            # Apply additional firewall rules
            response = requests.post(
                f"{self.firewall_api}/rules",
                json={
                    "hostname": hostname,
                    "rules": [
                        {"action": "deny", "direction": "inbound", "all": True},
                        {"action": "allow", "direction": "outbound", "ports": [53, 80, 443]}
                    ]
                }
            )
            return {
                'status': 'success',
                'message': f'Host {hostname} quarantined successfully',
                'timestamp': datetime.utcnow()
            }
        except Exception as e:
            logging.error(f"Failed to quarantine host {hostname}: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'timestamp': datetime.utcnow()
            }

    async def send_alert(self, alert_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Send email alert
            email_payload = {
                "to": alert_data.get("recipients", []),
                "subject": f"Security Alert: {alert_data.get('title')}",
                "body": alert_data.get("description", "")
            }
            email_response = requests.post(
                f"{self.notification_api}/email",
                json=email_payload
            )

            # Send SMS if phone numbers provided
            if "phone_numbers" in alert_data:
                sms_payload = {
                    "to": alert_data["phone_numbers"],
                    "message": f"ALERT: {alert_data.get('title')} - {alert_data.get('description')}"
                }
                sms_response = requests.post(
                    f"{self.notification_api}/sms",
                    json=sms_payload
                )

            # Log alert details
            logging.info(
                f"Alert sent - Title: {alert_data.get('title')}, "
                f"Recipients: {alert_data.get('recipients')}"
            )
            return {
                'status': 'success',
                'message': 'Alert sent successfully',
                'timestamp': datetime.utcnow()
            }
        except Exception as e:
            logging.error(f"Failed to send alert: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'timestamp': datetime.utcnow()
            }