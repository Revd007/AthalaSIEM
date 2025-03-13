from enum import Enum

class UserRole(str, Enum):
    ADMIN = "admin"
    ANALYST = "analyst"
    OPERATOR = "operator"
    VIEWER = "viewer"

    @classmethod
    def has_value(cls, value):
        return value in [item.value for item in cls]

class AgentType(str, Enum):
    WINDOWS_COLLECTOR = "windows_collector"
    LINUX_COLLECTOR = "linux_collector"
    MACOS_COLLECTOR = "macos_collector"
    NETWORK_COLLECTOR = "network_collector"
    CLOUD_COLLECTOR = "cloud_collector"

    @classmethod
    def get_permissions(cls, agent_type: str) -> list[str]:
        agent_permissions = {
            'windows_collector': ['event_log', 'syslog'],
            'linux_collector': ['syslog', 'network_flow'],
            'macos_collector': ['syslog', 'event_log'],
            'network_collector': ['network_flow', 'syslog'],
            'cloud_collector': ['aws_cloudtrail', 'azure_monitor', 'gcp_audit']
        }
        return agent_permissions.get(agent_type.lower(), [])

class CollectorType(str, Enum):
    EVENT_LOG = "event_log"
    SYSLOG = "syslog"
    NETWORK_FLOW = "network_flow"
    AWS_CLOUDTRAIL = "aws_cloudtrail"
    AZURE_MONITOR = "azure_monitor"
    GCP_AUDIT = "gcp_audit"

class AgentStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"