import win32evtlog
import win32con
import win32evtlogutil
import win32security
import wmi
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, AsyncGenerator, Optional
import logging
from database.models import Event
from database.connection import AsyncSessionLocal
from schemas.event import EventCreate
import win32api

class WindowsEventCollector:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Define all Windows event log types to collect
        self.log_types = [
            'Application',
            'Security', 
            'System',
            'Setup',
            'Windows PowerShell',
            'Microsoft-Windows-Sysmon/Operational',
            'Microsoft-Windows-TaskScheduler/Operational',
            'Microsoft-Windows-Windows Defender/Operational',
            'Microsoft-Windows-Windows Firewall With Advanced Security/Firewall',
            'Microsoft-Windows-DriverFrameworks-UserMode/Operational'
        ]
        
        self.handlers = {}
        self.last_read_times = {}
        
        try:
            self.wmi_connection = wmi.WMI()
        except Exception as e:
            logging.error(f"Failed to initialize WMI: {e}")
            
        self.get_security_privileges()
        self.initialize_handlers()

    def get_security_privileges(self):
        """Get required security privileges for accessing event logs"""
        try:
            th = win32security.OpenProcessToken(
                win32api.GetCurrentProcess(), 
                win32con.TOKEN_ADJUST_PRIVILEGES | win32con.TOKEN_QUERY
            )
            
            privileges = [
                'SeSecurityPrivilege',
                'SeBackupPrivilege',
                'SeSystemtimePrivilege',
                'SeDebugPrivilege',
                'SeAuditPrivilege'
            ]
            
            for privilege in privileges:
                try:
                    id = win32security.LookupPrivilegeValue(None, privilege)
                    win32security.AdjustTokenPrivileges(
                        th,
                        False,
                        [(id, win32con.SE_PRIVILEGE_ENABLED)]
                    )
                except Exception as e:
                    logging.error(f"Failed to enable {privilege}: {str(e)}")
                    
        except Exception as e:
            logging.error(f"Failed to adjust privileges: {str(e)}")

    def convert_windows_timestamp(self, win_timestamp) -> Optional[datetime]:
        """Convert Windows timestamp to datetime"""
        try:
            if isinstance(win_timestamp, (int, float)):
                return datetime.fromtimestamp(win_timestamp)
            elif hasattr(win_timestamp, 'timestamp'):
                return datetime.fromtimestamp(win_timestamp.timestamp())
            return None
        except Exception as e:
            logging.error(f"Error converting timestamp: {str(e)}")
            return None

    def initialize_handlers(self):
        """Initialize event log handlers for each log type"""
        for log_type in self.log_types:
            try:
                handle = None
                try:
                    handle = win32evtlog.OpenEventLog(None, log_type)
                except Exception:
                    flags = win32con.GENERIC_READ | win32con.STANDARD_RIGHTS_READ
                    handle = win32evtlog.OpenEventLog(None, log_type)
                
                if handle:
                    self.handlers[log_type] = handle
                    self.last_read_times[log_type] = datetime.utcnow() - timedelta(hours=24)
                    logging.info(f"Successfully initialized handler for {log_type}")
            except Exception as e:
                logging.error(f"Failed to initialize handler for {log_type}: {e}")

    async def collect_logs(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Collect logs from all configured Windows event sources"""
        while True:
            try:
                for log_type, handle in self.handlers.items():
                    if not handle:
                        continue
                        
                    flags = win32evtlog.EVENTLOG_BACKWARDS_READ | win32evtlog.EVENTLOG_SEQUENTIAL_READ
                    try:
                        events = win32evtlog.ReadEventLog(handle, flags, 0)
                        
                        for event in events:
                            event_time = self.convert_windows_timestamp(event.TimeGenerated)
                            
                            if event_time and (log_type not in self.last_read_times or 
                                             event_time > self.last_read_times[log_type]):
                                formatted_event = await self.format_event(event, log_type)
                                if formatted_event:
                                    yield formatted_event
                        
                        self.last_read_times[log_type] = datetime.utcnow()
                    except Exception as e:
                        logging.error(f"Error reading {log_type} log: {e}")
                        
            except Exception as e:
                logging.error(f"Error in collectors: {e}")
            
            await asyncio.sleep(self.config.get('collection_interval', 5))

    async def format_event(self, event, log_type: str) -> Dict[str, Any]:
        """Format a Windows event into a detailed standardized dictionary"""
        try:
            event_time = self.convert_windows_timestamp(event.TimeGenerated)
            time_written = self.convert_windows_timestamp(event.TimeWritten)
            
            if not event_time:
                logging.error("Could not convert event time")
                return None

            # Get detailed event message
            try:
                message = win32evtlogutil.SafeFormatMessage(event, log_type)
            except:
                message = f"Could not format message for event ID {event.EventID}"

            # Get detailed user information
            try:
                sid = win32security.ConvertSidToStringSid(event.Sid)
                try:
                    domain, user, _ = win32security.LookupAccountSid(None, event.Sid)
                    user = f"{domain}\\{user}" if domain else user
                except:
                    user = sid
            except:
                sid = "N/A"
                user = "N/A"

            # Get event category name
            try:
                category = win32evtlogutil.SafeFormatMessage(event, log_type, "CategoryMessageFile")
            except:
                category = str(event.EventCategory)

            # Determine detailed severity
            severity, severity_desc = self.determine_severity(event.EventType)

            # Extract event data
            event_data = {}
            try:
                for data in event.StringInserts or []:
                    if data:
                        event_data[f"data_{len(event_data)}"] = data
            except:
                pass

            formatted_event = {
                'timestamp': event_time,
                'source': event.SourceName,
                'event_type': log_type,
                'event_id': event.EventID & 0xFFFF,
                'severity': severity,
                'severity_description': severity_desc,
                'category': category,
                'message': message,
                'user': user,
                'sid': sid,
                'computer': event.ComputerName,
                'event_data': event_data,
                'raw_data': {
                    'event_category': event.EventCategory,
                    'record_number': event.RecordNumber,
                    'time_written': time_written or event_time,
                    'event_type': event.EventType,
                    'reserved_flags': getattr(event, 'ReservedFlags', None),
                    'reserved_data': getattr(event, 'ReservedData', None),
                    'string_inserts': event.StringInserts
                }
            }

            # Store event in database
            await self.store_event(formatted_event)

            return formatted_event

        except Exception as e:
            logging.error(f"Error formatting event: {e}")
            return None

    def determine_severity(self, event_type: int) -> tuple[int, str]:
        """Determine detailed severity level and description based on event type"""
        if event_type == win32con.EVENTLOG_ERROR_TYPE:
            return 1, "Critical Error"
        elif event_type == win32con.EVENTLOG_WARNING_TYPE:
            return 2, "Warning"
        elif event_type == win32con.EVENTLOG_INFORMATION_TYPE:
            return 3, "Information"
        elif event_type == win32con.EVENTLOG_AUDIT_FAILURE:
            return 1, "Audit Failure"
        elif event_type == win32con.EVENTLOG_AUDIT_SUCCESS:
            return 3, "Audit Success"
        elif event_type == win32con.EVENTLOG_SUCCESS:
            return 3, "Success"
        else:
            return 4, "Unknown"

    async def store_event(self, event_data: Dict):
        """Store detailed event in database"""
        try:
            event_create = EventCreate(
                timestamp=event_data['timestamp'],
                source=event_data['source'],
                event_type=event_data['event_type'],
                severity=event_data['severity'],
                message=event_data['message'],
                category=event_data['category'],
                user=event_data['user'],
                computer=event_data['computer'],
                event_id=event_data['event_id'],
                raw_data=event_data['raw_data']
            )
            
            async with AsyncSessionLocal() as session:
                db_event = Event(**event_create.model_dump())
                session.add(db_event)
                await session.commit()
                
        except Exception as e:
            self.logger.error(f"Error storing event in database: {e}")

    def cleanup(self):
        """Cleanup resources"""
        for handle in self.handlers.values():
            try:
                win32evtlog.CloseEventLog(handle)
            except:
                pass