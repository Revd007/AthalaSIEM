import win32evtlog
import win32con
import win32evtlogutil
import win32security
import wmi
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, AsyncGenerator
import logging
from database.models import Event
from database.connection import get_db

class WindowsEventCollector:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.log_types = config.get('log_types', ['System', 'Security', 'Application'])
        self.handlers = {}
        self.last_read_times = {}
        self.wmi_connection = wmi.WMI()
        self.initialize_handlers()

    def initialize_handlers(self):
        """Initialize event log handlers for each log type"""
        for log_type in self.log_types:
            try:
                handle = win32evtlog.OpenEventLog(None, log_type)
                self.handlers[log_type] = handle
                self.last_read_times[log_type] = datetime.utcnow() - timedelta(hours=1)
            except Exception as e:
                logging.error(f"Failed to initialize handler for {log_type}: {e}")

    async def collect_logs(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Collect logs from all configured sources"""
        while True:
            try:
                for log_type, handle in self.handlers.items():
                    flags = win32evtlog.EVENTLOG_BACKWARDS_READ | win32evtlog.EVENTLOG_SEQUENTIAL_READ
                    events = win32evtlog.ReadEventLog(handle, flags, 0)
                    
                    for event in events:
                        event_time = datetime.fromtimestamp(event.TimeGenerated)
                        
                        if event_time > self.last_read_times[log_type]:
                            formatted_event = await self.format_event(event, log_type)
                            if formatted_event:
                                yield formatted_event
                    
                    self.last_read_times[log_type] = datetime.utcnow()

            except Exception as e:
                logging.error(f"Error collecting Windows events: {e}")
            
            await asyncio.sleep(self.config.get('collection_interval', 10))

    async def format_event(self, event, log_type: str) -> Dict[str, Any]:
        """Format a Windows event into a standardized dictionary"""
        try:
            # Get event message
            try:
                message = win32evtlogutil.SafeFormatMessage(event, log_type)
            except:
                message = f"Could not format message for event ID {event.EventID}"

            # Get user information
            try:
                sid = win32security.ConvertSidToStringSid(event.Sid)
                try:
                    user = win32security.LookupAccountSid(None, event.Sid)[0]
                except:
                    user = sid
            except:
                sid = "N/A"
                user = "N/A"

            # Determine severity
            severity = self.determine_severity(event.EventType)

            formatted_event = {
                'timestamp': datetime.fromtimestamp(event.TimeGenerated),
                'source': event.SourceName,
                'event_type': log_type,
                'event_id': event.EventID & 0xFFFF,  # Mask out qualification bits
                'severity': severity,
                'message': message,
                'user': user,
                'sid': sid,
                'computer': event.ComputerName,
                'raw_data': {
                    'event_category': event.EventCategory,
                    'record_number': event.RecordNumber,
                    'time_written': datetime.fromtimestamp(event.TimeWritten),
                    'event_type': event.EventType,
                }
            }

            # Store event in database
            await self.store_event(formatted_event)

            return formatted_event

        except Exception as e:
            logging.error(f"Error formatting event: {e}")
            return None

    def determine_severity(self, event_type: int) -> int:
        """Determine severity level based on event type"""
        if event_type == win32con.EVENTLOG_ERROR_TYPE:
            return 1  # High severity
        elif event_type == win32con.EVENTLOG_WARNING_TYPE:
            return 2  # Medium severity
        elif event_type == win32con.EVENTLOG_INFORMATION_TYPE:
            return 3  # Low severity
        elif event_type == win32con.EVENTLOG_AUDIT_FAILURE:
            return 1  # High severity
        elif event_type == win32con.EVENTLOG_AUDIT_SUCCESS:
            return 3  # Low severity
        else:
            return 4  # Unknown severity

    async def store_event(self, event_data: Dict[str, Any]):
        """Store event in database"""
        try:
            async with get_db() as db:
                event = Event(
                    timestamp=event_data['timestamp'],
                    source=event_data['source'],
                    event_type=event_data['event_type'],
                    event_id=event_data['event_id'],
                    severity=event_data['severity'],
                    message=event_data['message'],
                    user=event_data['user'],
                    computer=event_data['computer'],
                    raw_data=event_data['raw_data']
                )
                db.add(event)
                await db.commit()
        except Exception as e:
            logging.error(f"Error storing event in database: {e}")

    async def monitor_system_changes(self):
        """Monitor for system changes using WMI"""
        try:
            # Monitor for process creation
            process_watcher = self.wmi_connection.Win32_Process.watch_for(
                notification_type="creation"
            )
            
            # Monitor for service changes
            service_watcher = self.wmi_connection.Win32_Service.watch_for(
                notification_type="modification"
            )
            
            while True:
                try:
                    process = await asyncio.to_thread(process_watcher)
                    if process:
                        await self.handle_process_event(process)
                    
                    service = await asyncio.to_thread(service_watcher)
                    if service:
                        await self.handle_service_event(service)
                        
                except Exception as e:
                    logging.error(f"Error in system monitoring: {e}")
                
                await asyncio.sleep(1)
                
        except Exception as e:
            logging.error(f"Error setting up system monitoring: {e}")

    async def handle_process_event(self, process):
        """Handle process creation events"""
        event_data = {
            'timestamp': datetime.utcnow(),
            'source': 'ProcessMonitor',
            'event_type': 'ProcessCreation',
            'severity': 2,
            'message': f"New process created: {process.Caption}",
            'user': process.GetOwner()[2] if process.GetOwner()[0] == 0 else 'N/A',
            'computer': process.CSName,
            'raw_data': {
                'process_id': process.ProcessId,
                'parent_process_id': process.ParentProcessId,
                'command_line': process.CommandLine,
                'executable_path': process.ExecutablePath
            }
        }
        await self.store_event(event_data)

    async def handle_service_event(self, service):
        """Handle service change events"""
        event_data = {
            'timestamp': datetime.utcnow(),
            'source': 'ServiceMonitor',
            'event_type': 'ServiceChange',
            'severity': 2,
            'message': f"Service changed: {service.Caption}",
            'user': 'SYSTEM',
            'computer': service.SystemName,
            'raw_data': {
                'service_name': service.Name,
                'display_name': service.DisplayName,
                'state': service.State,
                'start_mode': service.StartMode
            }
        }
        await self.store_event(event_data)

    def cleanup(self):
        """Cleanup resources"""
        for handle in self.handlers.values():
            try:
                win32evtlog.CloseEventLog(handle)
            except:
                pass