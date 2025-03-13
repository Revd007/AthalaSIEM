from typing import List, Dict, Any
import logging
import win32evtlog
import win32con
import win32evtlogutil
from datetime import datetime, timedelta
from .base_collector import BaseCollector

class WindowsCollector(BaseCollector):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.log_types = ['System', 'Application', 'Security']
        self.event_types = {
            win32con.EVENTLOG_AUDIT_SUCCESS: 'audit_success',
            win32con.EVENTLOG_AUDIT_FAILURE: 'audit_failure',
            win32con.EVENTLOG_INFORMATION_TYPE: 'information',
            win32con.EVENTLOG_WARNING_TYPE: 'warning',
            win32con.EVENTLOG_ERROR_TYPE: 'error'
        }

    async def check_availability(self) -> bool:
        """Check if Windows event logs are accessible"""
        try:
            handle = win32evtlog.OpenEventLog(None, 'System')
            win32evtlog.CloseEventLog(handle)
            return True
        except Exception as e:
            self.logger.error(f"Windows event logs not accessible: {e}")
            return False

    async def collect_events(self, hours: int = 1) -> List[Dict[str, Any]]:
        """Collect Windows event logs"""
        events = []
        try:
            for log_type in self.log_types:
                handle = win32evtlog.OpenEventLog(None, log_type)
                flags = win32evtlog.EVENTLOG_BACKWARDS_READ | win32evtlog.EVENTLOG_SEQUENTIAL_READ
                
                try:
                    windows_events = win32evtlog.ReadEventLog(handle, flags, 0)
                    
                    for event in windows_events:
                        # Convert Windows timestamp to datetime
                        event_time = event.TimeGenerated
                        
                        # Skip old events
                        if datetime.now() - event_time > timedelta(hours=hours):
                            continue
                            
                        # Format event data
                        formatted_event = {
                            'id': f"WIN-{event.EventID}",
                            'type': 'windows_event',
                            'source': event.SourceName,
                            'timestamp': event_time.isoformat(),
                            'severity': self._map_severity(event.EventType),
                            'description': win32evtlogutil.SafeFormatMessage(event, log_type),
                            'status': 'active',
                            'details': {
                                'event_type': self.event_types.get(event.EventType, 'unknown'),
                                'category': event.EventCategory,
                                'log_type': log_type,
                                'computer_name': event.ComputerName,
                                'sid': str(event.Sid) if event.Sid else None,
                                'raw_data': event.StringInserts if event.StringInserts else []
                            }
                        }
                        
                        events.append(formatted_event)
                        
                except Exception as e:
                    self.logger.error(f"Error reading {log_type} events: {e}")
                    continue
                    
                finally:
                    win32evtlog.CloseEventLog(handle)
                    
            return events
            
        except Exception as e:
            self.logger.error(f"Error collecting Windows events: {e}")
            return []

    def _map_severity(self, event_type: int) -> str:
        """Map Windows event type to severity"""
        if event_type == win32con.EVENTLOG_ERROR_TYPE:
            return 'high'
        elif event_type == win32con.EVENTLOG_WARNING_TYPE:
            return 'medium'
        elif event_type == win32con.EVENTLOG_AUDIT_FAILURE:
            return 'high'
        elif event_type in [win32con.EVENTLOG_INFORMATION_TYPE, win32con.EVENTLOG_AUDIT_SUCCESS]:
            return 'low'
        return 'unknown'

    async def get_collector_status(self) -> Dict[str, Any]:
        """Get collector status and metrics"""
        try:
            is_available = await self.check_availability()
            recent_events = await self.collect_events(hours=1)
            
            return {
                'status': 'active' if is_available else 'inactive',
                'collector_type': 'windows',
                'metrics': {
                    'events_collected': len(recent_events),
                    'log_types_available': len(self.log_types),
                    'last_collection': datetime.utcnow().isoformat()
                },
                'errors': [],
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting collector status: {e}")
            return {
                'status': 'error',
                'collector_type': 'windows',
                'errors': [str(e)],
                'timestamp': datetime.utcnow().isoformat()
            }