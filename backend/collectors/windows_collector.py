import win32evtlog
import win32con
import win32evtlogutil
import win32security
import wmi
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, AsyncGenerator, Optional
import logging
from database.models import Event, Alert
from database.connection import AsyncSessionLocal, get_db
from schemas.event import EventCreate
from schemas.alert import AlertCreate
import win32api
from ai_engine.core.model_manager import ModelManager
from ai_engine.correlation_engine import CorrelationEngine
from ai_engine.anomaly_detection import AnomalyDetector
from ai_engine.threat_intelligence import ThreatIntelligence
from ai_engine.processors.feature_engineering import FeatureEngineer

class WindowsEventCollector:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize model manager with config
        model_config = {
            'feature_toggles': {
                'anomaly_detection': True,
                'threat_detection': True
            }
        }
        self.model_manager = ModelManager(model_config)
        
        # Initialize components with model manager
        self.correlation_engine = CorrelationEngine(model_manager=self.model_manager)
        self.threat_intel = ThreatIntelligence(model_manager=self.model_manager)
        self.feature_engineer = FeatureEngineer()
        
        self.log_types = config.get('log_types', ['System', 'Security', 'Application'])
        self.handlers = {}
        self.last_read_times = {}
        
        try:
            self.wmi_connection = wmi.WMI()
        except Exception as e:
            logging.error(f"Failed to initialize WMI: {e}")
        self.get_security_privileges()
        self.initialize_handlers()

    def get_security_privileges(self):
        """Get required security privileges"""
        try:
            # Get the current process token
            th = win32security.OpenProcessToken(
                win32api.GetCurrentProcess(), 
                win32con.TOKEN_ADJUST_PRIVILEGES | win32con.TOKEN_QUERY
            )
            
            # Enable required privileges
            privileges = [
                'SeSecurityPrivilege',
                'SeBackupPrivilege',
                'SeSystemtimePrivilege'
            ]
            
            for privilege in privileges:
                try:
                    # Get the ID for the privilege
                    id = win32security.LookupPrivilegeValue(None, privilege)
                    # Enable the privilege
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
                # Try to run with elevated privileges
                handle = None
                try:
                    handle = win32evtlog.OpenEventLog(None, log_type)
                except Exception as e:
                    # If failed, try to open with backup privileges
                    flags = win32con.GENERIC_READ | win32con.STANDARD_RIGHTS_READ
                    handle = win32evtlog.OpenEventLog(None, log_type)
                
                if handle:
                    self.handlers[log_type] = handle
                    self.last_read_times[log_type] = datetime.utcnow() - timedelta(hours=1)
            except Exception as e:
                logging.error(f"Failed to initialize handler for {log_type}: {e}")

    async def collect_logs(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Collect logs from all configured sources"""
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
            
            await asyncio.sleep(self.config.get('collection_interval', 10))

    async def format_event(self, event, log_type: str) -> Dict[str, Any]:
        """Format a Windows event into a standardized dictionary"""
        try:
            # Convert timestamps using the new method
            event_time = self.convert_windows_timestamp(event.TimeGenerated)
            time_written = self.convert_windows_timestamp(event.TimeWritten)
            
            if not event_time:
                logging.error("Could not convert event time")
                return None

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
                'timestamp': event_time,  # Use converted time
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
                    'time_written': time_written or event_time,  # Use converted time
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

    async def process_with_ai(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process event data through AI engines"""
        try:
            # Correlation Engine Processing
            correlation_result = await self.correlation_engine.process_event(event_data)
            
            # Add AI insights to the event data
            event_data['ai_analysis'] = {
                'correlation_score': correlation_result.get('correlation_score', 0),
                'related_events': correlation_result.get('related_events', []),
                'threat_level': correlation_result.get('threat_level', 'low'),
                'recommendations': correlation_result.get('recommendations', [])
            }

            # Anomaly Detection (if available)
            if self.anomaly_detector:
                anomaly_result = await self.anomaly_detector.analyze(event_data)
                event_data['ai_analysis']['anomaly_score'] = anomaly_result.get('anomaly_score', 0)
                event_data['ai_analysis']['is_anomaly'] = anomaly_result.get('is_anomaly', False)

            # Threat Intelligence (if available)
            if self.threat_intel:
                threat_result = await self.threat_intel.analyze(event_data)
                event_data['ai_analysis']['threat_intel'] = threat_result

            return event_data
        except Exception as e:
            logging.error(f"Error in AI processing: {e}")
            return event_data

    async def store_event(self, event_data: Dict):
        """Store event in database"""
        try:
            # Create event data using Pydantic model
            event_create = EventCreate(
                timestamp=event_data['timestamp'],
                source=event_data['source'],
                event_type=event_data['event_type'],
                severity=event_data['severity'],
                message=event_data['message'],
                ai_analysis=event_data.get('ai_analysis')
            )
            
            async with AsyncSessionLocal() as session:
                # Convert Pydantic model to SQLAlchemy model
                db_event = Event(**event_create.model_dump())
                session.add(db_event)
                await session.commit()
                
        except Exception as e:
            self.logger.error(f"Error storing event in database: {e}")

    async def create_alert(self, event_id: int, severity: int, title: str, description: str, ai_analysis: Dict[str, Any]):
        """Create a new alert from an event"""
        try:
            alert_data = AlertCreate(
                title=title,
                description=description,
                severity=severity,
                source_event_id=event_id,
                ai_analysis=ai_analysis
            )
            
            async with AsyncSessionLocal() as session:
                db_alert = Alert(**alert_data.model_dump())
                session.add(db_alert)
                await session.commit()
                await session.refresh(db_alert)
                return db_alert
                
        except Exception as e:
            self.logger.error(f"Error creating alert: {e}")
            return None

    async def monitor_system_changes(self):
        try:
            # Initialize WMI with specific privileges and namespace
            wmi_service = None
            try:
                # Create WMI connection with explicit impersonation level
                wmi_service = wmi.WMI(privileges=['SecurityPrivilege', 'SystemProfile'])
            except Exception as e:
                logging.error(f"Failed to initialize WMI service: {e}")
                return

            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    # Create watchers with explicit WQL queries
                    process_watcher = wmi_service.watch_for(
                        raw_wql="SELECT * FROM __InstanceCreationEvent WITHIN 1 WHERE TargetInstance ISA 'Win32_Process'"
                    )
                    
                    service_watcher = wmi_service.watch_for(
                        raw_wql="SELECT * FROM __InstanceModificationEvent WITHIN 1 WHERE TargetInstance ISA 'Win32_Service'"
                    )
                    
                    logging.info("Successfully initialized WMI watchers")
                    
                    while True:
                        try:
                            # Use asyncio.to_thread with timeout
                            process = await asyncio.wait_for(
                                asyncio.to_thread(process_watcher, timeout_ms=1000), 
                                timeout=2.0
                            )
                            if process:
                                await self.handle_process_event(process.targetinstance)
                            
                            service = await asyncio.wait_for(
                                asyncio.to_thread(service_watcher, timeout_ms=1000),
                                timeout=2.0
                            )
                            if service:
                                await self.handle_service_event(service.targetinstance)
                                
                        except asyncio.TimeoutError:
                            # This is expected, just continue
                            pass
                        except Exception as e:
                            if "Unexpected COM Error" in str(e):
                                # If we get a COM error, break the inner loop to trigger a retry
                                raise
                            logging.error(f"Error processing WMI event: {e}")
                        
                        await asyncio.sleep(1)
                        
                except Exception as e:
                    retry_count += 1
                    if retry_count >= max_retries:
                        logging.error(f"Failed to monitor system after {max_retries} attempts: {e}")
                        return
                    
                    logging.warning(f"Retry {retry_count}/{max_retries} for system monitoring. Error: {str(e)}")
                    
                    # Clean up WMI connection before retry
                    try:
                        wmi_service = None
                        wmi_service = wmi.WMI(privileges=['SecurityPrivilege', 'SystemProfile'])
                    except Exception as cleanup_error:
                        logging.error(f"Failed to reinitialize WMI connection: {cleanup_error}")
                    
                    await asyncio.sleep(5)  # Wait longer between retries
                    
        except Exception as e:
            logging.error(f"Critical error in system monitoring: {e}")
        finally:
            # Cleanup WMI resources
            if wmi_service:
                try:
                    wmi_service = None
                except Exception as cleanup_error:
                    logging.error(f"Error during WMI cleanup: {cleanup_error}")

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

    async def process_event(self, event):
        try:
            # Extract features
            features = self.feature_engineer.process_features(event)
            
            # Get AI analysis results using model manager directly
            anomaly_result = await self.model_manager.detect_anomalies(features)
            threat_result = await self.model_manager.analyze_threats(features)
            
            return {
                'is_anomaly': bool(anomaly_result.get('is_anomaly', False)),
                'anomaly_score': float(anomaly_result.get('anomaly_score', 0.0)),
                'is_threat': bool(threat_result.get('is_threat', False)),
                'threat_score': float(threat_result.get('threat_score', 0.0)),
                'confidence': float(min(
                    anomaly_result.get('confidence', 0.0),
                    threat_result.get('confidence', 0.0)
                ))
            }
        except Exception as e:
            logging.error(f"Event processing error: {e}")
            return {
                'is_anomaly': False,
                'anomaly_score': 0.0,
                'is_threat': False,
                'threat_score': 0.0,
                'confidence': 0.0,
                'error': str(e)
            }