using System;
using System.Collections.Generic;

namespace AthalaSIEM.Agent.Models
{
    /// <summary>
    /// Represents the health state of an agent
    /// </summary>
    public enum AgentHealthState
    {
        /// <summary>
        /// The agent is healthy
        /// </summary>
        Healthy,
        
        /// <summary>
        /// The agent is degraded
        /// </summary>
        Degraded,
        
        /// <summary>
        /// The agent is in a warning state
        /// </summary>
        Warning,
        
        /// <summary>
        /// The agent is in a critical state
        /// </summary>
        Critical,
        
        /// <summary>
        /// The agent is offline
        /// </summary>
        Offline,
        
        /// <summary>
        /// The agent state is unknown
        /// </summary>
        Unknown
    }
    
    /// <summary>
    /// Represents the health status of an agent
    /// </summary>
    public class AgentHealthStatus
    {
        /// <summary>
        /// Gets or sets the overall status
        /// </summary>
        public string Status { get; set; } = "Healthy";
        
        /// <summary>
        /// Gets or sets the last update time
        /// </summary>
        public DateTime LastUpdated { get; set; } = DateTime.UtcNow;
        
        /// <summary>
        /// Gets or sets the last check time (alias for LastUpdated)
        /// </summary>
        public DateTime LastChecked { 
            get => LastUpdated; 
            set => LastUpdated = value; 
        }
        
        /// <summary>
        /// Gets or sets the start time
        /// </summary>
        public DateTime StartTime { get; set; } = DateTime.UtcNow;
        
        /// <summary>
        /// Gets or sets the uptime in seconds
        /// </summary>
        public long UptimeSeconds { get; set; }
        
        /// <summary>
        /// Gets or sets the uptime in seconds (alias for UptimeSeconds)
        /// </summary>
        public long Uptime { 
            get => UptimeSeconds; 
            set => UptimeSeconds = value; 
        }
        
        /// <summary>
        /// Gets or sets the component statuses
        /// </summary>
        public Dictionary<string, AgentComponentStatus> ComponentStatuses { get; set; } = new Dictionary<string, AgentComponentStatus>();
        
        /// <summary>
        /// Gets or sets the diagnostics
        /// </summary>
        public Dictionary<string, string> Diagnostics { get; set; } = new Dictionary<string, string>();
        
        /// <summary>
        /// Gets or sets the CPU usage
        /// </summary>
        public double? CpuUsage { get; set; }
        
        /// <summary>
        /// Gets or sets the memory usage
        /// </summary>
        public double? MemoryUsage { get; set; }
        
        /// <summary>
        /// Gets or sets the disk usage
        /// </summary>
        public double? DiskUsage { get; set; }
        
        /// <summary>
        /// Gets or sets whether the agent can connect to the backend
        /// </summary>
        public bool CanConnectToBackend { get; set; } = true;
        
        /// <summary>
        /// Gets or sets the agent ID
        /// </summary>
        public string AgentId { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the version
        /// </summary>
        public string Version { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the operating system
        /// </summary>
        public string OperatingSystem { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the hostname
        /// </summary>
        public string Hostname { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the IP address
        /// </summary>
        public string IpAddress { get; set; } = string.Empty;
    }
    
    /// <summary>
    /// Represents the status of an agent component
    /// </summary>
    public class AgentComponentStatus
    {
        /// <summary>
        /// Gets or sets the component name
        /// </summary>
        public string ComponentName { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the component name (alias for ComponentName)
        /// </summary>
        public string Name { 
            get => ComponentName; 
            set => ComponentName = value; 
        }
        
        /// <summary>
        /// Gets or sets the status
        /// </summary>
        public string Status { get; set; } = "Healthy";
        
        /// <summary>
        /// Gets or sets the status message
        /// </summary>
        public string Message { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the last update time
        /// </summary>
        public DateTime LastUpdated { get; set; } = DateTime.UtcNow;
        
        /// <summary>
        /// Gets or sets the last check time (alias for LastUpdated)
        /// </summary>
        public DateTime LastChecked { 
            get => LastUpdated; 
            set => LastUpdated = value; 
        }
        
        /// <summary>
        /// Gets or sets the details
        /// </summary>
        public Dictionary<string, string> Details { get; set; } = new Dictionary<string, string>();
        
        /// <summary>
        /// Gets or sets whether the component is critical
        /// </summary>
        public bool IsCritical { get; set; } = false;
    }
    
    /// <summary>
    /// Represents a health report for an agent
    /// </summary>
    public class AgentHealthReport
    {
        /// <summary>
        /// Gets or sets the report ID
        /// </summary>
        public string Id { get; set; } = Guid.NewGuid().ToString();
        
        /// <summary>
        /// Gets or sets the agent ID
        /// </summary>
        public string AgentId { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the timestamp
        /// </summary>
        public DateTime Timestamp { get; set; } = DateTime.UtcNow;
        
        /// <summary>
        /// Gets or sets the report timestamp (alias for Timestamp)
        /// </summary>
        public DateTime ReportedAt { 
            get => Timestamp; 
            set => Timestamp = value; 
        }
        
        /// <summary>
        /// Gets or sets the overall status
        /// </summary>
        public string Status { get; set; } = "Healthy";
        
        /// <summary>
        /// Gets or sets the overall status (alias for Status)
        /// </summary>
        public string OverallStatus { 
            get => Status; 
            set => Status = value; 
        }
        
        /// <summary>
        /// Gets or sets the uptime in seconds
        /// </summary>
        public long UptimeSeconds { get; set; }
        
        /// <summary>
        /// Gets or sets the uptime in seconds (alias for UptimeSeconds)
        /// </summary>
        public long Uptime { 
            get => UptimeSeconds; 
            set => UptimeSeconds = value; 
        }
        
        /// <summary>
        /// Gets or sets the system metrics
        /// </summary>
        public SystemMetrics SystemMetrics { get; set; } = new SystemMetrics
        {
            AgentId = string.Empty,
            Cpu = new CpuMetrics(),
            Memory = new MemoryMetrics(),
            Disk = new DiskMetrics { Drives = new List<DriveMeasurement>() },
            Network = new NetworkMetrics(),
            Process = new ProcessMetrics 
            { 
                CurrentProcess = new ProcessMemoryUsage { Name = "Unknown" }, 
                MemoryUsageProcesses = new List<ProcessMemoryUsage>() 
            }
        };
        
        /// <summary>
        /// Gets or sets the system metrics (alias for SystemMetrics)
        /// </summary>
        public SystemMetrics Metrics { 
            get => SystemMetrics; 
            set => SystemMetrics = value; 
        }
        
        /// <summary>
        /// Gets or sets the component statuses
        /// </summary>
        public Dictionary<string, string> ComponentStatuses { get; set; } = new Dictionary<string, string>();
        
        /// <summary>
        /// Gets or sets the component information
        /// </summary>
        public List<ComponentStatus> Components { get; set; } = new List<ComponentStatus>();
        
        /// <summary>
        /// Gets or sets the diagnostics
        /// </summary>
        public Dictionary<string, string> Diagnostics { get; set; } = new Dictionary<string, string>();
        
        /// <summary>
        /// Gets or sets the CPU usage
        /// </summary>
        public double? CpuUsage { get; set; }
        
        /// <summary>
        /// Gets or sets the memory usage
        /// </summary>
        public double? MemoryUsage { get; set; }
        
        /// <summary>
        /// Gets or sets the disk usage
        /// </summary>
        public double? DiskUsage { get; set; }
        
        /// <summary>
        /// Gets or sets the version
        /// </summary>
        public string Version { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the operating system
        /// </summary>
        public string OperatingSystem { get; set; } = string.Empty;
    }
    
    /// <summary>
    /// Represents a heartbeat from an agent
    /// </summary>
    public class AgentHeartbeat
    {
        /// <summary>
        /// Gets or sets the agent ID
        /// </summary>
        public string AgentId { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the timestamp
        /// </summary>
        public DateTime Timestamp { get; set; } = DateTime.UtcNow;
        
        /// <summary>
        /// Gets or sets the status
        /// </summary>
        public string Status { get; set; } = "Healthy";
        
        /// <summary>
        /// Gets or sets the version
        /// </summary>
        public string Version { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the uptime in seconds
        /// </summary>
        public long UptimeSeconds { get; set; }
        
        /// <summary>
        /// Gets or sets the uptime (alias for UptimeSeconds)
        /// </summary>
        public long Uptime { 
            get => UptimeSeconds; 
            set => UptimeSeconds = value; 
        }
        
        /// <summary>
        /// Gets or sets the IP address
        /// </summary>
        public string IpAddress { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the CPU usage percentage
        /// </summary>
        public double? CpuUsage { get; set; }
        
        /// <summary>
        /// Gets or sets the memory usage percentage
        /// </summary>
        public double? MemoryUsage { get; set; }
        
        /// <summary>
        /// Gets or sets the disk usage percentage
        /// </summary>
        public double? DiskUsage { get; set; }
        
        /// <summary>
        /// Gets or sets the number of logs collected
        /// </summary>
        public long LogsCollected { get; set; }
        
        /// <summary>
        /// Gets or sets the number of logs forwarded
        /// </summary>
        public long LogsForwarded { get; set; }
        
        /// <summary>
        /// Gets or sets the number of logs pending
        /// </summary>
        public long LogsPending { get; set; }
        
        /// <summary>
        /// Gets or sets the active collectors
        /// </summary>
        public List<string> ActiveCollectors { get; set; } = new List<string>();
        
        /// <summary>
        /// Gets or sets the operating system description
        /// </summary>
        public string OsDescription { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the machine name
        /// </summary>
        public string MachineName { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets additional details about the agent
        /// </summary>
        public Dictionary<string, string> AdditionalDetails { get; set; } = new Dictionary<string, string>();
        
        /// <summary>
        /// Gets or sets basic metrics
        /// </summary>
        public Dictionary<string, string> Metrics { get; set; } = new Dictionary<string, string>();
    }
} 