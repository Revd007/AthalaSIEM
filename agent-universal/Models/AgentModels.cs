using System;
using System.Collections.Generic;

namespace AthalaSIEM.UniversalAgent.Models
{
    /// <summary>
    /// Agent status enumeration
    /// </summary>
    public enum AgentStatus
    {
        Stopped,
        Starting,
        Running,
        Stopping,
        Error,
        Disconnected
    }

    /// <summary>
    /// Agent configuration class
    /// </summary>
    public class AgentConfiguration
    {
        public string AgentName { get; set; } = Environment.MachineName;
        public string BackendUrl { get; set; } = "http://localhost:9595";
        public string RegistrationKey { get; set; } = string.Empty;
        public int CollectionInterval { get; set; } = 30;
        public int BatchSize { get; set; } = 100;
        public int HeartbeatInterval { get; set; } = 60;
        public int MaxRetries { get; set; } = 3;
        public Dictionary<string, object> CollectorConfigs { get; set; } = new Dictionary<string, object>();
    }

    /// <summary>
    /// Comprehensive agent health information
    /// </summary>
    public class AgentHealth
    {
        public bool IsHealthy { get; set; }
        public AgentStatus Status { get; set; }
        public DateTime LastHeartbeat { get; set; }
        public TimeSpan Uptime { get; set; }
        public long TotalLogsCollected { get; set; }
        public long TotalLogsForwarded { get; set; }
        public List<CollectorHealth> CollectorHealth { get; set; } = new List<CollectorHealth>();
        public Dictionary<string, object> SystemMetrics { get; set; } = new Dictionary<string, object>();
        public List<string> Errors { get; set; } = new List<string>();
    }

    /// <summary>
    /// Connection test result
    /// </summary>
    public class ConnectionTestResult
    {
        public bool IsSuccessful { get; set; }
        public string Message { get; set; } = string.Empty;
        public TimeSpan ResponseTime { get; set; }
        public DateTime TestTime { get; set; } = DateTime.UtcNow;
        public string BackendVersion { get; set; } = string.Empty;
    }

    /// <summary>
    /// Event arguments for agent status changes
    /// </summary>
    public class AgentStatusChangedEventArgs : EventArgs
    {
        public AgentStatus OldStatus { get; set; }
        public AgentStatus NewStatus { get; set; }
        public DateTime ChangeTime { get; set; } = DateTime.UtcNow;
        public string Reason { get; set; } = string.Empty;
    }

    /// <summary>
    /// Event arguments for logs forwarded events
    /// </summary>
    public class LogsForwardedEventArgs : EventArgs
    {
        public int LogCount { get; set; }
        public DateTime ForwardTime { get; set; } = DateTime.UtcNow;
        public string Destination { get; set; } = string.Empty;
        public bool IsSuccessful { get; set; }
    }

    /// <summary>
    /// Event arguments for agent errors
    /// </summary>
    public class AgentErrorEventArgs : EventArgs
    {
        public Exception Exception { get; set; } = new Exception();
        public string Message { get; set; } = string.Empty;
        public DateTime ErrorTime { get; set; } = DateTime.UtcNow;
        public string Source { get; set; } = string.Empty;
        public bool IsCritical { get; set; }
    }

    /// <summary>
    /// Collector health information
    /// </summary>
    public class CollectorHealth
    {
        public bool IsHealthy { get; set; }
        public string Status { get; set; } = "Unknown";
        public long LogsCollected { get; set; }
        public DateTime LastCollection { get; set; }
        public TimeSpan Uptime { get; set; }
        public Dictionary<string, object> Metrics { get; set; } = new Dictionary<string, object>();
        public List<string> Errors { get; set; } = new List<string>();
    }
}
