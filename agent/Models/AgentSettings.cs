using System;
using System.Collections.Generic;
using System.IO;

namespace AthalaSIEM.Agent.Models
{
    /// <summary>
    /// Settings for agent configuration
    /// </summary>
    public class AgentSettings
    {
        /// <summary>
        /// Agent name
        /// </summary>
        public string AgentName { get; set; } = "AthalaSIEM Agent";

        /// <summary>
        /// Backend API URL
        /// </summary>
        public string BackendApiUrl { get; set; } = "https://localhost:7078";

        /// <summary>
        /// Backend gRPC URL
        /// </summary>
        public string BackendGrpcUrl { get; set; } = "https://localhost:7078";

        /// <summary>
        /// Backend URL (for backward compatibility)
        /// </summary>
        public string BackendUrl 
        { 
            get => BackendApiUrl;
            set => BackendApiUrl = value;
        }

        /// <summary>
        /// Log batch size for sending logs to the backend
        /// </summary>
        public int LogBatchSize { get; set; } = 100;

        /// <summary>
        /// Maximum log buffer size before auto-sending
        /// </summary>
        public int MaxLogBufferSize { get; set; } = 1000;

        /// <summary>
        /// Log sending interval in seconds
        /// </summary>
        public int LogSendingIntervalSeconds { get; set; } = 30;

        /// <summary>
        /// Health monitoring interval in minutes
        /// </summary>
        public int HealthMonitoringIntervalMinutes { get; set; } = 5;

        /// <summary>
        /// Heartbeat interval in minutes
        /// </summary>
        public int HeartbeatIntervalMinutes { get; set; } = 1;

        /// <summary>
        /// Maximum retries for sending logs
        /// </summary>
        public int MaxRetries { get; set; } = 3;

        /// <summary>
        /// Retry delay in seconds
        /// </summary>
        public int RetryDelaySeconds { get; set; } = 5;

        /// <summary>
        /// Whether to encrypt logs
        /// </summary>
        public bool EncryptLogs { get; set; } = false;

        /// <summary>
        /// Whether to use mutual TLS for communication
        /// </summary>
        public bool UseMutualTls { get; set; } = false;

        /// <summary>
        /// Path to the client certificate file
        /// </summary>
        public string ClientCertificatePath { get; set; } = string.Empty;

        /// <summary>
        /// Password for the client certificate
        /// </summary>
        public string ClientCertificatePassword { get; set; } = string.Empty;

        /// <summary>
        /// Path to the server CA certificate file
        /// </summary>
        public string ServerCaCertificatePath { get; set; } = string.Empty;

        /// <summary>
        /// Whether to validate the server certificate
        /// </summary>
        public bool ValidateServerCertificate { get; set; } = true;

        /// <summary>
        /// Whether to use traffic compression for all communication
        /// </summary>
        public bool UseTrafficCompression { get; set; } = true;

        /// <summary>
        /// Log collectors settings
        /// </summary>
        public List<CollectorSettings> Collectors { get; set; } = new List<CollectorSettings>();

        /// <summary>
        /// Config refresh interval in minutes
        /// </summary>
        public int ConfigRefreshIntervalMinutes { get; set; } = 15;

        /// <summary>
        /// Maximum log batch interval in seconds
        /// </summary>
        public int MaxLogBatchIntervalSeconds { get; set; } = 60;

        /// <summary>
        /// Whether to collect system metrics
        /// </summary>
        public bool CollectSystemMetrics { get; set; } = true;

        /// <summary>
        /// System metrics interval in minutes
        /// </summary>
        public int SystemMetricsIntervalMinutes { get; set; } = 5;

        /// <summary>
        /// Whether to use compression
        /// </summary>
        public bool UseCompression { get; set; } = true;

        /// <summary>
        /// Proxy settings
        /// </summary>
        public ProxySettings Proxy { get; set; } = new ProxySettings();
    }

    /// <summary>
    /// Settings for log collector configuration
    /// </summary>
    public class CollectorSettings
    {
        /// <summary>
        /// Type of collector (e.g., WindowsEventLog, LinuxSyslog)
        /// </summary>
        public required string Type { get; set; }

        /// <summary>
        /// Whether the collector is enabled
        /// </summary>
        public bool Enabled { get; set; } = true;

        /// <summary>
        /// Collection interval in seconds (0 for real-time)
        /// </summary>
        public int IntervalSeconds { get; set; } = 10;

        /// <summary>
        /// Additional collector-specific properties
        /// </summary>
        public Dictionary<string, string> Properties { get; set; } = new Dictionary<string, string>();
    }

    public class ProxySettings
    {
        public bool Enabled { get; set; } = false;
        public string Address { get; set; } = string.Empty;
        public int Port { get; set; } = 8080;
        public string Username { get; set; } = string.Empty;
        public string Password { get; set; } = string.Empty;
    }

    public class WindowsEventLogSettings
    {
        public string EventLogs { get; set; } = "Application,System,Security";
        public string CollectionMode { get; set; } = "Polling";
        public int MaxEvents { get; set; } = 100;
        public string QueryFilter { get; set; } = "*[System[TimeCreated[timediff(@SystemTime) <= 3600000]]]";
    }

    public class LinuxSyslogSettings
    {
        public string SyslogFiles { get; set; } = "/var/log/syslog,/var/log/messages";
        public string CollectionMode { get; set; } = "Polling";
        public int MaxLinesPerRead { get; set; } = 1000;
    }
} 