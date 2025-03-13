using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;

namespace Backend.DTOs
{
    /// <summary>
    /// Data transfer object for agent configuration
    /// </summary>
    public class AgentConfigDto
    {
        /// <summary>
        /// Gets or sets a value indicating whether the agent is enabled
        /// </summary>
        public bool Enabled { get; set; } = true;
        
        /// <summary>
        /// Gets or sets a value indicating whether to collect event logs
        /// </summary>
        public bool CollectEventLogs { get; set; } = true;
        
        /// <summary>
        /// Gets or sets a value indicating whether to collect system metrics
        /// </summary>
        public bool CollectSystemMetrics { get; set; } = true;
        
        /// <summary>
        /// Gets or sets the event logs to monitor
        /// </summary>
        public string EventLogsToMonitor { get; set; } = "Application,System,Security";
        
        /// <summary>
        /// Gets or sets the log collection interval in seconds
        /// </summary>
        [Range(10, 3600)]
        public int LogCollectionIntervalSeconds { get; set; } = 60;
        
        /// <summary>
        /// Gets or sets the maximum log buffer count
        /// </summary>
        [Range(100, 10000)]
        public int MaxLogBufferCount { get; set; } = 1000;
        
        /// <summary>
        /// Gets or sets the maximum log buffer time in seconds
        /// </summary>
        [Range(60, 3600)]
        public int MaxLogBufferTimeSeconds { get; set; } = 300;
        
        /// <summary>
        /// Gets or sets a value indicating whether to enable real-time monitoring
        /// </summary>
        public bool EnableRealTimeMonitoring { get; set; } = false;
        
        /// <summary>
        /// Gets or sets a value indicating whether to enable alerting
        /// </summary>
        public bool EnableAlerting { get; set; } = true;
        
        /// <summary>
        /// Gets or sets the CPU alert threshold percentage
        /// </summary>
        [Range(50, 100)]
        public int CpuAlertThresholdPercent { get; set; } = 90;
        
        /// <summary>
        /// Gets or sets the memory alert threshold percentage
        /// </summary>
        [Range(50, 100)]
        public int MemoryAlertThresholdPercent { get; set; } = 90;
        
        /// <summary>
        /// Gets or sets the disk alert threshold percentage
        /// </summary>
        [Range(50, 100)]
        public int DiskAlertThresholdPercent { get; set; } = 90;
        
        /// <summary>
        /// Gets or sets the configuration ID
        /// </summary>
        public string Id { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the agent ID
        /// </summary>
        public string AgentId { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the server URL
        /// </summary>
        public string ServerUrl { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the interval in minutes for refreshing the configuration
        /// </summary>
        public int ConfigRefreshIntervalMinutes { get; set; } = 60;
        
        /// <summary>
        /// Gets or sets a value indicating whether to include process details in logs
        /// </summary>
        public bool IncludeProcessDetails { get; set; } = true;
        
        /// <summary>
        /// Gets or sets the log level filters
        /// </summary>
        public string[] LogLevelFilters { get; set; } = new[] { "Information", "Warning", "Error", "Critical" };
        
        /// <summary>
        /// Gets or sets a value indicating whether to use SSL for communication
        /// </summary>
        public bool UseSSL { get; set; } = true;
        
        /// <summary>
        /// Gets or sets a value indicating whether to validate the server certificate
        /// </summary>
        public bool ValidateServerCertificate { get; set; } = true;
        
        /// <summary>
        /// Gets or sets the log sources to collect from
        /// </summary>
        public string[] LogSources { get; set; } = Array.Empty<string>();
        
        /// <summary>
        /// Gets or sets the log file paths to monitor
        /// </summary>
        public string[] LogFilePaths { get; set; } = Array.Empty<string>();
        
        /// <summary>
        /// Gets or sets the version of the configuration
        /// </summary>
        public int Version { get; set; } = 1;
        
        /// <summary>
        /// Gets or sets the date and time when the configuration was created
        /// </summary>
        public DateTime CreatedAt { get; set; }
        
        /// <summary>
        /// Gets or sets the date and time when the configuration was last updated
        /// </summary>
        public DateTime UpdatedAt { get; set; }
    }
} 