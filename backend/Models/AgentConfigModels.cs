using System;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using System.Text.Json.Serialization;

namespace Backend.Models
{
    /// <summary>
    /// Represents the configuration for an agent
    /// </summary>
    public class AgentConfigModels
    {
        /// <summary>
        /// Gets or sets the configuration ID
        /// </summary>
        [Key]
        public string Id { get; set; } = Guid.NewGuid().ToString();
        
        /// <summary>
        /// Gets or sets the agent ID
        /// </summary>
        [Required]
        public string AgentId { get; set; } = string.Empty;
        
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
        public int LogCollectionIntervalSeconds { get; set; } = 60;
        
        /// <summary>
        /// Gets or sets the maximum log buffer count
        /// </summary>
        public int MaxLogBufferCount { get; set; } = 1000;
        
        /// <summary>
        /// Gets or sets the maximum log buffer time in seconds
        /// </summary>
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
        public int CpuAlertThresholdPercent { get; set; } = 90;
        
        /// <summary>
        /// Gets or sets the memory alert threshold percentage
        /// </summary>
        public int MemoryAlertThresholdPercent { get; set; } = 90;
        
        /// <summary>
        /// Gets or sets the disk alert threshold percentage
        /// </summary>
        public int DiskAlertThresholdPercent { get; set; } = 90;
        
        /// <summary>
        /// Gets or sets the creation time
        /// </summary>
        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
        
        /// <summary>
        /// Gets or sets the last update time
        /// </summary>
        public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;
        
        /// <summary>
        /// Gets or sets the agent
        /// </summary>
        [ForeignKey("AgentId")]
        public AgentModels? Agent { get; set; }
        
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
        public string LogLevelFilters { get; set; } = "Information,Warning,Error,Critical";
        
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
        public string LogSources { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the log file paths to monitor
        /// </summary>
        public string LogFilePaths { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the version of the configuration
        /// </summary>
        public int Version { get; set; } = 1;
        
        /// <summary>
        /// Gets or sets the configuration as JSON string
        /// </summary>
        [Column("Configuration")]
        public string? Configuration { get; set; }
        
        /// <summary>
        /// Gets or sets the last updated timestamp
        /// </summary>
        [Column("LastUpdated")]
        public DateTime LastUpdated { get; set; } = DateTime.UtcNow;
        
        /// <summary>
        /// Gets or sets whether the agent requires restart after configuration change
        /// </summary>
        [Column("RequiresRestart")]
        public bool RequiresRestart { get; set; } = false;
    }
} 