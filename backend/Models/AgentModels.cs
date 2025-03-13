using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using System.Net;
using System.Text.Json;

namespace Backend.Models
{
    /// <summary>
    /// Represents an agent in the system
    /// </summary>
    [Table("agents")]
    public class AgentModels
    {
        /// <summary>
        /// Gets or sets the agent ID
        /// </summary>
        [Key]
        [Column("Id")]
        public string Id { get; set; } = Guid.NewGuid().ToString();
        
        /// <summary>
        /// Gets or sets the agent name
        /// </summary>
        [Column("Name")]
        [Required]
        [MaxLength(255)]
        public string Name { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the agent version
        /// </summary>
        [Column("Version")]
        [MaxLength(50)]
        public string Version { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the IP address of the agent
        /// </summary>
        [Column("IPAddress")]
        [Required]
        [MaxLength(45)]
        public string IPAddress { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the hostname of the machine running the agent
        /// </summary>
        [Column("Hostname")]
        [Required]
        [MaxLength(255)]
        public string Hostname { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the operating system of the machine running the agent
        /// </summary>
        [Column("OS")]
        [Required]
        [MaxLength(255)]
        public string OperatingSystem { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the operating system (alias for OperatingSystem)
        /// </summary>
        [NotMapped]
        public string OS 
        { 
            get => OperatingSystem;
            set => OperatingSystem = value;
        }
        
        /// <summary>
        /// Gets or sets the date and time when the agent was installed
        /// </summary>
        [Column("InstallDate")]
        public DateTime InstallDate { get; set; } = DateTime.UtcNow;
        
        /// <summary>
        /// Gets or sets the date and time when the agent last connected
        /// </summary>
        [Column("LastConnected")]
        public DateTime LastConnected { get; set; } = DateTime.UtcNow;
        
        /// <summary>
        /// Gets or sets the date and time of the last heartbeat
        /// </summary>
        [Column("LastHeartbeat")]
        public DateTime? LastHeartbeat { get; set; }
        
        /// <summary>
        /// Gets or sets the port used by the agent
        /// </summary>
        [Column("Port")]
        public int Port { get; set; } = 514;
        
        /// <summary>
        /// Gets or sets the status of the agent
        /// </summary>
        [Column("Status")]
        [EnumDataType(typeof(AgentStatus))]
        public AgentStatus Status { get; set; } = AgentStatus.Pending;
        
        /// <summary>
        /// Gets or sets the API key for the agent
        /// </summary>
        [Column("ApiKey")]
        [Required]
        public string ApiKey { get; set; } = Guid.NewGuid().ToString();
        
        /// <summary>
        /// Gets or sets the agent type
        /// </summary>
        [Column("Type")]
        public AgentType Type { get; set; } = AgentType.Windows;
        
        /// <summary>
        /// Gets or sets a value indicating whether the agent is enabled
        /// </summary>
        [Column("IsEnabled")]
        public bool IsEnabled { get; set; } = true;
        
        /// <summary>
        /// Gets or sets the date and time when the agent was created
        /// </summary>
        [Column("CreatedAt")]
        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
        
        /// <summary>
        /// Gets or sets the date and time when the agent was last updated
        /// </summary>
        [Column("UpdatedAt")]
        public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;
        
        /// <summary>
        /// Gets or sets the ID of the user who created the agent
        /// </summary>
        [Column("CreatedById")]
        public string? CreatedById { get; set; }
        
        /// <summary>
        /// Gets or sets the user who created the agent
        /// </summary>
        [ForeignKey("CreatedById")]
        public virtual UserModels? CreatedBy { get; set; }
        
        /// <summary>
        /// Gets or sets the agent configuration
        /// </summary>
        public virtual AgentConfigModels? Configuration { get; set; }
        
        /// <summary>
        /// Gets or sets the alerts for this agent
        /// </summary>
        public virtual ICollection<AlertModels> Alerts { get; set; } = new List<AlertModels>();
        
        /// <summary>
        /// Gets or sets the log entries from this agent
        /// </summary>
        public virtual ICollection<LogEntryModels> LogEntries { get; set; } = new List<LogEntryModels>();
        
        /// <summary>
        /// Gets or sets the health reports from this agent
        /// </summary>
        public virtual ICollection<AgentHealthReport> HealthReports { get; set; } = new List<AgentHealthReport>();
        
        /// <summary>
        /// Gets or sets the heartbeats from this agent
        /// </summary>
        public virtual ICollection<AgentHeartbeatModels> Heartbeats { get; set; } = new List<AgentHeartbeatModels>();
        
        /// <summary>
        /// Gets or sets the health metrics from this agent
        /// </summary>
        public virtual ICollection<HealthMetricModels> HealthMetrics { get; set; } = new List<HealthMetricModels>();
        
        /// <summary>
        /// Gets or sets the security events from this agent
        /// </summary>
        public virtual ICollection<SecurityEventModels> SecurityEvents { get; set; } = new List<SecurityEventModels>();
        
        /// <summary>
        /// Gets or sets additional information about the agent
        /// </summary>
        [Column("AdditionalInfo")]
        public string? AdditionalInfo { get; set; }
        
        // System metrics - not mapped to database columns
        [NotMapped]
        public double? CPU { get; set; }
        
        [NotMapped]
        public double? Memory { get; set; }
        
        [Column("CpuUsage")]
        public double? CpuUsage { get; set; }
        
        [Column("MemoryUsage")]
        public double? MemoryUsage { get; set; }
        
        [Column("DiskUsage")]
        public double? DiskUsage { get; set; }
        
        // Add collection settings
        [Column("CollectEventLogs")]
        public bool CollectEventLogs { get; set; } = true;
        
        [Column("CollectSystemMetrics")]
        public bool CollectSystemMetrics { get; set; } = true;
        
        [Column("EventLogsToMonitor")]
        public string EventLogsToMonitor { get; set; } = "Application,System,Security";
        
        // Helper method to get the default port based on agent type
        public static int GetDefaultPortForAgentType(string agentType)
        {
            if (string.IsNullOrEmpty(agentType))
                return 514; // Default to Syslog port
                
            return agentType.ToLowerInvariant() switch
            {
                "windows" => 445,     // Windows Event Log collection
                "linux" => 514,       // Syslog
                "syslog" => 514,   // Splunk forwarder
                "snmp" => 161,        // SNMP monitoring
                "aws" => 443,         // HTTPS for AWS API
                "azure" => 443,       // HTTPS for Azure API
                "gcp" => 443,         // HTTPS for GCP API
                _ => 514              // Default to Syslog port
            };
        }
    }

    /// <summary>
    /// Represents a health report from an agent
    /// </summary>
    public class AgentHealthReport
    {
        /// <summary>
        /// Gets or sets the health report ID
        /// </summary>
        [Key]
        public string Id { get; set; } = Guid.NewGuid().ToString();
        
        /// <summary>
        /// Gets or sets the agent ID
        /// </summary>
        public string AgentId { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the timestamp of the health report
        /// </summary>
        public DateTime Timestamp { get; set; } = DateTime.UtcNow;
        
        /// <summary>
        /// Gets or sets the overall status of the agent
        /// </summary>
        public string OverallStatus { get; set; } = "Healthy";
        
        /// <summary>
        /// Gets or sets the metrics as a JSON string
        /// </summary>
        public string Metrics { get; set; } = "{}";
        
        /// <summary>
        /// Gets or sets the agent
        /// </summary>
        public virtual AgentModels? Agent { get; set; }
    }

    /// <summary>
    /// Represents the type of an agent
    /// </summary>
    public enum AgentType
    {
        /// <summary>
        /// Windows agent
        /// </summary>
        Windows,
        
        /// <summary>
        /// Linux agent
        /// </summary>
        Linux,
        
        /// <summary>
        /// Syslog agent
        /// </summary>
        Syslog,
        
        /// <summary>
        /// AWS agent
        /// </summary>
        AWS,
        
        /// <summary>
        /// Azure agent
        /// </summary>
        Azure,
        
        /// <summary>
        /// GCP agent
        /// </summary>
        GCP,
        
        /// <summary>
        /// Custom agent
        /// </summary>
        Custom
    }

    /// <summary>
    /// Represents the status of an agent
    /// </summary>
    public enum AgentStatus
    {
        /// <summary>
        /// The agent is pending
        /// </summary>
        Pending,
        
        /// <summary>
        /// The agent is active
        /// </summary>
        Active,
        
        /// <summary>
        /// The agent is inactive
        /// </summary>
        Inactive,
        
        /// <summary>
        /// The agent is online and functioning normally
        /// </summary>
        Online,
        
        /// <summary>
        /// The agent is offline
        /// </summary>
        Offline,
        
        /// <summary>
        /// The agent is experiencing issues but is still functioning
        /// </summary>
        Warning,
        
        /// <summary>
        /// The agent is in maintenance mode
        /// </summary>
        Maintenance,
        
        /// <summary>
        /// The agent is disabled
        /// </summary>
        Disabled
    }
} 