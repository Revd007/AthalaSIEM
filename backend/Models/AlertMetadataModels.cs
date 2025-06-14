using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using System.Text.Json.Serialization;

namespace Backend.Models
{
    /// <summary>
    /// Enhanced alert metadata for multi-collector support
    /// </summary>
    public class AlertMetadataModels
    {
        /// <summary>
        /// Gets or sets the metadata ID
        /// </summary>
        [Key]
        public string Id { get; set; } = Guid.NewGuid().ToString();
        
        /// <summary>
        /// Gets or sets the alert ID
        /// </summary>
        [Required]
        public string AlertId { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the collector type
        /// </summary>
        [Required]
        [MaxLength(50)]
        public string CollectorType { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the threat level
        /// </summary>
        public int ThreatLevel { get; set; } // 0=None, 1=Low, 2=Medium, 3=High, 4=Critical
        
        /// <summary>
        /// Gets or sets the original log entry ID
        /// </summary>
        [MaxLength(100)]
        public string? OriginalLogId { get; set; }
        
        /// <summary>
        /// Gets or sets threat indicators as JSON
        /// </summary>
        public string? ThreatIndicators { get; set; } // JSON array
        
        /// <summary>
        /// Gets or sets collector-specific data as JSON
        /// </summary>
        public string? CollectorSpecificData { get; set; } // JSON object
        
        /// <summary>
        /// Gets or sets whether auto-escalation is enabled
        /// </summary>
        public bool AutoEscalationEnabled { get; set; }
        
        /// <summary>
        /// Gets or sets escalation thresholds as JSON
        /// </summary>
        public string? EscalationThresholds { get; set; } // JSON object
        
        /// <summary>
        /// Gets or sets notification channels as JSON
        /// </summary>
        public string? NotificationChannels { get; set; } // JSON array
        
        /// <summary>
        /// Gets or sets the threat score
        /// </summary>
        public double ThreatScore { get; set; }
        
        /// <summary>
        /// Gets or sets additional context data as JSON
        /// </summary>
        public string? ContextData { get; set; }
        
        /// <summary>
        /// Gets or sets the correlation ID for related alerts
        /// </summary>
        [MaxLength(100)]
        public string? CorrelationId { get; set; }
        
        /// <summary>
        /// Gets or sets the creation timestamp
        /// </summary>
        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
        
        /// <summary>
        /// Gets or sets the update timestamp
        /// </summary>
        public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;
        
        /// <summary>
        /// Gets or sets the alert
        /// </summary>
        [ForeignKey("AlertId")]
        [JsonIgnore]
        public virtual AlertModels? Alert { get; set; }
    }

    /// <summary>
    /// Alert correlation tracking for multi-collector analysis
    /// </summary>
    public class AlertCorrelationModels
    {
        /// <summary>
        /// Gets or sets the correlation ID
        /// </summary>
        [Key]
        public string Id { get; set; } = Guid.NewGuid().ToString();
        
        /// <summary>
        /// Gets or sets the correlation pattern
        /// </summary>
        [Required]
        [MaxLength(200)]
        public string Pattern { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the collector types involved
        /// </summary>
        public string CollectorTypes { get; set; } = string.Empty; // JSON array
        
        /// <summary>
        /// Gets or sets the severity level
        /// </summary>
        public AlertSeverityModels Severity { get; set; }
        
        /// <summary>
        /// Gets or sets the number of occurrences
        /// </summary>
        public int Occurrences { get; set; }
        
        /// <summary>
        /// Gets or sets the time window in minutes
        /// </summary>
        public int TimeWindowMinutes { get; set; }
        
        /// <summary>
        /// Gets or sets the first occurrence timestamp
        /// </summary>
        public DateTime FirstOccurrence { get; set; }
        
        /// <summary>
        /// Gets or sets the last occurrence timestamp
        /// </summary>
        public DateTime LastOccurrence { get; set; }
        
        /// <summary>
        /// Gets or sets the number of affected agents
        /// </summary>
        public int AffectedAgents { get; set; }
        
        /// <summary>
        /// Gets or sets recommended actions as JSON
        /// </summary>
        public string? RecommendedActions { get; set; } // JSON array
        
        /// <summary>
        /// Gets or sets additional analysis data as JSON
        /// </summary>
        public string? AnalysisData { get; set; }
        
        /// <summary>
        /// Gets or sets whether this correlation is active
        /// </summary>
        public bool IsActive { get; set; } = true;
        
        /// <summary>
        /// Gets or sets the creation timestamp
        /// </summary>
        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
        
        /// <summary>
        /// Gets or sets the update timestamp
        /// </summary>
        public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;
    }

    /// <summary>
    /// Alert rule configuration for automated alert generation
    /// </summary>
    public class AlertRuleModels
    {
        /// <summary>
        /// Gets or sets the rule ID
        /// </summary>
        [Key]
        public string Id { get; set; } = Guid.NewGuid().ToString();
        
        /// <summary>
        /// Gets or sets the rule name
        /// </summary>
        [Required]
        [MaxLength(200)]
        public string Name { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the description
        /// </summary>
        [MaxLength(1000)]
        public string? Description { get; set; }
        
        /// <summary>
        /// Gets or sets the collector type this rule applies to
        /// </summary>
        [Required]
        [MaxLength(50)]
        public string CollectorType { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the rule condition as JSON
        /// </summary>
        [Required]
        public string Condition { get; set; } = string.Empty; // JSON object with conditions
        
        /// <summary>
        /// Gets or sets the severity to assign to generated alerts
        /// </summary>
        public AlertSeverityModels Severity { get; set; }
        
        /// <summary>
        /// Gets or sets whether the rule is enabled
        /// </summary>
        public bool Enabled { get; set; } = true;
        
        /// <summary>
        /// Gets or sets the notification channels as JSON
        /// </summary>
        public string? NotificationChannels { get; set; } // JSON array
        
        /// <summary>
        /// Gets or sets actions to take as JSON
        /// </summary>
        public string? Actions { get; set; } // JSON array
        
        /// <summary>
        /// Gets or sets the rule priority
        /// </summary>
        public int Priority { get; set; } = 100;
        
        /// <summary>
        /// Gets or sets the evaluation frequency in minutes
        /// </summary>
        public int EvaluationFrequencyMinutes { get; set; } = 5;
        
        /// <summary>
        /// Gets or sets the time window for evaluation in minutes
        /// </summary>
        public int TimeWindowMinutes { get; set; } = 60;
        
        /// <summary>
        /// Gets or sets the number of matches required to trigger
        /// </summary>
        public int ThresholdCount { get; set; } = 1;
        
        /// <summary>
        /// Gets or sets additional configuration as JSON
        /// </summary>
        public string? Configuration { get; set; }
        
        /// <summary>
        /// Gets or sets the creation timestamp
        /// </summary>
        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
        
        /// <summary>
        /// Gets or sets the update timestamp
        /// </summary>
        public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;
        
        /// <summary>
        /// Gets or sets the user ID who created the rule
        /// </summary>
        [MaxLength(100)]
        public string? CreatedBy { get; set; }
        
        /// <summary>
        /// Gets or sets the user ID who last updated the rule
        /// </summary>
        [MaxLength(100)]
        public string? UpdatedBy { get; set; }
        
        /// <summary>
        /// Gets or sets the last execution timestamp
        /// </summary>
        public DateTime? LastExecuted { get; set; }
        
        /// <summary>
        /// Gets or sets the number of times this rule has fired
        /// </summary>
        public long ExecutionCount { get; set; }
        
        /// <summary>
        /// Gets or sets the number of alerts generated by this rule
        /// </summary>
        public long AlertsGenerated { get; set; }
    }

    /// <summary>
    /// Collector-specific configuration for enhanced features
    /// </summary>
    public class CollectorConfigurationModels
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
        /// Gets or sets the collector type
        /// </summary>
        [Required]
        [MaxLength(50)]
        public string CollectorType { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets whether the collector is enabled
        /// </summary>
        public bool Enabled { get; set; } = true;
        
        /// <summary>
        /// Gets or sets the configuration as JSON
        /// </summary>
        public string Configuration { get; set; } = "{}";
        
        /// <summary>
        /// Gets or sets the collection interval in seconds
        /// </summary>
        public int CollectionIntervalSeconds { get; set; } = 60;
        
        /// <summary>
        /// Gets or sets the maximum events per batch
        /// </summary>
        public int MaxEventsPerBatch { get; set; } = 100;
        
        /// <summary>
        /// Gets or sets the buffer size
        /// </summary>
        public int BufferSize { get; set; } = 1000;
        
        /// <summary>
        /// Gets or sets whether threat intelligence is enabled
        /// </summary>
        public bool EnableThreatIntelligence { get; set; } = true;
        
        /// <summary>
        /// Gets or sets whether real-time monitoring is enabled
        /// </summary>
        public bool EnableRealTimeMonitoring { get; set; } = true;
        
        /// <summary>
        /// Gets or sets the log level filter
        /// </summary>
        [MaxLength(100)]
        public string LogLevelFilter { get; set; } = "Information,Warning,Error,Critical";
        
        /// <summary>
        /// Gets or sets custom filters as JSON
        /// </summary>
        public string? CustomFilters { get; set; }
        
        /// <summary>
        /// Gets or sets the creation timestamp
        /// </summary>
        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
        
        /// <summary>
        /// Gets or sets the update timestamp
        /// </summary>
        public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;
        
        /// <summary>
        /// Gets or sets the agent
        /// </summary>
        [ForeignKey("AgentId")]
        [JsonIgnore]
        public virtual AgentModels? Agent { get; set; }
    }
} 