using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Text.Json.Serialization;

namespace Backend.Models
{
    /// <summary>
    /// File Integrity Monitoring event model
    /// </summary>
    public class FileIntegrityEvent
    {
        /// <summary>
        /// Gets or sets the event ID
        /// </summary>
        [Key]
        public string Id { get; set; } = Guid.NewGuid().ToString();

        /// <summary>
        /// Gets or sets the agent ID that detected the change
        /// </summary>
        [Required]
        [MaxLength(50)]
        public string AgentId { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the file path that was changed
        /// </summary>
        [Required]
        [MaxLength(500)]
        public string FilePath { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the type of change (Created, Modified, Deleted, Renamed)
        /// </summary>
        [Required]
        [MaxLength(20)]
        public string ChangeType { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the baseline file hash
        /// </summary>
        [MaxLength(128)]
        public string? BaselineHash { get; set; }

        /// <summary>
        /// Gets or sets the current file hash
        /// </summary>
        [MaxLength(128)]
        public string? CurrentHash { get; set; }

        /// <summary>
        /// Gets or sets the baseline file size
        /// </summary>
        public long? BaselineSize { get; set; }

        /// <summary>
        /// Gets or sets the current file size
        /// </summary>
        public long? CurrentSize { get; set; }

        /// <summary>
        /// Gets or sets the baseline last modified time
        /// </summary>
        public DateTime? BaselineModified { get; set; }

        /// <summary>
        /// Gets or sets the current last modified time
        /// </summary>
        public DateTime? CurrentModified { get; set; }

        /// <summary>
        /// Gets or sets the file attributes
        /// </summary>
        [MaxLength(100)]
        public string? FileAttributes { get; set; }

        /// <summary>
        /// Gets or sets the severity level
        /// </summary>
        [Required]
        [MaxLength(20)]
        public string Severity { get; set; } = "Medium";

        /// <summary>
        /// Gets or sets when the event was detected
        /// </summary>
        public DateTime DetectedAt { get; set; } = DateTime.UtcNow;

        /// <summary>
        /// Gets or sets when the event was processed
        /// </summary>
        public DateTime ProcessedAt { get; set; } = DateTime.UtcNow;

        /// <summary>
        /// Gets or sets whether the event has been acknowledged
        /// </summary>
        public bool IsAcknowledged { get; set; } = false;

        /// <summary>
        /// Gets or sets who acknowledged the event
        /// </summary>
        [MaxLength(50)]
        public string? AcknowledgedBy { get; set; }

        /// <summary>
        /// Gets or sets when the event was acknowledged
        /// </summary>
        public DateTime? AcknowledgedAt { get; set; }

        /// <summary>
        /// Gets or sets additional event details as JSON
        /// </summary>
        public string? Details { get; set; }

        /// <summary>
        /// Gets or sets the associated agent
        /// </summary>
        [JsonIgnore]
        public virtual AgentModels? Agent { get; set; }
    }

    /// <summary>
    /// File Integrity Monitoring rule model
    /// </summary>
    public class FileIntegrityRule
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
        [MaxLength(100)]
        public string Name { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the rule description
        /// </summary>
        [MaxLength(500)]
        public string? Description { get; set; }

        /// <summary>
        /// Gets or sets whether the rule is enabled
        /// </summary>
        public bool IsEnabled { get; set; } = true;

        /// <summary>
        /// Gets or sets the monitored paths (semicolon separated)
        /// </summary>
        [Required]
        public string MonitoredPaths { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the exclude patterns (semicolon separated)
        /// </summary>
        public string? ExcludePatterns { get; set; }

        /// <summary>
        /// Gets or sets whether real-time monitoring is enabled
        /// </summary>
        public bool RealTimeMonitoring { get; set; } = true;

        /// <summary>
        /// Gets or sets the scan interval in minutes
        /// </summary>
        public int ScanIntervalMinutes { get; set; } = 60;

        /// <summary>
        /// Gets or sets the severity level for alerts
        /// </summary>
        [Required]
        [MaxLength(20)]
        public string Severity { get; set; } = "Medium";

        /// <summary>
        /// Gets or sets whether to alert on file creation
        /// </summary>
        public bool AlertOnCreation { get; set; } = true;

        /// <summary>
        /// Gets or sets whether to alert on file modification
        /// </summary>
        public bool AlertOnModification { get; set; } = true;

        /// <summary>
        /// Gets or sets whether to alert on file deletion
        /// </summary>
        public bool AlertOnDeletion { get; set; } = true;

        /// <summary>
        /// Gets or sets whether to alert on file rename
        /// </summary>
        public bool AlertOnRename { get; set; } = true;

        /// <summary>
        /// Gets or sets when the rule was created
        /// </summary>
        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

        /// <summary>
        /// Gets or sets when the rule was last updated
        /// </summary>
        public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;

        /// <summary>
        /// Gets or sets who created the rule
        /// </summary>
        [MaxLength(50)]
        public string? CreatedBy { get; set; }

        /// <summary>
        /// Gets or sets the target agent IDs (semicolon separated, empty means all agents)
        /// </summary>
        public string? TargetAgents { get; set; }
    }

    /// <summary>
    /// File Integrity Monitoring baseline model
    /// </summary>
    public class FileIntegrityBaseline
    {
        /// <summary>
        /// Gets or sets the baseline ID
        /// </summary>
        [Key]
        public string Id { get; set; } = Guid.NewGuid().ToString();

        /// <summary>
        /// Gets or sets the agent ID
        /// </summary>
        [Required]
        [MaxLength(50)]
        public string AgentId { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the file path
        /// </summary>
        [Required]
        [MaxLength(500)]
        public string FilePath { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the file hash
        /// </summary>
        [Required]
        [MaxLength(128)]
        public string FileHash { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the file size
        /// </summary>
        public long FileSize { get; set; }

        /// <summary>
        /// Gets or sets the last modified time
        /// </summary>
        public DateTime LastModified { get; set; }

        /// <summary>
        /// Gets or sets the creation time
        /// </summary>
        public DateTime CreatedTime { get; set; }

        /// <summary>
        /// Gets or sets the file attributes
        /// </summary>
        [MaxLength(100)]
        public string? FileAttributes { get; set; }

        /// <summary>
        /// Gets or sets when the baseline was created
        /// </summary>
        public DateTime BaselineCreatedAt { get; set; } = DateTime.UtcNow;

        /// <summary>
        /// Gets or sets when the baseline was last updated
        /// </summary>
        public DateTime BaselineUpdatedAt { get; set; } = DateTime.UtcNow;

        /// <summary>
        /// Gets or sets whether the baseline is active
        /// </summary>
        public bool IsActive { get; set; } = true;

        /// <summary>
        /// Gets or sets the associated agent
        /// </summary>
        [JsonIgnore]
        public virtual AgentModels? Agent { get; set; }
    }

    /// <summary>
    /// DTO for File Integrity Event response
    /// </summary>
    public class FileIntegrityEventDto
    {
        public string Id { get; set; } = string.Empty;
        public string AgentId { get; set; } = string.Empty;
        public string AgentName { get; set; } = string.Empty;
        public string FilePath { get; set; } = string.Empty;
        public string ChangeType { get; set; } = string.Empty;
        public string? BaselineHash { get; set; }
        public string? CurrentHash { get; set; }
        public long? BaselineSize { get; set; }
        public long? CurrentSize { get; set; }
        public DateTime? BaselineModified { get; set; }
        public DateTime? CurrentModified { get; set; }
        public string? FileAttributes { get; set; }
        public string Severity { get; set; } = string.Empty;
        public DateTime DetectedAt { get; set; }
        public DateTime ProcessedAt { get; set; }
        public bool IsAcknowledged { get; set; }
        public string? AcknowledgedBy { get; set; }
        public DateTime? AcknowledgedAt { get; set; }
        public string? Details { get; set; }
    }

    /// <summary>
    /// DTO for creating File Integrity Rule
    /// </summary>
    public class CreateFileIntegrityRuleDto
    {
        [Required]
        [MaxLength(100)]
        public string Name { get; set; } = string.Empty;

        [MaxLength(500)]
        public string? Description { get; set; }

        public bool IsEnabled { get; set; } = true;

        [Required]
        public string MonitoredPaths { get; set; } = string.Empty;

        public string? ExcludePatterns { get; set; }

        public bool RealTimeMonitoring { get; set; } = true;

        public int ScanIntervalMinutes { get; set; } = 60;

        [Required]
        [MaxLength(20)]
        public string Severity { get; set; } = "Medium";

        public bool AlertOnCreation { get; set; } = true;
        public bool AlertOnModification { get; set; } = true;
        public bool AlertOnDeletion { get; set; } = true;
        public bool AlertOnRename { get; set; } = true;

        public string? TargetAgents { get; set; }
    }

    /// <summary>
    /// DTO for File Integrity Rule response
    /// </summary>
    public class FileIntegrityRuleDto
    {
        public string Id { get; set; } = string.Empty;
        public string Name { get; set; } = string.Empty;
        public string? Description { get; set; }
        public bool IsEnabled { get; set; }
        public string MonitoredPaths { get; set; } = string.Empty;
        public string? ExcludePatterns { get; set; }
        public bool RealTimeMonitoring { get; set; }
        public int ScanIntervalMinutes { get; set; }
        public string Severity { get; set; } = string.Empty;
        public bool AlertOnCreation { get; set; }
        public bool AlertOnModification { get; set; }
        public bool AlertOnDeletion { get; set; }
        public bool AlertOnRename { get; set; }
        public DateTime CreatedAt { get; set; }
        public DateTime UpdatedAt { get; set; }
        public string? CreatedBy { get; set; }
        public string? TargetAgents { get; set; }
    }

    /// <summary>
    /// Request for acknowledging FIM events
    /// </summary>
    public class AcknowledgeFimEventRequest
    {
        [Required]
        public List<string> EventIds { get; set; } = new();

        [MaxLength(500)]
        public string? Comments { get; set; }
    }
} 