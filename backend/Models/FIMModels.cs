using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace Backend.Models
{
    /// <summary>
    /// FIM Configuration Entity - Managed via web interface
    /// Following ManageEngine EventLog Analyzer and Splunk FIM patterns
    /// </summary>
    [Table("FIMConfigurations")]
    public class FIMConfiguration
    {
        [Key]
        public string Id { get; set; } = Guid.NewGuid().ToString();
        
        [Required]
        [MaxLength(255)]
        public string Name { get; set; } = "";
        
        [MaxLength(1000)]
        public string Description { get; set; } = "";
        
        public bool Enabled { get; set; } = true;
        
        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
        public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;
        
        [MaxLength(255)]
        public string CreatedBy { get; set; } = "";
        
        /// <summary>
        /// FIM rules for this configuration (JSON serialized)
        /// </summary>
        [Column(TypeName = "TEXT")]
        public string RulesJson { get; set; } = "[]";
        
        /// <summary>
        /// Global settings (JSON serialized)
        /// </summary>
        [Column(TypeName = "TEXT")]
        public string GlobalSettingsJson { get; set; } = "{}";
        
        /// <summary>
        /// Target agents (JSON serialized)
        /// </summary>
        [Column(TypeName = "TEXT")]
        public string TargetAgentsJson { get; set; } = "[]";
        
        /// <summary>
        /// Supported OS (JSON serialized)
        /// </summary>
        [Column(TypeName = "TEXT")]
        public string SupportedOSJson { get; set; } = "[\"Windows\",\"Linux\",\"macOS\"]";
        
        // Navigation properties - will be added when Agent model is available
        // public virtual ICollection<Agent> Agents { get; set; } = new List<Agent>();
    }

    /// <summary>
    /// FIM Template Entity for bulk configuration
    /// </summary>
    [Table("FIMTemplates")]
    public class FIMTemplate
    {
        [Key]
        public string Id { get; set; } = Guid.NewGuid().ToString();
        
        [Required]
        [MaxLength(255)]
        public string Name { get; set; } = "";
        
        [MaxLength(1000)]
        public string Description { get; set; } = "";
        
        [MaxLength(100)]
        public string Category { get; set; } = ""; // System, Application, Security, Custom
        
        /// <summary>
        /// Template rules (JSON serialized)
        /// </summary>
        [Column(TypeName = "TEXT")]
        public string TemplateRulesJson { get; set; } = "[]";
        
        /// <summary>
        /// Supported OS (JSON serialized)
        /// </summary>
        [Column(TypeName = "TEXT")]
        public string SupportedOSJson { get; set; } = "[]";
        
        /// <summary>
        /// Template variables (JSON serialized)
        /// </summary>
        [Column(TypeName = "TEXT")]
        public string VariablesJson { get; set; } = "{}";
        
        public bool IsBuiltIn { get; set; } = false;
        
        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
        public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;
        
        [MaxLength(255)]
        public string CreatedBy { get; set; } = "";
    }

    /// <summary>
    /// FIM Event Entity - generated when monitored files change
    /// </summary>
    [Table("FIMEvents")]
    public class FIMEvent
    {
        [Key]
        public string Id { get; set; } = Guid.NewGuid().ToString();
        
        [Required]
        [MaxLength(255)]
        public string RuleId { get; set; } = "";
        
        [MaxLength(255)]
        public string RuleName { get; set; } = "";
        
        [Required]
        [MaxLength(255)]
        public string AgentId { get; set; } = "";
        
        public DateTime Timestamp { get; set; } = DateTime.UtcNow;
        
        [Required]
        [MaxLength(1000)]
        public string FilePath { get; set; } = "";
        
        [MaxLength(100)]
        public string EventType { get; set; } = ""; // Created, Modified, Deleted, Renamed
        
        [MaxLength(1000)]
        public string OldFilePath { get; set; } = "";
        
        /// <summary>
        /// File information before change (JSON serialized)
        /// </summary>
        [Column(TypeName = "TEXT")]
        public string OldFileInfoJson { get; set; } = "{}";
        
        /// <summary>
        /// File information after change (JSON serialized)
        /// </summary>
        [Column(TypeName = "TEXT")]
        public string NewFileInfoJson { get; set; } = "{}";
        
        [MaxLength(255)]
        public string User { get; set; } = "";
        
        [MaxLength(255)]
        public string Process { get; set; } = "";
        
        public int? ProcessId { get; set; }
        
        [MaxLength(50)]
        public string SecurityLevel { get; set; } = "Medium";
        
        /// <summary>
        /// Additional metadata (JSON serialized)
        /// </summary>
        [Column(TypeName = "TEXT")]
        public string MetadataJson { get; set; } = "{}";
        
        public bool AlertGenerated { get; set; } = false;
        
        /// <summary>
        /// Tags (JSON serialized)
        /// </summary>
        [Column(TypeName = "TEXT")]
        public string TagsJson { get; set; } = "[]";
        
        // Navigation properties - will be added when Agent model is available
        // [ForeignKey("AgentId")]
        // public virtual Agent? Agent { get; set; }
    }

    // DTOs for API communication
    
    /// <summary>
    /// FIM Rule DTO for API communication
    /// </summary>
    public class FIMRuleDto
    {
        public string Id { get; set; } = Guid.NewGuid().ToString();
        public string Name { get; set; } = "";
        public string Description { get; set; } = "";
        public bool Enabled { get; set; } = true;
        
        [Required]
        public string MonitorPath { get; set; } = "";
        
        public FIMMonitoringMode MonitoringMode { get; set; } = FIMMonitoringMode.RealTime;
        public FIMMonitoringOptionsDto MonitoringOptions { get; set; } = new();
        public FIMFiltersDto Filters { get; set; } = new();
        public string SecurityLevel { get; set; } = "Medium";
        public FIMAlertSettingsDto AlertSettings { get; set; } = new();
        public Dictionary<string, object> OSSpecificSettings { get; set; } = new();
        public List<string> Tags { get; set; } = new();
        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
        public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;
    }

    /// <summary>
    /// FIM Monitoring Options DTO
    /// </summary>
    public class FIMMonitoringOptionsDto
    {
        public bool MonitorCreation { get; set; } = true;
        public bool MonitorModification { get; set; } = true;
        public bool MonitorDeletion { get; set; } = true;
        public bool MonitorRename { get; set; } = true;
        public bool MonitorPermissions { get; set; } = true;
        public bool MonitorOwnership { get; set; } = true;
        public bool MonitorAttributes { get; set; } = false;
        public bool MonitorHashes { get; set; } = true;
        public string HashAlgorithm { get; set; } = "SHA256";
        public bool RecursiveMonitoring { get; set; } = true;
        public int MaxRecursionDepth { get; set; } = 0;
    }

    /// <summary>
    /// FIM Filters DTO
    /// </summary>
    public class FIMFiltersDto
    {
        public List<string> IncludeExtensions { get; set; } = new();
        public List<string> ExcludeExtensions { get; set; } = new();
        public List<string> IncludeFilePatterns { get; set; } = new();
        public List<string> ExcludeFilePatterns { get; set; } = new();
        public List<string> ExcludeDirectories { get; set; } = new();
        public long MinFileSize { get; set; } = 0;
        public long MaxFileSize { get; set; } = 0;
        public int ModificationTimeWindow { get; set; } = 0;
    }

    /// <summary>
    /// FIM Alert Settings DTO
    /// </summary>
    public class FIMAlertSettingsDto
    {
        public bool EnableAlerts { get; set; } = true;
        public string AlertSeverity { get; set; } = "Medium";
        public List<string> AlertOnEvents { get; set; } = new();
        public int MaxAlertsPerHour { get; set; } = 100;
        public string AlertMessageTemplate { get; set; } = "";
        public List<string> AlertChannels { get; set; } = new();
    }

    /// <summary>
    /// FIM Global Settings DTO
    /// </summary>
    public class FIMGlobalSettingsDto
    {
        public int DefaultScanInterval { get; set; } = 60;
        public int MaxEventBuffer { get; set; } = 10000;
        public bool EnableCompression { get; set; } = true;
        public int RetentionDays { get; set; } = 90;
        public bool EnableBaseline { get; set; } = true;
        public int BaselineUpdateFrequency { get; set; } = 7;
        public FIMPerformanceSettingsDto Performance { get; set; } = new();
    }

    /// <summary>
    /// FIM Performance Settings DTO
    /// </summary>
    public class FIMPerformanceSettingsDto
    {
        public int MaxCpuUsage { get; set; } = 25;
        public int MaxMemoryUsage { get; set; } = 512;
        public int MaxDiskIOPS { get; set; } = 1000;
        public int ThreadPoolSize { get; set; } = 4;
        public int BatchSize { get; set; } = 100;
        public bool EnableThrottling { get; set; } = true;
    }

    /// <summary>
    /// FIM Configuration DTO for API communication
    /// </summary>
    public class FIMConfigurationDto
    {
        public string Id { get; set; } = Guid.NewGuid().ToString();
        public string Name { get; set; } = "";
        public string Description { get; set; } = "";
        public bool Enabled { get; set; } = true;
        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
        public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;
        public string CreatedBy { get; set; } = "";
        public List<FIMRuleDto> Rules { get; set; } = new();
        public FIMGlobalSettingsDto GlobalSettings { get; set; } = new();
        public List<string> TargetAgents { get; set; } = new();
        public List<string> SupportedOS { get; set; } = new() { "Windows", "Linux", "macOS" };
    }

    /// <summary>
    /// FIM Configuration Request DTO
    /// </summary>
    public class FIMConfigurationRequestDto
    {
        [Required]
        public string Name { get; set; } = "";
        
        public string Description { get; set; } = "";
        
        [Required]
        public List<FIMRuleRequestDto> Rules { get; set; } = new();
        
        public List<string> TargetAgents { get; set; } = new();
        
        public FIMGlobalSettingsDto? GlobalSettings { get; set; }
    }

    /// <summary>
    /// FIM Rule Request DTO
    /// </summary>
    public class FIMRuleRequestDto
    {
        [Required]
        public string Name { get; set; } = "";
        
        public string Description { get; set; } = "";
        
        [Required]
        public string MonitorPath { get; set; } = "";
        
        public FIMMonitoringMode MonitoringMode { get; set; } = FIMMonitoringMode.RealTime;
        
        public FIMMonitoringOptionsDto? MonitoringOptions { get; set; }
        
        public FIMFiltersDto? Filters { get; set; }
        
        public string SecurityLevel { get; set; } = "Medium";
        
        public FIMAlertSettingsDto? AlertSettings { get; set; }
        
        public List<string> Tags { get; set; } = new();
    }

    /// <summary>
    /// FIM Template DTO
    /// </summary>
    public class FIMTemplateDto
    {
        public string Id { get; set; } = Guid.NewGuid().ToString();
        public string Name { get; set; } = "";
        public string Description { get; set; } = "";
        public string Category { get; set; } = "";
        public List<FIMRuleDto> TemplateRules { get; set; } = new();
        public List<string> SupportedOS { get; set; } = new();
        public Dictionary<string, string> Variables { get; set; } = new();
        public bool IsBuiltIn { get; set; } = false;
        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
        public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;
        public string CreatedBy { get; set; } = "";
    }

    /// <summary>
    /// FIM Monitoring modes
    /// </summary>
    public enum FIMMonitoringMode
    {
        RealTime,
        Scheduled,
        Hybrid,
        OnDemand
    }

    /// <summary>
    /// FIM File Info DTO
    /// </summary>
    public class FIMFileInfoDto
    {
        public long Size { get; set; }
        public DateTime CreatedTime { get; set; }
        public DateTime ModifiedTime { get; set; }
        public DateTime AccessedTime { get; set; }
        public string Permissions { get; set; } = "";
        public string Owner { get; set; } = "";
        public string Group { get; set; } = "";
        public string Attributes { get; set; } = "";
        public Dictionary<string, string> Hashes { get; set; } = new();
        public string DigitalSignature { get; set; } = "";
        public string FileVersion { get; set; } = "";
        public string MimeType { get; set; } = "";
    }

    /// <summary>
    /// FIM Event DTO
    /// </summary>
    public class FIMEventDto
    {
        public string Id { get; set; } = Guid.NewGuid().ToString();
        public string RuleId { get; set; } = "";
        public string RuleName { get; set; } = "";
        public string AgentId { get; set; } = "";
        public DateTime Timestamp { get; set; } = DateTime.UtcNow;
        public string FilePath { get; set; } = "";
        public string EventType { get; set; } = "";
        public string OldFilePath { get; set; } = "";
        public FIMFileInfoDto? OldFileInfo { get; set; }
        public FIMFileInfoDto? NewFileInfo { get; set; }
        public string User { get; set; } = "";
        public string Process { get; set; } = "";
        public int? ProcessId { get; set; }
        public string SecurityLevel { get; set; } = "Medium";
        public Dictionary<string, object> Metadata { get; set; } = new();
        public bool AlertGenerated { get; set; } = false;
        public List<string> Tags { get; set; } = new();
    }
} 