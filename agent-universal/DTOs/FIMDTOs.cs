using System;
using System.Collections.Generic;

namespace AthalaSIEM.UniversalAgent.DTOs
{
    /// <summary>
    /// FIM Configuration DTO for agent-backend communication
    /// </summary>
    public class FIMConfigurationDto
    {
        public string Id { get; set; } = "";
        public string Name { get; set; } = "";
        public string Description { get; set; } = "";
        public bool Enabled { get; set; } = true;
        public List<FIMRuleDto> Rules { get; set; } = new();
        public FIMGlobalSettingsDto GlobalSettings { get; set; } = new();
        public List<string> TargetAgents { get; set; } = new();
        public List<string> SupportedOS { get; set; } = new();
    }

    /// <summary>
    /// FIM Rule DTO for agent communication
    /// </summary>
    public class FIMRuleDto
    {
        public string Id { get; set; } = "";
        public string Name { get; set; } = "";
        public string Description { get; set; } = "";
        public bool Enabled { get; set; } = true;
        public string MonitorPath { get; set; } = "";
        public FIMMonitoringMode MonitoringMode { get; set; } = FIMMonitoringMode.RealTime;
        public FIMMonitoringOptionsDto MonitoringOptions { get; set; } = new();
        public FIMFiltersDto Filters { get; set; } = new();
        public string SecurityLevel { get; set; } = "Medium";
        public FIMAlertSettingsDto AlertSettings { get; set; } = new();
        public List<string> Tags { get; set; } = new();
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
    /// FIM Template DTO
    /// </summary>
    public class FIMTemplateDto
    {
        public string Id { get; set; } = "";
        public string Name { get; set; } = "";
        public string Description { get; set; } = "";
        public string Category { get; set; } = "";
        public List<FIMRuleDto> TemplateRules { get; set; } = new();
        public List<string> SupportedOS { get; set; } = new();
        public Dictionary<string, string> Variables { get; set; } = new();
        public bool IsBuiltIn { get; set; } = false;
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
    /// FIM Monitoring modes
    /// </summary>
    public enum FIMMonitoringMode
    {
        RealTime,
        Scheduled,
        Hybrid,
        OnDemand
    }
} 