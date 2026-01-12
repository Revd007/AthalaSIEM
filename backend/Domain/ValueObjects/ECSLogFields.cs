namespace Backend.Domain.ValueObjects;

public class ECSLogFields
{
    // Core ECS fields
    public DateTime Timestamp { get; set; }
    public string? AgentId { get; set; }
    public string? AgentName { get; set; }
    public string? HostName { get; set; }
    public string? HostIp { get; set; }
    
    // User fields
    public string? UserName { get; set; }
    public string? UserId { get; set; }
    public string? UserDomain { get; set; }
    
    // Process fields
    public string? ProcessName { get; set; }
    public int? ProcessId { get; set; }
    public string? ProcessPath { get; set; }
    public string? ProcessCommandLine { get; set; }
    public string? ProcessHash { get; set; }
    public string? ParentProcessName { get; set; }
    public int? ParentProcessId { get; set; }
    
    // Network fields
    public string? SourceIp { get; set; }
    public int? SourcePort { get; set; }
    public string? DestinationIp { get; set; }
    public int? DestinationPort { get; set; }
    public string? Protocol { get; set; }
    
    // Event fields
    public string? EventAction { get; set; }
    public string? EventCategory { get; set; }
    public string? EventType { get; set; }
    public string? EventOutcome { get; set; }
    public string? EventCode { get; set; }
    
    // File fields
    public string? FilePath { get; set; }
    public string? FileName { get; set; }
    public string? FileHash { get; set; }
    public long? FileSize { get; set; }
    
    // SIEM extensions
    public string? SiemRuleId { get; set; }
    public string? SiemTechniqueId { get; set; }
    public double? SiemConfidence { get; set; }
    public int? SiemSeverity { get; set; }
    public string? SiemCorrelationId { get; set; }
    
    // Additional metadata
    public Dictionary<string, object>? Metadata { get; set; }
}
