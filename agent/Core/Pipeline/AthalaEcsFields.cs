namespace AthalaSIEM.Agent.Core.Pipeline;

public class AthalaEcsFields
{
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    
    public string? AgentId { get; set; }
    public string? AgentName { get; set; }
    
    public string? HostName { get; set; }
    public string? HostOs { get; set; }
    
    public string? EventCategory { get; set; }
    public string? EventAction { get; set; }
    public string? EventOutcome { get; set; }
    public string? LogLevel { get; set; }
    
    public string? UserName { get; set; }
    public string? UserId { get; set; }
    
    public string? ProcessName { get; set; }
    public int? ProcessId { get; set; }
    public string? ProcessCommandLine { get; set; }
    public string? ProcessParentName { get; set; }
    
    public string? SourceIp { get; set; }
    public int? SourcePort { get; set; }
    public string? DestinationIp { get; set; }
    public int? DestinationPort { get; set; }
    public string? NetworkProtocol { get; set; }
    
    public Dictionary<string, object> AdditionalFields { get; set; } = new();
}
