namespace Backend.Domain.Entities;

public class Agent
{
    public string Id { get; set; } = Guid.NewGuid().ToString();
    public string Name { get; set; } = string.Empty;
    public string Hostname { get; set; } = string.Empty;
    public string IpAddress { get; set; } = string.Empty;
    public string? OperatingSystem { get; set; }
    public string? AgentVersion { get; set; }
    public string ApiKey { get; set; } = string.Empty;
    public AgentStatus Status { get; set; } = AgentStatus.Offline;
    public DateTime? LastHeartbeat { get; set; }
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;
    
    // Configuration
    public Dictionary<string, object>? Configuration { get; set; }
    public List<string> EnabledCollectors { get; set; } = new();
    
    // Health metrics
    public double? CpuUsage { get; set; }
    public double? MemoryUsage { get; set; }
    public long? LogsSentCount { get; set; }
    public DateTime? LastLogSentAt { get; set; }
}

public enum AgentStatus
{
    Offline,
    Online,
    Degraded,
    Error
}
