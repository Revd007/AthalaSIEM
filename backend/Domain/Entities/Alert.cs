using Backend.Domain.ValueObjects;

namespace Backend.Domain.Entities;

public class Alert
{
    public string Id { get; set; } = Guid.NewGuid().ToString();
    public string? AgentId { get; set; }
    public string Title { get; set; } = string.Empty;
    public string Description { get; set; } = string.Empty;
    public string Message { get; set; } = string.Empty;
    
    public AlertSeverityLevel Severity { get; set; }
    public SeverityScore SeverityScore { get; set; } = new();
    public AlertStatus Status { get; set; } = AlertStatus.New;
    
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    public string Source { get; set; } = string.Empty;
    
    // Detection metadata
    public string? RuleId { get; set; }
    public List<string> TechniqueIds { get; set; } = new();
    public string? CorrelationId { get; set; }
    public double Confidence { get; set; }
    
    // Explainable detection
    public string? DetectionReason { get; set; }
    public Dictionary<string, object>? DetectionMetadata { get; set; }
    public List<string> RelatedLogIds { get; set; } = new();
    
    // Lifecycle
    public string? AcknowledgedBy { get; set; }
    public DateTime? AcknowledgedAt { get; set; }
    public string? ResolvedBy { get; set; }
    public DateTime? ResolvedAt { get; set; }
    public string? ResolutionNotes { get; set; }
    
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;
    
    // Deduplication
    public string? DeduplicationKey { get; set; }
    public int OccurrenceCount { get; set; } = 1;
    public DateTime? FirstOccurrence { get; set; }
    public DateTime? LastOccurrence { get; set; }
}

public enum AlertStatus
{
    New,
    Acknowledged,
    InProgress,
    Resolved,
    FalsePositive,
    Closed
}
