using Backend.Domain.ValueObjects;

namespace Backend.Domain.Entities;

public class LogEntry
{
    public string Id { get; set; } = Guid.NewGuid().ToString();
    public string AgentId { get; set; } = string.Empty;
    public DateTime Timestamp { get; set; }
    public DateTime ReceivedAt { get; set; } = DateTime.UtcNow;
    
    // Raw log data
    public string RawMessage { get; set; } = string.Empty;
    public string Source { get; set; } = string.Empty;
    public string? Category { get; set; }
    public long? EventId { get; set; }
    public string? RawProperties { get; set; } // JSON
    
    // Normalized ECS fields
    public ECSLogFields? NormalizedFields { get; set; }
    public bool IsNormalized { get; set; }
    public DateTime? NormalizedAt { get; set; }
    
    // Processing state
    public bool Processed { get; set; }
    public DateTime? ProcessedAt { get; set; }
    public bool Enriched { get; set; }
    public DateTime? EnrichedAt { get; set; }
    
    // Detection metadata
    public List<string> MatchedRuleIds { get; set; } = new();
    public List<string> TechniqueIds { get; set; } = new();
    public string? CorrelationId { get; set; }
    
    // Enrichment data
    public Dictionary<string, object>? EnrichmentData { get; set; }
}
