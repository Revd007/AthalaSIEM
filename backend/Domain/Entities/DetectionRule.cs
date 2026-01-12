namespace Backend.Domain.Entities;

public class DetectionRule
{
    public string Id { get; set; } = Guid.NewGuid().ToString();
    public string Name { get; set; } = string.Empty;
    public string Description { get; set; } = string.Empty;
    public string RuleDefinition { get; set; } = string.Empty; // YAML/JSON
    public RuleType Type { get; set; }
    
    public AlertSeverityLevel DefaultSeverity { get; set; }
    public bool Enabled { get; set; } = true;
    
    // MITRE ATT&CK mapping
    public List<string> TechniqueIds { get; set; } = new();
    public List<string> TacticIds { get; set; } = new();
    
    // Thresholds for threshold-based rules
    public int? ThresholdCount { get; set; }
    public TimeSpan? ThresholdWindow { get; set; }
    
    // Whitelisting
    public List<string> WhitelistIps { get; set; } = new();
    public List<string> WhitelistUsers { get; set; } = new();
    public List<string> WhitelistProcesses { get; set; } = new();
    
    // Statistics
    public int MatchCount { get; set; }
    public int FalsePositiveCount { get; set; }
    public DateTime? LastMatchedAt { get; set; }
    
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;
    public string? CreatedBy { get; set; }
}

public enum RuleType
{
    PatternMatch,
    Threshold,
    Statistical,
    Correlation,
    MLBased
}
