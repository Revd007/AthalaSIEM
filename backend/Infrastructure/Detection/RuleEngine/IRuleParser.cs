using Backend.Domain.Entities;
using Backend.Domain.ValueObjects;

namespace Backend.Infrastructure.Detection.RuleEngine;

public interface IRuleParser
{
    ParsedRule Parse(string ruleDefinition);
}

public class ParsedRule
{
    public string Id { get; set; } = string.Empty;
    public string Name { get; set; } = string.Empty;
    public string Description { get; set; } = string.Empty;
    public RuleType Type { get; set; }
    public Dictionary<string, object> Conditions { get; set; } = new();
    public List<string> TechniqueIds { get; set; } = new();
    public AlertSeverityLevel Severity { get; set; }
    public int? ThresholdCount { get; set; }
    public TimeSpan? ThresholdWindow { get; set; }
}
