using Backend.Domain.Entities;
using Backend.Domain.ValueObjects;

namespace Backend.Infrastructure.Detection.RuleEngine;

public interface IRuleExecutor
{
    Task<DetectionResult> ExecuteAsync(DetectionRule rule, LogEntry logEntry, CancellationToken cancellationToken = default);
    Task<IEnumerable<DetectionResult>> ExecuteBatchAsync(DetectionRule rule, IEnumerable<LogEntry> logEntries, CancellationToken cancellationToken = default);
}

public class DetectionResult
{
    public bool Matched { get; set; }
    public DetectionRule Rule { get; set; } = null!;
    public LogEntry LogEntry { get; set; } = null!;
    public Dictionary<string, object> MatchContext { get; set; } = new();
    public string? Reason { get; set; }
    public double Confidence { get; set; } = 1.0;
}
