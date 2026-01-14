using Backend.Domain.Entities;

namespace Backend.Infrastructure.Correlation;

public interface ICorrelationEngine
{
    Task<IEnumerable<CorrelationResult>> CorrelateAsync(LogEntry logEntry, CancellationToken cancellationToken = default);
    Task<IEnumerable<CorrelationResult>> CorrelateBatchAsync(IEnumerable<LogEntry> logEntries, CancellationToken cancellationToken = default);
}

public class CorrelationResult
{
    public string CorrelationId { get; set; } = Guid.NewGuid().ToString();
    public string? RuleName { get; set; }
    public string? RuleDescription { get; set; }
    public List<LogEntry> CorrelatedLogs { get; set; } = new();
    public CorrelationType Type { get; set; }
    public double Confidence { get; set; }
    public Dictionary<string, object> Metadata { get; set; } = new();
}

public enum CorrelationType
{
    Temporal,
    CrossAgent,
    Behavioral,
    AttackChain,
    RuleBased
}
