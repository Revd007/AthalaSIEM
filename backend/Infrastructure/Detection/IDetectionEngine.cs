using Backend.Domain.Entities;
using Backend.Infrastructure.Detection.RuleEngine;

namespace Backend.Infrastructure.Detection;

public interface IDetectionEngine
{
    Task<IEnumerable<DetectionResult>> DetectAsync(LogEntry logEntry, CancellationToken cancellationToken = default);
    Task<IEnumerable<DetectionResult>> DetectBatchAsync(IEnumerable<LogEntry> logEntries, CancellationToken cancellationToken = default);
    Task RegisterRuleAsync(DetectionRule rule, CancellationToken cancellationToken = default);
    Task UnregisterRuleAsync(string ruleId, CancellationToken cancellationToken = default);
}
