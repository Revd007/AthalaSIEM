using Microsoft.Extensions.Logging;
using Backend.Domain.Entities;
using Backend.Domain.Interfaces;

namespace Backend.Infrastructure.Correlation;

public class TemporalCorrelator : ICorrelationEngine
{
    private readonly ILogRepository _logRepository;
    private readonly ILogger<TemporalCorrelator> _logger;
    private readonly TimeSpan _defaultTimeWindow = TimeSpan.FromMinutes(15);

    public TemporalCorrelator(
        ILogRepository logRepository,
        ILogger<TemporalCorrelator> logger)
    {
        _logRepository = logRepository;
        _logger = logger;
    }

    public async Task<IEnumerable<CorrelationResult>> CorrelateAsync(LogEntry logEntry, CancellationToken cancellationToken = default)
    {
        var results = new List<CorrelationResult>();

        if (logEntry.NormalizedFields == null)
            return results;

        // Find logs within time window from same agent
        var startTime = logEntry.Timestamp.Subtract(_defaultTimeWindow);
        var endTime = logEntry.Timestamp.Add(_defaultTimeWindow);

        var relatedLogs = await _logRepository.GetByAgentIdAsync(
            logEntry.AgentId,
            startTime,
            endTime,
            cancellationToken);

        var correlated = relatedLogs
            .Where(l => l.Id != logEntry.Id && l.NormalizedFields != null)
            .ToList();

        if (correlated.Any())
        {
            var correlationId = Guid.NewGuid().ToString();
            
            // Set correlation ID on all logs
            foreach (var log in correlated)
            {
                log.CorrelationId = correlationId;
            }
            logEntry.CorrelationId = correlationId;

            results.Add(new CorrelationResult
            {
                CorrelationId = correlationId,
                CorrelatedLogs = new List<LogEntry> { logEntry }.Concat(correlated).ToList(),
                Type = CorrelationType.Temporal,
                Confidence = CalculateConfidence(correlated.Count),
                Metadata = new Dictionary<string, object>
                {
                    ["time_window_minutes"] = _defaultTimeWindow.TotalMinutes,
                    ["agent_id"] = logEntry.AgentId,
                    ["correlated_count"] = correlated.Count
                }
            });
        }

        return results;
    }

    public async Task<IEnumerable<CorrelationResult>> CorrelateBatchAsync(IEnumerable<LogEntry> logEntries, CancellationToken cancellationToken = default)
    {
        var allResults = new List<CorrelationResult>();

        foreach (var logEntry in logEntries)
        {
            var results = await CorrelateAsync(logEntry, cancellationToken);
            allResults.AddRange(results);
        }

        return allResults;
    }

    private double CalculateConfidence(int correlatedCount)
    {
        // More correlated events = higher confidence
        return Math.Min(0.5 + (correlatedCount * 0.1), 1.0);
    }
}
