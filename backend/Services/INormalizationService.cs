using Backend.Domain.ValueObjects;
using Backend.Domain.Entities;

namespace Backend.Services;

/// <summary>
/// Service interface for normalization operations
/// </summary>
public interface INormalizationService
{
    /// <summary>
    /// Normalize a single log entry
    /// </summary>
    Task<ECSLogFields?> NormalizeLogAsync(LogEntry logEntry, CancellationToken cancellationToken = default);

    /// <summary>
    /// Normalize a batch of log entries
    /// </summary>
    Task<IEnumerable<ECSLogFields>> NormalizeBatchAsync(IEnumerable<LogEntry> logEntries, CancellationToken cancellationToken = default);

    /// <summary>
    /// Get normalization statistics
    /// </summary>
    Task<object> GetStatisticsAsync(DateTime? startDate = null, DateTime? endDate = null, CancellationToken cancellationToken = default);
}
