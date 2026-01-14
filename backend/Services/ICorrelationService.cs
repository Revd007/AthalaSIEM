using Backend.Infrastructure.Correlation;

namespace Backend.Services;

/// <summary>
/// Service interface for correlation operations
/// </summary>
public interface ICorrelationService
{
    /// <summary>
    /// Get correlation statistics
    /// </summary>
    Task<object> GetStatisticsAsync(DateTime? startDate = null, DateTime? endDate = null, CancellationToken cancellationToken = default);

    /// <summary>
    /// Get active correlation rules
    /// </summary>
    Task<IEnumerable<object>> GetRulesAsync(CancellationToken cancellationToken = default);

    /// <summary>
    /// Trigger correlation for a specific log entry
    /// </summary>
    Task<CorrelationResult?> TriggerCorrelationAsync(string logEntryId, CancellationToken cancellationToken = default);
}
