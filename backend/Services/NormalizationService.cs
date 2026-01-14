using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging;
using Backend.Data;
using Backend.Domain.Entities;
using Backend.Domain.ValueObjects;
using Backend.Infrastructure.Normalizers;

namespace Backend.Services;

/// <summary>
/// Service for normalization operations
/// </summary>
public class NormalizationService : INormalizationService
{
    private readonly ILogNormalizer _normalizer;
    private readonly ApplicationDbContext _context;
    private readonly ILogger<NormalizationService> _logger;

    public NormalizationService(
        ILogNormalizer normalizer,
        ApplicationDbContext context,
        ILogger<NormalizationService> logger)
    {
        _normalizer = normalizer;
        _context = context;
        _logger = logger;
    }

    public async Task<ECSLogFields?> NormalizeLogAsync(LogEntry logEntry, CancellationToken cancellationToken = default)
    {
        return await _normalizer.NormalizeAsync(logEntry, cancellationToken);
    }

    public async Task<IEnumerable<ECSLogFields>> NormalizeBatchAsync(IEnumerable<LogEntry> logEntries, CancellationToken cancellationToken = default)
    {
        return await _normalizer.NormalizeBatchAsync(logEntries, cancellationToken);
    }

    public async Task<object> GetStatisticsAsync(DateTime? startDate = null, DateTime? endDate = null, CancellationToken cancellationToken = default)
    {
        var start = startDate ?? DateTime.UtcNow.AddDays(-7);
        var end = endDate ?? DateTime.UtcNow;

        var totalLogs = await _context.LogEntries
            .Where(l => l.Timestamp >= start && l.Timestamp <= end)
            .CountAsync(cancellationToken);

        var normalizedLogs = await _context.NormalizedLogs
            .Where(nl => nl.Timestamp >= start && nl.Timestamp <= end)
            .CountAsync(cancellationToken);

        var eventTypes = await _context.NormalizedLogs
            .Where(nl => nl.Timestamp >= start && nl.Timestamp <= end && !string.IsNullOrEmpty(nl.EventType))
            .GroupBy(nl => nl.EventType)
            .Select(g => new { EventType = g.Key, Count = g.Count() })
            .ToListAsync(cancellationToken);

        var severityDistribution = await _context.NormalizedLogs
            .Where(nl => nl.Timestamp >= start && nl.Timestamp <= end && nl.SiemSeverity.HasValue)
            .GroupBy(nl => nl.SiemSeverity)
            .Select(g => new { Severity = g.Key, Count = g.Count() })
            .ToListAsync(cancellationToken);

        return new
        {
            TotalLogs = totalLogs,
            NormalizedLogs = normalizedLogs,
            NormalizationRate = totalLogs > 0 ? (double)normalizedLogs / totalLogs * 100 : 0,
            EventTypeDistribution = eventTypes,
            SeverityDistribution = severityDistribution,
            TimeRange = new { Start = start, End = end }
        };
    }
}
