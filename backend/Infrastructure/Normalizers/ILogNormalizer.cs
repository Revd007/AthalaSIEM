using Backend.Domain.Entities;
using Backend.Domain.ValueObjects;

namespace Backend.Infrastructure.Normalizers;

public interface ILogNormalizer
{
    Task<ECSLogFields?> NormalizeAsync(LogEntry logEntry, CancellationToken cancellationToken = default);
    Task<IEnumerable<ECSLogFields>> NormalizeBatchAsync(IEnumerable<LogEntry> logEntries, CancellationToken cancellationToken = default);
}
