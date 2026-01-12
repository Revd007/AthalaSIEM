using Backend.Domain.Entities;

namespace Backend.Domain.Interfaces;

public interface ILogRepository
{
    Task<LogEntry?> GetByIdAsync(string id, CancellationToken cancellationToken = default);
    Task<IEnumerable<LogEntry>> GetByAgentIdAsync(string agentId, DateTime? startTime = null, DateTime? endTime = null, CancellationToken cancellationToken = default);
    Task<IEnumerable<LogEntry>> GetUnprocessedAsync(int limit = 1000, CancellationToken cancellationToken = default);
    Task<IEnumerable<LogEntry>> GetUnnormalizedAsync(int limit = 1000, CancellationToken cancellationToken = default);
    Task AddAsync(LogEntry logEntry, CancellationToken cancellationToken = default);
    Task AddRangeAsync(IEnumerable<LogEntry> logEntries, CancellationToken cancellationToken = default);
    Task UpdateAsync(LogEntry logEntry, CancellationToken cancellationToken = default);
    Task UpdateRangeAsync(IEnumerable<LogEntry> logEntries, CancellationToken cancellationToken = default);
}
