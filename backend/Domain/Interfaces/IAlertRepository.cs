using Backend.Domain.Entities;

namespace Backend.Domain.Interfaces;

public interface IAlertRepository
{
    Task<Alert?> GetByIdAsync(string id, CancellationToken cancellationToken = default);
    Task<Alert?> GetByDeduplicationKeyAsync(string key, CancellationToken cancellationToken = default);
    Task<IEnumerable<Alert>> GetByStatusAsync(AlertStatus status, CancellationToken cancellationToken = default);
    Task<IEnumerable<Alert>> GetByAgentIdAsync(string agentId, CancellationToken cancellationToken = default);
    Task AddAsync(Alert alert, CancellationToken cancellationToken = default);
    Task UpdateAsync(Alert alert, CancellationToken cancellationToken = default);
    Task DeleteAsync(string id, CancellationToken cancellationToken = default);
}
