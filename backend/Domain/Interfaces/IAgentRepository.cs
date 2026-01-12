using Backend.Domain.Entities;

namespace Backend.Domain.Interfaces;

public interface IAgentRepository
{
    Task<Agent?> GetByIdAsync(string id, CancellationToken cancellationToken = default);
    Task<Agent?> GetByApiKeyAsync(string apiKey, CancellationToken cancellationToken = default);
    Task<IEnumerable<Agent>> GetAllAsync(CancellationToken cancellationToken = default);
    Task<IEnumerable<Agent>> GetOnlineAgentsAsync(CancellationToken cancellationToken = default);
    Task<IEnumerable<Agent>> GetOfflineAgentsAsync(TimeSpan threshold, CancellationToken cancellationToken = default);
    Task AddAsync(Agent agent, CancellationToken cancellationToken = default);
    Task UpdateAsync(Agent agent, CancellationToken cancellationToken = default);
    Task DeleteAsync(string id, CancellationToken cancellationToken = default);
    Task<bool> ValidateApiKeyAsync(string agentId, string apiKey, CancellationToken cancellationToken = default);
}
