using Backend.Domain.Entities;

namespace Backend.Domain.Interfaces;

public interface IDetectionRuleRepository
{
    Task<DetectionRule?> GetByIdAsync(string id, CancellationToken cancellationToken = default);
    Task<IEnumerable<DetectionRule>> GetActiveRulesAsync(CancellationToken cancellationToken = default);
    Task<IEnumerable<DetectionRule>> GetAllAsync(CancellationToken cancellationToken = default);
    Task AddAsync(DetectionRule rule, CancellationToken cancellationToken = default);
    Task UpdateAsync(DetectionRule rule, CancellationToken cancellationToken = default);
    Task DeleteAsync(string id, CancellationToken cancellationToken = default);
}
