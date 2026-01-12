using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.Logging;
using Backend.Domain.Entities;

namespace Backend.Infrastructure.AlertProcessing;

public interface IAlertDeduplicator
{
    string GenerateDeduplicationKey(Alert alert);
    Task<Alert?> FindDuplicateAsync(Alert alert, CancellationToken cancellationToken = default);
    Task MergeDuplicateAsync(Alert existingAlert, Alert newAlert, CancellationToken cancellationToken = default);
}

public class AlertDeduplicator : IAlertDeduplicator
{
    private readonly ILogger<AlertDeduplicator> _logger;

    public AlertDeduplicator(ILogger<AlertDeduplicator> logger)
    {
        _logger = logger;
    }

    public string GenerateDeduplicationKey(Alert alert)
    {
        // Generate key based on rule, agent, and key fields
        var keyComponents = new List<string>
        {
            alert.RuleId ?? "unknown",
            alert.AgentId ?? "unknown",
            alert.Source
        };

        // Add technique IDs if available
        if (alert.TechniqueIds.Any())
        {
            keyComponents.AddRange(alert.TechniqueIds.OrderBy(t => t));
        }

        // Add correlation ID if available
        if (!string.IsNullOrEmpty(alert.CorrelationId))
        {
            keyComponents.Add(alert.CorrelationId);
        }

        var keyString = string.Join("|", keyComponents);
        
        // Generate hash for consistent key
        using var sha256 = SHA256.Create();
        var hashBytes = sha256.ComputeHash(Encoding.UTF8.GetBytes(keyString));
        return Convert.ToBase64String(hashBytes);
    }

    public async Task<Alert?> FindDuplicateAsync(Alert alert, CancellationToken cancellationToken = default)
    {
        // This would query the database for existing alerts with the same deduplication key
        // For now, return null (implementation would use IAlertRepository)
        await Task.CompletedTask;
        return null;
    }

    public async Task MergeDuplicateAsync(Alert existingAlert, Alert newAlert, CancellationToken cancellationToken = default)
    {
        existingAlert.OccurrenceCount++;
        existingAlert.LastOccurrence = newAlert.Timestamp;
        existingAlert.UpdatedAt = DateTime.UtcNow;

        // Merge related log IDs
        if (newAlert.RelatedLogIds.Any())
        {
            foreach (var logId in newAlert.RelatedLogIds)
            {
                if (!existingAlert.RelatedLogIds.Contains(logId))
                {
                    existingAlert.RelatedLogIds.Add(logId);
                }
            }
        }

        // Update severity if new alert is more severe
        if (newAlert.Severity > existingAlert.Severity)
        {
            existingAlert.Severity = newAlert.Severity;
            existingAlert.SeverityScore = newAlert.SeverityScore;
        }

        _logger.LogDebug("Merged duplicate alert {AlertId} with existing {ExistingAlertId}", 
            newAlert.Id, existingAlert.Id);

        await Task.CompletedTask;
    }
}
