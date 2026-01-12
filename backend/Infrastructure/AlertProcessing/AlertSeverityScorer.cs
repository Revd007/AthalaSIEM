using Microsoft.Extensions.Logging;
using Backend.Domain.Entities;
using Backend.Domain.ValueObjects;

namespace Backend.Infrastructure.AlertProcessing;

public interface IAlertSeverityScorer
{
    SeverityScore CalculateSeverity(Alert alert, LogEntry? logEntry = null);
}

public class AlertSeverityScorer : IAlertSeverityScorer
{
    private readonly ILogger<AlertSeverityScorer> _logger;

    public AlertSeverityScorer(ILogger<AlertSeverityScorer> logger)
    {
        _logger = logger;
    }

    public SeverityScore CalculateSeverity(Alert alert, LogEntry? logEntry = null)
    {
        var score = new SeverityScore
        {
            BaseSeverity = (int)alert.Severity
        };

        // Threat Intelligence Score
        score.ThreatIntelligenceScore = CalculateThreatIntelligenceScore(alert, logEntry);

        // Correlation Score
        score.CorrelationScore = CalculateCorrelationScore(alert);

        // Anomaly Score (if available in metadata)
        score.AnomalyScore = CalculateAnomalyScore(alert);

        // Technique Score
        score.TechniqueScore = CalculateTechniqueScore(alert);

        return score;
    }

    private int CalculateThreatIntelligenceScore(Alert alert, LogEntry? logEntry)
    {
        int score = 0;

        // Check enrichment data for threat intelligence indicators
        if (logEntry?.EnrichmentData != null)
        {
            if (logEntry.EnrichmentData.ContainsKey("threat_intelligence"))
            {
                var tiData = logEntry.EnrichmentData["threat_intelligence"];
                if (tiData != null)
                {
                    // Known malicious IP
                    if (tiData.ToString()?.Contains("malicious_ip", StringComparison.OrdinalIgnoreCase) == true)
                        score += 3;

                    // Known malicious hash
                    if (tiData.ToString()?.Contains("malicious_hash", StringComparison.OrdinalIgnoreCase) == true)
                        score += 5;

                    // Known C2 domain
                    if (tiData.ToString()?.Contains("c2_domain", StringComparison.OrdinalIgnoreCase) == true)
                        score += 4;
                }
            }
        }

        return score;
    }

    private int CalculateCorrelationScore(Alert alert)
    {
        if (string.IsNullOrEmpty(alert.CorrelationId))
            return 0;

        // If correlation ID exists, check occurrence count
        if (alert.OccurrenceCount > 1)
        {
            if (alert.OccurrenceCount >= 6)
                return 4;
            if (alert.OccurrenceCount >= 2)
                return 2;
        }

        // Attack chain detected (multiple techniques)
        if (alert.TechniqueIds.Count > 1)
            return 5;

        return 0;
    }

    private int CalculateAnomalyScore(Alert alert)
    {
        if (alert.DetectionMetadata != null)
        {
            if (alert.DetectionMetadata.TryGetValue("is_anomaly", out var isAnomaly))
            {
                if (isAnomaly?.ToString()?.ToLowerInvariant() == "true")
                {
                    // Check anomaly type
                    if (alert.DetectionMetadata.TryGetValue("anomaly_type", out var anomalyType))
                    {
                        if (anomalyType?.ToString()?.Contains("behavioral", StringComparison.OrdinalIgnoreCase) == true)
                            return 3;
                        return 2; // Statistical anomaly
                    }
                    return 2;
                }
            }
        }

        return 0;
    }

    private int CalculateTechniqueScore(Alert alert)
    {
        if (!alert.TechniqueIds.Any())
            return 0;

        // Score based on technique severity (simplified)
        // In production, this would query MITRE ATT&CK data for technique severity
        int maxScore = 0;
        foreach (var techniqueId in alert.TechniqueIds)
        {
            // High-severity techniques
            if (techniqueId.StartsWith("T1") || techniqueId.Contains("Execution") || techniqueId.Contains("Persistence"))
            {
                maxScore = Math.Max(maxScore, 3);
            }
            // Medium-severity techniques
            else if (techniqueId.StartsWith("T10") || techniqueId.Contains("Discovery") || techniqueId.Contains("Collection"))
            {
                maxScore = Math.Max(maxScore, 2);
            }
            // Default
            else
            {
                maxScore = Math.Max(maxScore, 1);
            }
        }

        return maxScore;
    }
}
