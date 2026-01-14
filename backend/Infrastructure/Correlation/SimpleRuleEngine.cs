using Microsoft.Extensions.Logging;
using Backend.Domain.Entities;
using Backend.Domain.ValueObjects;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace Backend.Infrastructure.Correlation;

/// <summary>
/// Simple Rule Engine for SIEM correlation
/// Detects patterns like "5 failed logins = Brute Force"
/// </summary>
public class SimpleRuleEngine
{
    private readonly ILogger<SimpleRuleEngine> _logger;
    private readonly Dictionary<string, List<LogEntry>> _eventBuffer;
    private readonly TimeSpan _defaultTimeWindow = TimeSpan.FromMinutes(5);

    // Predefined correlation rules
    private readonly List<CorrelationRule> _rules;

    public SimpleRuleEngine(ILogger<SimpleRuleEngine> logger)
    {
        _logger = logger;
        _eventBuffer = new Dictionary<string, List<LogEntry>>();
        _rules = InitializeRules();
    }

    /// <summary>
    /// Process a normalized log and check for correlation patterns
    /// </summary>
    public async Task<List<CorrelationResult>> ProcessLogAsync(
        LogEntry logEntry,
        Func<string, DateTime, DateTime, Task<List<LogEntry>>> getRelatedLogsAsync,
        CancellationToken cancellationToken = default)
    {
        var results = new List<CorrelationResult>();

        if (logEntry.NormalizedFields == null)
            return results;

        // Check each rule
        foreach (var rule in _rules)
        {
            if (MatchesRule(logEntry, rule))
            {
                var correlationResult = await CheckRuleConditionAsync(
                    logEntry,
                    rule,
                    getRelatedLogsAsync,
                    cancellationToken);

                if (correlationResult != null)
                {
                    results.Add(correlationResult);
                }
            }
        }

        return results;
    }

    /// <summary>
    /// Check if log entry matches rule pattern
    /// </summary>
    private bool MatchesRule(LogEntry logEntry, CorrelationRule rule)
    {
        var fields = logEntry.NormalizedFields;
        if (fields == null)
            return false;

        // Check event type
        if (!string.IsNullOrEmpty(rule.EventType) && 
            fields.EventType?.Equals(rule.EventType, StringComparison.OrdinalIgnoreCase) != true)
            return false;

        // Check event action
        if (!string.IsNullOrEmpty(rule.EventAction) && 
            fields.EventAction?.Equals(rule.EventAction, StringComparison.OrdinalIgnoreCase) != true)
            return false;

        // Check event outcome
        if (!string.IsNullOrEmpty(rule.EventOutcome) && 
            fields.EventOutcome?.Equals(rule.EventOutcome, StringComparison.OrdinalIgnoreCase) != true)
            return false;

        // Check severity threshold
        if (rule.MinSeverity.HasValue && 
            (fields.SiemSeverity ?? 0) < rule.MinSeverity.Value)
            return false;

        return true;
    }

    /// <summary>
    /// Check rule condition (e.g., count threshold)
    /// </summary>
    private async Task<CorrelationResult?> CheckRuleConditionAsync(
        LogEntry logEntry,
        CorrelationRule rule,
        Func<string, DateTime, DateTime, Task<List<LogEntry>>> getRelatedLogsAsync,
        CancellationToken cancellationToken)
    {
        var fields = logEntry.NormalizedFields;
        if (fields == null)
            return null;

        // Build correlation key (e.g., "source_ip" or "user_name")
        var correlationKey = BuildCorrelationKey(logEntry, rule);

        // Get time window
        var timeWindow = rule.TimeWindowMinutes.HasValue 
            ? TimeSpan.FromMinutes(rule.TimeWindowMinutes.Value)
            : _defaultTimeWindow;

        var startTime = logEntry.Timestamp.Subtract(timeWindow);
        var endTime = logEntry.Timestamp;

        // Get related logs within time window
        var relatedLogs = await getRelatedLogsAsync(correlationKey, startTime, endTime);

        // Filter by rule criteria
        var matchingLogs = relatedLogs
            .Where(l => MatchesRule(l, rule))
            .ToList();

        // Check threshold
        if (matchingLogs.Count >= rule.Threshold)
        {
            _logger.LogWarning(
                "Correlation rule '{RuleName}' triggered: {Count} events in {Minutes} minutes",
                rule.Name,
                matchingLogs.Count,
                timeWindow.TotalMinutes);

            return new CorrelationResult
            {
                CorrelationId = Guid.NewGuid().ToString(),
                RuleName = rule.Name,
                RuleDescription = rule.Description,
                CorrelatedLogs = matchingLogs,
                Type = CorrelationType.RuleBased,
                Confidence = CalculateConfidence(matchingLogs.Count, rule.Threshold),
                Metadata = new Dictionary<string, object>
                {
                    ["correlation_key"] = correlationKey,
                    ["event_count"] = matchingLogs.Count,
                    ["threshold"] = rule.Threshold,
                    ["time_window_minutes"] = timeWindow.TotalMinutes,
                    ["alert_severity"] = rule.AlertSeverity
                }
            };
        }

        return null;
    }

    /// <summary>
    /// Build correlation key based on rule grouping
    /// </summary>
    private string BuildCorrelationKey(LogEntry logEntry, CorrelationRule rule)
    {
        var fields = logEntry.NormalizedFields;
        if (fields == null)
            return logEntry.AgentId ?? "unknown";

        // Group by source IP (for brute force detection)
        if (rule.GroupBy == "source_ip" && !string.IsNullOrEmpty(fields.SourceIp))
            return $"ip:{fields.SourceIp}";

        // Group by user name
        if (rule.GroupBy == "user_name" && !string.IsNullOrEmpty(fields.UserName))
            return $"user:{fields.UserName}";

        // Group by destination IP
        if (rule.GroupBy == "destination_ip" && !string.IsNullOrEmpty(fields.DestinationIp))
            return $"dst_ip:{fields.DestinationIp}";

        // Default: group by agent
        return logEntry.AgentId ?? "unknown";
    }

    private double CalculateConfidence(int eventCount, int threshold)
    {
        // Higher event count relative to threshold = higher confidence
        var ratio = (double)eventCount / threshold;
        return Math.Min(0.5 + (ratio * 0.3), 1.0);
    }

    /// <summary>
    /// Initialize predefined correlation rules
    /// </summary>
    private List<CorrelationRule> InitializeRules()
    {
        return new List<CorrelationRule>
        {
            // Brute Force Detection: 5+ failed logins from same IP
            new CorrelationRule
            {
                Name = "Brute Force Attack",
                Description = "Multiple failed authentication attempts from same source IP",
                EventType = "authentication",
                EventOutcome = "failure",
                GroupBy = "source_ip",
                Threshold = 5,
                TimeWindowMinutes = 5,
                AlertSeverity = 7, // High
                MinSeverity = 2
            },

            // Credential Stuffing: Multiple users from same IP
            new CorrelationRule
            {
                Name = "Credential Stuffing",
                Description = "Multiple failed logins for different users from same IP",
                EventType = "authentication",
                EventOutcome = "failure",
                GroupBy = "source_ip",
                Threshold = 10,
                TimeWindowMinutes = 10,
                AlertSeverity = 8, // High
                MinSeverity = 2
            },

            // Privilege Escalation: Successful login after failures
            new CorrelationRule
            {
                Name = "Privilege Escalation Attempt",
                Description = "Successful authentication after multiple failures",
                EventType = "authentication",
                EventOutcome = "success",
                GroupBy = "user_name",
                Threshold = 1,
                TimeWindowMinutes = 15,
                AlertSeverity = 6, // Medium-High
                MinSeverity = 2,
                RequiresPreviousFailures = true
            },

            // Port Scanning: Multiple connection attempts to different ports
            new CorrelationRule
            {
                Name = "Port Scanning",
                Description = "Multiple network connection attempts to different ports",
                EventType = "network",
                GroupBy = "source_ip",
                Threshold = 20,
                TimeWindowMinutes = 5,
                AlertSeverity = 5, // Medium
                MinSeverity = 2
            },

            // Suspicious Process Execution
            new CorrelationRule
            {
                Name = "Suspicious Process Execution",
                Description = "Multiple process creation events in short time",
                EventType = "process",
                EventAction = "process_creation",
                GroupBy = "source_ip",
                Threshold = 10,
                TimeWindowMinutes = 2,
                AlertSeverity = 6, // Medium-High
                MinSeverity = 4
            }
        };
    }
}

/// <summary>
/// Correlation rule definition
/// </summary>
public class CorrelationRule
{
    public string Name { get; set; } = string.Empty;
    public string Description { get; set; } = string.Empty;
    public string? EventType { get; set; }
    public string? EventAction { get; set; }
    public string? EventOutcome { get; set; }
    public string GroupBy { get; set; } = "source_ip"; // source_ip, user_name, destination_ip
    public int Threshold { get; set; } = 5; // Number of events to trigger
    public int? TimeWindowMinutes { get; set; } = 5; // Time window in minutes
    public int AlertSeverity { get; set; } = 5; // Alert severity (1-10)
    public int? MinSeverity { get; set; } // Minimum log severity to consider
    public bool RequiresPreviousFailures { get; set; } = false;
}
