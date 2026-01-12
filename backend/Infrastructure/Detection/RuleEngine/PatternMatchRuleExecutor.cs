using System.Text.RegularExpressions;
using Microsoft.Extensions.Logging;
using Backend.Domain.Entities;
using Backend.Domain.ValueObjects;

namespace Backend.Infrastructure.Detection.RuleEngine;

public class PatternMatchRuleExecutor : IRuleExecutor
{
    private readonly IRuleParser _ruleParser;
    private readonly ILogger<PatternMatchRuleExecutor> _logger;

    public PatternMatchRuleExecutor(IRuleParser ruleParser, ILogger<PatternMatchRuleExecutor> logger)
    {
        _ruleParser = ruleParser;
        _logger = logger;
    }

    public Task<DetectionResult> ExecuteAsync(DetectionRule rule, LogEntry logEntry, CancellationToken cancellationToken = default)
    {
        try
        {
            var parsedRule = _ruleParser.Parse(rule.RuleDefinition);
            var result = new DetectionResult
            {
                Rule = rule,
                LogEntry = logEntry,
                Matched = false
            };

            if (logEntry.NormalizedFields == null)
            {
                return Task.FromResult(result);
            }

            // Check conditions against normalized fields
            var matched = EvaluateConditions(parsedRule.Conditions, logEntry.NormalizedFields, logEntry);
            
            result.Matched = matched;
            if (matched)
            {
                result.Reason = $"Rule '{rule.Name}' matched on log {logEntry.Id}";
                result.Confidence = CalculateConfidence(parsedRule, logEntry);
                result.MatchContext = BuildMatchContext(parsedRule, logEntry);
            }

            return Task.FromResult(result);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error executing rule {RuleId} on log {LogId}", rule.Id, logEntry.Id);
            return Task.FromResult(new DetectionResult
            {
                Rule = rule,
                LogEntry = logEntry,
                Matched = false
            });
        }
    }

    public async Task<IEnumerable<DetectionResult>> ExecuteBatchAsync(DetectionRule rule, IEnumerable<LogEntry> logEntries, CancellationToken cancellationToken = default)
    {
        var results = new List<DetectionResult>();
        
        foreach (var logEntry in logEntries)
        {
            var result = await ExecuteAsync(rule, logEntry, cancellationToken);
            results.Add(result);
        }

        return results;
    }

    private bool EvaluateConditions(Dictionary<string, object> conditions, ECSLogFields fields, LogEntry logEntry)
    {
        foreach (var condition in conditions)
        {
            if (!EvaluateCondition(condition.Key, condition.Value, fields, logEntry))
            {
                return false;
            }
        }

        return conditions.Count > 0;
    }

    private bool EvaluateCondition(string field, object expectedValue, ECSLogFields fields, LogEntry logEntry)
    {
        object? actualValue = field.ToLowerInvariant() switch
        {
            "eventid" or "event_code" => fields.EventCode,
            "processname" or "process.name" => fields.ProcessName,
            "username" or "user.name" => fields.UserName,
            "sourceip" or "source.ip" => fields.SourceIp,
            "destinationip" or "destination.ip" => fields.DestinationIp,
            "eventaction" or "event.action" => fields.EventAction,
            "eventcategory" or "event.category" => fields.EventCategory,
            "filepath" or "file.path" => fields.FilePath,
            _ => null
        };

        if (actualValue == null)
            return false;

        var expectedStr = expectedValue.ToString() ?? string.Empty;
        var actualStr = actualValue.ToString() ?? string.Empty;

        // Support wildcard matching
        if (expectedStr.Contains('*'))
        {
            var pattern = "^" + Regex.Escape(expectedStr).Replace("\\*", ".*") + "$";
            return Regex.IsMatch(actualStr, pattern, RegexOptions.IgnoreCase);
        }

        // Support endsWith matching (common in Sigma rules)
        if (expectedStr.StartsWith("|endswith"))
        {
            var value = expectedStr.Replace("|endswith", "").Trim();
            return actualStr.EndsWith(value, StringComparison.OrdinalIgnoreCase);
        }

        // Support contains matching
        if (expectedStr.StartsWith("|contains"))
        {
            var value = expectedStr.Replace("|contains", "").Trim();
            return actualStr.Contains(value, StringComparison.OrdinalIgnoreCase);
        }

        // Exact match
        return string.Equals(actualStr, expectedStr, StringComparison.OrdinalIgnoreCase);
    }

    private double CalculateConfidence(ParsedRule parsedRule, LogEntry logEntry)
    {
        double confidence = 0.8; // Base confidence

        // Increase confidence if technique IDs match
        if (parsedRule.TechniqueIds.Any() && logEntry.TechniqueIds.Any())
        {
            if (parsedRule.TechniqueIds.Intersect(logEntry.TechniqueIds).Any())
            {
                confidence = 0.95;
            }
        }

        return confidence;
    }

    private Dictionary<string, object> BuildMatchContext(ParsedRule parsedRule, LogEntry logEntry)
    {
        return new Dictionary<string, object>
        {
            ["rule_id"] = parsedRule.Id,
            ["rule_name"] = parsedRule.Name,
            ["log_id"] = logEntry.Id,
            ["matched_fields"] = GetMatchedFields(parsedRule, logEntry),
            ["technique_ids"] = parsedRule.TechniqueIds
        };
    }

    private List<string> GetMatchedFields(ParsedRule parsedRule, LogEntry logEntry)
    {
        var matchedFields = new List<string>();
        
        if (logEntry.NormalizedFields == null)
            return matchedFields;

        foreach (var condition in parsedRule.Conditions.Keys)
        {
            matchedFields.Add(condition);
        }

        return matchedFields;
    }
}
