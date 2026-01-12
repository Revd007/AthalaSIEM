using System.Text.RegularExpressions;
using Microsoft.Extensions.Logging;
using Backend.Domain.Entities;
using Backend.Domain.ValueObjects;

namespace Backend.Infrastructure.Detection.RuleEngine;

public class YamlRuleParser : IRuleParser
{
    private readonly ILogger<YamlRuleParser> _logger;

    public YamlRuleParser(ILogger<YamlRuleParser> logger)
    {
        _logger = logger;
    }

    public ParsedRule Parse(string ruleDefinition)
    {
        try
        {
            var rule = new ParsedRule();
            var lines = ruleDefinition.Split('\n');
            
            foreach (var line in lines)
            {
                var trimmed = line.Trim();
                if (string.IsNullOrEmpty(trimmed) || trimmed.StartsWith('#'))
                    continue;

                if (trimmed.StartsWith("title:", StringComparison.OrdinalIgnoreCase))
                    rule.Name = ExtractValue(trimmed);
                else if (trimmed.StartsWith("id:", StringComparison.OrdinalIgnoreCase))
                    rule.Id = ExtractValue(trimmed);
                else if (trimmed.StartsWith("description:", StringComparison.OrdinalIgnoreCase))
                    rule.Description = ExtractValue(trimmed);
                else if (trimmed.StartsWith("level:", StringComparison.OrdinalIgnoreCase))
                    rule.Severity = ParseSeverity(ExtractValue(trimmed));
                else if (trimmed.Contains("technique.", StringComparison.OrdinalIgnoreCase))
                    rule.TechniqueIds.AddRange(ExtractTechniques(trimmed));
            }

            // Determine rule type from conditions
            rule.Type = DetermineRuleType(ruleDefinition);
            
            // Parse detection conditions (simplified - full YAML parser would be better)
            rule.Conditions = ParseConditions(ruleDefinition);

            return rule;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error parsing rule definition");
            throw;
        }
    }

    private string ExtractValue(string line)
    {
        var colonIndex = line.IndexOf(':');
        if (colonIndex >= 0 && colonIndex < line.Length - 1)
        {
            return line.Substring(colonIndex + 1).Trim().Trim('"', '\'');
        }
        return string.Empty;
    }

    private AlertSeverityLevel ParseSeverity(string level)
    {
        return level.ToLowerInvariant() switch
        {
            "critical" => AlertSeverityLevel.Critical,
            "high" => AlertSeverityLevel.High,
            "medium" => AlertSeverityLevel.Medium,
            "low" => AlertSeverityLevel.Low,
            _ => AlertSeverityLevel.Info
        };
    }

    private List<string> ExtractTechniques(string line)
    {
        var techniques = new List<string>();
        var pattern = @"T\d{4}(\.\d{3})?";
        var matches = Regex.Matches(line, pattern);
        foreach (Match match in matches)
        {
            techniques.Add(match.Value);
        }
        return techniques;
    }

    private RuleType DetermineRuleType(string definition)
    {
        if (definition.Contains("threshold", StringComparison.OrdinalIgnoreCase) ||
            definition.Contains("count", StringComparison.OrdinalIgnoreCase))
            return RuleType.Threshold;
        
        if (definition.Contains("correlation", StringComparison.OrdinalIgnoreCase))
            return RuleType.Correlation;
        
        if (definition.Contains("statistical", StringComparison.OrdinalIgnoreCase) ||
            definition.Contains("anomaly", StringComparison.OrdinalIgnoreCase))
            return RuleType.Statistical;
        
        return RuleType.PatternMatch;
    }

    private Dictionary<string, object> ParseConditions(string definition)
    {
        var conditions = new Dictionary<string, object>();
        
        // Extract selection criteria (simplified)
        var selectionMatch = Regex.Match(definition, @"selection:\s*\n((?:\s+.*\n?)+)", RegexOptions.Multiline);
        if (selectionMatch.Success)
        {
            var selectionLines = selectionMatch.Groups[1].Value.Split('\n');
            foreach (var line in selectionLines)
            {
                var trimmed = line.Trim();
                if (trimmed.Contains(':'))
                {
                    var parts = trimmed.Split(':', 2);
                    if (parts.Length == 2)
                    {
                        var key = parts[0].Trim();
                        var value = parts[1].Trim().Trim('"', '\'');
                        conditions[key] = value;
                    }
                }
            }
        }

        return conditions;
    }
}
