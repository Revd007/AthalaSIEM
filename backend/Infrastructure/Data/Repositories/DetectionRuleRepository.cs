using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging;
using Backend.Data;
using Backend.Domain.Entities;
using Backend.Domain.Interfaces;
using System.Text.Json;
using Backend.Domain.ValueObjects;

namespace Backend.Infrastructure.Data.Repositories;

public class DetectionRuleRepository : IDetectionRuleRepository
{
    private readonly ApplicationDbContext _context;
    private readonly ILogger<DetectionRuleRepository> _logger;

    public DetectionRuleRepository(ApplicationDbContext context, ILogger<DetectionRuleRepository> logger)
    {
        _context = context;
        _logger = logger;
    }

    public async Task<DetectionRule?> GetByIdAsync(string id, CancellationToken cancellationToken = default)
    {
        // Query from AlertRulesNew table and map to DetectionRule
        var model = await _context.AlertRulesNew.FindAsync(new object[] { id }, cancellationToken);
        return model != null ? MapToDomain(model) : null;
    }

    public async Task<IEnumerable<DetectionRule>> GetActiveRulesAsync(CancellationToken cancellationToken = default)
    {
        var models = await _context.AlertRulesNew
            .Where(r => r.Enabled)
            .ToListAsync(cancellationToken);
        
        return models.Select(MapToDomain);
    }

    public async Task<IEnumerable<DetectionRule>> GetAllAsync(CancellationToken cancellationToken = default)
    {
        var models = await _context.AlertRulesNew.ToListAsync(cancellationToken);
        return models.Select(MapToDomain);
    }

    public async Task AddAsync(DetectionRule rule, CancellationToken cancellationToken = default)
    {
        var model = MapToModel(rule);
        await _context.AlertRulesNew.AddAsync(model, cancellationToken);
        await _context.SaveChangesAsync(cancellationToken);
    }

    public async Task UpdateAsync(DetectionRule rule, CancellationToken cancellationToken = default)
    {
        var model = await _context.AlertRulesNew.FindAsync(new object[] { rule.Id }, cancellationToken);
        if (model != null)
        {
            UpdateModel(model, rule);
            await _context.SaveChangesAsync(cancellationToken);
        }
    }

    public async Task DeleteAsync(string id, CancellationToken cancellationToken = default)
    {
        var model = await _context.AlertRulesNew.FindAsync(new object[] { id }, cancellationToken);
        if (model != null)
        {
            _context.AlertRulesNew.Remove(model);
            await _context.SaveChangesAsync(cancellationToken);
        }
    }

    private DetectionRule MapToDomain(Models.AlertRuleModels model)
    {
        var rule = new DetectionRule
        {
            Id = model.Id,
            Name = model.Name,
            Description = model.Description ?? string.Empty,
            RuleDefinition = model.Condition,
            Enabled = model.Enabled ?? true,
            CreatedAt = model.CreatedAt,
            UpdatedAt = model.UpdatedAt ?? model.CreatedAt,
            CreatedBy = model.CreatedBy
        };

        // Parse rule type from condition
        rule.Type = DetermineRuleType(model.Condition);

        // Parse technique IDs from condition or metadata
        rule.TechniqueIds = ExtractTechniqueIds(model.Condition);

        // Parse default severity - try to get from model, fallback to Medium
        var severityStr = "Medium";
        if (model.Severity != null)
        {
            severityStr = model.Severity.ToString() ?? "Medium";
        }
        rule.DefaultSeverity = ParseSeverity(severityStr);

        return rule;
    }

    private Models.AlertRuleModels MapToModel(DetectionRule rule)
    {
        return new Models.AlertRuleModels
        {
            Id = rule.Id,
            Name = rule.Name,
            Description = rule.Description,
            Condition = rule.RuleDefinition,
            Enabled = rule.Enabled,
            CreatedAt = rule.CreatedAt,
            UpdatedAt = rule.UpdatedAt,
            CreatedBy = rule.CreatedBy
        };
    }

    private void UpdateModel(Models.AlertRuleModels model, DetectionRule rule)
    {
        model.Name = rule.Name;
        model.Description = rule.Description;
        model.Condition = rule.RuleDefinition;
        model.Enabled = rule.Enabled;
        model.UpdatedAt = rule.UpdatedAt;
    }

    private RuleType DetermineRuleType(string condition)
    {
        if (string.IsNullOrEmpty(condition))
            return RuleType.PatternMatch;

        var lower = condition.ToLowerInvariant();
        if (lower.Contains("threshold") || lower.Contains("count"))
            return RuleType.Threshold;
        if (lower.Contains("correlation"))
            return RuleType.Correlation;
        if (lower.Contains("statistical") || lower.Contains("anomaly"))
            return RuleType.Statistical;
        
        return RuleType.PatternMatch;
    }

    private List<string> ExtractTechniqueIds(string condition)
    {
        var techniques = new List<string>();
        if (string.IsNullOrEmpty(condition))
            return techniques;

        var pattern = @"T\d{4}(\.\d{3})?";
        var matches = System.Text.RegularExpressions.Regex.Matches(condition, pattern);
        foreach (System.Text.RegularExpressions.Match match in matches)
        {
            techniques.Add(match.Value);
        }
        return techniques.Distinct().ToList();
    }

    private AlertSeverityLevel ParseSeverity(string severity)
    {
        return severity.ToLowerInvariant() switch
        {
            "critical" => AlertSeverityLevel.Critical,
            "high" => AlertSeverityLevel.High,
            "medium" => AlertSeverityLevel.Medium,
            "low" => AlertSeverityLevel.Low,
            _ => AlertSeverityLevel.Info
        };
    }
}
