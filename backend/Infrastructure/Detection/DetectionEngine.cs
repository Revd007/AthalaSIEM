using Microsoft.Extensions.Logging;
using Backend.Domain.Entities;
using Backend.Domain.Interfaces;
using Backend.Infrastructure.Detection.RuleEngine;

namespace Backend.Infrastructure.Detection;

public class DetectionEngine : IDetectionEngine
{
    private readonly Backend.Domain.Interfaces.IDetectionRuleRepository _ruleRepository;
    private readonly IRuleExecutor _ruleExecutor;
    private readonly ILogger<DetectionEngine> _logger;
    private readonly Dictionary<string, DetectionRule> _activeRules = new();

    public DetectionEngine(
        IDetectionRuleRepository ruleRepository,
        IRuleExecutor ruleExecutor,
        ILogger<DetectionEngine> logger)
    {
        _ruleRepository = ruleRepository;
        _ruleExecutor = ruleExecutor;
        _logger = logger;
    }

    public async Task<IEnumerable<DetectionResult>> DetectAsync(LogEntry logEntry, CancellationToken cancellationToken = default)
    {
        var results = new List<DetectionResult>();

        if (logEntry.NormalizedFields == null)
        {
            _logger.LogDebug("Log {LogId} not normalized, skipping detection", logEntry.Id);
            return results;
        }

        // Load active rules if not loaded
        if (_activeRules.Count == 0)
        {
            await LoadActiveRulesAsync(cancellationToken);
        }

        foreach (var rule in _activeRules.Values.Where(r => r.Enabled))
        {
            try
            {
                var result = await _ruleExecutor.ExecuteAsync(rule, logEntry, cancellationToken);
                if (result.Matched)
                {
                    results.Add(result);
                    _logger.LogInformation("Rule {RuleId} matched on log {LogId}", rule.Id, logEntry.Id);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error executing rule {RuleId} on log {LogId}", rule.Id, logEntry.Id);
            }
        }

        return results;
    }

    public async Task<IEnumerable<DetectionResult>> DetectBatchAsync(IEnumerable<LogEntry> logEntries, CancellationToken cancellationToken = default)
    {
        var allResults = new List<DetectionResult>();

        foreach (var logEntry in logEntries)
        {
            var results = await DetectAsync(logEntry, cancellationToken);
            allResults.AddRange(results);
        }

        return allResults;
    }

    public async Task RegisterRuleAsync(DetectionRule rule, CancellationToken cancellationToken = default)
    {
        _activeRules[rule.Id] = rule;
        _logger.LogInformation("Rule {RuleId} registered", rule.Id);
        await Task.CompletedTask;
    }

    public async Task UnregisterRuleAsync(string ruleId, CancellationToken cancellationToken = default)
    {
        if (_activeRules.Remove(ruleId))
        {
            _logger.LogInformation("Rule {RuleId} unregistered", ruleId);
        }
        await Task.CompletedTask;
    }

    private async Task LoadActiveRulesAsync(CancellationToken cancellationToken)
    {
        try
        {
            var rules = await _ruleRepository.GetActiveRulesAsync(cancellationToken);
            foreach (var rule in rules)
            {
                _activeRules[rule.Id] = rule;
            }
            _logger.LogInformation("Loaded {Count} active detection rules", _activeRules.Count);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error loading active rules");
        }
    }
}
