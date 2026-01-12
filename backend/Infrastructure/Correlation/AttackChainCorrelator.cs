using Microsoft.Extensions.Logging;
using Backend.Domain.Entities;
using Backend.Domain.Interfaces;

namespace Backend.Infrastructure.Correlation;

public class AttackChainCorrelator
{
    private readonly ILogRepository _logRepository;
    private readonly ILogger<AttackChainCorrelator> _logger;
    private readonly TimeSpan _chainTimeWindow = TimeSpan.FromHours(24);

    // MITRE ATT&CK tactic progression
    private readonly Dictionary<string, List<string>> _tacticProgression = new()
    {
        { "Initial Access", new List<string> { "Execution", "Persistence", "Privilege Escalation" } },
        { "Execution", new List<string> { "Persistence", "Privilege Escalation", "Defense Evasion" } },
        { "Persistence", new List<string> { "Privilege Escalation", "Defense Evasion", "Credential Access" } },
        { "Privilege Escalation", new List<string> { "Defense Evasion", "Credential Access", "Discovery" } },
        { "Defense Evasion", new List<string> { "Credential Access", "Discovery", "Lateral Movement" } },
        { "Credential Access", new List<string> { "Discovery", "Lateral Movement", "Collection" } },
        { "Discovery", new List<string> { "Lateral Movement", "Collection", "Command and Control" } },
        { "Lateral Movement", new List<string> { "Collection", "Command and Control", "Exfiltration" } },
        { "Collection", new List<string> { "Command and Control", "Exfiltration", "Impact" } },
        { "Command and Control", new List<string> { "Exfiltration", "Impact" } },
        { "Exfiltration", new List<string> { "Impact" } }
    };

    public AttackChainCorrelator(
        ILogRepository logRepository,
        ILogger<AttackChainCorrelator> logger)
    {
        _logRepository = logRepository;
        _logger = logger;
    }

    public async Task<List<CorrelationResult>> DetectAttackChainsAsync(
        string agentId,
        DateTime startTime,
        DateTime endTime,
        CancellationToken cancellationToken = default)
    {
        var results = new List<CorrelationResult>();

        // Get all logs with technique IDs in time window
        var logs = await _logRepository.GetByAgentIdAsync(agentId, startTime, endTime, cancellationToken);
        var logsWithTechniques = logs
            .Where(l => l.TechniqueIds.Any() && l.NormalizedFields != null)
            .OrderBy(l => l.Timestamp)
            .ToList();

        if (logsWithTechniques.Count < 2)
            return results;

        // Group by technique sequences
        var chains = new List<List<LogEntry>>();
        var currentChain = new List<LogEntry> { logsWithTechniques[0] };

        for (int i = 1; i < logsWithTechniques.Count; i++)
        {
            var currentLog = logsWithTechniques[i];
            var previousLog = currentChain.Last();

            // Check if techniques form a progression
            if (IsTechniqueProgression(previousLog.TechniqueIds, currentLog.TechniqueIds))
            {
                currentChain.Add(currentLog);
            }
            else
            {
                if (currentChain.Count >= 2)
                {
                    chains.Add(new List<LogEntry>(currentChain));
                }
                currentChain = new List<LogEntry> { currentLog };
            }
        }

        if (currentChain.Count >= 2)
        {
            chains.Add(currentChain);
        }

        // Create correlation results for each chain
        foreach (var chain in chains)
        {
            var correlationId = Guid.NewGuid().ToString();
            foreach (var log in chain)
            {
                log.CorrelationId = correlationId;
            }

            results.Add(new CorrelationResult
            {
                CorrelationId = correlationId,
                CorrelatedLogs = chain,
                Type = CorrelationType.AttackChain,
                Confidence = CalculateChainConfidence(chain),
                Metadata = new Dictionary<string, object>
                {
                    ["chain_length"] = chain.Count,
                    ["techniques"] = chain.SelectMany(l => l.TechniqueIds).Distinct().ToList(),
                    ["time_span_minutes"] = (chain.Last().Timestamp - chain.First().Timestamp).TotalMinutes
                }
            });
        }

        return results;
    }

    private bool IsTechniqueProgression(List<string> previousTechniques, List<string> currentTechniques)
    {
        // Simplified: check if techniques are related
        // In production, use MITRE ATT&CK data to check tactic progression
        if (!previousTechniques.Any() || !currentTechniques.Any())
            return false;

        // Check if any technique IDs are sequential or related
        return previousTechniques.Any(pt => 
            currentTechniques.Any(ct => 
                pt.StartsWith("T1") && ct.StartsWith("T1") && 
                Math.Abs(int.Parse(pt.Substring(1)) - int.Parse(ct.Substring(1))) < 100));
    }

    private double CalculateChainConfidence(List<LogEntry> chain)
    {
        // Longer chains = higher confidence
        var baseConfidence = 0.6;
        var lengthBonus = Math.Min(chain.Count * 0.1, 0.3);
        return Math.Min(baseConfidence + lengthBonus, 1.0);
    }
}
