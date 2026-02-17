using Microsoft.Extensions.Logging;
using Backend.Domain.Entities;

namespace Backend.Services;

/// <summary>
/// Maps Windows Event Logs to MITRE ATT&CK Techniques
/// Provides threat intelligence enrichment for SIEM analytics
/// </summary>
public class MitreAttackMapper
{
    private readonly ILogger<MitreAttackMapper> _logger;
    
    // Mapping of Windows Event IDs to MITRE ATT&CK Techniques
    private static readonly Dictionary<long, MitreTechnique> EventIdToTechnique = new()
    {
        // Authentication & Access
        { 4625, new MitreTechnique { TechniqueId = "T1110", TechniqueName = "Brute Force", Tactic = "Credential Access", Severity = "High" } },
        { 4648, new MitreTechnique { TechniqueId = "T1078", TechniqueName = "Valid Accounts", Tactic = "Defense Evasion, Persistence, Privilege Escalation, Initial Access", Severity = "Medium" } },
        { 4624, new MitreTechnique { TechniqueId = "T1078", TechniqueName = "Valid Accounts", Tactic = "Defense Evasion, Persistence, Privilege Escalation, Initial Access", Severity = "Low" } },
        
        // Privilege Escalation
        { 4672, new MitreTechnique { TechniqueId = "T1078", TechniqueName = "Valid Accounts", Tactic = "Privilege Escalation", Severity = "High" } },
        { 4673, new MitreTechnique { TechniqueId = "T1078", TechniqueName = "Valid Accounts", Tactic = "Privilege Escalation", Severity = "High" } },
        
        // Process Execution
        { 4688, new MitreTechnique { TechniqueId = "T1055", TechniqueName = "Process Injection", Tactic = "Defense Evasion, Privilege Escalation", Severity = "Medium" } },
        { 1, new MitreTechnique { TechniqueId = "T1055", TechniqueName = "Process Injection", Tactic = "Defense Evasion, Privilege Escalation", Severity = "Medium" } }, // Sysmon
        
        // File System
        { 4656, new MitreTechnique { TechniqueId = "T1083", TechniqueName = "File and Directory Discovery", Tactic = "Discovery", Severity = "Low" } },
        { 11, new MitreTechnique { TechniqueId = "T1083", TechniqueName = "File and Directory Discovery", Tactic = "Discovery", Severity = "Low" } }, // Sysmon
        
        // Network
        { 5156, new MitreTechnique { TechniqueId = "T1043", TechniqueName = "Commonly Used Port", Tactic = "Command and Control", Severity = "Medium" } },
        { 5157, new MitreTechnique { TechniqueId = "T1043", TechniqueName = "Commonly Used Port", Tactic = "Command and Control", Severity = "High" } },
        { 3, new MitreTechnique { TechniqueId = "T1043", TechniqueName = "Commonly Used Port", Tactic = "Command and Control", Severity = "Medium" } }, // Sysmon
        
        // Persistence
        { 4698, new MitreTechnique { TechniqueId = "T1543", TechniqueName = "Create or Modify System Process", Tactic = "Persistence", Severity = "High" } },
        
        // Defense Evasion
        { 4657, new MitreTechnique { TechniqueId = "T1070", TechniqueName = "Indicator Removal on Host", Tactic = "Defense Evasion", Severity = "High" } },
        
        // Discovery
        { 4697, new MitreTechnique { TechniqueId = "T1082", TechniqueName = "System Information Discovery", Tactic = "Discovery", Severity = "Low" } },
    };

    // Mapping based on event patterns/actions
    private static readonly Dictionary<string, MitreTechnique> PatternToTechnique = new()
    {
        { "failed_logon", new MitreTechnique { TechniqueId = "T1110", TechniqueName = "Brute Force", Tactic = "Credential Access", Severity = "High" } },
        { "privilege_escalation", new MitreTechnique { TechniqueId = "T1078", TechniqueName = "Valid Accounts", Tactic = "Privilege Escalation", Severity = "High" } },
        { "process_injection", new MitreTechnique { TechniqueId = "T1055", TechniqueName = "Process Injection", Tactic = "Defense Evasion", Severity = "High" } },
        { "file_deletion", new MitreTechnique { TechniqueId = "T1070", TechniqueName = "Indicator Removal on Host", Tactic = "Defense Evasion", Severity = "Medium" } },
        { "suspicious_connection", new MitreTechnique { TechniqueId = "T1043", TechniqueName = "Commonly Used Port", Tactic = "Command and Control", Severity = "High" } },
    };

    public MitreAttackMapper(ILogger<MitreAttackMapper> logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// Maps a log entry to MITRE ATT&CK techniques
    /// </summary>
    public List<MitreTechnique> MapToTechniques(LogEntry logEntry)
    {
        var techniques = new List<MitreTechnique>();

        try
        {
            // Map by Event ID
            if (logEntry.EventId.HasValue && EventIdToTechnique.TryGetValue(logEntry.EventId.Value, out var technique))
            {
                techniques.Add(technique);
            }

            // Map by event action/pattern
            if (!string.IsNullOrEmpty(logEntry.RawProperties))
            {
                var properties = System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, object>>(logEntry.RawProperties);
                if (properties != null)
                {
                    // Check for event action patterns
                    if (properties.TryGetValue("EventAction", out var action) ||
                        properties.TryGetValue("action", out action))
                    {
                        var actionStr = action?.ToString()?.ToLowerInvariant();
                        if (!string.IsNullOrEmpty(actionStr))
                        {
                            foreach (var pattern in PatternToTechnique.Keys)
                            {
                                if (actionStr.Contains(pattern, StringComparison.OrdinalIgnoreCase))
                                {
                                    techniques.Add(PatternToTechnique[pattern]);
                                    break;
                                }
                            }
                        }
                    }
                }
            }

            // Additional heuristics based on message content
            if (techniques.Count == 0 && !string.IsNullOrEmpty(logEntry.RawMessage))
            {
                var messageLower = logEntry.RawMessage.ToLowerInvariant();
                
                // Failed logon attempts
                if (messageLower.Contains("failed logon") || messageLower.Contains("logon failure"))
                {
                    techniques.Add(PatternToTechnique["failed_logon"]);
                }
                
                // Suspicious network connections
                if (messageLower.Contains("blocked") && messageLower.Contains("connection"))
                {
                    techniques.Add(PatternToTechnique["suspicious_connection"]);
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Error mapping log {LogId} to MITRE techniques", logEntry.Id);
        }

        return techniques.DistinctBy(t => t.TechniqueId).ToList();
    }

    /// <summary>
    /// Gets all MITRE techniques for analytics/dashboard
    /// </summary>
    public Dictionary<string, int> GetTechniqueCounts(IEnumerable<LogEntry> logs)
    {
        var counts = new Dictionary<string, int>();
        
        foreach (var log in logs)
        {
            var techniques = MapToTechniques(log);
            foreach (var technique in techniques)
            {
                if (!counts.ContainsKey(technique.TechniqueId))
                    counts[technique.TechniqueId] = 0;
                counts[technique.TechniqueId]++;
            }
        }
        
        return counts;
    }
}

/// <summary>
/// Represents a MITRE ATT&CK Technique
/// </summary>
public class MitreTechnique
{
    public string TechniqueId { get; set; } = string.Empty;
    public string TechniqueName { get; set; } = string.Empty;
    public string Tactic { get; set; } = string.Empty;
    public string Severity { get; set; } = "Low";
}
