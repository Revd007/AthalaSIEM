using System.Text.Json;
using System.Text.RegularExpressions;
using Microsoft.Extensions.Logging;
using Backend.Domain.Entities;
using Backend.Domain.ValueObjects;

namespace Backend.Infrastructure.Normalizers;

/// <summary>
/// Enhanced ECS Log Normalizer - Ensures ALL logs have required fields:
/// - timestamp (always present)
/// - source_ip (extracted or inferred)
/// - event_type (categorized)
/// - severity (mapped from log level)
/// </summary>
public class EnhancedECSLogNormalizer : ILogNormalizer
{
    private readonly ILogger<EnhancedECSLogNormalizer> _logger;

    public EnhancedECSLogNormalizer(ILogger<EnhancedECSLogNormalizer> logger)
    {
        _logger = logger;
    }

    public Task<ECSLogFields?> NormalizeAsync(LogEntry logEntry, CancellationToken cancellationToken = default)
    {
        try
        {
            var ecsFields = new ECSLogFields
            {
                // REQUIRED: timestamp (always present)
                Timestamp = logEntry.Timestamp,
                
                AgentId = logEntry.AgentId,
                HostName = ExtractHostName(logEntry),
            };

            // REQUIRED: Extract source_ip (from properties, message, or agent)
            ecsFields.SourceIp = ExtractSourceIp(logEntry) ?? ExtractIpFromMessage(logEntry.RawMessage) ?? logEntry.AgentId;

            // REQUIRED: Determine event_type (categorize the event)
            ecsFields.EventType = DetermineEventType(logEntry);

            // REQUIRED: Map severity from log level (use Category or Level if available)
            var severitySource = logEntry.Category ?? logEntry.Source ?? "Information";
            ecsFields.SiemSeverity = MapSeverity(severitySource);

            // Parse properties if available
            if (!string.IsNullOrEmpty(logEntry.RawProperties))
            {
                try
                {
                    var properties = JsonSerializer.Deserialize<Dictionary<string, object>>(logEntry.RawProperties);
                    if (properties != null)
                    {
                        MapPropertiesToECS(properties, ecsFields);
                    }
                }
                catch (JsonException ex)
                {
                    _logger.LogWarning(ex, "Failed to parse properties for log {LogId}", logEntry.Id);
                }
            }

            // Parse message for additional patterns
            ParseMessage(logEntry.RawMessage, ecsFields);

            // Source-specific parsing
            ParseBySource(logEntry, ecsFields);

            // Ensure event_type is set (fallback)
            if (string.IsNullOrEmpty(ecsFields.EventType))
            {
                ecsFields.EventType = "unknown";
            }

            return Task.FromResult<ECSLogFields?>(ecsFields);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error normalizing log {LogId}", logEntry.Id);
            return Task.FromResult<ECSLogFields?>(null);
        }
    }

    public async Task<IEnumerable<ECSLogFields>> NormalizeBatchAsync(IEnumerable<LogEntry> logEntries, CancellationToken cancellationToken = default)
    {
        var results = new List<ECSLogFields>();
        
        foreach (var logEntry in logEntries)
        {
            var normalized = await NormalizeAsync(logEntry, cancellationToken);
            if (normalized != null)
            {
                results.Add(normalized);
            }
        }

        return results;
    }

    /// <summary>
    /// REQUIRED: Extract source_ip from log entry
    /// </summary>
    private string? ExtractSourceIp(LogEntry logEntry)
    {
        if (!string.IsNullOrEmpty(logEntry.RawProperties))
        {
            try
            {
                var properties = JsonSerializer.Deserialize<Dictionary<string, object>>(logEntry.RawProperties);
                if (properties != null)
                {
                    // Try common IP field names
                    var ipFields = new[] { "SourceIp", "src_ip", "source_ip", "ip", "client_ip", "remote_ip", "SourceAddress" };
                    foreach (var field in ipFields)
                    {
                        if (properties.TryGetValue(field, out var ip) && ip != null)
                        {
                            var ipStr = ip.ToString();
                            if (IsValidIpAddress(ipStr))
                                return ipStr;
                        }
                    }
                }
            }
            catch { }
        }

        return null;
    }

    /// <summary>
    /// Extract IP from message using regex
    /// </summary>
    private string? ExtractIpFromMessage(string? message)
    {
        if (string.IsNullOrEmpty(message))
            return null;

        var ipPattern = @"\b(?:\d{1,3}\.){3}\d{1,3}\b";
        var match = Regex.Match(message, ipPattern);
        return match.Success ? match.Value : null;
    }

    /// <summary>
    /// REQUIRED: Determine event_type (categorize event)
    /// </summary>
    private string DetermineEventType(LogEntry logEntry)
    {
        // Check event code/ID first
        if (logEntry.EventId.HasValue)
        {
            return CategorizeByEventId(logEntry.EventId.Value);
        }

        // Check source
        if (!string.IsNullOrEmpty(logEntry.Source))
        {
            var sourceLower = logEntry.Source.ToLowerInvariant();
            if (sourceLower.Contains("auth") || sourceLower.Contains("login") || sourceLower.Contains("logon"))
                return "authentication";
            if (sourceLower.Contains("process") || sourceLower.Contains("exec"))
                return "process";
            if (sourceLower.Contains("network") || sourceLower.Contains("connection"))
                return "network";
            if (sourceLower.Contains("file") || sourceLower.Contains("access"))
                return "file";
            if (sourceLower.Contains("security") || sourceLower.Contains("audit"))
                return "security";
        }

        // Check message content
        if (!string.IsNullOrEmpty(logEntry.RawMessage))
        {
            var msgLower = logEntry.RawMessage.ToLowerInvariant();
            if (msgLower.Contains("login") || msgLower.Contains("logon") || msgLower.Contains("auth"))
                return "authentication";
            if (msgLower.Contains("process") || msgLower.Contains("exec"))
                return "process";
            if (msgLower.Contains("connection") || msgLower.Contains("network"))
                return "network";
            if (msgLower.Contains("file") || msgLower.Contains("access"))
                return "file";
        }

        return "general";
    }

    /// <summary>
    /// Categorize by Windows Event ID
    /// </summary>
    private string CategorizeByEventId(long eventId)
    {
        return eventId switch
        {
            // Authentication events
            4624 or 4625 or 4648 or 4776 => "authentication",
            // Process events
            4688 or 4689 => "process",
            // Network events
            5156 or 5157 => "network",
            // File events
            4656 or 4658 => "file",
            // Security events
            4672 or 4702 or 4703 => "security",
            _ => "general"
        };
    }

    /// <summary>
    /// REQUIRED: Map severity from log level
    /// </summary>
    private int MapSeverity(string? severity)
    {
        if (string.IsNullOrEmpty(severity))
            return 1; // Info

        return severity.ToLowerInvariant() switch
        {
            "critical" or "fatal" => 10,
            "error" or "high" => 7,
            "warning" or "medium" => 4,
            "information" or "info" or "low" => 2,
            "debug" or "verbose" => 1,
            _ => 2 // Default to info
        };
    }

    private void MapPropertiesToECS(Dictionary<string, object> properties, ECSLogFields ecsFields)
    {
        // Map common property names to ECS fields
        if (properties.TryGetValue("UserName", out var userName) || properties.TryGetValue("user", out userName))
            ecsFields.UserName = userName?.ToString();

        if (properties.TryGetValue("ProcessName", out var processName) || properties.TryGetValue("process", out processName))
            ecsFields.ProcessName = processName?.ToString();

        if (properties.TryGetValue("ProcessId", out var processId) || properties.TryGetValue("pid", out processId))
        {
            if (int.TryParse(processId?.ToString(), out var pid))
                ecsFields.ProcessId = pid;
        }

        if (properties.TryGetValue("SourceIp", out var sourceIp) || properties.TryGetValue("src_ip", out sourceIp))
        {
            var ipStr = sourceIp?.ToString();
            if (IsValidIpAddress(ipStr))
                ecsFields.SourceIp = ipStr;
        }

        if (properties.TryGetValue("DestinationIp", out var destIp) || properties.TryGetValue("dst_ip", out destIp))
            ecsFields.DestinationIp = destIp?.ToString();

        if (properties.TryGetValue("FilePath", out var filePath) || properties.TryGetValue("file_path", out filePath))
            ecsFields.FilePath = filePath?.ToString();

        if (properties.TryGetValue("EventAction", out var action) || properties.TryGetValue("action", out action))
            ecsFields.EventAction = action?.ToString();
    }

    private void ParseMessage(string? message, ECSLogFields ecsFields)
    {
        if (string.IsNullOrEmpty(message))
            return;

        // IP address patterns (if not already set)
        if (string.IsNullOrEmpty(ecsFields.SourceIp))
        {
            var ipPattern = @"\b(?:\d{1,3}\.){3}\d{1,3}\b";
            var ipMatches = Regex.Matches(message, ipPattern);
            if (ipMatches.Count > 0)
            {
                ecsFields.SourceIp = ipMatches[0].Value;
                if (ipMatches.Count > 1)
                    ecsFields.DestinationIp = ipMatches[1].Value;
            }
        }

        // Process name patterns
        var processPattern = @"\b([A-Za-z0-9_-]+\.(exe|dll|bat|ps1|sh))\b";
        var processMatch = Regex.Match(message, processPattern, RegexOptions.IgnoreCase);
        if (processMatch.Success && string.IsNullOrEmpty(ecsFields.ProcessName))
        {
            ecsFields.ProcessName = processMatch.Groups[1].Value;
        }
    }

    private void ParseBySource(LogEntry logEntry, ECSLogFields ecsFields)
    {
        // Windows Event Log parsing
        if (logEntry.Source.Contains("Windows", StringComparison.OrdinalIgnoreCase) || 
            logEntry.Source.Contains("EventLog", StringComparison.OrdinalIgnoreCase))
        {
            ParseWindowsEventLog(logEntry, ecsFields);
        }

        // Sysmon parsing
        if (logEntry.Source.Contains("Sysmon", StringComparison.OrdinalIgnoreCase))
        {
            ParseSysmon(logEntry, ecsFields);
        }
    }

    private void ParseWindowsEventLog(LogEntry logEntry, ECSLogFields ecsFields)
    {
        if (logEntry.EventId.HasValue)
        {
            ecsFields.EventCode = logEntry.EventId.Value.ToString();
            
            // Map common Windows Event IDs
            switch (logEntry.EventId.Value)
            {
                case 4624: // Successful logon
                    ecsFields.EventAction = "logon";
                    ecsFields.EventOutcome = "success";
                    ecsFields.EventType = "authentication";
                    break;
                case 4625: // Failed logon
                    ecsFields.EventAction = "logon";
                    ecsFields.EventOutcome = "failure";
                    ecsFields.EventType = "authentication";
                    break;
                case 4672: // Special privileges assigned
                    ecsFields.EventAction = "privilege_assignment";
                    ecsFields.EventType = "security";
                    break;
                case 4688: // Process creation
                    ecsFields.EventAction = "process_creation";
                    ecsFields.EventType = "process";
                    break;
            }
        }
    }

    private void ParseSysmon(LogEntry logEntry, ECSLogFields ecsFields)
    {
        if (logEntry.EventId.HasValue)
        {
            ecsFields.EventCode = logEntry.EventId.Value.ToString();
            
            // Sysmon Event IDs
            switch (logEntry.EventId.Value)
            {
                case 1: // Process creation
                    ecsFields.EventAction = "process_creation";
                    ecsFields.EventType = "process";
                    break;
                case 3: // Network connection
                    ecsFields.EventAction = "network_connection";
                    ecsFields.EventType = "network";
                    break;
                case 7: // Image loaded
                    ecsFields.EventAction = "image_loaded";
                    ecsFields.EventType = "process";
                    break;
                case 11: // File creation
                    ecsFields.EventAction = "file_creation";
                    ecsFields.EventType = "file";
                    break;
            }
        }
    }

    private string? ExtractHostName(LogEntry logEntry)
    {
        if (!string.IsNullOrEmpty(logEntry.RawProperties))
        {
            try
            {
                var properties = JsonSerializer.Deserialize<Dictionary<string, object>>(logEntry.RawProperties);
                if (properties != null)
                {
                    if (properties.TryGetValue("MachineName", out var machineName) ||
                        properties.TryGetValue("hostname", out machineName) ||
                        properties.TryGetValue("HostName", out machineName))
                    {
                        return machineName?.ToString();
                    }
                }
            }
            catch { }
        }

        return null;
    }

    private bool IsValidIpAddress(string? ip)
    {
        if (string.IsNullOrEmpty(ip))
            return false;

        var parts = ip.Split('.');
        if (parts.Length != 4)
            return false;

        foreach (var part in parts)
        {
            if (!int.TryParse(part, out var num) || num < 0 || num > 255)
                return false;
        }

        return true;
    }
}
