using System.Text.Json;
using System.Text.RegularExpressions;
using Microsoft.Extensions.Logging;
using Backend.Domain.Entities;
using Backend.Domain.ValueObjects;

namespace Backend.Infrastructure.Normalizers;

public class ECSLogNormalizer : ILogNormalizer
{
        private readonly ILogger<ECSLogNormalizer> _logger;
        private readonly Backend.Infrastructure.Data.Repositories.INormalizedLogRepository _normalizedLogRepository;

    public ECSLogNormalizer(
        ILogger<ECSLogNormalizer> logger,
        Backend.Infrastructure.Data.Repositories.INormalizedLogRepository normalizedLogRepository)
    {
        _logger = logger;
        _normalizedLogRepository = normalizedLogRepository;
    }

    public Task<ECSLogFields?> NormalizeAsync(LogEntry logEntry, CancellationToken cancellationToken = default)
    {
        try
        {
            var ecsFields = new ECSLogFields
            {
                Timestamp = logEntry.Timestamp,
                AgentId = logEntry.AgentId,
                HostName = ExtractHostName(logEntry),
            };

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

            // Parse message for common patterns
            ParseMessage(logEntry.RawMessage, ecsFields);

            // Source-specific parsing
            ParseBySource(logEntry, ecsFields);

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
            ecsFields.SourceIp = sourceIp?.ToString();

        if (properties.TryGetValue("DestinationIp", out var destIp) || properties.TryGetValue("dst_ip", out destIp))
            ecsFields.DestinationIp = destIp?.ToString();

        if (properties.TryGetValue("FilePath", out var filePath) || properties.TryGetValue("file_path", out filePath))
            ecsFields.FilePath = filePath?.ToString();

        if (properties.TryGetValue("EventAction", out var action) || properties.TryGetValue("action", out action))
            ecsFields.EventAction = action?.ToString();

        if (properties.TryGetValue("EventCategory", out var category) || properties.TryGetValue("category", out category))
            ecsFields.EventCategory = category?.ToString();
    }

    private void ParseMessage(string message, ECSLogFields ecsFields)
    {
        if (string.IsNullOrEmpty(message))
            return;

        // IP address patterns
        var ipPattern = @"\b(?:\d{1,3}\.){3}\d{1,3}\b";
        var ipMatches = Regex.Matches(message, ipPattern);
        if (ipMatches.Count > 0)
        {
            ecsFields.SourceIp = ipMatches[0].Value;
            if (ipMatches.Count > 1)
                ecsFields.DestinationIp = ipMatches[1].Value;
        }

        // Process name patterns (common executables)
        var processPattern = @"\b([A-Za-z0-9_-]+\.(exe|dll|bat|ps1|sh))\b";
        var processMatch = Regex.Match(message, processPattern, RegexOptions.IgnoreCase);
        if (processMatch.Success)
        {
            ecsFields.ProcessName = processMatch.Groups[1].Value;
        }

        // File path patterns
        var pathPattern = @"[A-Za-z]:\\(?:[^\\/:*?""<>|\r\n]+\\)*[^\\/:*?""<>|\r\n]*";
        var pathMatch = Regex.Match(message, pathPattern);
        if (pathMatch.Success)
        {
            ecsFields.FilePath = pathMatch.Value;
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
                    break;
                case 4625: // Failed logon
                    ecsFields.EventAction = "logon";
                    ecsFields.EventOutcome = "failure";
                    break;
                case 4672: // Special privileges assigned
                    ecsFields.EventAction = "privilege_assignment";
                    break;
                case 4688: // Process creation
                    ecsFields.EventAction = "process_creation";
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
                    break;
                case 3: // Network connection
                    ecsFields.EventAction = "network_connection";
                    break;
                case 7: // Image loaded
                    ecsFields.EventAction = "image_loaded";
                    break;
                case 11: // File creation
                    ecsFields.EventAction = "file_creation";
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
}
