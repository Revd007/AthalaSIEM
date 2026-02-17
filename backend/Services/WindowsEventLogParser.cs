using System.Text.RegularExpressions;
using Microsoft.Extensions.Logging;
using Backend.Domain.Entities;

namespace Backend.Services;

/// <summary>
/// Enhanced Windows Event Log Parser with regex-based field extraction
/// Extracts structured fields from Windows Event Log messages, especially WFP (Windows Filtering Platform) logs
/// </summary>
public class WindowsEventLogParser
{
    private readonly ILogger<WindowsEventLogParser> _logger;
    
    // Regex patterns for common Windows Event Log fields
    private static readonly Regex SourceAddressPattern = new Regex(@"SourceAddress[:\s]+([0-9a-fA-F:\.]+)", RegexOptions.IgnoreCase | RegexOptions.Compiled);
    private static readonly Regex SourcePortPattern = new Regex(@"SourcePort[:\s]+(\d+)", RegexOptions.IgnoreCase | RegexOptions.Compiled);
    private static readonly Regex DestinationAddressPattern = new Regex(@"DestinationAddress[:\s]+([0-9a-fA-F:\.]+)", RegexOptions.IgnoreCase | RegexOptions.Compiled);
    private static readonly Regex DestinationPortPattern = new Regex(@"DestinationPort[:\s]+(\d+)", RegexOptions.IgnoreCase | RegexOptions.Compiled);
    private static readonly Regex ApplicationNamePattern = new Regex(@"ApplicationName[:\s]+([^\r\n]+)", RegexOptions.IgnoreCase | RegexOptions.Compiled);
    private static readonly Regex UserNamePattern = new Regex(@"Account Name[:\s]+([^\r\n]+)|UserName[:\s]+([^\r\n]+)|SubjectUserName[:\s]+([^\r\n]+)", RegexOptions.IgnoreCase | RegexOptions.Compiled);
    private static readonly Regex DomainPattern = new Regex(@"Account Domain[:\s]+([^\r\n]+)|Domain[:\s]+([^\r\n]+)|SubjectDomainName[:\s]+([^\r\n]+)", RegexOptions.IgnoreCase | RegexOptions.Compiled);
    private static readonly Regex ProcessNamePattern = new Regex(@"Process Name[:\s]+([^\r\n]+)|Image[:\s]+([^\r\n]+)|NewProcessName[:\s]+([^\r\n]+)", RegexOptions.IgnoreCase | RegexOptions.Compiled);
    private static readonly Regex ProcessIdPattern = new Regex(@"Process Id[:\s]+(0x[0-9a-fA-F]+|\d+)|ProcessID[:\s]+(0x[0-9a-fA-F]+|\d+)|NewProcessId[:\s]+(0x[0-9a-fA-F]+|\d+)", RegexOptions.IgnoreCase | RegexOptions.Compiled);
    private static readonly Regex CommandLinePattern = new Regex(@"Command Line[:\s]+([^\r\n]+)|ProcessCommandLine[:\s]+([^\r\n]+)", RegexOptions.IgnoreCase | RegexOptions.Compiled);
    private static readonly Regex FilePathPattern = new Regex(@"(?:File Name|TargetFileName|ObjectName)[:\s]+([^\r\n]+)", RegexOptions.IgnoreCase | RegexOptions.Compiled);
    private static readonly Regex IpAddressPattern = new Regex(@"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b", RegexOptions.Compiled);
    private static readonly Regex PortPattern = new Regex(@"\b(\d{1,5})\b", RegexOptions.Compiled);

    public WindowsEventLogParser(ILogger<WindowsEventLogParser> logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// Parses a Windows Event Log entry and extracts structured fields
    /// </summary>
    public ParsedLogFields Parse(LogEntry logEntry)
    {
        var fields = new ParsedLogFields
        {
            EventId = logEntry.EventId,
            Source = logEntry.Source,
            Message = logEntry.RawMessage
        };

        if (string.IsNullOrEmpty(logEntry.RawMessage))
            return fields;

        try
        {
            // Parse based on Event ID
            if (logEntry.EventId.HasValue)
            {
                ParseByEventId(logEntry, fields);
            }

            // Parse WFP (Windows Filtering Platform) logs
            if (logEntry.Source.Contains("Filtering Platform", StringComparison.OrdinalIgnoreCase) ||
                logEntry.RawMessage.Contains("SourceAddress", StringComparison.OrdinalIgnoreCase))
            {
                ParseWfpLog(logEntry.RawMessage, fields);
            }

            // Parse authentication events (4624, 4625, etc.)
            if (logEntry.EventId.HasValue && 
                (logEntry.EventId.Value == 4624 || logEntry.EventId.Value == 4625 || 
                 logEntry.EventId.Value == 4648 || logEntry.EventId.Value == 4672))
            {
                ParseAuthenticationEvent(logEntry.RawMessage, fields);
            }

            // Parse process creation events (4688, Sysmon 1)
            if (logEntry.EventId.HasValue && 
                (logEntry.EventId.Value == 4688 || logEntry.EventId.Value == 1))
            {
                ParseProcessCreationEvent(logEntry.RawMessage, fields);
            }

            // Parse file events (Sysmon 11, 4656, etc.)
            if (logEntry.EventId.HasValue && 
                (logEntry.EventId.Value == 11 || logEntry.EventId.Value == 4656))
            {
                ParseFileEvent(logEntry.RawMessage, fields);
            }

            // Generic IP/Port extraction as fallback
            ExtractNetworkFields(logEntry.RawMessage, fields);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Error parsing log entry {LogId}", logEntry.Id);
        }

        return fields;
    }

    private void ParseWfpLog(string message, ParsedLogFields fields)
    {
        // Extract SourceAddress
        var sourceAddrMatch = SourceAddressPattern.Match(message);
        if (sourceAddrMatch.Success)
        {
            fields.SourceAddress = sourceAddrMatch.Groups[1].Value.Trim();
        }

        // Extract SourcePort
        var sourcePortMatch = SourcePortPattern.Match(message);
        if (sourcePortMatch.Success && int.TryParse(sourcePortMatch.Groups[1].Value, out var sourcePort))
        {
            fields.SourcePort = sourcePort;
        }

        // Extract DestinationAddress
        var destAddrMatch = DestinationAddressPattern.Match(message);
        if (destAddrMatch.Success)
        {
            fields.DestinationAddress = destAddrMatch.Groups[1].Value.Trim();
        }

        // Extract DestinationPort
        var destPortMatch = DestinationPortPattern.Match(message);
        if (destPortMatch.Success && int.TryParse(destPortMatch.Groups[1].Value, out var destPort))
        {
            fields.DestinationPort = destPort;
        }

        // Extract ApplicationName
        var appMatch = ApplicationNamePattern.Match(message);
        if (appMatch.Success)
        {
            fields.ApplicationName = appMatch.Groups[1].Value.Trim();
        }
    }

    private void ParseAuthenticationEvent(string message, ParsedLogFields fields)
    {
        // Extract UserName
        var userMatch = UserNamePattern.Match(message);
        if (userMatch.Success)
        {
            fields.UserName = userMatch.Groups[1].Value.Trim();
            if (string.IsNullOrEmpty(fields.UserName))
                fields.UserName = userMatch.Groups[2].Value.Trim();
            if (string.IsNullOrEmpty(fields.UserName))
                fields.UserName = userMatch.Groups[3].Value.Trim();
        }

        // Extract Domain
        var domainMatch = DomainPattern.Match(message);
        if (domainMatch.Success)
        {
            fields.Domain = domainMatch.Groups[1].Value.Trim();
            if (string.IsNullOrEmpty(fields.Domain))
                fields.Domain = domainMatch.Groups[2].Value.Trim();
            if (string.IsNullOrEmpty(fields.Domain))
                fields.Domain = domainMatch.Groups[3].Value.Trim();
        }

        // Extract Source IP (often in authentication events)
        var ipMatches = IpAddressPattern.Matches(message);
        if (ipMatches.Count > 0)
        {
            fields.SourceAddress = ipMatches[0].Value;
        }
    }

    private void ParseProcessCreationEvent(string message, ParsedLogFields fields)
    {
        // Extract Process Name
        var processMatch = ProcessNamePattern.Match(message);
        if (processMatch.Success)
        {
            fields.ProcessName = processMatch.Groups[1].Value.Trim();
            if (string.IsNullOrEmpty(fields.ProcessName))
                fields.ProcessName = processMatch.Groups[2].Value.Trim();
            if (string.IsNullOrEmpty(fields.ProcessName))
                fields.ProcessName = processMatch.Groups[3].Value.Trim();
        }

        // Extract Process ID
        var pidMatch = ProcessIdPattern.Match(message);
        if (pidMatch.Success)
        {
            var pidStr = pidMatch.Groups[1].Value;
            if (string.IsNullOrEmpty(pidStr))
                pidStr = pidMatch.Groups[2].Value;
            if (string.IsNullOrEmpty(pidStr))
                pidStr = pidMatch.Groups[3].Value;

            // Handle hex format
            if (pidStr.StartsWith("0x", StringComparison.OrdinalIgnoreCase))
            {
                if (int.TryParse(pidStr.Substring(2), System.Globalization.NumberStyles.HexNumber, null, out var hexPid))
                    fields.ProcessId = hexPid;
            }
            else if (int.TryParse(pidStr, out var pid))
            {
                fields.ProcessId = pid;
            }
        }

        // Extract Command Line
        var cmdMatch = CommandLinePattern.Match(message);
        if (cmdMatch.Success)
        {
            fields.CommandLine = cmdMatch.Groups[1].Value.Trim();
            if (string.IsNullOrEmpty(fields.CommandLine))
                fields.CommandLine = cmdMatch.Groups[2].Value.Trim();
        }
    }

    private void ParseFileEvent(string message, ParsedLogFields fields)
    {
        // Extract File Path
        var fileMatch = FilePathPattern.Match(message);
        if (fileMatch.Success)
        {
            fields.FilePath = fileMatch.Groups[1].Value.Trim();
        }
    }

    private void ParseByEventId(LogEntry logEntry, ParsedLogFields fields)
    {
        // Map common Event IDs to event types
        if (!logEntry.EventId.HasValue)
            return;

        var eventId = logEntry.EventId.Value;
        switch (eventId)
        {
            case 4624: // Successful logon
                fields.EventType = "authentication";
                fields.EventAction = "logon";
                fields.EventOutcome = "success";
                break;
            case 4625: // Failed logon
                fields.EventType = "authentication";
                fields.EventAction = "logon";
                fields.EventOutcome = "failure";
                break;
            case 4648: // Logon with explicit credentials
                fields.EventType = "authentication";
                fields.EventAction = "logon_explicit";
                break;
            case 4672: // Special privileges assigned
                fields.EventType = "authorization";
                fields.EventAction = "privilege_assignment";
                break;
            case 4688: // Process creation
                fields.EventType = "process";
                fields.EventAction = "process_creation";
                break;
            case 4656: // File access
                fields.EventType = "file";
                fields.EventAction = "file_access";
                break;
            case 5156: // WFP connection allowed
            case 5157: // WFP connection blocked
                fields.EventType = "network";
                fields.EventAction = eventId == 5156 ? "connection_allowed" : "connection_blocked";
                break;
        }
    }

    private void ExtractNetworkFields(string message, ParsedLogFields fields)
    {
        // If we haven't extracted network fields yet, try generic extraction
        if (string.IsNullOrEmpty(fields.SourceAddress))
        {
            var ipMatches = IpAddressPattern.Matches(message);
            if (ipMatches.Count > 0)
            {
                fields.SourceAddress = ipMatches[0].Value;
                if (ipMatches.Count > 1)
                    fields.DestinationAddress = ipMatches[1].Value;
            }
        }

        // Extract ports if not already extracted
        if (!fields.SourcePort.HasValue || !fields.DestinationPort.HasValue)
        {
            var portMatches = PortPattern.Matches(message);
            var ports = portMatches.Cast<Match>()
                .Select(m => int.TryParse(m.Value, out var p) ? p : (int?)null)
                .Where(p => p.HasValue && p.Value > 0 && p.Value <= 65535)
                .Select(p => p!.Value)
                .Distinct()
                .ToList();

            if (ports.Count > 0 && !fields.SourcePort.HasValue)
                fields.SourcePort = ports[0];
            if (ports.Count > 1 && !fields.DestinationPort.HasValue)
                fields.DestinationPort = ports[1];
        }
    }
}

/// <summary>
/// Structured fields extracted from log messages
/// </summary>
public class ParsedLogFields
{
    public long? EventId { get; set; }
    public string Source { get; set; } = string.Empty;
    public string Message { get; set; } = string.Empty;
    
    // Network fields
    public string? SourceAddress { get; set; }
    public int? SourcePort { get; set; }
    public string? DestinationAddress { get; set; }
    public int? DestinationPort { get; set; }
    public string? ApplicationName { get; set; }
    
    // User fields
    public string? UserName { get; set; }
    public string? Domain { get; set; }
    
    // Process fields
    public string? ProcessName { get; set; }
    public int? ProcessId { get; set; }
    public string? CommandLine { get; set; }
    
    // File fields
    public string? FilePath { get; set; }
    
    // Event classification
    public string? EventType { get; set; }
    public string? EventAction { get; set; }
    public string? EventOutcome { get; set; }
}
