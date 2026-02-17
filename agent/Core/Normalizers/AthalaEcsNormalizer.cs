using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using AthalaSIEM.Agent.Security;
using Microsoft.Extensions.Logging;
using AthalaSIEM.Agent.Core.Pipeline;

namespace AthalaSIEM.Agent.Core.Normalizers;

public class AthalaEcsNormalizer : INormalizer
{
    private const string PlaceholderAgentId = "unregistered-agent";
    private readonly ILogger<AthalaEcsNormalizer> _logger;
    private readonly string _agentName;
    private readonly string _hostName;
    private readonly IAgentIdentityService? _identityService;

    public AthalaEcsNormalizer(
        ILogger<AthalaEcsNormalizer> logger,
        string agentName,
        string hostName,
        IAgentIdentityService? identityService = null)
    {
        _logger = logger;
        _agentName = agentName;
        _hostName = hostName;
        _identityService = identityService;
    }

    public string Name => "AthalaEcsNormalizer";

    public Task<INormalizedEvent> NormalizeAsync(IParsedEvent parsedEvent, CancellationToken cancellationToken)
    {
        return Task.Run(() => NormalizeInternal(parsedEvent), cancellationToken);
    }

    /// <summary>
    /// Resolve agent ID lazily so we do not call GetAgentIdAsync at DI/startup (avoids "Cannot get agent ID" before registration).
    /// </summary>
    private string GetCurrentAgentId()
    {
        if (_identityService == null)
            return PlaceholderAgentId;
        try
        {
            var id = _identityService.GetAgentIdAsync().GetAwaiter().GetResult();
            return !string.IsNullOrEmpty(id) ? id : PlaceholderAgentId;
        }
        catch
        {
            return PlaceholderAgentId;
        }
    }

    private INormalizedEvent NormalizeInternal(IParsedEvent parsedEvent)
    {
        var ecs = new AthalaEcsFields
        {
            Timestamp = parsedEvent.Timestamp,
            AgentId = GetCurrentAgentId(),
            AgentName = _agentName,
            HostName = _hostName,
            HostOs = GetHostOs()
        };

        var rawEvent = new Dictionary<string, object>();
        var extensions = new Dictionary<string, object>
        {
            ["athala.collector"] = parsedEvent.CollectorName,
            ["athala.source_type"] = parsedEvent.SourceType,
            ["athala.pipeline_stage"] = "normalized"
        };

        // CRITICAL: Extract and preserve human-readable message
        string? humanReadableMessage = null;
        if (parsedEvent.StructuredData.TryGetValue("message", out var msgObj))
        {
            humanReadableMessage = msgObj?.ToString();
        }
        // Fallback: check raw_message
        if (string.IsNullOrEmpty(humanReadableMessage) && parsedEvent.StructuredData.TryGetValue("raw_message", out var rawMsgObj))
        {
            humanReadableMessage = rawMsgObj?.ToString();
        }

        foreach (var kvp in parsedEvent.StructuredData)
        {
            if (kvp.Value != null)
            {
                rawEvent[kvp.Key] = kvp.Value;
            }

            var key = kvp.Key.ToLowerInvariant();
            var value = kvp.Value?.ToString() ?? string.Empty;

            MapToEcs(ecs, key, value, kvp.Value);
        }

        // Ensure message is always in RawEvent for frontend display
        if (!string.IsNullOrEmpty(humanReadableMessage) && !rawEvent.ContainsKey("message"))
        {
            rawEvent["message"] = humanReadableMessage;
        }

        var rawBytes = parsedEvent.OriginalRawEvent?.RawData;
        rawEvent["athala.raw_event"] = (rawBytes != null && rawBytes.Length > 0)
            ? System.Text.Encoding.UTF8.GetString(rawBytes)
            : string.Empty;

        return new NormalizedEvent
        {
            Id = parsedEvent.Id,
            Timestamp = parsedEvent.Timestamp,
            Ecs = ecs,
            RawEvent = rawEvent,
            Extensions = extensions
        };
    }

    private void MapToEcs(AthalaEcsFields ecs, string key, string value, object? originalValue)
    {
        switch (key)
        {
            case "event_category":
            case "category":
                ecs.EventCategory = value;
                break;
            case "event_action":
            case "action":
                ecs.EventAction = value;
                break;
            case "event_outcome":
            case "outcome":
                ecs.EventOutcome = value;
                break;
            case "log_level":
            case "level":
            case "severity":
                ecs.LogLevel = value;
                break;
            case "user_name":
            case "username":
            case "user":
                ecs.UserName = value;
                break;
            case "user_id":
            case "uid":
                ecs.UserId = value;
                break;
            case "process_name":
            case "process":
                ecs.ProcessName = value;
                break;
            case "process_id":
            case "pid":
                if (int.TryParse(value, out var pid))
                    ecs.ProcessId = pid;
                break;
            case "process_command_line":
            case "command_line":
            case "cmdline":
                ecs.ProcessCommandLine = value;
                break;
            case "parent_process_name":
            case "parent_process":
                ecs.ProcessParentName = value;
                break;
            case "source_ip":
            case "src_ip":
            case "source.address":
                ecs.SourceIp = value;
                break;
            case "source_port":
            case "src_port":
            case "source.port":
                if (int.TryParse(value, out var srcPort))
                    ecs.SourcePort = srcPort;
                break;
            case "destination_ip":
            case "dst_ip":
            case "destination.address":
                ecs.DestinationIp = value;
                break;
            case "destination_port":
            case "dst_port":
            case "destination.port":
                if (int.TryParse(value, out var dstPort))
                    ecs.DestinationPort = dstPort;
                break;
            case "network_protocol":
            case "protocol":
                ecs.NetworkProtocol = value;
                break;
            default:
                if (!ecs.AdditionalFields.ContainsKey(key))
                {
                    var fieldValue = originalValue ?? value;
                    if (fieldValue != null)
                        ecs.AdditionalFields[key] = fieldValue;
                }
                break;
        }
    }

    private string GetHostOs()
    {
        if (OperatingSystem.IsWindows())
            return "Windows";
        if (OperatingSystem.IsLinux())
            return "Linux";
        if (OperatingSystem.IsMacOS())
            return "macOS";
        return "Unknown";
    }
}
