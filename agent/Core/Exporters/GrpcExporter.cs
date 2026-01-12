using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AthalaSIEM.Agent.Communication;
using Microsoft.Extensions.Logging;
using AthalaSIEM.Agent.Core.Pipeline;

namespace AthalaSIEM.Agent.Core.Exporters;

public class GrpcExporter : IExporter
{
    private readonly ILogForwarder _logForwarder;
    private readonly ILogger<GrpcExporter> _logger;
    private readonly bool _enabled;

    public GrpcExporter(
        ILogForwarder logForwarder,
        ILogger<GrpcExporter> logger,
        bool enabled = true)
    {
        _logForwarder = logForwarder;
        _logger = logger;
        _enabled = enabled;
    }

    public string Name => "GrpcExporter";
    public bool IsEnabled => _enabled;

    public async Task<bool> ExportAsync(IEnumerable<INormalizedEvent> events, CancellationToken cancellationToken)
    {
        if (!_enabled)
            return true;

        try
        {
            var normalizedLogs = events.Select(ConvertToNormalizedLogEntry).ToArray();
            await _logForwarder.ForwardLogBatchAsync(normalizedLogs);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "gRPC export failed");
            return false;
        }
    }

    public async Task<bool> ExportAsync(INormalizedEvent normalizedEvent, CancellationToken cancellationToken)
    {
        if (!_enabled)
            return true;

        try
        {
            var normalizedLog = ConvertToNormalizedLogEntry(normalizedEvent);
            await _logForwarder.ForwardLogAsync(normalizedLog);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "gRPC export failed for event {EventId}", normalizedEvent.Id);
            return false;
        }
    }

    private Models.NormalizedLogEntry ConvertToNormalizedLogEntry(INormalizedEvent evt)
    {
        var ecs = evt.Ecs;
        var metadata = new Dictionary<string, string>();

        foreach (var kvp in evt.RawEvent)
        {
            metadata[kvp.Key] = kvp.Value?.ToString() ?? string.Empty;
        }

        var message = evt.RawEvent.GetValueOrDefault("message")?.ToString() 
            ?? evt.RawEvent.GetValueOrDefault("raw_message")?.ToString() 
            ?? string.Empty;

        var result = new Models.NormalizedLogEntry
        {
            Id = evt.Id,
            Timestamp = evt.Timestamp,
            Source = ecs.EventCategory ?? "Unknown",
            SourceType = evt.Extensions.GetValueOrDefault("athala.source_type")?.ToString() ?? "Unknown",
            Level = ecs.LogLevel ?? "Information",
            Message = message,
            Metadata = metadata,
            AgentId = ecs.AgentId ?? string.Empty,
            Hostname = ecs.HostName ?? string.Empty,
            AdditionalFields = new Dictionary<string, string>()
        };
        
        if (!string.IsNullOrEmpty(ecs.UserName))
            result.AdditionalFields["user_name"] = ecs.UserName;
        if (!string.IsNullOrEmpty(ecs.ProcessName))
            result.AdditionalFields["process_name"] = ecs.ProcessName;
        if (ecs.ProcessId.HasValue)
            result.AdditionalFields["process_id"] = ecs.ProcessId.Value.ToString();
        if (!string.IsNullOrEmpty(ecs.SourceIp))
            result.AdditionalFields["source_ip"] = ecs.SourceIp;
        if (!string.IsNullOrEmpty(ecs.DestinationIp))
            result.AdditionalFields["destination_ip"] = ecs.DestinationIp;
        if (ecs.SourcePort.HasValue)
            result.AdditionalFields["source_port"] = ecs.SourcePort.Value.ToString();
        if (ecs.DestinationPort.HasValue)
            result.AdditionalFields["destination_port"] = ecs.DestinationPort.Value.ToString();
        
        return result;
    }
}
