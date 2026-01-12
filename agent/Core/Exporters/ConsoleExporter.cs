using System;
using System.Collections.Generic;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using AthalaSIEM.Agent.Core.Pipeline;

namespace AthalaSIEM.Agent.Core.Exporters;

public class ConsoleExporter : IExporter
{
    private readonly ILogger<ConsoleExporter> _logger;
    private readonly bool _enabled;

    public ConsoleExporter(ILogger<ConsoleExporter> logger, bool enabled = true)
    {
        _logger = logger;
        _enabled = enabled;
    }

    public string Name => "ConsoleExporter";
    public bool IsEnabled => _enabled;

    public async Task<bool> ExportAsync(IEnumerable<INormalizedEvent> events, CancellationToken cancellationToken)
    {
        if (!_enabled)
            return true;

        foreach (var evt in events)
        {
            await ExportAsync(evt, cancellationToken);
        }
        return true;
    }

    public Task<bool> ExportAsync(INormalizedEvent normalizedEvent, CancellationToken cancellationToken)
    {
        if (!_enabled)
            return Task.FromResult(true);

        try
        {
            var json = JsonSerializer.Serialize(normalizedEvent, new JsonSerializerOptions
            {
                WriteIndented = true
            });

            Console.WriteLine($"[{normalizedEvent.Timestamp:yyyy-MM-dd HH:mm:ss}] {json}");
            return Task.FromResult(true);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Console export failed");
            return Task.FromResult(false);
        }
    }
}
