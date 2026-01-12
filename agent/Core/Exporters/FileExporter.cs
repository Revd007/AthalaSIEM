using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using AthalaSIEM.Agent.Core.Pipeline;

namespace AthalaSIEM.Agent.Core.Exporters;

public class FileExporter : IExporter
{
    private readonly string _outputPath;
    private readonly ILogger<FileExporter> _logger;
    private readonly bool _enabled;

    public FileExporter(
        ILogger<FileExporter> logger,
        string outputPath,
        bool enabled = true)
    {
        _logger = logger;
        _outputPath = outputPath;
        _enabled = enabled;

        if (_enabled)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(_outputPath)!);
        }
    }

    public string Name => "FileExporter";
    public bool IsEnabled => _enabled;

    public async Task<bool> ExportAsync(IEnumerable<INormalizedEvent> events, CancellationToken cancellationToken)
    {
        if (!_enabled)
            return true;

        try
        {
            foreach (var evt in events)
            {
                await ExportAsync(evt, cancellationToken);
            }
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "File export failed");
            return false;
        }
    }

    public async Task<bool> ExportAsync(INormalizedEvent normalizedEvent, CancellationToken cancellationToken)
    {
        if (!_enabled)
            return true;

        try
        {
            var json = JsonSerializer.Serialize(normalizedEvent, new JsonSerializerOptions
            {
                WriteIndented = false
            });

            var filePath = Path.Combine(
                Path.GetDirectoryName(_outputPath)!,
                $"events_{DateTime.UtcNow:yyyyMMdd}.jsonl");

            await File.AppendAllTextAsync(filePath, json + Environment.NewLine, cancellationToken);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "File export failed for event {EventId}", normalizedEvent.Id);
            return false;
        }
    }
}
