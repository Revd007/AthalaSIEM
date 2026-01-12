using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace AthalaSIEM.Agent.Core.Pipeline;

public class ReplayRunner
{
    private readonly ILogger<ReplayRunner> _logger;
    private readonly IEnumerable<IParser> _parsers;
    private readonly INormalizer _normalizer;
    private readonly IExporter _exporter;

    public ReplayRunner(
        ILogger<ReplayRunner> logger,
        IEnumerable<IParser> parsers,
        INormalizer normalizer,
        IExporter exporter)
    {
        _logger = logger;
        _parsers = parsers;
        _normalizer = normalizer;
        _exporter = exporter;
    }

    public async Task ReplayFromFileAsync(string filePath, CancellationToken cancellationToken)
    {
        if (!File.Exists(filePath))
        {
            _logger.LogError("Replay file not found: {FilePath}", filePath);
            return;
        }

        _logger.LogInformation("Starting replay from {FilePath}", filePath);

        var lines = await File.ReadAllLinesAsync(filePath, cancellationToken);
        var count = 0;

        foreach (var line in lines)
        {
            if (cancellationToken.IsCancellationRequested)
                break;

            try
            {
                var rawEvent = JsonSerializer.Deserialize<RawEvent>(line);
                if (rawEvent == null)
                    continue;

                IParsedEvent? parsedEvent = null;
                foreach (var parser in _parsers)
                {
                    if (parser.CanParse(rawEvent))
                    {
                        parsedEvent = await parser.ParseAsync(rawEvent, cancellationToken);
                        break;
                    }
                }

                if (parsedEvent == null)
                {
                    _logger.LogWarning("No parser found for event {EventId}", rawEvent.Id);
                    continue;
                }

                var normalizedEvent = await _normalizer.NormalizeAsync(parsedEvent, cancellationToken);
                await _exporter.ExportAsync(normalizedEvent, cancellationToken);
                count++;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error replaying line {LineNumber}", count + 1);
            }
        }

        _logger.LogInformation("Replay completed: {Count} events processed", count);
    }
}
