using Microsoft.Extensions.Logging;

namespace AthalaSIEM.Agent.Core.Pipeline;

public class PipelineOrchestrator : IHostedService
{
    private readonly IEnumerable<ICollector> _collectors;
    private readonly IEnumerable<IParser> _parsers;
    private readonly INormalizer _normalizer;
    private readonly IBuffer _buffer;
    private readonly IEnumerable<IExporter> _exporters;
    private readonly ILogger<PipelineOrchestrator> _logger;
    private readonly CancellationTokenSource _cancellationTokenSource = new();
    private Task? _processingTask;

    public PipelineOrchestrator(
        IEnumerable<ICollector> collectors,
        IEnumerable<IParser> parsers,
        INormalizer normalizer,
        IBuffer buffer,
        IEnumerable<IExporter> exporters,
        ILogger<PipelineOrchestrator> logger)
    {
        _collectors = collectors;
        _parsers = parsers;
        _normalizer = normalizer;
        _buffer = buffer;
        _exporters = exporters;
        _logger = logger;
    }

    public async Task StartAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("Starting pipeline orchestrator");

        foreach (var collector in _collectors.Where(c => c.IsEnabled))
        {
            collector.EventCollected += OnEventCollected;
            await collector.StartAsync(cancellationToken);
            _logger.LogInformation("Started collector: {CollectorName}", collector.Name);
        }

        _processingTask = Task.Run(() => ProcessPipelineAsync(_cancellationTokenSource.Token), cancellationToken);

        _logger.LogInformation("Pipeline orchestrator started");
    }

    public async Task StopAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("Stopping pipeline orchestrator");

        _cancellationTokenSource.Cancel();

        foreach (var collector in _collectors)
        {
            collector.EventCollected -= OnEventCollected;
            await collector.StopAsync(cancellationToken);
        }

        if (_processingTask != null)
        {
            await _processingTask;
        }

        _logger.LogInformation("Pipeline orchestrator stopped");
    }

    private void OnEventCollected(object? sender, IRawEvent rawEvent)
    {
        _ = Task.Run(async () =>
        {
            try
            {
                await ProcessRawEventAsync(rawEvent, _cancellationTokenSource.Token);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error processing raw event {EventId}", rawEvent.Id);
            }
        }, _cancellationTokenSource.Token);
    }

    private async Task ProcessRawEventAsync(IRawEvent rawEvent, CancellationToken cancellationToken)
    {
        IParsedEvent? parsedEvent = null;

        foreach (var parser in _parsers)
        {
            if (parser.CanParse(rawEvent))
            {
                try
                {
                    parsedEvent = await parser.ParseAsync(rawEvent, cancellationToken);
                    break;
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Parser {ParserName} failed for event {EventId}", parser.Name, rawEvent.Id);
                }
            }
        }

        if (parsedEvent == null)
        {
            _logger.LogWarning("No parser found for event {EventId} from collector {CollectorName}", rawEvent.Id, rawEvent.CollectorName);
            return;
        }

        INormalizedEvent normalizedEvent;
        try
        {
            normalizedEvent = await _normalizer.NormalizeAsync(parsedEvent, cancellationToken);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Normalization failed for event {EventId}", parsedEvent.Id);
            return;
        }

        var added = await _buffer.AddAsync(normalizedEvent, cancellationToken);
        if (!added)
        {
            _logger.LogWarning("Buffer rejected event {EventId}", normalizedEvent.Id);
        }
    }

    private async Task ProcessPipelineAsync(CancellationToken cancellationToken)
    {
        while (!cancellationToken.IsCancellationRequested)
        {
            try
            {
                var events = await _buffer.DrainAsync(100, cancellationToken);
                if (events.Any())
                {
                    foreach (var exporter in _exporters.Where(e => e.IsEnabled))
                    {
                        try
                        {
                            var success = await exporter.ExportAsync(events, cancellationToken);
                            if (!success)
                            {
                                _logger.LogWarning("Exporter {ExporterName} failed", exporter.Name);
                            }
                        }
                        catch (Exception ex)
                        {
                            _logger.LogError(ex, "Exporter {ExporterName} error", exporter.Name);
                        }
                    }
                }
                else
                {
                    await Task.Delay(100, cancellationToken);
                }
            }
            catch (OperationCanceledException)
            {
                break;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Pipeline processing error");
                await Task.Delay(1000, cancellationToken);
            }
        }
    }
}
