using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using AthalaSIEM.UniversalAgent.Models;
using AthalaSIEM.UniversalAgent.Core.Parser;
using AthalaSIEM.UniversalAgent.Core.Normalizer;
using AthalaSIEM.UniversalAgent.Core.Exporter;

namespace AthalaSIEM.UniversalAgent.Core.Pipeline
{
    /// <summary>
    /// Production-grade Event Pipeline
    /// Implements the specification pipeline: Collector → Parser → Normalizer → Buffer → Exporter
    /// 
    /// HARD RULES:
    /// - Collector never blocks
    /// - Parser never detects
    /// - Normalizer never enriches
    /// - Exporter never mutates events
    /// </summary>
    public class EventPipeline : IAsyncDisposable
    {
        private readonly ILogger<EventPipeline> _logger;
        private readonly List<IParser> _parsers;
        private readonly INormalizer _normalizer;
        private readonly List<IExporter> _exporters;
        private readonly Queue<AthalaEcsLiteEvent> _buffer;
        private readonly int _maxBufferSize;
        private readonly object _bufferLock = new();

        // Metrics
        private long _eventsProcessed = 0;
        private long _eventsDropped = 0;
        private long _pipelineErrors = 0;
        private readonly DateTime _startTime = DateTime.UtcNow;

        public EventPipeline(
            ILogger<EventPipeline> logger,
            IEnumerable<IParser> parsers,
            INormalizer normalizer,
            IEnumerable<IExporter> exporters,
            int maxBufferSize = 10000)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _parsers = parsers?.ToList() ?? throw new ArgumentNullException(nameof(parsers));
            _normalizer = normalizer ?? throw new ArgumentNullException(nameof(normalizer));
            _exporters = exporters?.ToList() ?? throw new ArgumentNullException(nameof(exporters));
            _maxBufferSize = maxBufferSize;
            _buffer = new Queue<AthalaEcsLiteEvent>();

            _logger.LogInformation("Event pipeline initialized - Parsers: {ParserCount}, Normalizer: {NormalizerName}, Exporters: {ExporterCount}",
                _parsers.Count, _normalizer.Name, _exporters.Count);
        }

        /// <summary>
        /// Processes a raw event through the complete pipeline
        /// </summary>
        public async Task<bool> ProcessEventAsync(object rawEvent)
        {
            if (rawEvent == null)
            {
                return false;
            }

            try
            {
                // Step 1: Parse (Collector → Parser)
                var parser = FindParser(rawEvent);
                if (parser == null)
                {
                    _logger.LogWarning("No parser found for event type: {Type}", rawEvent.GetType().Name);
                    return false;
                }

                var parsedEvent = await parser.ParseAsync(rawEvent);

                // Step 2: Normalize (Parser → Normalizer)
                var normalizedEvent = await _normalizer.NormalizeAsync(parsedEvent);

                // Step 3: Buffer (Normalizer → Buffer)
                if (!AddToBuffer(normalizedEvent))
                {
                    _eventsDropped++;
                    _logger.LogWarning("Buffer full, event dropped");
                    return false;
                }

                _eventsProcessed++;
                return true;
            }
            catch (Exception ex)
            {
                _pipelineErrors++;
                _logger.LogError(ex, "Error processing event through pipeline: {Message}", ex.Message);
                return false;
            }
        }

        /// <summary>
        /// Processes a batch of raw events
        /// </summary>
        public async Task<PipelineBatchResult> ProcessBatchAsync(IEnumerable<object> rawEvents)
        {
            var events = rawEvents?.ToList() ?? new List<object>();
            var processed = 0;
            var failed = 0;

            foreach (var rawEvent in events)
            {
                var success = await ProcessEventAsync(rawEvent);
                if (success)
                    processed++;
                else
                    failed++;
            }

            return new PipelineBatchResult
            {
                TotalEvents = events.Count,
                ProcessedCount = processed,
                FailedCount = failed,
                BufferSize = GetBufferSize()
            };
        }

        /// <summary>
        /// Flushes buffered events to exporters
        /// </summary>
        public async Task<ExportResult> FlushBufferAsync()
        {
            List<AthalaEcsLiteEvent> eventsToExport;

            lock (_bufferLock)
            {
                eventsToExport = _buffer.ToList();
                _buffer.Clear();
            }

            if (!eventsToExport.Any())
            {
                return new ExportResult
                {
                    Success = true,
                    ExportedCount = 0,
                    FailedCount = 0
                };
            }

            // Export to all configured exporters
            var results = new List<ExportResult>();
            foreach (var exporter in _exporters)
            {
                try
                {
                    var result = await exporter.ExportAsync(eventsToExport);
                    results.Add(result);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error exporting to {ExporterName}: {Message}", exporter.Name, ex.Message);
                    results.Add(new ExportResult
                    {
                        Success = false,
                        ExportedCount = 0,
                        FailedCount = eventsToExport.Count,
                        ErrorMessage = ex.Message
                    });
                }
            }

            // Aggregate results
            var totalExported = results.Sum(r => r.ExportedCount);
            var totalFailed = results.Sum(r => r.FailedCount);

            return new ExportResult
            {
                Success = totalFailed == 0,
                ExportedCount = totalExported,
                FailedCount = totalFailed
            };
        }

        /// <summary>
        /// Finds the appropriate parser for a raw event
        /// </summary>
        private IParser? FindParser(object rawEvent)
        {
            return _parsers.FirstOrDefault(p => p.CanParse(rawEvent));
        }

        /// <summary>
        /// Adds normalized event to buffer
        /// </summary>
        private bool AddToBuffer(AthalaEcsLiteEvent normalizedEvent)
        {
            lock (_bufferLock)
            {
                if (_buffer.Count >= _maxBufferSize)
                {
                    return false; // Buffer full
                }

                _buffer.Enqueue(normalizedEvent);
                return true;
            }
        }

        /// <summary>
        /// Gets current buffer size
        /// </summary>
        public int GetBufferSize()
        {
            lock (_bufferLock)
            {
                return _buffer.Count;
            }
        }

        /// <summary>
        /// Gets pipeline metrics
        /// </summary>
        public Dictionary<string, object> GetMetrics()
        {
            var uptime = DateTime.UtcNow - _startTime;
            return new Dictionary<string, object>
            {
                ["EventsProcessed"] = _eventsProcessed,
                ["EventsDropped"] = _eventsDropped,
                ["PipelineErrors"] = _pipelineErrors,
                ["BufferSize"] = GetBufferSize(),
                ["MaxBufferSize"] = _maxBufferSize,
                ["BufferUtilization"] = _maxBufferSize > 0
                    ? (double)GetBufferSize() / _maxBufferSize * 100
                    : 0.0,
                ["UptimeSeconds"] = uptime.TotalSeconds,
                ["EventsPerSecond"] = uptime.TotalSeconds > 0
                    ? _eventsProcessed / uptime.TotalSeconds
                    : 0.0,
                ["ParserMetrics"] = _parsers.ToDictionary(p => p.Name, p => p.GetMetrics()),
                ["NormalizerMetrics"] = _normalizer.GetMetrics(),
                ["ExporterMetrics"] = _exporters.ToDictionary(e => e.Name, e => e.GetMetrics())
            };
        }

        public async ValueTask DisposeAsync()
        {
            // Flush remaining events
            await FlushBufferAsync();

            // Dispose exporters
            foreach (var exporter in _exporters.OfType<IAsyncDisposable>())
            {
                try
                {
                    await exporter.DisposeAsync();
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Error disposing exporter");
                }
            }

            _logger.LogInformation("Event pipeline disposed");
        }
    }

    /// <summary>
    /// Pipeline batch processing result
    /// </summary>
    public class PipelineBatchResult
    {
        public int TotalEvents { get; set; }
        public int ProcessedCount { get; set; }
        public int FailedCount { get; set; }
        public int BufferSize { get; set; }
    }
}
