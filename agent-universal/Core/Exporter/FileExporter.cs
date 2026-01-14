using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using AthalaSIEM.UniversalAgent.Core.Normalizer;

namespace AthalaSIEM.UniversalAgent.Core.Exporter
{
    /// <summary>
    /// File Exporter for Test Mode
    /// Exports normalized events to JSON Lines file format
    /// 
    /// GOLDEN RULE: Agent MUST run without backend (test mode)
    /// This exporter enables testing without backend connectivity
    /// </summary>
    public class FileExporter : IExporter, IAsyncDisposable
    {
        private readonly ILogger<FileExporter> _logger;
        private readonly string _outputDirectory;
        private readonly string _outputFileName;
        private readonly bool _appendMode;
        private readonly JsonSerializerOptions _jsonOptions;

        // Metrics
        private long _eventsExported = 0;
        private long _exportErrors = 0;
        private readonly DateTime _startTime = DateTime.UtcNow;
        private StreamWriter? _writer;
        private readonly object _writeLock = new();

        public string Name => "FileExporter";
        public string Mode => "File";

        public FileExporter(
            ILogger<FileExporter> logger,
            string outputDirectory = "./test-output",
            string outputFileName = "athalasiem-events.jsonl",
            bool appendMode = false)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _outputDirectory = outputDirectory ?? throw new ArgumentNullException(nameof(outputDirectory));
            _outputFileName = outputFileName ?? throw new ArgumentNullException(nameof(outputFileName));
            _appendMode = appendMode;

            _jsonOptions = new JsonSerializerOptions
            {
                WriteIndented = false, // JSON Lines format (one JSON per line)
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
                DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull
            };
        }

        public async Task<bool> InitializeAsync()
        {
            try
            {
                // Create output directory if it doesn't exist
                if (!Directory.Exists(_outputDirectory))
                {
                    Directory.CreateDirectory(_outputDirectory);
                    _logger.LogInformation("Created output directory: {Directory}", _outputDirectory);
                }

                // Create or open file
                var filePath = Path.Combine(_outputDirectory, _outputFileName);
                var fileMode = _appendMode ? FileMode.Append : FileMode.Create;

                _writer = new StreamWriter(filePath, append: _appendMode)
                {
                    AutoFlush = true // Auto-flush for real-time viewing
                };

                _logger.LogInformation("File exporter initialized - Output: {FilePath}, Mode: {Mode}",
                    filePath, _appendMode ? "Append" : "Create");

                await Task.CompletedTask;
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to initialize file exporter");
                return false;
            }
        }

        public Task<ExportResult> ExportAsync(IEnumerable<AthalaEcsLiteEvent> events)
        {
            if (events == null)
            {
                throw new ArgumentNullException(nameof(events));
            }

            if (_writer == null)
            {
                return Task.FromResult(new ExportResult
                {
                    Success = false,
                    ExportedCount = 0,
                    FailedCount = 0,
                    ErrorMessage = "Exporter not initialized. Call InitializeAsync() first."
                });
            }

            var eventList = events.ToList();
            var exported = 0;
            var failed = 0;
            var errors = new List<string>();

            lock (_writeLock)
            {
                foreach (var evt in eventList)
                {
                    try
                    {
                        // Serialize to JSON
                        var json = JsonSerializer.Serialize(evt, _jsonOptions);

                        // Write as JSON Lines (one JSON object per line)
                        _writer.WriteLine(json);

                        exported++;
                        _eventsExported++;
                    }
                    catch (Exception ex)
                    {
                        failed++;
                        _exportErrors++;
                        errors.Add($"Error exporting event {evt.Athala?.OriginalEventId}: {ex.Message}");
                        _logger.LogWarning(ex, "Error exporting event: {EventId}", evt.Athala?.OriginalEventId);
                    }
                }
            }

            var result = new ExportResult
            {
                Success = failed == 0,
                ExportedCount = exported,
                FailedCount = failed,
                ErrorMessage = errors.Any() ? string.Join("; ", errors) : null
            };

            if (exported > 0)
            {
                _logger.LogDebug("Exported {Count} events to file", exported);
            }

            if (failed > 0)
            {
                _logger.LogWarning("Failed to export {Count} events", failed);
            }

            return Task.FromResult(result);
        }

        public Dictionary<string, object> GetMetrics()
        {
            var uptime = DateTime.UtcNow - _startTime;
            return new Dictionary<string, object>
            {
                ["Name"] = Name,
                ["Mode"] = Mode,
                ["OutputDirectory"] = _outputDirectory,
                ["OutputFileName"] = _outputFileName,
                ["EventsExported"] = _eventsExported,
                ["ExportErrors"] = _exportErrors,
                ["SuccessRate"] = _eventsExported > 0
                    ? (double)(_eventsExported - _exportErrors) / _eventsExported * 100
                    : 100.0,
                ["UptimeSeconds"] = uptime.TotalSeconds,
                ["EventsPerSecond"] = uptime.TotalSeconds > 0
                    ? _eventsExported / uptime.TotalSeconds
                    : 0.0
            };
        }

        /// <summary>
        /// Disposes the exporter and closes the file
        /// </summary>
        public async ValueTask DisposeAsync()
        {
            if (_writer != null)
            {
                await _writer.FlushAsync();
                _writer.Dispose();
                _writer = null;
                _logger.LogInformation("File exporter disposed");
            }
        }
    }
}
