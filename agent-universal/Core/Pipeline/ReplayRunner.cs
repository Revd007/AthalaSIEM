using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace AthalaSIEM.UniversalAgent.Core.Pipeline
{
    /// <summary>
    /// Replay Runner for Test Mode
    /// Replays recorded logs through the pipeline for testing
    /// 
    /// GOLDEN RULE: Agent MUST run without backend
    /// Replay mode enables testing with recorded data
    /// </summary>
    public class ReplayRunner
    {
        private readonly ILogger<ReplayRunner> _logger;
        private readonly EventPipeline _pipeline;

        // Metrics
        private long _eventsReplayed = 0;
        private long _eventsSuccessful = 0;
        private long _eventsFailed = 0;
        private readonly DateTime _startTime = DateTime.UtcNow;

        public ReplayRunner(ILogger<ReplayRunner> logger, EventPipeline pipeline)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _pipeline = pipeline ?? throw new ArgumentNullException(nameof(pipeline));
        }

        /// <summary>
        /// Replay logs from a JSON Lines file
        /// </summary>
        public async Task<ReplayResult> ReplayFromFileAsync(string filePath)
        {
            if (!File.Exists(filePath))
            {
                throw new FileNotFoundException($"Replay file not found: {filePath}");
            }

            _logger.LogInformation("Starting replay from file: {FilePath}", filePath);
            var startTime = DateTime.UtcNow;

            try
            {
                var lines = await File.ReadAllLinesAsync(filePath);
                var events = new List<object>();

                foreach (var line in lines)
                {
                    if (string.IsNullOrWhiteSpace(line))
                        continue;

                    // Try to parse as JSON
                    if (line.TrimStart().StartsWith("{"))
                    {
                        try
                        {
                            var json = JsonDocument.Parse(line);
                            events.Add(json.RootElement);
                        }
                        catch
                        {
                            // Treat as raw text
                            events.Add(line);
                        }
                    }
                    else
                    {
                        // Raw text event
                        events.Add(line);
                    }
                }

                return await ReplayEventsAsync(events);
            }
            finally
            {
                var duration = DateTime.UtcNow - startTime;
                _logger.LogInformation("Replay completed in {Duration:F2}s", duration.TotalSeconds);
            }
        }

        /// <summary>
        /// Replay events from a directory of log files
        /// </summary>
        public async Task<ReplayResult> ReplayFromDirectoryAsync(string directoryPath, string pattern = "*.jsonl")
        {
            if (!Directory.Exists(directoryPath))
            {
                throw new DirectoryNotFoundException($"Replay directory not found: {directoryPath}");
            }

            _logger.LogInformation("Starting replay from directory: {DirectoryPath} (pattern: {Pattern})",
                directoryPath, pattern);

            var files = Directory.GetFiles(directoryPath, pattern);
            var results = new List<ReplayResult>();

            foreach (var file in files)
            {
                var result = await ReplayFromFileAsync(file);
                results.Add(result);
            }

            // Aggregate results
            return new ReplayResult
            {
                TotalEvents = results.Sum(r => r.TotalEvents),
                SuccessfulEvents = results.Sum(r => r.SuccessfulEvents),
                FailedEvents = results.Sum(r => r.FailedEvents),
                FilesProcessed = files.Length
            };
        }

        /// <summary>
        /// Replay a list of events
        /// </summary>
        public async Task<ReplayResult> ReplayEventsAsync(IEnumerable<object> events)
        {
            var eventList = events.ToList();
            _logger.LogInformation("Replaying {Count} events", eventList.Count);

            var successful = 0;
            var failed = 0;

            foreach (var evt in eventList)
            {
                try
                {
                    var result = await _pipeline.ProcessEventAsync(evt);
                    _eventsReplayed++;

                    if (result)
                    {
                        successful++;
                        _eventsSuccessful++;
                    }
                    else
                    {
                        failed++;
                        _eventsFailed++;
                    }
                }
                catch (Exception ex)
                {
                    failed++;
                    _eventsFailed++;
                    _logger.LogWarning(ex, "Error replaying event");
                }
            }

            // Flush buffer after replay
            var exportResult = await _pipeline.FlushBufferAsync();
            _logger.LogInformation("Exported {Count} events", exportResult.ExportedCount);

            return new ReplayResult
            {
                TotalEvents = eventList.Count,
                SuccessfulEvents = successful,
                FailedEvents = failed,
                ExportedEvents = exportResult.ExportedCount
            };
        }

        /// <summary>
        /// Replay events at a specified rate (events per second)
        /// </summary>
        public async Task<ReplayResult> ReplayWithRateAsync(IEnumerable<object> events, int eventsPerSecond)
        {
            if (eventsPerSecond <= 0)
            {
                throw new ArgumentException("Events per second must be positive", nameof(eventsPerSecond));
            }

            var eventList = events.ToList();
            _logger.LogInformation("Replaying {Count} events at {Rate} events/second", eventList.Count, eventsPerSecond);

            var delayMs = 1000.0 / eventsPerSecond;
            var successful = 0;
            var failed = 0;

            foreach (var evt in eventList)
            {
                try
                {
                    var result = await _pipeline.ProcessEventAsync(evt);
                    _eventsReplayed++;

                    if (result)
                        successful++;
                    else
                        failed++;
                }
                catch
                {
                    failed++;
                }

                await Task.Delay(TimeSpan.FromMilliseconds(delayMs));
            }

            // Flush buffer after replay
            var exportResult = await _pipeline.FlushBufferAsync();

            return new ReplayResult
            {
                TotalEvents = eventList.Count,
                SuccessfulEvents = successful,
                FailedEvents = failed,
                ExportedEvents = exportResult.ExportedCount
            };
        }

        /// <summary>
        /// Get replay metrics
        /// </summary>
        public Dictionary<string, object> GetMetrics()
        {
            var uptime = DateTime.UtcNow - _startTime;
            return new Dictionary<string, object>
            {
                ["EventsReplayed"] = _eventsReplayed,
                ["EventsSuccessful"] = _eventsSuccessful,
                ["EventsFailed"] = _eventsFailed,
                ["SuccessRate"] = _eventsReplayed > 0
                    ? (double)_eventsSuccessful / _eventsReplayed * 100
                    : 100.0,
                ["UptimeSeconds"] = uptime.TotalSeconds,
                ["EventsPerSecond"] = uptime.TotalSeconds > 0
                    ? _eventsReplayed / uptime.TotalSeconds
                    : 0.0
            };
        }
    }

    /// <summary>
    /// Replay result
    /// </summary>
    public class ReplayResult
    {
        public int TotalEvents { get; set; }
        public int SuccessfulEvents { get; set; }
        public int FailedEvents { get; set; }
        public int ExportedEvents { get; set; }
        public int FilesProcessed { get; set; }
    }
}
