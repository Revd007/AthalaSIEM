using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using AthalaSIEM.UniversalAgent.Core;
using AthalaSIEM.UniversalAgent.Models;

namespace AthalaSIEM.UniversalAgent.Core.Collector
{
    /// <summary>
    /// Adapter to bridge existing ILogCollector to new ICollector interface
    /// Allows existing collectors to work with the new pipeline architecture
    /// </summary>
    public class CollectorAdapter : ICollector
    {
        private readonly ILogCollector _legacyCollector;
        private readonly ILogger<CollectorAdapter> _logger;

        public string Name => _legacyCollector.CollectorName;
        public string SourceType => _legacyCollector.CollectorName;
        public bool IsActive => _legacyCollector.IsActive;

        public event EventHandler<RawEventsCollectedEventArgs>? RawEventsCollected;

        public CollectorAdapter(ILogCollector legacyCollector, ILogger<CollectorAdapter> logger)
        {
            _legacyCollector = legacyCollector ?? throw new ArgumentNullException(nameof(legacyCollector));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));

            // Wire up events
            _legacyCollector.LogCollected += OnLogCollected;
        }

        public async Task<bool> InitializeAsync(Dictionary<string, object> config)
        {
            try
            {
                var result = await _legacyCollector.InitializeAsync(config);
                return result;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error initializing collector adapter for {CollectorName}", Name);
                return false;
            }
        }

        public async Task StartAsync(CancellationToken cancellationToken)
        {
            await _legacyCollector.StartCollectionAsync(cancellationToken);
        }

        public async Task StopAsync()
        {
            await _legacyCollector.StopCollectionAsync();
        }

        public Dictionary<string, object> GetMetrics()
        {
            return new Dictionary<string, object>
            {
                ["Name"] = Name,
                ["SourceType"] = SourceType,
                ["IsActive"] = IsActive,
                ["LogsCollected"] = _legacyCollector.LogsCollected
            };
        }

        /// <summary>
        /// Convert LogCollectedEventArgs to RawEventsCollectedEventArgs
        /// </summary>
        private void OnLogCollected(object? sender, LogCollectedEventArgs e)
        {
            // Convert LogEntry objects to raw events
            var rawEvents = new List<object>();
            foreach (var log in e.Logs)
            {
                // Preserve original log entry as raw event
                rawEvents.Add(log);
            }

            RawEventsCollected?.Invoke(this, new RawEventsCollectedEventArgs
            {
                RawEvents = rawEvents,
                Source = e.Source ?? Name,
                CollectionTime = e.CollectionTime
            });
        }
    }
}
