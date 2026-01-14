using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace AthalaSIEM.UniversalAgent.Core.Collector
{
    /// <summary>
    /// Interface for event collectors
    /// Collectors acquire raw telemetry from various sources
    /// 
    /// HARD RULES (from specification):
    /// - Collector MUST NOT block
    /// - Collector MUST NOT parse (that's Parser's job)
    /// - Collector MUST NOT normalize (that's Normalizer's job)
    /// - Collector MUST NOT detect (that's backend's job)
    /// - Collector outputs RawEvent only
    /// </summary>
    public interface ICollector
    {
        /// <summary>
        /// Gets the name of the collector
        /// </summary>
        string Name { get; }

        /// <summary>
        /// Gets the source type this collector handles
        /// </summary>
        string SourceType { get; }

        /// <summary>
        /// Gets whether the collector is currently active
        /// </summary>
        bool IsActive { get; }

        /// <summary>
        /// Initializes the collector with configuration
        /// </summary>
        /// <param name="config">Collector configuration from backend/config file</param>
        /// <returns>True if initialization succeeded</returns>
        Task<bool> InitializeAsync(Dictionary<string, object> config);

        /// <summary>
        /// Starts collecting events
        /// </summary>
        /// <param name="cancellationToken">Cancellation token to stop collection</param>
        Task StartAsync(CancellationToken cancellationToken);

        /// <summary>
        /// Stops collecting events
        /// </summary>
        Task StopAsync();

        /// <summary>
        /// Event raised when raw events are collected
        /// </summary>
        event EventHandler<RawEventsCollectedEventArgs>? RawEventsCollected;

        /// <summary>
        /// Gets metrics about collection operations
        /// </summary>
        /// <returns>Dictionary of metrics</returns>
        Dictionary<string, object> GetMetrics();
    }

    /// <summary>
    /// Event args for raw events collected
    /// </summary>
    public class RawEventsCollectedEventArgs : EventArgs
    {
        /// <summary>
        /// Raw events collected (unstructured)
        /// </summary>
        public IEnumerable<object> RawEvents { get; set; } = Array.Empty<object>();

        /// <summary>
        /// Source of the events
        /// </summary>
        public string Source { get; set; } = "";

        /// <summary>
        /// Time of collection
        /// </summary>
        public DateTime CollectionTime { get; set; } = DateTime.UtcNow;
    }
}
