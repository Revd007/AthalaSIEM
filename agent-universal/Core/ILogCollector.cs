using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using AthalaSIEM.UniversalAgent.Models;
using LocalLogEntry = AthalaSIEM.UniversalAgent.Models.LogEntry;

namespace AthalaSIEM.UniversalAgent.Core
{
    /// <summary>
    /// Core interface for log collection across different platforms and sources.
    /// Follows ManageEngine EventLog Analyzer pattern for universal log collection.
    /// </summary>
    public interface ILogCollector : IAsyncDisposable
    {
        /// <summary>
        /// Gets the name of this log collector (e.g., "Windows Event Log", "Syslog", etc.)
        /// </summary>
        string CollectorName { get; }

        /// <summary>
        /// Gets the supported operating system for this collector
        /// </summary>
        OperatingSystem SupportedOS { get; }

        /// <summary>
        /// Indicates whether this collector is currently active and collecting logs
        /// </summary>
        bool IsActive { get; }

        /// <summary>
        /// Gets the number of logs collected since startup
        /// </summary>
        long LogsCollected { get; }

        /// <summary>
        /// Initializes the log collector with the specified configuration
        /// </summary>
        /// <param name="config">Configuration parameters for the collector</param>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>True if initialization was successful</returns>
        Task<bool> InitializeAsync(Dictionary<string, object> config, CancellationToken cancellationToken = default);

        /// <summary>
        /// Starts collecting logs from the configured sources
        /// </summary>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>Task representing the collection operation</returns>
        Task StartCollectionAsync(CancellationToken cancellationToken = default);

        /// <summary>
        /// Stops log collection gracefully
        /// </summary>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>Task representing the stop operation</returns>
        Task StopCollectionAsync(CancellationToken cancellationToken = default);

        /// <summary>
        /// Retrieves collected logs in batches
        /// </summary>
        /// <param name="batchSize">Maximum number of logs to retrieve</param>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>Collection of log entries</returns>
        Task<IEnumerable<LocalLogEntry>> GetLogsAsync(int batchSize = 100, CancellationToken cancellationToken = default);

        /// <summary>
        /// Event fired when new logs are collected
        /// </summary>
        event EventHandler<LogCollectedEventArgs> LogCollected;

        /// <summary>
        /// Event fired when an error occurs during collection
        /// </summary>
        event EventHandler<LogCollectionErrorEventArgs> CollectionError;

        /// <summary>
        /// Gets health status and metrics for this collector
        /// </summary>
        /// <returns>Collector health information</returns>
        Task<CollectorHealth> GetHealthAsync();
    }

    /// <summary>
    /// Supported operating systems for log collection
    /// </summary>
    public enum OperatingSystem
    {
        Windows,
        Linux,
        MacOS,
        Universal
    }

    /// <summary>
    /// Event arguments for log collection events
    /// </summary>
    public class LogCollectedEventArgs : EventArgs
    {
        public IEnumerable<LocalLogEntry> Logs { get; set; } = new List<LocalLogEntry>();
        public DateTime CollectionTime { get; set; } = DateTime.UtcNow;
        public string Source { get; set; } = string.Empty;
    }

    /// <summary>
    /// Event arguments for log collection errors
    /// </summary>
    public class LogCollectionErrorEventArgs : EventArgs
    {
        public Exception Exception { get; set; } = new Exception();
        public string Message { get; set; } = string.Empty;
        public DateTime ErrorTime { get; set; } = DateTime.UtcNow;
        public string Source { get; set; } = string.Empty;
    }

    /// <summary>
    /// Health status for a log collector
    /// </summary>

} 
