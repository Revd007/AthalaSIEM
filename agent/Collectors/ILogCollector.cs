using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using AthalaSIEM.Agent.Models;

namespace AthalaSIEM.Agent.Collectors
{
    /// <summary>
    /// Event arguments for when logs are collected
    /// </summary>
    public class LogsCollectedEventArgs : EventArgs
    {
        public NormalizedLogEntry[] LogEntries { get; }

        public LogsCollectedEventArgs(NormalizedLogEntry[] logEntries)
        {
            LogEntries = logEntries ?? Array.Empty<NormalizedLogEntry>();
        }
    }

    /// <summary>
    /// Interface for log collectors
    /// </summary>
    public interface ILogCollector
    {
        /// <summary>
        /// Event raised when a log is collected
        /// </summary>
        event EventHandler<NormalizedLogEntry> LogCollected;

        /// <summary>
        /// Gets the type of the collector
        /// </summary>
        string CollectorType { get; }

        /// <summary>
        /// Gets the status of the collector
        /// </summary>
        CollectorStatus Status { get; }

        /// <summary>
        /// Gets the error message if the collector is in an error state
        /// </summary>
        string ErrorMessage { get; }

        /// <summary>
        /// Initializes the collector with the provided settings
        /// </summary>
        /// <param name="settings">Collector settings</param>
        /// <returns>True if initialization was successful, otherwise false</returns>
        bool Initialize(CollectorSettings settings);

        /// <summary>
        /// Starts the log collector
        /// </summary>
        /// <returns>A task representing the asynchronous operation</returns>
        Task StartAsync();

        /// <summary>
        /// Stops the log collector
        /// </summary>
        /// <returns>A task representing the asynchronous operation</returns>
        Task StopAsync();

        /// <summary>
        /// Pauses the log collector
        /// </summary>
        /// <returns>A task representing the asynchronous operation</returns>
        Task PauseAsync();

        /// <summary>
        /// Resumes the log collector
        /// </summary>
        /// <returns>A task representing the asynchronous operation</returns>
        Task ResumeAsync();

        /// <summary>
        /// Collects logs on demand
        /// </summary>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>The number of logs collected</returns>
        Task<int> CollectLogsAsync(CancellationToken cancellationToken);
    }

    /// <summary>
    /// Interface for the log collector factory
    /// </summary>
    public interface ILogCollectorFactory
    {
        /// <summary>
        /// Creates a log collector based on the provided settings
        /// </summary>
        /// <param name="settings">Collector settings</param>
        /// <returns>An initialized log collector</returns>
        ILogCollector CreateCollector(CollectorSettings settings);
        
        /// <summary>
        /// Registers a collector factory
        /// </summary>
        /// <param name="collectorType">Collector type</param>
        /// <param name="factory">Factory function</param>
        void RegisterCollectorFactory(string collectorType, Func<ILogCollector> factory);
        
        /// <summary>
        /// Checks if a collector type is supported
        /// </summary>
        /// <param name="collectorType">The collector type to check</param>
        /// <returns>True if the collector type is supported, otherwise false</returns>
        bool IsCollectorTypeSupported(string collectorType);
    }
    
    /// <summary>
    /// Interface for log normalization
    /// </summary>
    public interface ILogNormalizer
    {
        /// <summary>
        /// Normalizes raw log data into a standardized format
        /// </summary>
        /// <param name="rawLog">Raw log data to normalize</param>
        /// <returns>Normalized log entry</returns>
        NormalizedLogEntry Normalize(RawLogData rawLog);
    }

    /// <summary>
    /// Event arguments for raw log data collection
    /// </summary>
    public class LogCollectedEventArgs : EventArgs
    {
        public IEnumerable<RawLogData> Logs { get; }

        public LogCollectedEventArgs(IEnumerable<RawLogData> logs)
        {
            Logs = logs;
        }
    }

    /// <summary>
    /// Status of a log collector
    /// </summary>
    public enum CollectorStatus
    {
        /// <summary>
        /// The collector is stopped
        /// </summary>
        Stopped,

        /// <summary>
        /// The collector is running
        /// </summary>
        Running,

        /// <summary>
        /// The collector is paused
        /// </summary>
        Paused,

        /// <summary>
        /// The collector is in an error state
        /// </summary>
        Error
    }
} 