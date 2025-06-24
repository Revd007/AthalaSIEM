using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using AthalaSIEM.UniversalAgent.Models;

namespace AthalaSIEM.UniversalAgent.Services.Interfaces
{
    /// <summary>
    /// Interface for backend communication service.
    /// Defines contract for communicating with the SIEM backend.
    /// </summary>
    public interface IBackendCommunicationService : IAsyncDisposable
    {
        /// <summary>
        /// Gets a value indicating whether the service is connected to the backend.
        /// </summary>
        bool IsConnected { get; }

        /// <summary>
        /// Gets the number of logs currently queued for sending.
        /// </summary>
        long QueuedLogs { get; }

        /// <summary>
        /// Gets the timestamp of the last successful send operation.
        /// </summary>
        DateTime LastSuccessfulSend { get; }

        /// <summary>
        /// Gets the total number of logs sent since startup.
        /// </summary>
        long TotalLogsSent { get; }

        /// <summary>
        /// Gets the total number of send errors since startup.
        /// </summary>
        long TotalSendErrors { get; }

        /// <summary>
        /// Event raised when logs are successfully sent to the backend.
        /// </summary>
        event EventHandler<LogsSentEventArgs>? LogsSent;

        /// <summary>
        /// Event raised when a communication error occurs.
        /// </summary>
        event EventHandler<CommunicationErrorEventArgs>? CommunicationError;

        /// <summary>
        /// Event raised when the connection status changes.
        /// </summary>
        event EventHandler<ConnectionStatusChangedEventArgs>? ConnectionStatusChanged;

        /// <summary>
        /// Initializes the communication service.
        /// </summary>
        /// <returns>True if initialization was successful.</returns>
        Task<bool> InitializeAsync();

        /// <summary>
        /// Queues a single log entry for sending to the backend.
        /// </summary>
        /// <param name="log">The log entry to queue.</param>
        void QueueLog(LogEntry log);

        /// <summary>
        /// Queues multiple log entries for sending to the backend.
        /// </summary>
        /// <param name="logs">The log entries to queue.</param>
        void QueueLogs(IEnumerable<LogEntry> logs);

        /// <summary>
        /// Forces immediate sending of all queued logs.
        /// </summary>
        /// <returns>True if logs were sent successfully.</returns>
        Task<bool> FlushLogsAsync();

        /// <summary>
        /// Tests the connection to the backend.
        /// </summary>
        /// <returns>True if connection test was successful.</returns>
        Task<bool> TestConnectionAsync();

        /// <summary>
        /// Gets the current health status of the communication service.
        /// </summary>
        /// <returns>Health status information.</returns>
        CommunicationHealth GetHealthStatus();
    }
} 