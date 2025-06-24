using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;

namespace AthalaSIEM.UniversalAgent.Models
{
    /// <summary>
    /// Event arguments for logs sent to backend.
    /// </summary>
    public sealed class LogsSentEventArgs : EventArgs
    {
        [Range(0, int.MaxValue)]
        public int LogCount { get; init; }

        public DateTime SentAt { get; init; } = DateTime.UtcNow;

        public TimeSpan ProcessingDuration { get; init; }

        public long BatchSize { get; init; }

        /// <summary>
        /// Validates the event arguments.
        /// </summary>
        /// <returns>True if the data is valid.</returns>
        public bool IsValid() => LogCount >= 0 && SentAt <= DateTime.UtcNow;
    }

    /// <summary>
    /// Event arguments for communication errors.
    /// </summary>
    public sealed class CommunicationErrorEventArgs : EventArgs
    {
        [Required]
        public string ErrorMessage { get; init; } = string.Empty;

        [Range(0, int.MaxValue)]
        public int LogCount { get; init; }

        public DateTime ErrorTime { get; init; } = DateTime.UtcNow;

        public Exception? Exception { get; init; }

        public string ErrorCategory { get; init; } = "Unknown";

        public bool IsRetryable { get; init; } = true;

        /// <summary>
        /// Validates the error event arguments.
        /// </summary>
        /// <returns>True if the data is valid.</returns>
        public bool IsValid() => !string.IsNullOrWhiteSpace(ErrorMessage) && LogCount >= 0;
    }

    /// <summary>
    /// Event arguments for connection status changes.
    /// </summary>
    public sealed class ConnectionStatusChangedEventArgs : EventArgs
    {
        public bool IsConnected { get; init; }

        [Required]
        public string StatusMessage { get; init; } = string.Empty;

        public DateTime StatusTime { get; init; } = DateTime.UtcNow;

        public string? PreviousStatus { get; init; }

        public TimeSpan? Downtime { get; init; }

        /// <summary>
        /// Validates the connection status event arguments.
        /// </summary>
        /// <returns>True if the data is valid.</returns>
        public bool IsValid() => !string.IsNullOrWhiteSpace(StatusMessage);
    }
} 