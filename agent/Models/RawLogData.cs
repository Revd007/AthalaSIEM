using System;
using System.Collections.Generic;

namespace AthalaSIEM.Agent.Models
{
    /// <summary>
    /// Represents raw log data collected from various sources
    /// </summary>
    public class RawLogData
    {
        /// <summary>
        /// Gets or sets the unique identifier for this log entry
        /// </summary>
        public required string Id { get; set; }

        /// <summary>
        /// Gets or sets the timestamp when the log was generated
        /// </summary>
        public DateTime Timestamp { get; set; } = DateTime.UtcNow;

        /// <summary>
        /// Gets or sets the source of the log (e.g., application name, system component)
        /// </summary>
        public required string Source { get; set; }

        /// <summary>
        /// Gets or sets the type of the log source (e.g., Windows Event Log, Linux Syslog)
        /// </summary>
        public required string SourceType { get; set; }

        /// <summary>
        /// Gets or sets the specific identifier for the source (e.g., event log name, log file path)
        /// </summary>
        public required string SourceIdentifier { get; set; }

        /// <summary>
        /// Gets or sets the type of collector that collected this log
        /// </summary>
        public required string CollectorType { get; set; }

        /// <summary>
        /// Gets or sets the severity/level of the log
        /// </summary>
        public required string LogLevel { get; set; }

        /// <summary>
        /// Gets or sets the raw content/message of the log
        /// </summary>
        public required string Content { get; set; }

        /// <summary>
        /// Gets or sets the raw content/message of the log (alias for Content)
        /// </summary>
        public string RawContent { 
            get => Content; 
            set => Content = value; 
        }

        /// <summary>
        /// Gets or sets the severity of the log (alias for LogLevel)
        /// </summary>
        public string Severity {
            get => LogLevel;
            set => LogLevel = value;
        }

        /// <summary>
        /// Gets or sets the host that generated the log
        /// </summary>
        public string SourceHost { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets additional metadata associated with the log
        /// </summary>
        public Dictionary<string, string> Metadata { get; set; } = new Dictionary<string, string>();
    }
}