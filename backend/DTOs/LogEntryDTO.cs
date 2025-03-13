using System;
using System.Collections.Generic;

namespace backend.DTOs
{
    /// <summary>
    /// Data transfer object for log entries
    /// </summary>
    public class LogEntryDTO
    {
        /// <summary>
        /// Gets or sets the log entry ID
        /// </summary>
        public string Id { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the agent ID that generated the log
        /// </summary>
        public string AgentId { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the timestamp of the log entry
        /// </summary>
        public DateTime Timestamp { get; set; } = DateTime.UtcNow;
        
        /// <summary>
        /// Gets or sets the log level (Information, Warning, Error, Critical)
        /// </summary>
        public string Level { get; set; } = "Information";
        
        /// <summary>
        /// Gets or sets the log source (Application, System, Security, etc.)
        /// </summary>
        public string Source { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the log category
        /// </summary>
        public string Category { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the log message
        /// </summary>
        public string Message { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the exception details if applicable
        /// </summary>
        public string Exception { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the stack trace if applicable
        /// </summary>
        public string StackTrace { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the process ID that generated the log
        /// </summary>
        public int? ProcessId { get; set; }
        
        /// <summary>
        /// Gets or sets the process name that generated the log
        /// </summary>
        public string ProcessName { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the thread ID that generated the log
        /// </summary>
        public int? ThreadId { get; set; }
        
        /// <summary>
        /// Gets or sets the user that was associated with the log
        /// </summary>
        public string Username { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets additional properties associated with the log
        /// </summary>
        public Dictionary<string, string> Properties { get; set; } = new Dictionary<string, string>();
        
        /// <summary>
        /// Gets or sets the date and time when the log entry was received by the server
        /// </summary>
        public DateTime ReceivedAt { get; set; } = DateTime.UtcNow;
    }
} 