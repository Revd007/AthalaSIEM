using System;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using System.Text.Json.Serialization;

namespace Backend.Models
{
    /// <summary>
    /// Represents a log entry in the system
    /// </summary>
    public class LogEntryModels
    {
        /// <summary>
        /// Gets or sets the log entry ID
        /// </summary>
        [Key]
        public string Id { get; set; } = Guid.NewGuid().ToString();
        
        /// <summary>
        /// Gets or sets the agent ID that generated the log
        /// </summary>
        [Required]
        public string AgentId { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the agent
        /// </summary>
        [JsonIgnore]
        public AgentModels? Agent { get; set; }
        
        /// <summary>
        /// Gets or sets the timestamp of the log entry
        /// </summary>
        [Required]
        public DateTime Timestamp { get; set; } = DateTime.UtcNow;
        
        /// <summary>
        /// Gets or sets the log level
        /// </summary>
        [Required]
        public string Level { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the log message
        /// </summary>
        [Required]
        public string Message { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the log source
        /// </summary>
        [Required]
        public string Source { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the log category
        /// </summary>
        [MaxLength(100)]
        public string? Category { get; set; }
        
        /// <summary>
        /// Gets or sets the event ID
        /// </summary>
        public long EventId { get; set; }

        [Column("IPAddress")]
        public string IPAddress { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the exception details
        /// </summary>
        public string? Exception { get; set; }
        
        /// <summary>
        /// Gets or sets the name of the machine that generated the log
        /// </summary>
        [Required]
        public string MachineName { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the process ID
        /// </summary>
        public int ProcessId { get; set; }
        
        /// <summary>
        /// Gets or sets the thread ID
        /// </summary>
        public int ThreadId { get; set; }
        
        /// <summary>
        /// Gets or sets the user ID
        /// </summary>
        public string? UserId { get; set; }
        
        /// <summary>
        /// Gets or sets the request path (for web applications)
        /// </summary>
        public string? RequestPath { get; set; }
        
        /// <summary>
        /// Gets or sets the request ID (for web applications)
        /// </summary>
        public string? RequestId { get; set; }
        
        /// <summary>
        /// Gets or sets the IP address of the client (for web applications)
        /// </summary>
        public string? ClientIp { get; set; }
        
        /// <summary>
        /// Gets or sets additional properties as JSON
        /// </summary>
        public string? Properties { get; set; }
        
        /// <summary>
        /// Gets or sets the timestamp when the log was received by the server
        /// </summary>
        public DateTime ReceivedAt { get; set; } = DateTime.UtcNow;
        
        /// <summary>
        /// Gets or sets a value indicating whether the log has been processed
        /// </summary>
        public bool Processed { get; set; } = false;
        
        /// <summary>
        /// Gets or sets the timestamp when the log was processed
        /// </summary>
        public DateTime? ProcessedAt { get; set; }
        
        /// <summary>
        /// Gets or sets the date and time when the log entry was created in the database
        /// </summary>
        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

        public string? StackTrace { get; set; }
    }
} 