using System;
using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace AthalaSIEM.Agent.Models
{
    /// <summary>
    /// Represents raw log data collected from a source
    /// </summary>
    public class LogModels
    {
        /// <summary>
        /// Gets or sets the log ID
        /// </summary>
        public string Id { get; set; } = Guid.NewGuid().ToString();

        /// <summary>
        /// Gets or sets the log source
        /// </summary>
        public string Source { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the source host
        /// </summary>
        public string SourceHost { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the source type
        /// </summary>
        public string SourceType { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the log timestamp
        /// </summary>
        public DateTime Timestamp { get; set; } = DateTime.UtcNow;
        
        /// <summary>
        /// Gets or sets the raw log content
        /// </summary>
        public string RawContent { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets additional metadata
        /// </summary>
        public Dictionary<string, string> Metadata { get; set; } = new Dictionary<string, string>();
        
        /// <summary>
        /// Gets or sets the log severity if available
        /// </summary>
        public string? Severity { get; set; }
    }
    
    /// <summary>
    /// Represents a normalized log entry ready for transmission
    /// </summary>
    public class NormalizedLogEntry
    {
        /// <summary>
        /// Gets or sets the log ID
        /// </summary>
        public string Id { get; set; } = Guid.NewGuid().ToString();
        
        /// <summary>
        /// Gets or sets the agent ID
        /// </summary>
        public string AgentId { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the log source
        /// </summary>
        public string Source { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the source host
        /// </summary>
        public string SourceHost { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the source type
        /// </summary>
        public string SourceType { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the collector type that collected this log
        /// </summary>
        public string CollectorType { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the specific identifier for the source
        /// </summary>
        public string SourceIdentifier { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the log timestamp
        /// </summary>
        public DateTime Timestamp { get; set; } = DateTime.UtcNow;
        
        /// <summary>
        /// Gets or sets the collection timestamp
        /// </summary>
        public DateTime CollectedAt { get; set; } = DateTime.UtcNow;
        
        /// <summary>
        /// Gets or sets the log message
        /// </summary>
        public string Message { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the raw content of the log
        /// </summary>
        public string RawContent { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the hash of the log content
        /// </summary>
        public string ContentHash { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the log severity
        /// </summary>
        public string Severity { get; set; } = "Information";
        
        /// <summary>
        /// Gets or sets the log category
        /// </summary>
        public string Category { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the hostname where the log was generated
        /// </summary>
        public string Hostname { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the application name
        /// </summary>
        public string? Application { get; set; }
        
        /// <summary>
        /// Gets or sets the process ID
        /// </summary>
        public string? ProcessId { get; set; }
        
        /// <summary>
        /// Gets or sets the thread ID
        /// </summary>
        public string? ThreadId { get; set; }
        
        /// <summary>
        /// Gets or sets the log format
        /// </summary>
        public string Format { get; set; } = "Raw";
        
        /// <summary>
        /// Gets or sets additional fields associated with the log
        /// </summary>
        public Dictionary<string, string> AdditionalFields { get; set; } = new Dictionary<string, string>();
        
        /// <summary>
        /// Gets or sets the hash value for integrity checking
        /// </summary>
        public string Hash { get; set; } = string.Empty;
    }
    
    /// <summary>
    /// Represents a log batch for efficient transmission
    /// </summary>
    public class LogBatch
    {
        /// <summary>
        /// Gets or sets the batch ID
        /// </summary>
        public string BatchId { get; set; } = Guid.NewGuid().ToString();
        
        /// <summary>
        /// Gets or sets the agent ID
        /// </summary>
        public string AgentId { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the batch creation timestamp
        /// </summary>
        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
        
        /// <summary>
        /// Gets or sets whether the batch is compressed
        /// </summary>
        public bool Compressed { get; set; } = false;
        
        /// <summary>
        /// Gets or sets whether the batch is encrypted
        /// </summary>
        public bool Encrypted { get; set; } = false;
        
        /// <summary>
        /// Gets or sets the compression algorithm used
        /// </summary>
        public string? CompressionAlgorithm { get; set; }
        
        /// <summary>
        /// Gets or sets the encryption algorithm used
        /// </summary>
        public string? EncryptionAlgorithm { get; set; }
        
        /// <summary>
        /// Gets or sets the batch size (number of log entries)
        /// </summary>
        public int BatchSize { get; set; }
        
        /// <summary>
        /// Gets or sets the logs data (serialized logs)
        /// When compressed/encrypted, this will contain the binary data
        /// </summary>
        public byte[] LogsData { get; set; } = Array.Empty<byte>();
        
        /// <summary>
        /// Gets or sets the logs (only used when not compressed/encrypted)
        /// </summary>
        [JsonIgnore]
        public List<NormalizedLogEntry>? Logs { get; set; }
    }
} 