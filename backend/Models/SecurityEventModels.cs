using System;
using System.ComponentModel.DataAnnotations;
using System.Text.Json.Serialization;

namespace Backend.Models
{
    /// <summary>
    /// Represents a security event in the system
    /// </summary>
    public class SecurityEventModels
    {
        /// <summary>
        /// Gets or sets the event ID
        /// </summary>
        [Key]
        public string Id { get; set; } = Guid.NewGuid().ToString();
        
        /// <summary>
        /// Gets or sets the agent ID
        /// </summary>
        [Required]
        public string AgentId { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the agent
        /// </summary>
        [JsonIgnore]
        public AgentModels? Agent { get; set; }
        
        /// <summary>
        /// Gets or sets the timestamp
        /// </summary>
        public DateTime Timestamp { get; set; } = DateTime.UtcNow;
        
        /// <summary>
        /// Gets or sets the event type
        /// </summary>
        [Required]
        [MaxLength(100)]
        public string EventType { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the event source
        /// </summary>
        [MaxLength(100)]
        public string LogSource { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the event severity
        /// </summary>
        public AlertSeverityModels Severity { get; set; } = AlertSeverityModels.Medium;


        [Required]
        public required string RawLog { get; set; }

        /// <summary>
        /// Gets or sets the event message
        /// </summary>
        [Required]
        public string Message { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the user associated with the event
        /// </summary>
        [MaxLength(100)]
        public string? Username { get; set; }
        
        /// <summary>
        /// Gets or sets the source IP address
        /// </summary>
        [MaxLength(50)]
        public string? SourceIp { get; set; }
        
        /// <summary>
        /// Gets or sets the destination IP address
        /// </summary>
        [MaxLength(50)]
        public string? DestinationIp { get; set; }
        
        /// <summary>
        /// Gets or sets additional details as JSON
        /// </summary>
        public string? Details { get; set; }
        
        /// <summary>
        /// Gets or sets a value indicating whether an alert was generated
        /// </summary>
        public bool AlertGenerated { get; set; } = false;
        
        /// <summary>
        /// Gets or sets the related alert ID
        /// </summary>
        public string? AlertId { get; set; }
        
        /// <summary>
        /// Gets or sets a value indicating whether the event has been processed
        /// </summary>
        public bool Processed { get; set; }
        
        /// <summary>
        /// Gets or sets the date and time when the event was processed
        /// </summary>
        public DateTime? ProcessedAt { get; set; }
        
        /// <summary>
        /// Gets or sets the date and time when the event was created in the database
        /// </summary>
        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    }
} 