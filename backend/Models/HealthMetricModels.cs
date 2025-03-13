using System;
using System.ComponentModel.DataAnnotations;
using System.Text.Json.Serialization;

namespace Backend.Models
{
    /// <summary>
    /// Represents a health metric in the system
    /// </summary>
    public class HealthMetricModels
    {
        /// <summary>
        /// Gets or sets the metric ID
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
        /// Gets or sets the metric name
        /// </summary>
        [Required]
        [MaxLength(100)]
        public string Name { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the metric value
        /// </summary>
        public double Value { get; set; }
        
        /// <summary>
        /// Gets or sets the metric unit
        /// </summary>
        [MaxLength(20)]
        public string? Unit { get; set; }
        
        /// <summary>
        /// Gets or sets the metric category
        /// </summary>
        [MaxLength(50)]
        public string Category { get; set; } = "System";
        
        /// <summary>
        /// Gets or sets the status of the metric
        /// </summary>
        public HealthStatus Status { get; set; }
        
        /// <summary>
        /// Gets or sets additional properties as JSON
        /// </summary>
        public string? Properties { get; set; }
        
        /// <summary>
        /// Gets or sets the warning threshold for the metric
        /// </summary>
        public double WarningThreshold { get; set; }
        
        /// <summary>
        /// Gets or sets the critical threshold for the metric
        /// </summary>
        public double CriticalThreshold { get; set; }
        
        /// <summary>
        /// Gets or sets additional details about the metric
        /// </summary>
        public string? Details { get; set; }
        
        /// <summary>
        /// Gets or sets a value indicating whether the metric has been processed
        /// </summary>
        public bool Processed { get; set; }
        
        /// <summary>
        /// Gets or sets the date and time when the metric was processed
        /// </summary>
        public DateTime? ProcessedAt { get; set; }
        
        /// <summary>
        /// Gets or sets the date and time when the metric was created in the database
        /// </summary>
        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    }
    
    /// <summary>
    /// Represents the health status of a component
    /// </summary>
    public enum HealthStatus
    {
        /// <summary>
        /// The health status is unknown
        /// </summary>
        Unknown = 0,
        
        /// <summary>
        /// The component is healthy
        /// </summary>
        Healthy = 1,
        
        /// <summary>
        /// The component is degraded but still functioning
        /// </summary>
        Degraded = 2,
        
        /// <summary>
        /// The component is in a warning state
        /// </summary>
        Warning = 3,
        
        /// <summary>
        /// The component is in a critical state
        /// </summary>
        Critical = 4,
        
        /// <summary>
        /// The component is unhealthy
        /// </summary>
        Unhealthy = 5
    }
} 