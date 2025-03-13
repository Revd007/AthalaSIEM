using System;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using System.Text.Json.Serialization;

namespace Backend.Models
{
    /// <summary>
    /// Represents an alert in the system
    /// </summary>
    public class AlertModels
    {
        /// <summary>
        /// Gets or sets the alert ID
        /// </summary>
        [Key]
        public string Id { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the agent ID associated with the alert
        /// </summary>
        public string? AgentId { get; set; }

        /// <summary>
        /// Gets or sets the alert title
        /// </summary>
        [Required]
        [MaxLength(200)]
        public string Title { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the alert description
        /// </summary>
        [MaxLength(2000)]
        public string Description { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the alert message
        /// </summary>
        [MaxLength(2000)]
        public string Message { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the alert severity
        /// </summary>
        public AlertSeverityModels Severity { get; set; }

        /// <summary>
        /// Gets or sets the alert status
        /// </summary>
        public AlertStatusModels Status { get; set; }

        /// <summary>
        /// Gets or sets the timestamp when the alert occurred
        /// </summary>
        public DateTime Timestamp { get; set; }

        /// <summary>
        /// Gets or sets the source of the alert
        /// </summary>
        [MaxLength(100)]
        public string Source { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the resolution notes
        /// </summary>
        [MaxLength(2000)]
        public string? ResolutionNotes { get; set; }

        /// <summary>
        /// Gets or sets the user ID who acknowledged the alert
        /// </summary>
        public string? AcknowledgedBy { get; set; }

        /// <summary>
        /// Gets or sets the timestamp when the alert was acknowledged
        /// </summary>
        public DateTime? AcknowledgedAt { get; set; }

        /// <summary>
        /// Gets or sets the user ID who resolved the alert
        /// </summary>
        public string? ResolvedBy { get; set; }

        /// <summary>
        /// Gets or sets the timestamp when the alert was resolved
        /// </summary>
        public DateTime? ResolvedAt { get; set; }

        /// <summary>
        /// Gets or sets the timestamp when the alert was created
        /// </summary>
        public DateTime CreatedAt { get; set; }

        /// <summary>
        /// Gets or sets the timestamp when the alert was last updated
        /// </summary>
        public DateTime UpdatedAt { get; set; }

        // Navigation properties
        public AgentModels? Agent { get; set; }
    }
}