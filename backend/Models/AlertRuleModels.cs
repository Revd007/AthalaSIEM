using System;
using System.ComponentModel.DataAnnotations;

namespace Backend.Models
{
    public class AlertRuleModels
    {
        [Key]
        public string Id { get; set; } = Guid.NewGuid().ToString();

        [Required]
        [MaxLength(100)]
        public string Name { get; set; } = string.Empty;

        [MaxLength(500)]
        public string? Description { get; set; }

        [Required]
        public string Condition { get; set; } = string.Empty;

        [Required]
        public AlertSeverityModels Severity { get; set; }

        public bool Enabled { get; set; } = true;

        [Required]
        public string CreatedBy { get; set; } = string.Empty;

        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

        public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;

        public string? NotificationChannels { get; set; }

        public string? Tags { get; set; }
    }
} 