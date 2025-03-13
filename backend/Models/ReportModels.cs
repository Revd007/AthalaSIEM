using System;
using System.ComponentModel.DataAnnotations;
using System.Text.Json.Serialization;
using System.Collections.Generic;

namespace Backend.Models
{
    /// <summary>
    /// Represents a report in the system
    /// </summary>
    public class ReportModels
    {
        /// <summary>
        /// Gets or sets the report ID
        /// </summary>
        [Key]
        public string Id { get; set; } = Guid.NewGuid().ToString();
        
        /// <summary>
        /// Gets or sets the report name
        /// </summary>
        [Required]
        [MaxLength(100)]
        public string Name { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the report description
        /// </summary>
        [MaxLength(500)]
        public string? Description { get; set; }
        
        /// <summary>
        /// Gets or sets the user ID who created the report
        /// </summary>
        [Required]
        public string UserId { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the user who created the report
        /// </summary>
        public UserModels? User { get; set; }
        
        /// <summary>
        /// Gets or sets the report type
        /// </summary>
        [MaxLength(50)]
        public string Type { get; set; } = "custom";
        
        /// <summary>
        /// Gets or sets the report parameters as Dictionary
        /// </summary>
        public Dictionary<string, string> Parameters { get; set; } = new Dictionary<string, string>();
        
        /// <summary>
        /// Gets or sets the report query
        /// </summary>
        public string? Query { get; set; }
        
        /// <summary>
        /// Gets or sets the report schedule (cron expression)
        /// </summary>
        [MaxLength(100)]
        public string? Schedule { get; set; }
        
        /// <summary>
        /// Gets or sets the timestamp when the report was last generated
        /// </summary>
        public DateTime? LastGeneratedAt { get; set; }
        
        /// <summary>
        /// Gets or sets the timestamp when the report was created
        /// </summary>
        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
        
        /// <summary>
        /// Gets or sets the timestamp when the report was last updated
        /// </summary>
        public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;
        
        /// <summary>
        /// Gets or sets the report format
        /// </summary>
        [MaxLength(20)]
        public string Format { get; set; } = "pdf";
        
        /// <summary>
        /// Gets or sets the email recipients for scheduled reports
        /// </summary>
        [MaxLength(1000)]
        public string? EmailRecipients { get; set; }
    }
} 