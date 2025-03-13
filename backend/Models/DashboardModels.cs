using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Text.Json.Serialization;

namespace Backend.Models
{
    /// <summary>
    /// Represents a dashboard in the system
    /// </summary>
    public class DashboardModels
    {
        /// <summary>
        /// Gets or sets the dashboard ID
        /// </summary>
        [Key]
        public string Id { get; set; } = Guid.NewGuid().ToString();
        
        /// <summary>
        /// Gets or sets the dashboard name
        /// </summary>
        [Required]
        [MaxLength(100)]
        public string Name { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the dashboard description
        /// </summary>
        [MaxLength(500)]
        public string? Description { get; set; }
        
        /// <summary>
        /// Gets or sets the dashboard type
        /// </summary>
        [MaxLength(50)]
        public string Type { get; set; } = "custom";
        
        /// <summary>
        /// Gets or sets the user ID who created the dashboard
        /// </summary>
        [Required]
        public string UserId { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets a value indicating whether the dashboard is shared
        /// </summary>
        public bool IsShared { get; set; } = false;
        
        /// <summary>
        /// Gets or sets the dashboard layout as JSON
        /// </summary>
        public string LayoutJson { get; set; } = "[]";
        
        /// <summary>
        /// Gets or sets the dashboard widgets
        /// </summary>
        public List<DashboardWidgetModels> Widgets { get; set; } = new List<DashboardWidgetModels>();
        
        /// <summary>
        /// Gets or sets the timestamp when the dashboard was created
        /// </summary>
        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
        
        /// <summary>
        /// Gets or sets the timestamp when the dashboard was last updated
        /// </summary>
        public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;
        
        /// <summary>
        /// Gets or sets the dashboard theme
        /// </summary>
        [MaxLength(50)]
        public string? Theme { get; set; } = "default";
        
        /// <summary>
        /// Gets or sets the dashboard refresh interval in seconds
        /// </summary>
        public int RefreshIntervalSeconds { get; set; } = 300;
        
        /// <summary>
        /// Gets or sets the user
        /// </summary>
        [JsonIgnore]
        public UserModels? User { get; set; }
        
        /// <summary>
        /// Gets or sets a value indicating whether the dashboard is the default
        /// </summary>
        public bool IsDefault { get; set; }
    }
    
    /// <summary>
    /// Represents a widget on a dashboard
    /// </summary>
    public class DashboardWidgetModels
    {
        /// <summary>
        /// Gets or sets the widget ID
        /// </summary>
        public string Id { get; set; } = Guid.NewGuid().ToString();
        
        /// <summary>
        /// Gets or sets the widget type
        /// </summary>
        [Required]
        [MaxLength(50)]
        public string Type { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the widget title
        /// </summary>
        [MaxLength(100)]
        public string Title { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the widget configuration as JSON
        /// </summary>
        public string ConfigJson { get; set; } = "{}";
        
        /// <summary>
        /// Gets or sets the widget position (x)
        /// </summary>
        public int X { get; set; }
        
        /// <summary>
        /// Gets or sets the widget position (y)
        /// </summary>
        public int Y { get; set; }
        
        /// <summary>
        /// Gets or sets the widget width
        /// </summary>
        public int Width { get; set; } = 4;
        
        /// <summary>
        /// Gets or sets the widget height
        /// </summary>
        public int Height { get; set; } = 4;
    }
} 