using System;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace Backend.Models
{
    /// <summary>
    /// Represents a heartbeat from an agent
    /// </summary>
    public class AgentHeartbeatModels
    {
        /// <summary>
        /// Gets or sets the heartbeat ID
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
        [ForeignKey("AgentId")]
        public virtual AgentModels Agent { get; set; } = null!;
        
        /// <summary>
        /// Gets or sets the timestamp
        /// </summary>
        public DateTime Timestamp { get; set; } = DateTime.UtcNow;
        
        /// <summary>
        /// Gets or sets the status
        /// </summary>
        public AgentStatus Status { get; set; } = AgentStatus.Online;
        
        /// <summary>
        /// Gets or sets the CPU usage percentage
        /// </summary>
        public double CpuUsage { get; set; }
        
        /// <summary>
        /// Gets or sets the memory usage percentage
        /// </summary>
        public double MemoryUsage { get; set; }
        
        /// <summary>
        /// Gets or sets the disk usage percentage
        /// </summary>
        public double DiskUsage { get; set; }
        
        /// <summary>
        /// Gets or sets the IP address
        /// </summary>
        [MaxLength(45)]
        public string? IpAddress { get; set; }
        
        /// <summary>
        /// Gets or sets additional information as JSON
        /// </summary>
        public string? AdditionalInfo { get; set; }
        
        /// <summary>
        /// Gets or sets the creation time
        /// </summary>
        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    }
} 