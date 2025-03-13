using System;
using System.ComponentModel.DataAnnotations;
using Backend.Models;

namespace Backend.DTOs
{
    /// <summary>
    /// Data transfer object for agent heartbeats
    /// </summary>
    public class AgentHeartbeatDto
    {
        /// <summary>
        /// Gets or sets the timestamp
        /// </summary>
        public DateTime Timestamp { get; set; }
        
        /// <summary>
        /// Gets or sets the status
        /// </summary>
        [Required]
        public AgentStatus Status { get; set; }
        
        /// <summary>
        /// Gets or sets the CPU usage percentage
        /// </summary>
        [Required]
        [Range(0, 100)]
        public double CpuUsage { get; set; }
        
        /// <summary>
        /// Gets or sets the memory usage percentage
        /// </summary>
        [Required]
        [Range(0, 100)]
        public double MemoryUsage { get; set; }
        
        /// <summary>
        /// Gets or sets the disk usage percentage
        /// </summary>
        public double DiskUsage { get; set; }
        
        /// <summary>
        /// Gets or sets the IP address
        /// </summary>
        [Required]
        [RegularExpression(@"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$|^([a-f0-9:]+:+)+[a-f0-9]+$")]
        public string? IpAddress { get; set; }
        
        /// <summary>
        /// Gets or sets additional information as JSON
        /// </summary>
        [Required]
        public string? AdditionalInfo { get; set; }
    }
}