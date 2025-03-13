// This file is being kept as a backup but is not used
// The HeartbeatDto class is defined elsewhere

using System;
using Backend.Models;

namespace Backend.DTOs
{
    /// <summary>
    /// Data transfer object for agent heartbeats
    /// </summary>
    public class HeartbeatDto
    {
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
        public string? IpAddress { get; set; }
        
        /// <summary>
        /// Gets or sets additional information as JSON
        /// </summary>
        public string? AdditionalInfo { get; set; }
    }
}
