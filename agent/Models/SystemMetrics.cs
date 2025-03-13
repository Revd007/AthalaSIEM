using System;
using System.Collections.Generic;

namespace AthalaSIEM.Agent.Models
{
    public class SystemMetrics
    {
        /// <summary>
        /// Gets or sets the agent ID
        /// </summary>
        public required string AgentId { get; set; }
        
        public DateTime Timestamp { get; set; } = DateTime.UtcNow;
        public required CpuMetrics Cpu { get; set; }
        public required MemoryMetrics Memory { get; set; }
        public required DiskMetrics Disk { get; set; }
        public required NetworkMetrics Network { get; set; }
        public required ProcessMetrics Process { get; set; }
        
        /// <summary>
        /// Gets or sets the list of disk metrics
        /// </summary>
        public List<DiskMetrics> Disks { get; set; } = new List<DiskMetrics>();
        
        /// <summary>
        /// Gets or sets the list of process metrics
        /// </summary>
        public List<ProcessMetrics> Processes { get; set; } = new List<ProcessMetrics>();
    }
} 