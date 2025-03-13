using System;

namespace AthalaSIEM.Agent.Models
{
    public class ProcessMemoryUsage
    {
        public required string Name { get; set; }
        public int Id { get; set; }
        public long MemoryUsageBytes { get; set; }
        public double CpuUsagePercent { get; set; }
        public int ThreadCount { get; set; }
        public DateTime StartTime { get; set; } = DateTime.UtcNow;
    }
} 