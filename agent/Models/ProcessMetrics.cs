using System.Collections.Generic;

namespace AthalaSIEM.Agent.Models
{
    public class ProcessMetrics
    {
        public required ProcessMemoryUsage CurrentProcess { get; set; }
        public required List<ProcessMemoryUsage> MemoryUsageProcesses { get; set; } = new List<ProcessMemoryUsage>();
    }
} 