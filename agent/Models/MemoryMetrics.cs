using System;

namespace AthalaSIEM.Agent.Models
{
    public class MemoryMetrics
    {
        public double UsedPercentage { get; set; }
        public long AvailableBytes { get; set; }
        public long TotalBytes { get; set; }
        public long UsedBytes { get; set; }
        public long ProcessUsedBytes { get; set; }
        public long ProcessPrivateBytes { get; set; }
        
        /// <summary>
        /// Gets the free bytes (calculated from available bytes).
        /// </summary>
        public long FreeBytes => AvailableBytes;
        
        /// <summary>
        /// Gets or sets the swap memory usage percentage.
        /// </summary>
        public double SwapUsagePercentage { get; set; }
    }
} 