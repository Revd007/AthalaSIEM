using System.Collections.Generic;

namespace AthalaSIEM.Agent.Models
{
    /// <summary>
    /// Represents the disk metrics.
    /// </summary>
    public class DiskMetrics
    {
        /// <summary>
        /// Gets or sets the list of drives.
        /// </summary>
        public required List<DriveMeasurement> Drives { get; set; } = new List<DriveMeasurement>();

        /// <summary>
        /// Gets or sets the total bytes.
        /// </summary>
        public long TotalBytes { get; set; }

        /// <summary>
        /// Gets or sets the available bytes.
        /// </summary>
        public long AvailableBytes { get; set; }

        /// <summary>
        /// Gets or sets the used bytes.
        /// </summary>
        public long UsedBytes { get; set; }

        /// <summary>
        /// Gets or sets the used percentage.
        /// </summary>
        public double UsedPercentage { get; set; }
    }
} 