using System.Collections.Generic;

namespace AthalaSIEM.Agent.Models
{
    /// <summary>
    /// Represents the CPU metrics.
    /// </summary>
    public class CpuMetrics
    {
        /// <summary>
        /// Gets or sets the CPU usage percentage.
        /// </summary>
        public double? Usage { get; set; }

        /// <summary>
        /// Gets or sets the CPU usage percentage (compatibility property).
        /// </summary>
        public double? UsagePercentage { 
            get => Usage;
            set => Usage = value;
        }

        /// <summary>
        /// Gets or sets the number of cores.
        /// </summary>
        public int NumberOfCores { get; set; }

        /// <summary>
        /// Gets or sets the load average (Linux only).
        /// </summary>
        public double[] LoadAverage { get; set; } = new double[0];
        
        /// <summary>
        /// Gets or sets the CPU temperature in Celsius.
        /// </summary>
        public double? Temperature { get; set; }
        
        /// <summary>
        /// Gets or sets the per-core CPU usage percentages.
        /// </summary>
        public List<double> CoreUsages { get; set; } = new List<double>();
    }
} 