using System.Collections.Generic;

namespace AthalaSIEM.UniversalAgent.Models
{
    /// <summary>
    /// Configuration model for log collectors in the AthalaSIEM Universal Agent.
    /// Defines the type, enabled state, and properties for each collector.
    /// </summary>
    public class CollectorConfiguration
    {
        /// <summary>
        /// Gets or sets the type of the collector (e.g., "WindowsEventLog", "FileIntegrity", "Registry").
        /// </summary>
        public string Type { get; set; } = "";

        /// <summary>
        /// Gets or sets a value indicating whether this collector is enabled.
        /// </summary>
        public bool Enabled { get; set; } = true;

        /// <summary>
        /// Gets or sets the configuration properties specific to this collector type.
        /// </summary>
        public Dictionary<string, object> Properties { get; set; } = new();
    }
} 
