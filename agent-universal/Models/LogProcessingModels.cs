using System;
using System.Collections.Generic;
using System.Linq;
using AthalaSIEM.UniversalAgent.Models;

namespace AthalaSIEM.UniversalAgent.Models
{
    /// <summary>
    /// Represents a batch of processed logs with processing metadata and error information.
    /// Used in the ManageEngine-style processing pipeline for batch operations.
    /// </summary>
    public class ProcessedLogBatch
    {
        /// <summary>
        /// Gets or sets the collection of successfully processed log entries.
        /// </summary>
        public List<LogEntry> ProcessedLogs { get; set; } = new();

        /// <summary>
        /// Gets or sets the collection of errors that occurred during processing.
        /// </summary>
        public List<string> Errors { get; set; } = new();

        /// <summary>
        /// Gets or sets the timestamp when processing was completed.
        /// </summary>
        public DateTime ProcessingTime { get; set; }

        /// <summary>
        /// Gets or sets the total number of logs processed in this batch.
        /// </summary>
        public int TotalProcessed { get; set; }

        /// <summary>
        /// Gets or sets the processing duration in milliseconds.
        /// </summary>
        public long ProcessingDurationMs { get; set; }

        /// <summary>
        /// Gets or sets the processing statistics for this batch.
        /// </summary>
        public Dictionary<string, object> ProcessingStats { get; set; } = new();
    }

    /// <summary>
    /// Represents a detected security correlation between multiple log entries.
    /// Used for attack chain detection and threat correlation in enterprise SIEM systems.
    /// </summary>
    public class LogCorrelation
    {
        /// <summary>
        /// Gets or sets the unique identifier for this correlation.
        /// </summary>
        public string CorrelationId { get; set; } = Guid.NewGuid().ToString();

        /// <summary>
        /// Gets or sets the correlation name (e.g., "Brute Force Attack", "Lateral Movement").
        /// </summary>
        public string Name { get; set; } = "";

        /// <summary>
        /// Gets or sets the detailed description of what was detected.
        /// </summary>
        public string Description { get; set; } = "";

        /// <summary>
        /// Gets or sets the severity level (Critical, High, Medium, Low).
        /// </summary>
        public string Severity { get; set; } = "Medium";

        /// <summary>
        /// Gets or sets the confidence score (0.0 to 1.0) of the correlation.
        /// </summary>
        public double ConfidenceScore { get; set; } = 0.5;

        /// <summary>
        /// Gets or sets the MITRE ATT&CK technique IDs associated with this correlation.
        /// </summary>
        public List<string> MitreTechniques { get; set; } = new();

        /// <summary>
        /// Gets or sets the list of log entries that form this correlation.
        /// </summary>
        public List<LogEntry> RelatedLogs { get; set; } = new();

        /// <summary>
        /// Gets or sets the timestamp when this correlation was detected.
        /// </summary>
        public DateTime DetectedAt { get; set; } = DateTime.UtcNow;

        /// <summary>
        /// Gets or sets additional properties and metadata for this correlation.
        /// </summary>
        public Dictionary<string, object> Properties { get; set; } = new();

        /// <summary>
        /// Gets or sets the attack timeline based on related log timestamps.
        /// </summary>
        public TimeSpan AttackDuration => RelatedLogs.Count > 1 
            ? RelatedLogs.Max(l => l.Timestamp) - RelatedLogs.Min(l => l.Timestamp)
            : TimeSpan.Zero;
    }

    /// <summary>
    /// Configuration settings for log processing filters and enrichers.
    /// Supports dynamic configuration without hardcoded values.
    /// </summary>
    public class LogProcessingConfiguration
    {
        /// <summary>
        /// Gets or sets the security event IDs to monitor for Windows Event Logs.
        /// </summary>
        public HashSet<string> SecurityEventIds { get; set; } = new();

        /// <summary>
        /// Gets or sets the minimum security relevance level to process.
        /// </summary>
        public string MinimumSecurityRelevance { get; set; } = "Medium";

        /// <summary>
        /// Gets or sets whether to enable real-time correlation processing.
        /// </summary>
        public bool EnableCorrelation { get; set; } = true;

        /// <summary>
        /// Gets or sets the correlation processing interval in seconds.
        /// </summary>
        public int CorrelationIntervalSeconds { get; set; } = 30;

        /// <summary>
        /// Gets or sets the correlation buffer size per entity.
        /// </summary>
        public int CorrelationBufferSize { get; set; } = 100;

        /// <summary>
        /// Gets or sets the thresholds for various security detections.
        /// </summary>
        public Dictionary<string, int> DetectionThresholds { get; set; } = new();

        /// <summary>
        /// Gets or sets the enrichment providers configuration.
        /// </summary>
        public Dictionary<string, object> EnrichmentConfig { get; set; } = new();
    }

    /// <summary>
    /// Represents threat intelligence data for IP addresses, domains, and other indicators.
    /// </summary>
    public class ThreatIntelligenceData
    {
        /// <summary>
        /// Gets or sets the indicator value (IP, domain, hash, etc.).
        /// </summary>
        public string Indicator { get; set; } = "";

        /// <summary>
        /// Gets or sets the indicator type (IP, Domain, Hash, URL).
        /// </summary>
        public string IndicatorType { get; set; } = "";

        /// <summary>
        /// Gets or sets the threat classification (Malware, C2, Phishing, etc.).
        /// </summary>
        public string ThreatType { get; set; } = "";

        /// <summary>
        /// Gets or sets the confidence score from threat intelligence provider.
        /// </summary>
        public double Confidence { get; set; }

        /// <summary>
        /// Gets or sets the threat intelligence provider source.
        /// </summary>
        public string Source { get; set; } = "";

        /// <summary>
        /// Gets or sets when this intelligence was last updated.
        /// </summary>
        public DateTime LastUpdated { get; set; }

        /// <summary>
        /// Gets or sets additional metadata from the threat intelligence provider.
        /// </summary>
        public Dictionary<string, object> Metadata { get; set; } = new();
    }

    /// <summary>
    /// Asset information for enriching logs with organizational context.
    /// </summary>
    public class AssetInformation
    {
        /// <summary>
        /// Gets or sets the asset identifier (hostname, IP, etc.).
        /// </summary>
        public string AssetId { get; set; } = "";

        /// <summary>
        /// Gets or sets the asset name or hostname.
        /// </summary>
        public string AssetName { get; set; } = "";

        /// <summary>
        /// Gets or sets the asset type (Server, Workstation, Network Device).
        /// </summary>
        public string AssetType { get; set; } = "";

        /// <summary>
        /// Gets or sets the business criticality level (Critical, High, Medium, Low).
        /// </summary>
        public string Criticality { get; set; } = "Medium";

        /// <summary>
        /// Gets or sets the asset owner or responsible team.
        /// </summary>
        public string Owner { get; set; } = "";

        /// <summary>
        /// Gets or sets the business unit or department.
        /// </summary>
        public string BusinessUnit { get; set; } = "";

        /// <summary>
        /// Gets or sets the physical or logical location.
        /// </summary>
        public string Location { get; set; } = "";

        /// <summary>
        /// Gets or sets additional asset metadata and properties.
        /// </summary>
        public Dictionary<string, object> Properties { get; set; } = new();
    }
} 
