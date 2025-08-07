using System;
using AthalaSIEM.UniversalAgent.Models;

namespace AthalaSIEM.UniversalAgent.Core
{
    /// <summary>
    /// Event arguments for when a batch of logs has been processed.
    /// Contains the processed batch and processing metadata for monitoring and metrics.
    /// </summary>
    public class LogProcessedEventArgs : EventArgs
    {
        /// <summary>
        /// Gets or sets the batch of processed logs with metadata.
        /// </summary>
        public ProcessedLogBatch ProcessedBatch { get; set; } = new();

        /// <summary>
        /// Gets or sets the processing pipeline stage where this event was raised.
        /// </summary>
        public string ProcessingStage { get; set; } = "";

        /// <summary>
        /// Gets or sets additional context information about the processing.
        /// </summary>
        public string Context { get; set; } = "";

        /// <summary>
        /// Gets or sets the agent instance that processed the logs.
        /// </summary>
        public string AgentId { get; set; } = "";
    }

    /// <summary>
    /// Event arguments for when a security correlation has been detected.
    /// Contains the correlation details and detection metadata for alerting and response.
    /// </summary>
    public class CorrelationDetectedEventArgs : EventArgs
    {
        /// <summary>
        /// Gets or sets the detected security correlation.
        /// </summary>
        public LogCorrelation Correlation { get; set; } = new();

        /// <summary>
        /// Gets or sets when this correlation was detected.
        /// </summary>
        public DateTime DetectedAt { get; set; } = DateTime.UtcNow;

        /// <summary>
        /// Gets or sets the correlator that detected this pattern.
        /// </summary>
        public string DetectorName { get; set; } = "";

        /// <summary>
        /// Gets or sets the processing context when correlation was detected.
        /// </summary>
        public string ProcessingContext { get; set; } = "";

        /// <summary>
        /// Gets or sets whether this correlation requires immediate attention.
        /// </summary>
        public bool IsHighPriority { get; set; }

        /// <summary>
        /// Gets or sets the agent instance that detected the correlation.
        /// </summary>
        public string AgentId { get; set; } = "";
    }

    /// <summary>
    /// Event arguments for log processing errors and exceptions.
    /// Contains error details and context for troubleshooting and monitoring.
    /// </summary>
    public class LogProcessingErrorEventArgs : EventArgs
    {
        /// <summary>
        /// Gets or sets the exception that occurred during processing.
        /// </summary>
        public Exception Exception { get; set; } = new();

        /// <summary>
        /// Gets or sets a descriptive error message.
        /// </summary>
        public string ErrorMessage { get; set; } = "";

        /// <summary>
        /// Gets or sets when the error occurred.
        /// </summary>
        public DateTime ErrorTime { get; set; } = DateTime.UtcNow;

        /// <summary>
        /// Gets or sets the processing stage where the error occurred.
        /// </summary>
        public string ProcessingStage { get; set; } = "";

        /// <summary>
        /// Gets or sets the log entry that caused the error (if applicable).
        /// </summary>
        public LogEntry? FailedLog { get; set; }

        /// <summary>
        /// Gets or sets the severity of this error (Low, Medium, High, Critical).
        /// </summary>
        public string Severity { get; set; } = "Medium";

        /// <summary>
        /// Gets or sets whether processing can continue after this error.
        /// </summary>
        public bool IsCritical { get; set; }

        /// <summary>
        /// Gets or sets the component that raised this error.
        /// </summary>
        public string Source { get; set; } = "";

        /// <summary>
        /// Gets or sets the agent instance where the error occurred.
        /// </summary>
        public string AgentId { get; set; } = "";
    }

    /// <summary>
    /// Event arguments for filter performance and statistics.
    /// Contains metrics about filter execution for monitoring and optimization.
    /// </summary>
    public class FilterMetricsEventArgs : EventArgs
    {
        /// <summary>
        /// Gets or sets the name of the filter.
        /// </summary>
        public string FilterName { get; set; } = "";

        /// <summary>
        /// Gets or sets the number of logs processed by this filter.
        /// </summary>
        public long LogsProcessed { get; set; }

        /// <summary>
        /// Gets or sets the number of logs that passed this filter.
        /// </summary>
        public long LogsPassed { get; set; }

        /// <summary>
        /// Gets or sets the number of logs that were filtered out.
        /// </summary>
        public long LogsFiltered { get; set; }

        /// <summary>
        /// Gets or sets the average processing time per log in milliseconds.
        /// </summary>
        public double AverageProcessingTimeMs { get; set; }

        /// <summary>
        /// Gets or sets the filter efficiency percentage (0-100).
        /// </summary>
        public double FilterEfficiency => LogsProcessed > 0 
            ? (double)LogsFiltered / LogsProcessed * 100 
            : 0;

        /// <summary>
        /// Gets or sets when these metrics were collected.
        /// </summary>
        public DateTime MetricsTime { get; set; } = DateTime.UtcNow;
    }

    /// <summary>
    /// Event arguments for enricher performance and statistics.
    /// Contains metrics about enricher execution for monitoring and optimization.
    /// </summary>
    public class EnricherMetricsEventArgs : EventArgs
    {
        /// <summary>
        /// Gets or sets the name of the enricher.
        /// </summary>
        public string EnricherName { get; set; } = "";

        /// <summary>
        /// Gets or sets the number of logs enriched.
        /// </summary>
        public long LogsEnriched { get; set; }

        /// <summary>
        /// Gets or sets the number of successful enrichments.
        /// </summary>
        public long SuccessfulEnrichments { get; set; }

        /// <summary>
        /// Gets or sets the number of failed enrichments.
        /// </summary>
        public long FailedEnrichments { get; set; }

        /// <summary>
        /// Gets or sets the average enrichment time per log in milliseconds.
        /// </summary>
        public double AverageEnrichmentTimeMs { get; set; }

        /// <summary>
        /// Gets or sets the enrichment success rate percentage (0-100).
        /// </summary>
        public double SuccessRate => LogsEnriched > 0 
            ? (double)SuccessfulEnrichments / LogsEnriched * 100 
            : 0;

        /// <summary>
        /// Gets or sets when these metrics were collected.
        /// </summary>
        public DateTime MetricsTime { get; set; } = DateTime.UtcNow;
    }

    /// <summary>
    /// Event arguments for correlator performance and statistics.
    /// Contains metrics about correlation detection for monitoring and tuning.
    /// </summary>
    public class CorrelatorMetricsEventArgs : EventArgs
    {
        /// <summary>
        /// Gets or sets the name of the correlator.
        /// </summary>
        public string CorrelatorName { get; set; } = "";

        /// <summary>
        /// Gets or sets the number of log batches analyzed.
        /// </summary>
        public long BatchesAnalyzed { get; set; }

        /// <summary>
        /// Gets or sets the number of correlations detected.
        /// </summary>
        public long CorrelationsDetected { get; set; }

        /// <summary>
        /// Gets or sets the number of false positive correlations.
        /// </summary>
        public long FalsePositives { get; set; }

        /// <summary>
        /// Gets or sets the average analysis time per batch in milliseconds.
        /// </summary>
        public double AverageAnalysisTimeMs { get; set; }

        /// <summary>
        /// Gets or sets the detection accuracy percentage (0-100).
        /// </summary>
        public double DetectionAccuracy => CorrelationsDetected > 0 
            ? (double)(CorrelationsDetected - FalsePositives) / CorrelationsDetected * 100 
            : 0;

        /// <summary>
        /// Gets or sets when these metrics were collected.
        /// </summary>
        public DateTime MetricsTime { get; set; } = DateTime.UtcNow;
    }
} 
