using Backend.Models;
using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace Backend.Services
{
    /// <summary>
    /// Interface for threat intelligence service with multi-collector support
    /// </summary>
    public interface IThreatIntelligenceService
    {
        /// <summary>
        /// Analyzes a log entry for threats
        /// </summary>
        /// <param name="logEntry">The log entry to analyze</param>
        /// <returns>Threat analysis result</returns>
        Task<ThreatAnalysisResult> AnalyzeLogEntryAsync(LogEntryModels logEntry);
        
        /// <summary>
        /// Gets threat summary for a specific collector type
        /// </summary>
        /// <param name="collectorType">The collector type</param>
        /// <param name="since">Optional start date for analysis period</param>
        /// <returns>Collector threat summary</returns>
        Task<CollectorThreatSummary> GetCollectorThreatSummaryAsync(string collectorType, DateTime? since = null);
        
        /// <summary>
        /// Finds threat correlations across collectors
        /// </summary>
        /// <param name="timeWindow">Time window for correlation analysis</param>
        /// <param name="minimumOccurrences">Minimum occurrences to consider correlation</param>
        /// <returns>List of threat correlations</returns>
        Task<List<ThreatCorrelation>> FindThreatCorrelationsAsync(TimeSpan timeWindow, int minimumOccurrences);
        
        /// <summary>
        /// Updates threat intelligence feeds
        /// </summary>
        /// <param name="feedId">Feed ID to update</param>
        /// <returns>Task</returns>
        Task UpdateFeedAsync(string feedId);
        
        /// <summary>
        /// Enriches a threat indicator with additional data
        /// </summary>
        /// <param name="indicator">The threat indicator</param>
        /// <returns>Enriched indicator data</returns>
        Task<object> EnrichIndicatorAsync(ThreatIndicator indicator);
        
        /// <summary>
        /// Searches for threats based on criteria
        /// </summary>
        /// <param name="request">Search request</param>
        /// <returns>Threat matches</returns>
        Task<IEnumerable<ThreatMatchDto>> SearchThreatsAsync(ThreatSearchRequest request);
        
        /// <summary>
        /// Checks if a value matches known threat indicators
        /// </summary>
        /// <param name="value">Value to check</param>
        /// <param name="type">Type of indicator</param>
        /// <returns>True if threat detected</returns>
        Task<bool> CheckIndicatorAsync(string value, string type);
        
        /// <summary>
        /// Processes a log entry for threat analysis
        /// </summary>
        /// <param name="logEntry">The log entry</param>
        /// <returns>Task</returns>
        Task ProcessLogEntryAsync(LogEntryModels logEntry);
    }
} 