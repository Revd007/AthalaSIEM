using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using AthalaSIEM.UniversalAgent.Models;

namespace AthalaSIEM.Agent.Core
{
    /// <summary>
    /// Defines the contract for log filtering components in the processing pipeline.
    /// Filters determine which logs should be processed based on security relevance and business rules.
    /// </summary>
    public interface ILogFilter
    {
        /// <summary>
        /// Gets the human-readable name of this filter.
        /// </summary>
        string Name { get; }

        /// <summary>
        /// Gets the description of what this filter does.
        /// </summary>
        string Description { get; }

        /// <summary>
        /// Gets the priority level for filter execution (higher numbers execute first).
        /// </summary>
        int Priority { get; }

        /// <summary>
        /// Determines whether a log entry should be processed based on filter criteria.
        /// </summary>
        /// <param name="log">The log entry to evaluate.</param>
        /// <returns>True if the log should be processed, false if it should be filtered out.</returns>
        Task<bool> ShouldProcessAsync(LogEntry log);

        /// <summary>
        /// Gets performance metrics for this filter.
        /// </summary>
        /// <returns>Dictionary containing filter performance statistics.</returns>
        Dictionary<string, object> GetMetrics();
    }

    /// <summary>
    /// Defines the contract for log enrichment components in the processing pipeline.
    /// Enrichers add contextual information and metadata to log entries.
    /// </summary>
    public interface ILogEnricher
    {
        /// <summary>
        /// Gets the human-readable name of this enricher.
        /// </summary>
        string Name { get; }

        /// <summary>
        /// Gets the description of what this enricher adds to logs.
        /// </summary>
        string Description { get; }

        /// <summary>
        /// Gets the priority level for enricher execution (higher numbers execute first).
        /// </summary>
        int Priority { get; }

        /// <summary>
        /// Enriches a log entry with additional contextual information.
        /// </summary>
        /// <param name="log">The log entry to enrich.</param>
        /// <returns>Task representing the enrichment operation.</returns>
        Task EnrichAsync(LogEntry log);

        /// <summary>
        /// Initializes the enricher with configuration settings.
        /// </summary>
        /// <param name="config">Configuration parameters for the enricher.</param>
        /// <returns>True if initialization was successful.</returns>
        Task<bool> InitializeAsync(Dictionary<string, object> config);

        /// <summary>
        /// Gets performance metrics for this enricher.
        /// </summary>
        /// <returns>Dictionary containing enricher performance statistics.</returns>
        Dictionary<string, object> GetMetrics();
    }

    /// <summary>
    /// Defines the contract for log correlation components that detect attack patterns.
    /// Correlators analyze multiple log entries to identify security threats and attack chains.
    /// </summary>
    public interface ILogCorrelator
    {
        /// <summary>
        /// Gets the human-readable name of this correlator.
        /// </summary>
        string Name { get; }

        /// <summary>
        /// Gets the description of what attack patterns this correlator detects.
        /// </summary>
        string Description { get; }

        /// <summary>
        /// Gets the MITRE ATT&CK techniques this correlator can detect.
        /// </summary>
        List<string> DetectedTechniques { get; }

        /// <summary>
        /// Gets the minimum confidence threshold for correlations (0.0 to 1.0).
        /// </summary>
        double MinimumConfidence { get; }

        /// <summary>
        /// Analyzes a collection of log entries to detect security correlations.
        /// </summary>
        /// <param name="logs">The log entries to analyze for correlations.</param>
        /// <returns>Collection of detected security correlations.</returns>
        IEnumerable<LogCorrelation> DetectCorrelations(IEnumerable<LogEntry> logs);

        /// <summary>
        /// Initializes the correlator with configuration settings and threat intelligence.
        /// </summary>
        /// <param name="config">Configuration parameters for the correlator.</param>
        /// <returns>True if initialization was successful.</returns>
        Task<bool> InitializeAsync(Dictionary<string, object> config);

        /// <summary>
        /// Gets performance metrics for this correlator.
        /// </summary>
        /// <returns>Dictionary containing correlator performance statistics.</returns>
        Dictionary<string, object> GetMetrics();
    }

    /// <summary>
    /// Defines the contract for threat intelligence providers.
    /// Provides threat intelligence data for enriching logs and correlations.
    /// </summary>
    public interface IThreatIntelligenceProvider
    {
        /// <summary>
        /// Gets the name of this threat intelligence provider.
        /// </summary>
        string ProviderName { get; }

        /// <summary>
        /// Gets the types of indicators this provider supports.
        /// </summary>
        List<string> SupportedIndicatorTypes { get; }

        /// <summary>
        /// Looks up threat intelligence for an indicator (IP, domain, hash, etc.).
        /// </summary>
        /// <param name="indicator">The indicator to look up.</param>
        /// <param name="indicatorType">The type of indicator (IP, Domain, Hash, URL).</param>
        /// <returns>Threat intelligence data if found, null otherwise.</returns>
        Task<ThreatIntelligenceData?> LookupAsync(string indicator, string indicatorType);

        /// <summary>
        /// Performs bulk lookup of multiple indicators for performance.
        /// </summary>
        /// <param name="indicators">Dictionary of indicators and their types.</param>
        /// <returns>Dictionary of indicators and their threat intelligence data.</returns>
        Task<Dictionary<string, ThreatIntelligenceData?>> BulkLookupAsync(Dictionary<string, string> indicators);

        /// <summary>
        /// Initializes the threat intelligence provider with API keys and configuration.
        /// </summary>
        /// <param name="config">Configuration parameters including API keys.</param>
        /// <returns>True if initialization was successful.</returns>
        Task<bool> InitializeAsync(Dictionary<string, object> config);
    }

    /// <summary>
    /// Defines the contract for asset information providers.
    /// Provides organizational context about assets for log enrichment.
    /// </summary>
    public interface IAssetProvider
    {
        /// <summary>
        /// Gets the name of this asset provider.
        /// </summary>
        string ProviderName { get; }

        /// <summary>
        /// Looks up asset information by identifier (hostname, IP, etc.).
        /// </summary>
        /// <param name="assetId">The asset identifier to look up.</param>
        /// <returns>Asset information if found, null otherwise.</returns>
        Task<AssetInformation?> GetAssetAsync(string assetId);

        /// <summary>
        /// Performs bulk asset lookup for performance.
        /// </summary>
        /// <param name="assetIds">Collection of asset identifiers.</param>
        /// <returns>Dictionary of asset identifiers and their information.</returns>
        Task<Dictionary<string, AssetInformation?>> GetAssetsAsync(IEnumerable<string> assetIds);

        /// <summary>
        /// Initializes the asset provider with data sources and configuration.
        /// </summary>
        /// <param name="config">Configuration parameters for the asset provider.</param>
        /// <returns>True if initialization was successful.</returns>
        Task<bool> InitializeAsync(Dictionary<string, object> config);

        /// <summary>
        /// Refreshes asset information from the data source.
        /// </summary>
        /// <returns>Task representing the refresh operation.</returns>
        Task RefreshAsync();
    }

    /// <summary>
    /// Defines the contract for GeoIP providers.
    /// Provides geographical information for IP addresses.
    /// </summary>
    public interface IGeoIpProvider
    {
        /// <summary>
        /// Gets the name of this GeoIP provider.
        /// </summary>
        string ProviderName { get; }

        /// <summary>
        /// Looks up geographical information for an IP address.
        /// </summary>
        /// <param name="ipAddress">The IP address to look up.</param>
        /// <returns>Dictionary containing geographical information.</returns>
        Task<Dictionary<string, object>?> LookupAsync(string ipAddress);

        /// <summary>
        /// Initializes the GeoIP provider with database paths and configuration.
        /// </summary>
        /// <param name="config">Configuration parameters for the GeoIP provider.</param>
        /// <returns>True if initialization was successful.</returns>
        Task<bool> InitializeAsync(Dictionary<string, object> config);

        /// <summary>
        /// Determines if an IP address is from a private network range.
        /// </summary>
        /// <param name="ipAddress">The IP address to check.</param>
        /// <returns>True if the IP is private, false otherwise.</returns>
        bool IsPrivateIP(string ipAddress);
    }
} 