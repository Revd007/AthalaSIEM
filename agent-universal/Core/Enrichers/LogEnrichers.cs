using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Net;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using AthalaSIEM.UniversalAgent.Models;

namespace AthalaSIEM.Agent.Core.Enrichers
{
    /// <summary>
    /// Enterprise GeoIP enricher that adds geographical information to logs based on IP addresses.
    /// Supports configurable GeoIP providers and offline databases for performance.
    /// </summary>
    public class EnterpriseGeoIpEnricher : ILogEnricher
    {
        private readonly ILogger<EnterpriseGeoIpEnricher> _logger;
        private readonly Stopwatch _processingTimer = new();
        private long _logsEnriched;
        private long _successfulEnrichments;
        private long _failedEnrichments;
        
        private IGeoIpProvider? _geoIpProvider;
        private bool _isInitialized;
        private HashSet<string> _privateNetworkRanges = new();

        /// <inheritdoc />
        public string Name => "Enterprise GeoIP Enricher";

        /// <inheritdoc />
        public string Description => "Adds geographical information for IP addresses using configurable GeoIP providers";

        /// <inheritdoc />
        public int Priority => 80;

        /// <summary>
        /// Initializes a new instance of the EnterpriseGeoIpEnricher.
        /// </summary>
        /// <param name="logger">Logger instance for this enricher.</param>
        public EnterpriseGeoIpEnricher(ILogger<EnterpriseGeoIpEnricher> logger)
        {
            _logger = logger;
            InitializePrivateNetworkRanges();
        }

        /// <summary>
        /// Initializes private network ranges for IP classification.
        /// </summary>
        private void InitializePrivateNetworkRanges()
        {
            _privateNetworkRanges = new HashSet<string>
            {
                "10.0.0.0/8",
                "172.16.0.0/12", 
                "192.168.0.0/16",
                "127.0.0.0/8",
                "169.254.0.0/16",
                "224.0.0.0/4",
                "240.0.0.0/4",
                "::1/128",
                "fe80::/10",
                "fc00::/7"
            };
        }

        /// <inheritdoc />
        public async Task<bool> InitializeAsync(Dictionary<string, object> config)
        {
            try
            {
                // This would initialize actual GeoIP provider based on configuration
                // For now, we'll use a placeholder implementation
                _geoIpProvider = new PlaceholderGeoIpProvider(_logger);
                await _geoIpProvider.InitializeAsync(config);
                
                _isInitialized = true;
                _logger.LogInformation("Enterprise GeoIP enricher initialized with provider: {ProviderName}", 
                    _geoIpProvider.ProviderName);
                
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to initialize GeoIP enricher");
                return false;
            }
        }

        /// <inheritdoc />
        public async Task EnrichAsync(LogEntry log)
        {
            if (!_isInitialized || _geoIpProvider == null)
            {
                return;
            }

            _processingTimer.Start();
            _logsEnriched++;

            try
            {
                var ipAddress = ExtractIpAddress(log);
                if (string.IsNullOrEmpty(ipAddress) || IsPrivateOrLocalIp(ipAddress))
                {
                    return;
                }

                var geoData = await _geoIpProvider.LookupAsync(ipAddress);
                if (geoData != null)
                {
                    foreach (var kvp in geoData)
                    {
                        log.Properties[$"GeoIP_{kvp.Key}"] = kvp.Value;
                    }
                    
                    _successfulEnrichments++;
                }
                else
                {
                    _failedEnrichments++;
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Error enriching log with GeoIP data for IP: {IpAddress}", 
                    ExtractIpAddress(log));
                _failedEnrichments++;
            }
            finally
            {
                _processingTimer.Stop();
            }
        }

        /// <summary>
        /// Extracts IP address from log entry.
        /// </summary>
        /// <param name="log">The log entry to extract IP from.</param>
        /// <returns>IP address string or null if not found.</returns>
        private string? ExtractIpAddress(LogEntry log)
        {
            // Check common IP address fields
            if (!string.IsNullOrEmpty(log.IpAddress))
            {
                return log.IpAddress;
            }

            // Check properties for IP addresses
            foreach (var kvp in log.Properties)
            {
                if (kvp.Key.Contains("IP", StringComparison.OrdinalIgnoreCase) ||
                    kvp.Key.Contains("Address", StringComparison.OrdinalIgnoreCase))
                {
                    var value = kvp.Value?.ToString();
                    if (!string.IsNullOrEmpty(value) && IPAddress.TryParse(value, out _))
                    {
                        return value;
                    }
                }
            }

            return null;
        }

        /// <summary>
        /// Determines if an IP address is private or local.
        /// </summary>
        /// <param name="ipAddress">The IP address to check.</param>
        /// <returns>True if the IP is private or local.</returns>
        private bool IsPrivateOrLocalIp(string ipAddress)
        {
            if (!IPAddress.TryParse(ipAddress, out var ip))
            {
                return true; // Invalid IP
            }

            // Check against private ranges
            // This is a simplified implementation - production would use proper CIDR matching
            var ipStr = ip.ToString();
            return ipStr.StartsWith("10.") ||
                   ipStr.StartsWith("192.168.") ||
                   ipStr.StartsWith("172.") ||
                   ipStr.StartsWith("127.") ||
                   ipStr.StartsWith("169.254.") ||
                   ip.Equals(IPAddress.Loopback) ||
                   ip.Equals(IPAddress.IPv6Loopback);
        }

        /// <inheritdoc />
        public Dictionary<string, object> GetMetrics()
        {
            return new Dictionary<string, object>
            {
                ["LogsEnriched"] = _logsEnriched,
                ["SuccessfulEnrichments"] = _successfulEnrichments,
                ["FailedEnrichments"] = _failedEnrichments,
                ["SuccessRate"] = _logsEnriched > 0 ? (double)_successfulEnrichments / _logsEnriched * 100 : 0,
                ["AverageEnrichmentTimeMs"] = _logsEnriched > 0 ? _processingTimer.ElapsedMilliseconds / (double)_logsEnriched : 0,
                ["IsInitialized"] = _isInitialized,
                ["ProviderName"] = _geoIpProvider?.ProviderName ?? "None"
            };
        }
    }

    /// <summary>
    /// Enterprise threat intelligence enricher that adds threat context to logs.
    /// Supports multiple threat intelligence providers and caching for performance.
    /// </summary>
    public class EnterpriseThreatIntelligenceEnricher : ILogEnricher
    {
        private readonly ILogger<EnterpriseThreatIntelligenceEnricher> _logger;
        private readonly Stopwatch _processingTimer = new();
        private long _logsEnriched;
        private long _successfulEnrichments;
        private long _failedEnrichments;
        
        private List<IThreatIntelligenceProvider> _threatProviders = new();
        private Dictionary<string, ThreatIntelligenceData> _cache = new();
        private bool _isInitialized;
        private int _cacheMaxSize;
        private TimeSpan _cacheExpiry = TimeSpan.FromHours(1);

        /// <inheritdoc />
        public string Name => "Enterprise Threat Intelligence Enricher";

        /// <inheritdoc />
        public string Description => "Adds threat intelligence context from multiple providers with caching";

        /// <inheritdoc />
        public int Priority => 85;

        /// <summary>
        /// Initializes a new instance of the EnterpriseThreatIntelligenceEnricher.
        /// </summary>
        /// <param name="logger">Logger instance for this enricher.</param>
        public EnterpriseThreatIntelligenceEnricher(ILogger<EnterpriseThreatIntelligenceEnricher> logger)
        {
            _logger = logger;
            _cacheMaxSize = 10000; // Default value, will be configured during initialization
        }

        /// <inheritdoc />
        public async Task<bool> InitializeAsync(Dictionary<string, object> config)
        {
            try
            {
                // Configure cache settings - prefer config parameter, fallback to appsettings
                if (config.TryGetValue("CacheMaxSize", out var maxSize) && maxSize is int size)
                {
                    _cacheMaxSize = size;
                }
                else
                {
                    // Use default if not provided in config
                    _cacheMaxSize = 10000;
                }

                if (config.TryGetValue("CacheExpiryHours", out var expiry) && expiry is int hours)
                {
                    _cacheExpiry = TimeSpan.FromHours(hours);
                }

                // Initialize threat intelligence providers
                // This would load actual providers based on configuration
                _threatProviders.Add(new PlaceholderThreatProvider(_logger));

                foreach (var provider in _threatProviders)
                {
                    await provider.InitializeAsync(config);
                }

                _isInitialized = true;
                _logger.LogInformation("Enterprise threat intelligence enricher initialized with {ProviderCount} providers", 
                    _threatProviders.Count);
                
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to initialize threat intelligence enricher");
                return false;
            }
        }

        /// <inheritdoc />
        public async Task EnrichAsync(LogEntry log)
        {
            if (!_isInitialized)
            {
                return;
            }

            _processingTimer.Start();
            _logsEnriched++;

            try
            {
                var indicators = ExtractIndicators(log);
                var foundThreats = false;

                foreach (var indicator in indicators)
                {
                    var threatData = await GetThreatIntelligence(indicator.Key, indicator.Value);
                    if (threatData != null)
                    {
                        log.Properties[$"ThreatIntel_{indicator.Value}_{threatData.ThreatType}"] = threatData.Confidence;
                        log.Properties[$"ThreatIntel_Source"] = threatData.Source;
                        log.Properties[$"ThreatIntel_Indicator"] = indicator.Key;
                        
                        foundThreats = true;
                    }
                }

                if (foundThreats)
                {
                    _successfulEnrichments++;
                    log.SecurityRelevance = EscalateSecurityRelevance(log.SecurityRelevance);
                }
                else
                {
                    // Mark as clean if checked
                    log.Properties["ThreatIntel_Status"] = "Clean";
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Error enriching log with threat intelligence");
                _failedEnrichments++;
            }
            finally
            {
                _processingTimer.Stop();
            }
        }

        /// <summary>
        /// Extracts potential threat indicators from log entry.
        /// </summary>
        /// <param name="log">The log entry to extract indicators from.</param>
        /// <returns>Dictionary of indicators and their types.</returns>
        private Dictionary<string, string> ExtractIndicators(LogEntry log)
        {
            var indicators = new Dictionary<string, string>();

            // Extract IP addresses
            if (!string.IsNullOrEmpty(log.IpAddress) && IPAddress.TryParse(log.IpAddress, out _))
            {
                indicators[log.IpAddress] = "IP";
            }

            // Extract from properties
            foreach (var kvp in log.Properties)
            {
                var value = kvp.Value?.ToString();
                if (string.IsNullOrEmpty(value)) continue;

                // Check for IP addresses
                if (IPAddress.TryParse(value, out _))
                {
                    indicators[value] = "IP";
                }
                // Check for domains (simplified)
                else if (value.Contains('.') && !value.Contains(' ') && value.Length > 4)
                {
                    indicators[value] = "Domain";
                }
                // Check for hashes (simplified)
                else if (value.Length is 32 or 40 or 64 && IsHex(value))
                {
                    indicators[value] = "Hash";
                }
            }

            return indicators;
        }

        /// <summary>
        /// Gets threat intelligence for an indicator.
        /// </summary>
        /// <param name="indicator">The indicator to look up.</param>
        /// <param name="indicatorType">The type of indicator.</param>
        /// <returns>Threat intelligence data or null.</returns>
        private async Task<ThreatIntelligenceData?> GetThreatIntelligence(string indicator, string indicatorType)
        {
            // Check cache first
            if (_cache.TryGetValue(indicator, out var cachedData))
            {
                if (DateTime.UtcNow - cachedData.LastUpdated < _cacheExpiry)
                {
                    return cachedData;
                }
                else
                {
                    _cache.Remove(indicator);
                }
            }

            // Query providers
            foreach (var provider in _threatProviders)
            {
                try
                {
                    var threatData = await provider.LookupAsync(indicator, indicatorType);
                    if (threatData != null)
                    {
                        // Cache the result
                        if (_cache.Count < _cacheMaxSize)
                        {
                            _cache[indicator] = threatData;
                        }
                        
                        return threatData;
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Error querying threat provider {Provider} for indicator {Indicator}", 
                        provider.ProviderName, indicator);
                }
            }

            return null;
        }

        /// <summary>
        /// Escalates security relevance if threats are found.
        /// </summary>
        /// <param name="currentRelevance">Current security relevance level.</param>
        /// <returns>Escalated security relevance level.</returns>
        private string EscalateSecurityRelevance(string currentRelevance)
        {
            return currentRelevance.ToLowerInvariant() switch
            {
                "low" => "Medium",
                "medium" => "High", 
                "high" => "Critical",
                _ => "High"
            };
        }

        /// <summary>
        /// Checks if a string contains only hexadecimal characters.
        /// </summary>
        /// <param name="value">The string to check.</param>
        /// <returns>True if the string is hexadecimal.</returns>
        private bool IsHex(string value)
        {
            return System.Text.RegularExpressions.Regex.IsMatch(value, @"^[0-9A-Fa-f]+$");
        }

        /// <inheritdoc />
        public Dictionary<string, object> GetMetrics()
        {
            return new Dictionary<string, object>
            {
                ["LogsEnriched"] = _logsEnriched,
                ["SuccessfulEnrichments"] = _successfulEnrichments,
                ["FailedEnrichments"] = _failedEnrichments,
                ["SuccessRate"] = _logsEnriched > 0 ? (double)_successfulEnrichments / _logsEnriched * 100 : 0,
                ["AverageEnrichmentTimeMs"] = _logsEnriched > 0 ? _processingTimer.ElapsedMilliseconds / (double)_logsEnriched : 0,
                ["CacheSize"] = _cache.Count,
                ["CacheMaxSize"] = _cacheMaxSize,
                ["ProviderCount"] = _threatProviders.Count,
                ["IsInitialized"] = _isInitialized
            };
        }
    }

    /// <summary>
    /// Enterprise asset enricher that adds organizational context to logs.
    /// Provides asset criticality, ownership, and business context information.
    /// </summary>
    public class EnterpriseAssetEnricher : ILogEnricher
    {
        private readonly ILogger<EnterpriseAssetEnricher> _logger;
        private readonly Stopwatch _processingTimer = new();
        private long _logsEnriched;
        private long _successfulEnrichments;
        
        private IAssetProvider? _assetProvider;
        private bool _isInitialized;

        /// <inheritdoc />
        public string Name => "Enterprise Asset Enricher";

        /// <inheritdoc />
        public string Description => "Adds organizational asset context including criticality and ownership";

        /// <inheritdoc />
        public int Priority => 75;

        /// <summary>
        /// Initializes a new instance of the EnterpriseAssetEnricher.
        /// </summary>
        /// <param name="logger">Logger instance for this enricher.</param>
        public EnterpriseAssetEnricher(ILogger<EnterpriseAssetEnricher> logger)
        {
            _logger = logger;
        }

        /// <inheritdoc />
        public async Task<bool> InitializeAsync(Dictionary<string, object> config)
        {
            try
            {
                // Initialize asset provider
                _assetProvider = new PlaceholderAssetProvider(_logger);
                await _assetProvider.InitializeAsync(config);
                
                _isInitialized = true;
                _logger.LogInformation("Enterprise asset enricher initialized with provider: {ProviderName}", 
                    _assetProvider.ProviderName);
                
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to initialize asset enricher");
                return false;
            }
        }

        /// <inheritdoc />
        public async Task EnrichAsync(LogEntry log)
        {
            if (!_isInitialized || _assetProvider == null)
            {
                return;
            }

            _processingTimer.Start();
            _logsEnriched++;

            try
            {
                var assetId = ExtractAssetId(log);
                if (string.IsNullOrEmpty(assetId))
                {
                    return;
                }

                var assetInfo = await _assetProvider.GetAssetAsync(assetId);
                if (assetInfo != null)
                {
                    log.Properties["Asset_Criticality"] = assetInfo.Criticality;
                    log.Properties["Asset_Owner"] = assetInfo.Owner;
                    log.Properties["Asset_BusinessUnit"] = assetInfo.BusinessUnit;
                    log.Properties["Asset_Location"] = assetInfo.Location;
                    log.Properties["Asset_Type"] = assetInfo.AssetType;
                    
                    // Escalate security relevance for critical assets
                    if (assetInfo.Criticality.Equals("Critical", StringComparison.OrdinalIgnoreCase))
                    {
                        log.SecurityRelevance = EscalateForCriticalAsset(log.SecurityRelevance);
                    }
                    
                    _successfulEnrichments++;
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Error enriching log with asset information for asset: {AssetId}", 
                    ExtractAssetId(log));
            }
            finally
            {
                _processingTimer.Stop();
            }
        }

        /// <summary>
        /// Extracts asset identifier from log entry.
        /// </summary>
        /// <param name="log">The log entry to extract asset ID from.</param>
        /// <returns>Asset identifier or null if not found.</returns>
        private string? ExtractAssetId(LogEntry log)
        {
            // Try computer name first
            if (!string.IsNullOrEmpty(log.ComputerName))
            {
                return log.ComputerName;
            }

            // Try IP address
            if (!string.IsNullOrEmpty(log.IpAddress))
            {
                return log.IpAddress;
            }

            // Check properties for asset identifiers
            foreach (var kvp in log.Properties)
            {
                if (kvp.Key.Contains("Computer", StringComparison.OrdinalIgnoreCase) ||
                    kvp.Key.Contains("Host", StringComparison.OrdinalIgnoreCase) ||
                    kvp.Key.Contains("Server", StringComparison.OrdinalIgnoreCase))
                {
                    return kvp.Value?.ToString();
                }
            }

            return null;
        }

        /// <summary>
        /// Escalates security relevance for critical assets.
        /// </summary>
        /// <param name="currentRelevance">Current security relevance level.</param>
        /// <returns>Escalated security relevance level.</returns>
        private string EscalateForCriticalAsset(string currentRelevance)
        {
            return currentRelevance.ToLowerInvariant() switch
            {
                "low" => "Medium",
                "medium" => "High",
                "high" => "Critical",
                _ => currentRelevance
            };
        }

        /// <inheritdoc />
        public Dictionary<string, object> GetMetrics()
        {
            return new Dictionary<string, object>
            {
                ["LogsEnriched"] = _logsEnriched,
                ["SuccessfulEnrichments"] = _successfulEnrichments,
                ["SuccessRate"] = _logsEnriched > 0 ? (double)_successfulEnrichments / _logsEnriched * 100 : 0,
                ["AverageEnrichmentTimeMs"] = _logsEnriched > 0 ? _processingTimer.ElapsedMilliseconds / (double)_logsEnriched : 0,
                ["IsInitialized"] = _isInitialized,
                ["ProviderName"] = _assetProvider?.ProviderName ?? "None"
            };
        }
    }

    #region Placeholder Providers (To be replaced with actual implementations)

    /// <summary>
    /// Placeholder GeoIP provider for demonstration purposes.
    /// In production, this would integrate with MaxMind, IP2Location, or similar services.
    /// </summary>
    internal class PlaceholderGeoIpProvider : IGeoIpProvider
    {
        private readonly ILogger _logger;

        public string ProviderName => "Placeholder GeoIP Provider";

        public PlaceholderGeoIpProvider(ILogger logger)
        {
            _logger = logger;
        }

        public Task<bool> InitializeAsync(Dictionary<string, object> config)
        {
            return Task.FromResult(true);
        }

        public Task<Dictionary<string, object>?> LookupAsync(string ipAddress)
        {
            // Placeholder implementation
            if (IsPrivateIP(ipAddress))
            {
                return Task.FromResult<Dictionary<string, object>?>(null);
            }

            return Task.FromResult<Dictionary<string, object>?>(new Dictionary<string, object>
            {
                ["Country"] = "Unknown",
                ["Region"] = "Unknown", 
                ["City"] = "Unknown",
                ["Latitude"] = 0.0,
                ["Longitude"] = 0.0
            });
        }

        public bool IsPrivateIP(string ipAddress)
        {
            if (!IPAddress.TryParse(ipAddress, out var ip))
            {
                return true;
            }

            var bytes = ip.GetAddressBytes();
            return bytes[0] == 10 ||
                   (bytes[0] == 172 && bytes[1] >= 16 && bytes[1] <= 31) ||
                   (bytes[0] == 192 && bytes[1] == 168) ||
                   bytes[0] == 127;
        }
    }

    /// <summary>
    /// Placeholder threat intelligence provider for demonstration purposes.
    /// In production, this would integrate with VirusTotal, AbuseIPDB, or similar services.
    /// </summary>
    internal class PlaceholderThreatProvider : IThreatIntelligenceProvider
    {
        private readonly ILogger _logger;

        public string ProviderName => "Placeholder Threat Provider";

        public List<string> SupportedIndicatorTypes => new() { "IP", "Domain", "Hash", "URL" };

        public PlaceholderThreatProvider(ILogger logger)
        {
            _logger = logger;
        }

        public Task<bool> InitializeAsync(Dictionary<string, object> config)
        {
            return Task.FromResult(true);
        }

        public Task<ThreatIntelligenceData?> LookupAsync(string indicator, string indicatorType)
        {
            // Placeholder implementation - always returns clean
            return Task.FromResult<ThreatIntelligenceData?>(null);
        }

        public Task<Dictionary<string, ThreatIntelligenceData?>> BulkLookupAsync(Dictionary<string, string> indicators)
        {
            var results = new Dictionary<string, ThreatIntelligenceData?>();
            foreach (var indicator in indicators.Keys)
            {
                results[indicator] = null;
            }
            return Task.FromResult(results);
        }
    }

    /// <summary>
    /// Placeholder asset provider for demonstration purposes.
    /// In production, this would integrate with CMDB, AD, or asset management systems.
    /// </summary>
    internal class PlaceholderAssetProvider : IAssetProvider
    {
        private readonly ILogger _logger;

        public string ProviderName => "Placeholder Asset Provider";

        public PlaceholderAssetProvider(ILogger logger)
        {
            _logger = logger;
        }

        public Task<bool> InitializeAsync(Dictionary<string, object> config)
        {
            return Task.FromResult(true);
        }

        public Task<AssetInformation?> GetAssetAsync(string assetId)
        {
            // Placeholder implementation
            return Task.FromResult<AssetInformation?>(new AssetInformation
            {
                AssetId = assetId,
                AssetName = assetId,
                AssetType = "Server",
                Criticality = "Medium",
                Owner = "IT Operations",
                BusinessUnit = "Technology",
                Location = "Data Center"
            });
        }

        public async Task<Dictionary<string, AssetInformation?>> GetAssetsAsync(IEnumerable<string> assetIds)
        {
            var results = new Dictionary<string, AssetInformation?>();
            foreach (var assetId in assetIds)
            {
                results[assetId] = await GetAssetAsync(assetId);
            }
            return results;
        }

        public Task RefreshAsync()
        {
            return Task.CompletedTask;
        }
    }

    #endregion
} 