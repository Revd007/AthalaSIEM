using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using AthalaSIEM.UniversalAgent.Models;

namespace AthalaSIEM.Agent.Core.Filters
{
    /// <summary>
    /// Enterprise-grade security relevance filter that processes logs based on configurable security levels.
    /// Supports dynamic configuration for different organizational security requirements.
    /// </summary>
    public class EnterpriseSecurityRelevanceFilter : ILogFilter
    {
        private readonly ILogger<EnterpriseSecurityRelevanceFilter> _logger;
        private readonly Stopwatch _processingTimer = new();
        private long _logsProcessed;
        private long _logsPassed;
        private HashSet<string> _allowedSecurityLevels = new();

        /// <inheritdoc />
        public string Name => "Enterprise Security Relevance Filter";

        /// <inheritdoc />
        public string Description => "Filters logs based on configurable security relevance levels for enterprise environments";

        /// <inheritdoc />
        public int Priority => 100; // High priority for security filtering

        /// <summary>
        /// Initializes a new instance of the EnterpriseSecurityRelevanceFilter.
        /// </summary>
        /// <param name="logger">Logger instance for this filter.</param>
        public EnterpriseSecurityRelevanceFilter(ILogger<EnterpriseSecurityRelevanceFilter> logger)
        {
            _logger = logger;
            
            // Default security levels - can be overridden by configuration
            _allowedSecurityLevels = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
            {
                "Critical", "High", "Medium"
            };
        }

        /// <summary>
        /// Initializes the filter with configuration settings.
        /// </summary>
        /// <param name="config">Configuration dictionary containing filter settings.</param>
        public void Initialize(Dictionary<string, object> config)
        {
            if (config.TryGetValue("AllowedSecurityLevels", out var levels))
            {
                if (levels is string[] levelArray)
                {
                    _allowedSecurityLevels = new HashSet<string>(levelArray, StringComparer.OrdinalIgnoreCase);
                }
                else if (levels is string levelString)
                {
                    _allowedSecurityLevels = new HashSet<string>(
                        levelString.Split(',').Select(s => s.Trim()),
                        StringComparer.OrdinalIgnoreCase);
                }
            }

            _logger.LogInformation("Security relevance filter initialized with levels: {Levels}", 
                string.Join(", ", _allowedSecurityLevels));
        }

        /// <inheritdoc />
        public Task<bool> ShouldProcessAsync(LogEntry log)
        {
            _processingTimer.Start();
            _logsProcessed++;

            try
            {
                var shouldProcess = _allowedSecurityLevels.Contains(log.SecurityRelevance);
                
                if (shouldProcess)
                {
                    _logsPassed++;
                }

                return Task.FromResult(shouldProcess);
            }
            finally
            {
                _processingTimer.Stop();
            }
        }

        /// <inheritdoc />
        public Dictionary<string, object> GetMetrics()
        {
            return new Dictionary<string, object>
            {
                ["LogsProcessed"] = _logsProcessed,
                ["LogsPassed"] = _logsPassed,
                ["LogsFiltered"] = _logsProcessed - _logsPassed,
                ["FilterEfficiency"] = _logsProcessed > 0 ? (double)(_logsProcessed - _logsPassed) / _logsProcessed * 100 : 0,
                ["AverageProcessingTimeMs"] = _logsProcessed > 0 ? _processingTimer.ElapsedMilliseconds / (double)_logsProcessed : 0,
                ["AllowedSecurityLevels"] = _allowedSecurityLevels.ToArray()
            };
        }
    }

    /// <summary>
    /// Enterprise Windows Event ID filter - COMPLETELY CONFIGURABLE from Backend.
    /// NO HARDCODED VALUES - All Event IDs and categories are fetched from Backend dynamically.
    /// Backend controls what events to monitor via Web Interface.
    /// </summary>
    public class EnterpriseWindowsEventIdFilter : ILogFilter
    {
        private readonly ILogger<EnterpriseWindowsEventIdFilter> _logger;
        private readonly Stopwatch _processingTimer = new();
        private long _logsProcessed;
        private long _logsPassed;
        
        // Backend-configured event monitoring - NO DEFAULTS, NO HARDCODE
        private HashSet<string> _monitoredEventIds = new();
        private bool _collectAllEvents = true; // Default: collect everything like real SIEM
        private bool _filteringEnabled = false; // Default: no filtering (collect all)

        /// <inheritdoc />
        public string Name => "Enterprise Windows Event ID Filter (Backend Configured)";

        /// <inheritdoc />
        public string Description => "Backend-configurable Windows Event ID filtering - NO HARDCODED VALUES";

        /// <inheritdoc />
        public int Priority => 90;

        /// <summary>
        /// Initializes a new instance of the EnterpriseWindowsEventIdFilter.
        /// </summary>
        /// <param name="logger">Logger instance for this filter.</param>
        public EnterpriseWindowsEventIdFilter(ILogger<EnterpriseWindowsEventIdFilter> logger)
        {
            _logger = logger;
            // NO DEFAULT INITIALIZATION - Everything is backend-controlled
            _logger.LogInformation("Windows Event ID filter initialized in 'COLLECT ALL' mode - Backend will configure filtering");
        }

        /// <summary>
        /// Updates filter configuration from backend.
        /// This method is called when backend sends new filtering configuration.
        /// </summary>
        /// <param name="config">Backend configuration containing Event IDs and settings.</param>
        public void UpdateFromBackendConfig(Dictionary<string, object> config)
        {
            try
            {
                _logger.LogInformation("Updating Event ID filter configuration from backend...");

                // Check if filtering is enabled by backend
                if (config.TryGetValue("EnableEventFiltering", out var enableFiltering))
                {
                    _filteringEnabled = ParseBooleanFromConfig(enableFiltering, false);
                }

                // Check if we should collect all events (default SIEM behavior)
                if (config.TryGetValue("CollectAllEvents", out var collectAll))
                {
                    _collectAllEvents = ParseBooleanFromConfig(collectAll, true);
                }

                // Load monitored Event IDs from backend
                if (config.TryGetValue("MonitoredEventIds", out var eventIdsObj))
                {
                    LoadMonitoredEventIds(eventIdsObj);
                }

                var mode = _collectAllEvents ? "COLLECT ALL EVENTS" : $"FILTER MODE - {_monitoredEventIds.Count} Event IDs";
                _logger.LogInformation("Event ID filter updated: {Mode}, Filtering: {Enabled}", mode, _filteringEnabled);

            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error updating event ID filter configuration from backend");
            }
        }

        /// <summary>
        /// Legacy initialize method for local configuration (now deprecated).
        /// Use UpdateFromBackendConfig() instead.
        /// </summary>
        /// <param name="config">Local configuration dictionary.</param>
        public void Initialize(Dictionary<string, object> config)
        {
            _logger.LogWarning("Local Event ID configuration is deprecated. Use Backend configuration instead.");
            
            // For backward compatibility, still load local config if no backend config available
            if (config != null && config.Any())
            {
                UpdateFromBackendConfig(config);
            }
            else
            {
                _logger.LogInformation("No local Event ID configuration - running in 'COLLECT ALL' mode until Backend provides configuration");
            }
        }

        /// <summary>
        /// Loads monitored event IDs from backend configuration.
        /// </summary>
        /// <param name="eventIdsObj">Backend configuration object containing Event IDs.</param>
        private void LoadMonitoredEventIds(object eventIdsObj)
        {
            try
            {
                _monitoredEventIds.Clear();

                if (eventIdsObj is string[] stringArray)
                {
                    foreach (var eventId in stringArray)
                    {
                        if (!string.IsNullOrWhiteSpace(eventId))
                        {
                            _monitoredEventIds.Add(eventId.Trim());
                        }
                    }
                }
                else if (eventIdsObj is System.Text.Json.JsonElement jsonElement && jsonElement.ValueKind == System.Text.Json.JsonValueKind.Array)
                {
                    foreach (var element in jsonElement.EnumerateArray())
                    {
                        if (element.ValueKind == System.Text.Json.JsonValueKind.String)
                        {
                            var eventId = element.GetString();
                            if (!string.IsNullOrWhiteSpace(eventId))
                            {
                                _monitoredEventIds.Add(eventId.Trim());
                            }
                        }
                    }
                }
                else if (eventIdsObj is string csvString)
                {
                    var eventIds = csvString.Split(',');
                    foreach (var eventId in eventIds)
                    {
                        var trimmed = eventId.Trim();
                        if (!string.IsNullOrWhiteSpace(trimmed))
                        {
                            _monitoredEventIds.Add(trimmed);
                        }
                    }
                }
                else if (eventIdsObj is System.Collections.IEnumerable enumerable && !(eventIdsObj is string))
                {
                    foreach (var item in enumerable)
                    {
                        if (item != null)
                        {
                            var eventId = item.ToString()?.Trim();
                            if (!string.IsNullOrWhiteSpace(eventId))
                            {
                                _monitoredEventIds.Add(eventId);
                            }
                        }
                    }
                }

                _logger.LogInformation("Loaded {Count} monitored Event IDs from backend", _monitoredEventIds.Count);
                
                if (_monitoredEventIds.Count > 0)
                {
                    _logger.LogDebug("Monitoring Event IDs: {EventIds}", string.Join(", ", _monitoredEventIds.Take(20)));
                    if (_monitoredEventIds.Count > 20)
                    {
                        _logger.LogDebug("... and {More} more Event IDs", _monitoredEventIds.Count - 20);
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to load monitored Event IDs from backend configuration");
            }
        }

        /// <inheritdoc />
        public Task<bool> ShouldProcessAsync(LogEntry log)
        {
            _processingTimer.Start();
            _logsProcessed++;

            try
            {
                // If filtering is disabled, or collect all mode is enabled, process everything
                if (!_filteringEnabled || _collectAllEvents)
                {
                    _logsPassed++;
                    return Task.FromResult(true);
                }

                // If no Event IDs configured for filtering, collect everything (fail-safe)
                if (_monitoredEventIds.Count == 0)
                {
                    _logsPassed++;
                    return Task.FromResult(true);
                }

                // Process non-Windows events by default (don't filter out non-Windows logs)
                if (string.IsNullOrEmpty(log.EventId))
                {
                    _logsPassed++;
                    return Task.FromResult(true);
                }

                // Check if this Event ID is in the monitored list
                var shouldProcess = _monitoredEventIds.Contains(log.EventId);
                
                if (shouldProcess)
                {
                    _logsPassed++;
                }

                return Task.FromResult(shouldProcess);
            }
            finally
            {
                _processingTimer.Stop();
            }
        }

        /// <inheritdoc />
        public Dictionary<string, object> GetMetrics()
        {
            return new Dictionary<string, object>
            {
                ["LogsProcessed"] = _logsProcessed,
                ["LogsPassed"] = _logsPassed,
                ["LogsFiltered"] = _logsProcessed - _logsPassed,
                ["FilterEfficiency"] = _logsProcessed > 0 ? (double)(_logsProcessed - _logsPassed) / _logsProcessed * 100 : 0,
                ["AverageProcessingTimeMs"] = _logsProcessed > 0 ? _processingTimer.ElapsedMilliseconds / (double)_logsProcessed : 0,
                ["FilteringEnabled"] = _filteringEnabled,
                ["CollectAllEvents"] = _collectAllEvents,
                ["MonitoredEventIdCount"] = _monitoredEventIds.Count,
                ["ConfigurationStatus"] = _filteringEnabled && !_collectAllEvents ? 
                    (_monitoredEventIds.Count > 0 ? "Backend Configured" : "NO EVENT IDS CONFIGURED") : 
                    "COLLECT ALL MODE",
                ["ConfigurationSource"] = "Backend Controlled"
            };
        }

        /// <summary>
        /// Parses a boolean value from various configuration object types.
        /// </summary>
        /// <param name="configObj">The configuration object to parse.</param>
        /// <param name="defaultValue">Default value if parsing fails.</param>
        /// <returns>Parsed boolean value.</returns>
        private bool ParseBooleanFromConfig(object configObj, bool defaultValue)
        {
            try
            {
                return configObj switch
                {
                    null => defaultValue,
                    bool boolValue => boolValue,
                    string stringValue => bool.TryParse(stringValue, out var result) ? result : defaultValue,
                    System.Text.Json.JsonElement jsonElement when jsonElement.ValueKind == System.Text.Json.JsonValueKind.True => true,
                    System.Text.Json.JsonElement jsonElement when jsonElement.ValueKind == System.Text.Json.JsonValueKind.False => false,
                    System.Text.Json.JsonElement jsonElement when jsonElement.ValueKind == System.Text.Json.JsonValueKind.String => 
                        bool.TryParse(jsonElement.GetString(), out var result) ? result : defaultValue,
                    _ => defaultValue
                };
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Failed to parse boolean from config object {Type}: {Value}", 
                    configObj?.GetType().Name ?? "null", configObj?.ToString() ?? "null");
                return defaultValue;
            }
        }
    }

    /// <summary>
    /// Enterprise log level filter that processes logs based on configurable severity levels.
    /// Supports different filtering strategies for various operational modes.
    /// </summary>
    public class EnterpriseLogLevelFilter : ILogFilter
    {
        private readonly ILogger<EnterpriseLogLevelFilter> _logger;
        private readonly Stopwatch _processingTimer = new();
        private long _logsProcessed;
        private long _logsPassed;
        
        private HashSet<string> _allowedLogLevels = new();
        private bool _invertFilter; // If true, filter OUT the specified levels instead of filtering IN

        /// <inheritdoc />
        public string Name => "Enterprise Log Level Filter";

        /// <inheritdoc />
        public string Description => "Filters logs based on configurable log levels with support for inclusion/exclusion modes";

        /// <inheritdoc />
        public int Priority => 85;

        /// <summary>
        /// Initializes a new instance of the EnterpriseLogLevelFilter.
        /// </summary>
        /// <param name="logger">Logger instance for this filter.</param>
        public EnterpriseLogLevelFilter(ILogger<EnterpriseLogLevelFilter> logger)
        {
            _logger = logger;
            
            // Default: Filter out Debug and Verbose levels
            _allowedLogLevels = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
            {
                "Critical", "Error", "Warning", "Information"
            };
        }

        /// <summary>
        /// Initializes the filter with configuration settings.
        /// </summary>
        /// <param name="config">Configuration dictionary containing filter settings.</param>
        public void Initialize(Dictionary<string, object> config)
        {
            if (config.TryGetValue("AllowedLogLevels", out var levels))
            {
                if (levels is string[] levelArray)
                {
                    _allowedLogLevels = new HashSet<string>(levelArray, StringComparer.OrdinalIgnoreCase);
                }
                else if (levels is string levelString)
                {
                    _allowedLogLevels = new HashSet<string>(
                        levelString.Split(',').Select(s => s.Trim()),
                        StringComparer.OrdinalIgnoreCase);
                }
            }

            if (config.TryGetValue("InvertFilter", out var invert) && invert is bool invertValue)
            {
                _invertFilter = invertValue;
            }

            var filterMode = _invertFilter ? "exclusion" : "inclusion";
            _logger.LogInformation("Enterprise log level filter initialized in {FilterMode} mode with levels: {Levels}",
                filterMode, string.Join(", ", _allowedLogLevels));
        }

        /// <inheritdoc />
        public Task<bool> ShouldProcessAsync(LogEntry log)
        {
            _processingTimer.Start();
            _logsProcessed++;

            try
            {
                var levelMatches = _allowedLogLevels.Contains(log.Level);
                var shouldProcess = _invertFilter ? !levelMatches : levelMatches;
                
                if (shouldProcess)
                {
                    _logsPassed++;
                }

                return Task.FromResult(shouldProcess);
            }
            finally
            {
                _processingTimer.Stop();
            }
        }

        /// <inheritdoc />
        public Dictionary<string, object> GetMetrics()
        {
            return new Dictionary<string, object>
            {
                ["LogsProcessed"] = _logsProcessed,
                ["LogsPassed"] = _logsPassed,
                ["LogsFiltered"] = _logsProcessed - _logsPassed,
                ["FilterEfficiency"] = _logsProcessed > 0 ? (double)(_logsProcessed - _logsPassed) / _logsProcessed * 100 : 0,
                ["AverageProcessingTimeMs"] = _logsProcessed > 0 ? _processingTimer.ElapsedMilliseconds / (double)_logsProcessed : 0,
                ["AllowedLogLevels"] = _allowedLogLevels.ToArray(),
                ["FilterMode"] = _invertFilter ? "Exclusion" : "Inclusion"
            };
        }
    }
} 