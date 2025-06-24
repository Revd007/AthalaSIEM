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
    /// Enterprise Windows Event ID filter for configurable security monitoring.
    /// All event IDs and categories are fully configurable via appsettings.json or registry.
    /// NO HARDCODED VALUES - Users have complete control over what gets monitored.
    /// </summary>
    public class EnterpriseWindowsEventIdFilter : ILogFilter
    {
        private readonly ILogger<EnterpriseWindowsEventIdFilter> _logger;
        private readonly Stopwatch _processingTimer = new();
        private long _logsProcessed;
        private long _logsPassed;
        
        // User-configured event ID collections - NO DEFAULTS
        private Dictionary<string, HashSet<string>> _eventIdCategories = new();
        private HashSet<string> _enabledCategories = new();
        private HashSet<string> _allMonitoredEventIds = new();

        /// <inheritdoc />
        public string Name => "Enterprise Windows Event ID Filter";

        /// <inheritdoc />
        public string Description => "Fully configurable Windows Event ID filtering - NO HARDCODED VALUES";

        /// <inheritdoc />
        public int Priority => 90;

        /// <summary>
        /// Initializes a new instance of the EnterpriseWindowsEventIdFilter.
        /// </summary>
        /// <param name="logger">Logger instance for this filter.</param>
        public EnterpriseWindowsEventIdFilter(ILogger<EnterpriseWindowsEventIdFilter> logger)
        {
            _logger = logger;
            // NO DEFAULT INITIALIZATION - Everything must be configured by user
        }

        /// <summary>
        /// Initializes the filter with user-provided configuration.
        /// Configuration must include EventIdCategories and EnabledCategories.
        /// If no configuration is provided, NO events will be processed (fail-secure).
        /// </summary>
        /// <param name="config">Configuration dictionary containing user-defined event ID categories.</param>
        /// <exception cref="InvalidOperationException">Thrown when configuration is invalid or missing.</exception>
        public void Initialize(Dictionary<string, object> config)
        {
            if (config == null || !config.Any())
            {
                _logger.LogWarning("No configuration provided for Windows Event ID filter. NO EVENTS WILL BE PROCESSED.");
                return;
            }

            // Load user-defined event ID categories
            if (config.TryGetValue("EventIdCategories", out var categoriesObj))
            {
                LoadEventIdCategories(categoriesObj);
            }
            else
            {
                _logger.LogWarning("EventIdCategories not configured. NO EVENTS WILL BE PROCESSED.");
                _logger.LogInformation("Configure EventIdCategories in appsettings.json under LogProcessing:Filters:EventIdCategories");
                return;
            }

            // Load enabled categories
            if (config.TryGetValue("EnabledCategories", out var categories))
            {
                LoadEnabledCategories(categories);
            }
            else
            {
                // If no enabled categories specified, enable all configured categories
                _enabledCategories = new HashSet<string>(_eventIdCategories.Keys, StringComparer.OrdinalIgnoreCase);
                _logger.LogInformation("No EnabledCategories specified. Enabling all configured categories.");
            }

            UpdateMonitoredEventIds();

            _logger.LogInformation("Windows Event ID filter initialized with {CategoryCount} categories and {EventIdCount} event IDs",
                _enabledCategories.Count, _allMonitoredEventIds.Count);

            if (_allMonitoredEventIds.Count == 0)
            {
                _logger.LogWarning("NO EVENT IDS CONFIGURED FOR MONITORING. This filter will block all events.");
                _logger.LogInformation("Example configuration:");
                _logger.LogInformation("\"EventIdCategories\": {");
                _logger.LogInformation("  \"Authentication\": [\"4624\", \"4625\", \"4634\"],");
                _logger.LogInformation("  \"AccountManagement\": [\"4720\", \"4722\", \"4723\"]");
                _logger.LogInformation("}");
            }
        }

        /// <summary>
        /// Loads event ID categories from configuration object.
        /// Supports both dictionary and JSON string formats.
        /// </summary>
        /// <param name="categoriesObj">Configuration object containing event ID categories.</param>
        private void LoadEventIdCategories(object categoriesObj)
        {
            try
            {
                Dictionary<string, object>? categoriesDict = null;

                if (categoriesObj is Dictionary<string, object> directDict)
                {
                    categoriesDict = directDict;
                }
                else if (categoriesObj is string jsonString)
                {
                    categoriesDict = System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, object>>(jsonString);
                }

                if (categoriesDict != null)
                {
                    foreach (var kvp in categoriesDict)
                    {
                        var categoryName = kvp.Key;
                        var eventIds = new HashSet<string>();

                        if (kvp.Value is System.Text.Json.JsonElement jsonElement && jsonElement.ValueKind == System.Text.Json.JsonValueKind.Array)
                        {
                            foreach (var element in jsonElement.EnumerateArray())
                            {
                                if (element.ValueKind == System.Text.Json.JsonValueKind.String)
                                {
                                    var eventId = element.GetString();
                                    if (!string.IsNullOrEmpty(eventId))
                                    {
                                        eventIds.Add(eventId);
                                    }
                                }
                            }
                        }
                        else if (kvp.Value is string[] stringArray)
                        {
                            foreach (var eventId in stringArray)
                            {
                                if (!string.IsNullOrEmpty(eventId))
                                {
                                    eventIds.Add(eventId);
                                }
                            }
                        }
                        else if (kvp.Value is string csvString)
                        {
                            var eventIdArray = csvString.Split(',');
                            foreach (var eventId in eventIdArray)
                            {
                                var trimmed = eventId.Trim();
                                if (!string.IsNullOrEmpty(trimmed))
                                {
                                    eventIds.Add(trimmed);
                                }
                            }
                        }

                        if (eventIds.Any())
                        {
                            _eventIdCategories[categoryName] = eventIds;
                            _logger.LogDebug("Loaded category '{Category}' with {Count} event IDs", categoryName, eventIds.Count);
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to load event ID categories from configuration");
            }
        }

        /// <summary>
        /// Loads enabled categories from configuration.
        /// </summary>
        /// <param name="categories">Configuration object containing enabled categories.</param>
        private void LoadEnabledCategories(object categories)
        {
            if (categories is string[] categoryArray)
            {
                _enabledCategories = new HashSet<string>(categoryArray, StringComparer.OrdinalIgnoreCase);
            }
            else if (categories is string categoryString)
            {
                _enabledCategories = new HashSet<string>(
                    categoryString.Split(',').Select(s => s.Trim()).Where(s => !string.IsNullOrEmpty(s)),
                    StringComparer.OrdinalIgnoreCase);
            }
            else if (categories is System.Text.Json.JsonElement jsonElement && jsonElement.ValueKind == System.Text.Json.JsonValueKind.Array)
            {
                _enabledCategories = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
                foreach (var element in jsonElement.EnumerateArray())
                {
                    if (element.ValueKind == System.Text.Json.JsonValueKind.String)
                    {
                        var category = element.GetString();
                        if (!string.IsNullOrEmpty(category))
                        {
                            _enabledCategories.Add(category);
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Updates the collection of all monitored event IDs based on enabled categories.
        /// </summary>
        private void UpdateMonitoredEventIds()
        {
            _allMonitoredEventIds.Clear();
            foreach (var category in _enabledCategories)
            {
                if (_eventIdCategories.TryGetValue(category, out var eventIds))
                {
                    foreach (var eventId in eventIds)
                    {
                        _allMonitoredEventIds.Add(eventId);
                    }
                }
                else
                {
                    _logger.LogWarning("Enabled category '{Category}' not found in configured event ID categories", category);
                }
            }
        }

        /// <inheritdoc />
        public Task<bool> ShouldProcessAsync(LogEntry log)
        {
            _processingTimer.Start();
            _logsProcessed++;

            try
            {
                // If no event IDs are configured, process nothing (fail-secure)
                if (_allMonitoredEventIds.Count == 0)
                {
                    return Task.FromResult(false);
                }

                // Process all non-Windows events by default
                if (string.IsNullOrEmpty(log.EventId))
                {
                    _logsPassed++;
                    return Task.FromResult(true);
                }

                var shouldProcess = _allMonitoredEventIds.Contains(log.EventId);
                
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
                ["EnabledCategories"] = _enabledCategories.ToArray(),
                ["MonitoredEventIdCount"] = _allMonitoredEventIds.Count,
                ["ConfiguredCategories"] = _eventIdCategories.Keys.ToArray(),
                ["ConfigurationStatus"] = _allMonitoredEventIds.Count > 0 ? "Configured" : "NOT CONFIGURED - NO EVENTS WILL BE PROCESSED"
            };
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