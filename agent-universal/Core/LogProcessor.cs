using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Configuration;
using AthalaSIEM.UniversalAgent.Models;
using AthalaSIEM.UniversalAgent.Core.Filters;
using AthalaSIEM.UniversalAgent.Core.Enrichers;
using AthalaSIEM.UniversalAgent.Core.Correlators;

namespace AthalaSIEM.UniversalAgent.Core
{
    /// <summary>
    /// Enterprise log processor implementing ManageEngine EventLog Analyzer processing pipeline.
    /// Provides configurable, secure, and scalable log processing with comprehensive documentation.
    /// Architecture: Raw Logs → Security Filters → Parser → Enrichment → Indexing → Correlation
    /// </summary>
    public class LogProcessor : IAsyncDisposable
    {
        private readonly ILogger<LogProcessor> _logger;
        private readonly ILoggerFactory _loggerFactory;
        private readonly IConfiguration _configuration;
        private readonly List<ILogFilter> _securityFilters = new();
        private readonly List<ILogEnricher> _enrichers = new();
        private readonly List<ILogCorrelator> _correlators = new();
        private readonly Dictionary<string, List<LogEntry>> _correlationBuffer = new();
        private Timer _correlationTimer;
        private readonly object _processingLock = new();
        private LogProcessingConfiguration _processingConfig;

        /// <summary>
        /// Gets a value indicating whether the processor is currently processing logs.
        /// </summary>
        public bool IsProcessing { get; private set; }

        /// <summary>
        /// Gets the total number of logs processed since startup.
        /// </summary>
        public long ProcessedLogs { get; private set; }

        /// <summary>
        /// Gets the total number of logs filtered out since startup.
        /// </summary>
        public long FilteredLogs { get; private set; }

        /// <summary>
        /// Gets a value indicating whether the processor is initialized and ready.
        /// </summary>
        public bool IsInitialized { get; private set; }

        /// <summary>
        /// Event raised when a batch of logs has been processed.
        /// </summary>
        public event EventHandler<LogProcessedEventArgs>? LogProcessed;

        /// <summary>
        /// Event raised when a security correlation is detected.
        /// </summary>
        public event EventHandler<CorrelationDetectedEventArgs>? CorrelationDetected;

        /// <summary>
        /// Event raised when an error occurs during processing.
        /// </summary>
        public event EventHandler<LogProcessingErrorEventArgs>? ProcessingError;

        /// <summary>
        /// Event raised when backend configuration is updated.
        /// </summary>
        public event EventHandler<ConfigurationUpdatedEventArgs>? ConfigurationUpdated;

        /// <summary>
        /// Initializes a new instance of the LogProcessor class.
        /// </summary>
        /// <param name="logger">Logger instance for this processor.</param>
        /// <param name="loggerFactory">Logger factory for creating specific loggers.</param>
        /// <param name="configuration">Configuration provider for settings.</param>
        public LogProcessor(ILogger<LogProcessor> logger, ILoggerFactory loggerFactory, IConfiguration configuration)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _loggerFactory = loggerFactory ?? throw new ArgumentNullException(nameof(loggerFactory));
            _configuration = configuration ?? throw new ArgumentNullException(nameof(configuration));
            
            // Load processing configuration from settings
            _processingConfig = LoadProcessingConfiguration();
            
            // Setup correlation timer with configured interval
            var intervalSeconds = _processingConfig.CorrelationIntervalSeconds;
            _correlationTimer = new Timer(ProcessCorrelations, null, 
                TimeSpan.FromSeconds(intervalSeconds), 
                TimeSpan.FromSeconds(intervalSeconds));

            _logger.LogInformation("LogProcessor initialized - Backend configuration support enabled");
        }

        /// <summary>
        /// Initializes the log processor with configured filters, enrichers, and correlators.
        /// Must be called before processing any logs.
        /// </summary>
        /// <returns>Task representing the initialization operation.</returns>
        public async Task<bool> InitializeAsync()
        {
            try
            {
                _logger.LogInformation("Initializing enterprise log processor...");

                // Initialize filters
                await InitializeSecurityFiltersAsync();
            
            // Initialize enrichers
                await InitializeEnrichersAsync();
            
            // Initialize correlators
                await InitializeCorrelatorsAsync();
                
                IsInitialized = true;
                _logger.LogInformation("Enterprise log processor initialized successfully with {FilterCount} filters, {EnricherCount} enrichers, {CorrelatorCount} correlators",
                    _securityFilters.Count, _enrichers.Count, _correlators.Count);
                
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to initialize log processor");
                IsInitialized = false;
                return false;
            }
        }

        /// <summary>
        /// Processes a batch of logs through the enterprise ManageEngine-style pipeline.
        /// </summary>
        /// <param name="logs">The logs to process.</param>
        /// <returns>Processed log batch with results and metadata.</returns>
        public async Task<ProcessedLogBatch> ProcessLogBatchAsync(IEnumerable<LogEntry> logs)
        {
            if (!IsInitialized)
            {
                throw new InvalidOperationException("LogProcessor must be initialized before processing logs. Call InitializeAsync() first.");
            }

            var startTime = DateTime.UtcNow;
            var processedBatch = new ProcessedLogBatch();
            
            try
            {
                lock (_processingLock)
                {
                    IsProcessing = true;
                }

                var logArray = logs.ToArray();
                _logger.LogDebug("Processing batch of {LogCount} logs", logArray.Length);

                foreach (var log in logArray)
                {
                    try
                    {
                        var processedLog = await ProcessSingleLogAsync(log);
                        if (processedLog != null)
                        {
                            processedBatch.ProcessedLogs.Add(processedLog);
                            ProcessedLogs++;
                        }
                        else
                        {
                            FilteredLogs++;
                        }
                    }
                    catch (Exception ex)
                    {
                        _logger.LogError(ex, "Error processing log: {Message}", log.Message);
                        processedBatch.Errors.Add($"Error processing log: {ex.Message}");
                        
                        // Fire error event
                        ProcessingError?.Invoke(this, new LogProcessingErrorEventArgs
                        {
                            Exception = ex,
                            ErrorMessage = $"Error processing log: {ex.Message}",
                            FailedLog = log,
                            ProcessingStage = "LogProcessing",
                            Severity = "Medium"
                        });
                    }
                }

                // Add to correlation buffer if correlation is enabled
                if (_processingConfig.EnableCorrelation)
                {
                AddToCorrelationBuffer(processedBatch.ProcessedLogs);
                }

                // Set batch metadata
                var endTime = DateTime.UtcNow;
                processedBatch.ProcessingTime = endTime;
                processedBatch.TotalProcessed = processedBatch.ProcessedLogs.Count;
                processedBatch.ProcessingDurationMs = (long)(endTime - startTime).TotalMilliseconds;
                processedBatch.ProcessingStats = new Dictionary<string, object>
                {
                    ["InputLogCount"] = logArray.Length,
                    ["ProcessedLogCount"] = processedBatch.ProcessedLogs.Count,
                    ["FilteredLogCount"] = logArray.Length - processedBatch.ProcessedLogs.Count,
                    ["ErrorCount"] = processedBatch.Errors.Count,
                    ["ProcessingDurationMs"] = processedBatch.ProcessingDurationMs
                };
                
                // Fire event
                LogProcessed?.Invoke(this, new LogProcessedEventArgs 
                { 
                    ProcessedBatch = processedBatch,
                    ProcessingStage = "Complete",
                    AgentId = Environment.MachineName
                });
                
                return processedBatch;
            }
            finally
            {
                lock (_processingLock)
                {
                    IsProcessing = false;
                }
            }
        }

        /// <summary>
        /// Process a single log through the complete pipeline
        /// </summary>
        private async Task<LogEntry?> ProcessSingleLogAsync(LogEntry log)
        {
            // Step 1: Apply security filters (ManageEngine pattern)
            if (!await ApplySecurityFiltersAsync(log))
            {
                return null; // Log filtered out
            }

            // Step 2: Parse and normalize (ManageEngine parsing pattern)
            await ParseAndNormalizeAsync(log);

            // Step 3: Enrich with context (ManageEngine enrichment pattern)
            await EnrichLogAsync(log);

            // Step 4: Create search index (ManageEngine indexing pattern)
            await CreateSearchIndexAsync(log);

            // Step 5: Generate integrity hash
            await GenerateLogHashAsync(log);

            return log;
        }

        /// <summary>
        /// Applies security-focused filters through the enterprise filter pipeline.
        /// Filters are executed in priority order to optimize performance.
        /// </summary>
        /// <param name="log">The log entry to filter.</param>
        /// <returns>True if the log should be processed, false if filtered out.</returns>
        private async Task<bool> ApplySecurityFiltersAsync(LogEntry log)
        {
            foreach (var filter in _securityFilters.OrderByDescending(f => f.Priority))
            {
                try
            {
                if (!await filter.ShouldProcessAsync(log))
                {
                    _logger.LogDebug("Log filtered by {FilterName}: {LogMessage}", filter.Name, log.Message);
                    return false;
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Error in filter {FilterName}, allowing log to pass", filter.Name);
                    // Continue processing if filter fails (fail-open for availability)
                }
            }
            return true;
        }

        /// <summary>
        /// Parse and normalize log following ManageEngine's parsing pattern
        /// Breaks down logs into structured, searchable components
        /// </summary>
        private async Task ParseAndNormalizeAsync(LogEntry log)
        {
            // Extract structured data based on log type
            if (log is WindowsLogEntry windowsLog)
            {
                await ParseWindowsEventAsync(windowsLog);
            }
            else if (log is SyslogEntry syslogEntry)
            {
                await ParseSyslogAsync(syslogEntry);
            }
            else if (log is IISLogEntry iisLog)
            {
                await ParseIISLogAsync(iisLog);
            }

            // Normalize timestamp to UTC
            if (log.Timestamp.Kind != DateTimeKind.Utc)
            {
                log.Timestamp = log.Timestamp.ToUniversalTime();
            }

            // Normalize level
            log.Level = NormalizeLogLevel(log.Level);
        }

        /// <summary>
        /// Enriches log with additional context through the enterprise enrichment pipeline.
        /// Enrichers are executed in priority order with error handling.
        /// </summary>
        /// <param name="log">The log entry to enrich.</param>
        /// <returns>Task representing the enrichment operation.</returns>
        private async Task EnrichLogAsync(LogEntry log)
        {
            foreach (var enricher in _enrichers.OrderByDescending(e => e.Priority))
            {
                try
            {
                await enricher.EnrichAsync(log);
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Error in enricher {EnricherName}, continuing with next enricher", enricher.Name);
                    // Continue with other enrichers if one fails
                }
            }
        }

        /// <summary>
        /// Create search index for fast querying (ManageEngine search pattern)
        /// </summary>
        private Task CreateSearchIndexAsync(LogEntry log)
        {
            var indexBuilder = new StringBuilder();
            
            // Add main fields to index
            indexBuilder.Append($"{log.Source} {log.Level} {log.Category} {log.Message} ");
            
            // Add properties to index
            foreach (var prop in log.Properties)
            {
                indexBuilder.Append($"{prop.Key}:{prop.Value} ");
            }
            
            // Add parsed fields
            if (!string.IsNullOrEmpty(log.ComputerName))
                indexBuilder.Append($"computer:{log.ComputerName} ");
            if (!string.IsNullOrEmpty(log.Username))
                indexBuilder.Append($"user:{log.Username} ");
            if (!string.IsNullOrEmpty(log.IpAddress))
                indexBuilder.Append($"ip:{log.IpAddress} ");

            log.SearchIndex = indexBuilder.ToString().ToLowerInvariant().Trim();
            
            return Task.CompletedTask;
        }

        /// <summary>
        /// Generate integrity hash for log verification
        /// </summary>
        private Task GenerateLogHashAsync(LogEntry log)
        {
            var logData = JsonSerializer.Serialize(new
            {
                log.Timestamp,
                log.Source,
                log.Level,
                log.Message,
                log.EventId,
                log.ComputerName,
                log.Username
            });

            using var sha256 = SHA256.Create();
            var hashBytes = sha256.ComputeHash(Encoding.UTF8.GetBytes(logData));
            log.LogHash = Convert.ToBase64String(hashBytes);
            
            return Task.CompletedTask;
        }

        /// <summary>
        /// Adds logs to correlation buffer for attack chain detection.
        /// Buffer is organized by entity key (computer/user combination) for efficient correlation.
        /// </summary>
        /// <param name="logs">The logs to add to the correlation buffer.</param>
        private void AddToCorrelationBuffer(List<LogEntry> logs)
        {
            foreach (var log in logs)
            {
                var key = GenerateCorrelationKey(log);
                if (!_correlationBuffer.ContainsKey(key))
                {
                    _correlationBuffer[key] = new List<LogEntry>();
                }
                
                _correlationBuffer[key].Add(log);
                
                // Keep buffer size manageable (configurable)
                var maxBufferSize = _processingConfig.CorrelationBufferSize;
                if (_correlationBuffer[key].Count > maxBufferSize)
                {
                    var removeCount = maxBufferSize / 2;
                    _correlationBuffer[key].RemoveRange(0, removeCount);
                }
            }
        }

        /// <summary>
        /// Processes correlations to detect attack chains using enterprise correlation algorithms.
        /// Correlations are processed periodically to detect multi-event attack patterns.
        /// </summary>
        /// <param name="state">Timer state (not used).</param>
        private void ProcessCorrelations(object? state)
        {
            if (!_processingConfig.EnableCorrelation || !_correlators.Any())
            {
                return;
            }

            try
            {
                _logger.LogDebug("Processing correlations for {BufferCount} entity buffers", _correlationBuffer.Count);
                var correlationsDetected = 0;

                foreach (var correlator in _correlators)
                {
                    try
                    {
                        foreach (var bufferEntry in _correlationBuffer.ToList()) // ToList to avoid collection modification
                    {
                        var correlations = correlator.DetectCorrelations(bufferEntry.Value);
                        foreach (var correlation in correlations)
                        {
                                if (correlation.ConfidenceScore >= correlator.MinimumConfidence)
                                {
                                    correlationsDetected++;
                                    
                            CorrelationDetected?.Invoke(this, new CorrelationDetectedEventArgs
                            {
                                Correlation = correlation,
                                        DetectedAt = DateTime.UtcNow,
                                        DetectorName = correlator.Name,
                                        IsHighPriority = correlation.Severity == "Critical" || correlation.Severity == "High",
                                        AgentId = Environment.MachineName
                            });
                        }
                    }
                        }
                    }
                    catch (Exception ex)
                    {
                        _logger.LogWarning(ex, "Error in correlator {CorrelatorName}", correlator.Name);
                    }
                }

                if (correlationsDetected > 0)
                {
                    _logger.LogInformation("Processed correlations: {CorrelationCount} patterns detected", correlationsDetected);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error processing correlations");
            }
        }

        /// <summary>
        /// Generates a correlation key for grouping related logs.
        /// </summary>
        /// <param name="log">The log entry to generate a key for.</param>
        /// <returns>Correlation key string.</returns>
        private string GenerateCorrelationKey(LogEntry log)
        {
            var computer = log.ComputerName ?? "Unknown";
            var user = log.Username ?? "Unknown";
            var ip = log.IpAddress ?? "Unknown";
            
            return $"{computer}_{user}_{ip}";
        }

        /// <summary>
        /// Updates processing configuration from backend.
        /// This replaces hardcoded thresholds with dynamic backend-controlled settings.
        /// </summary>
        /// <param name="configType">Type of configuration being updated.</param>
        /// <param name="config">Backend configuration data.</param>
        /// <returns>True if configuration was successfully applied.</returns>
        public async Task<bool> UpdateFromBackendConfigAsync(string configType, Dictionary<string, object> config)
        {
            try
            {
                _logger.LogInformation("Updating {ConfigType} configuration from backend...", configType);

                switch (configType)
                {
                    case Constants.BackendConfig.ConfigurationTypeEventFiltering:
                        await UpdateEventFilteringConfigAsync(config);
                        break;

                    case Constants.BackendConfig.ConfigurationTypeDetectionThresholds:
                        await UpdateDetectionThresholdsAsync(config);
                        break;

                    case Constants.BackendConfig.ConfigurationTypeMonitoring:
                        await UpdateMonitoringSettingsAsync(config);
                        break;

                    default:
                        _logger.LogWarning("Unknown configuration type: {ConfigType}", configType);
                        return false;
                }

                // Fire configuration updated event
                ConfigurationUpdated?.Invoke(this, new ConfigurationUpdatedEventArgs
                {
                    ConfigurationType = configType,
                    Configuration = config,
                    UpdateTime = DateTime.UtcNow,
                    Success = true
                });

                _logger.LogInformation(" {ConfigType} configuration updated successfully", configType);
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to update {ConfigType} configuration from backend", configType);
                
                ConfigurationUpdated?.Invoke(this, new ConfigurationUpdatedEventArgs
                {
                    ConfigurationType = configType,
                    Configuration = config,
                    UpdateTime = DateTime.UtcNow,
                    Success = false,
                    Error = ex.Message
                });

                return false;
            }
        }

        /// <summary>
        /// Updates event filtering configuration from backend.
        /// </summary>
        /// <param name="config">Backend event filtering configuration.</param>
        /// <returns>Task representing the update operation.</returns>
        private async Task UpdateEventFilteringConfigAsync(Dictionary<string, object> config)
        {
            try
            {
                _logger.LogDebug("Updating event filtering configuration...");

                // Find and update the Event ID filter
                var eventIdFilter = _securityFilters.OfType<EnterpriseWindowsEventIdFilter>().FirstOrDefault();
                if (eventIdFilter != null)
                {
                    eventIdFilter.UpdateFromBackendConfig(config);
                    _logger.LogInformation("Event ID filter updated with backend configuration");
                }
                else
                {
                    _logger.LogWarning("Event ID filter not found - creating new instance");
                    var newFilter = new EnterpriseWindowsEventIdFilter(_loggerFactory.CreateLogger<EnterpriseWindowsEventIdFilter>());
                    newFilter.UpdateFromBackendConfig(config);
                    _securityFilters.Add(newFilter);
                    
                    // Re-sort filters by priority
                    _securityFilters.Sort((x, y) => y.Priority.CompareTo(x.Priority));
                }

                await Task.CompletedTask;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error updating event filtering configuration");
                throw;
            }
        }

        /// <summary>
        /// Updates detection thresholds from backend configuration.
        /// This replaces ALL hardcoded detection values with backend-controlled settings.
        /// </summary>
        /// <param name="config">Backend detection thresholds configuration.</param>
        /// <returns>Task representing the update operation.</returns>
        private async Task UpdateDetectionThresholdsAsync(Dictionary<string, object> config)
        {
            try
            {
                _logger.LogDebug("Updating detection thresholds from backend...");

                // Update processing configuration thresholds
                foreach (var kvp in config)
                {
                    if (int.TryParse(kvp.Value.ToString(), out var threshold))
                    {
                        _processingConfig.DetectionThresholds[kvp.Key] = threshold;
                        _logger.LogDebug("Updated threshold: {Key} = {Value}", kvp.Key, threshold);
                    }
                }

                // Update correlators with new thresholds and preserve UAT test environment flags
                var thresholdConfig = new Dictionary<string, object>(_processingConfig.DetectionThresholds.ToDictionary(k => k.Key, v => (object)v.Value));
                
                // Include UAT test environment flags if they exist in the original config
                foreach (var kvp in config)
                {
                    if (kvp.Key.Contains("TestMode", StringComparison.OrdinalIgnoreCase) ||
                        kvp.Key.Contains("UAT", StringComparison.OrdinalIgnoreCase) ||
                        kvp.Key.Contains("IsUATEnvironment", StringComparison.OrdinalIgnoreCase))
                    {
                        thresholdConfig[kvp.Key] = kvp.Value;
                        _logger.LogDebug("Preserving UAT test flag: {Key} = {Value}", kvp.Key, kvp.Value);
                    }
                }
                
                foreach (var correlator in _correlators)
                {
                    try
                    {
                        await correlator.InitializeAsync(thresholdConfig);
                        _logger.LogDebug("Updated correlator: {CorrelatorName}", correlator.Name);
                    }
                    catch (Exception ex)
                    {
                        _logger.LogWarning(ex, "Failed to update correlator {CorrelatorName} with new thresholds", correlator.Name);
                    }
                }

                _logger.LogInformation("Detection thresholds updated: {Count} thresholds configured", _processingConfig.DetectionThresholds.Count);
                await Task.CompletedTask;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error updating detection thresholds");
                throw;
            }
        }

        /// <summary>
        /// Updates monitoring settings from backend configuration.
        /// </summary>
        /// <param name="config">Backend monitoring configuration.</param>
        /// <returns>Task representing the update operation.</returns>
        private async Task UpdateMonitoringSettingsAsync(Dictionary<string, object> config)
        {
            try
            {
                _logger.LogDebug("Updating monitoring settings from backend...");

                // Update correlation settings
                if (config.TryGetValue("CorrelationIntervalSeconds", out var correlationInterval) && 
                    int.TryParse(correlationInterval.ToString(), out var interval))
                {
                    _processingConfig.CorrelationIntervalSeconds = interval;
                    
                    // Update correlation timer
                    _correlationTimer?.Dispose();
                    _correlationTimer = new Timer(ProcessCorrelations, null, 
                        TimeSpan.FromSeconds(interval), 
                        TimeSpan.FromSeconds(interval));
                    
                    _logger.LogInformation("Correlation interval updated to {Interval} seconds", interval);
                }

                if (config.TryGetValue("CorrelationBufferSize", out var bufferSize) && 
                    int.TryParse(bufferSize.ToString(), out var size))
                {
                    _processingConfig.CorrelationBufferSize = size;
                    _logger.LogInformation("Correlation buffer size updated to {Size}", size);
                }

                if (config.TryGetValue("EnableCorrelation", out var enableCorrelation) && 
                    bool.TryParse(enableCorrelation.ToString(), out var enable))
                {
                    _processingConfig.EnableCorrelation = enable;
                    _logger.LogInformation("Correlation enabled: {Enabled}", enable);
                }

                await Task.CompletedTask;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error updating monitoring settings");
                throw;
            }
        }

        #region Initialization Methods

        /// <summary>
        /// Loads processing configuration from the configuration provider.
        /// Now uses minimal defaults - backend will provide actual configuration.
        /// </summary>
        /// <returns>Log processing configuration instance.</returns>
        private LogProcessingConfiguration LoadProcessingConfiguration()
        {
            var config = new LogProcessingConfiguration();

            // Load configuration from appsettings.json
            var processingSection = _configuration.GetSection("LogProcessing");
            if (processingSection.Exists())
            {
                processingSection.Bind(config);
            }

            // Minimal default values - backend will override these
            if (config.DetectionThresholds.Count == 0)
            {
                _logger.LogInformation("No local detection thresholds configured - using minimal defaults until backend provides configuration");
                config.DetectionThresholds = new Dictionary<string, int>
                {
                    ["BruteForceThreshold"] = 5,
                    ["CredentialStuffingThreshold"] = 10,
                    ["PrivilegeEscalationThreshold"] = 3,
                    ["TimeWindowMinutes"] = 15
                };
            }

            // Log configuration source
            _logger.LogInformation("Processing configuration loaded - Backend will provide final configuration");
            return config;
        }

        /// <summary>
        /// Initializes security filters with enterprise-grade implementations.
        /// </summary>
        /// <returns>Task representing the initialization operation.</returns>
        private async Task InitializeSecurityFiltersAsync()
        {
            try
        {
                _logger.LogDebug("Initializing security filters...");

                // Get filter configuration - use a more robust approach
                var filterConfig = new Dictionary<string, object>();
                
                // Load security relevance levels
                var allowedLevels = _configuration.GetSection("LogProcessing:Filters:AllowedSecurityLevels").Get<string[]>();
                if (allowedLevels != null)
                {
                    filterConfig["AllowedSecurityLevels"] = allowedLevels;
                }

                // Load Event ID categories and enabled categories
                var eventIdCategoriesSection = _configuration.GetSection("LogProcessing:Filters:EventIdCategories");
                if (eventIdCategoriesSection.Exists())
                {
                    var eventIdCategories = new Dictionary<string, object>();
                    foreach (var categorySection in eventIdCategoriesSection.GetChildren())
                    {
                        var eventIds = categorySection.Get<string[]>();
                        if (eventIds != null)
                        {
                            eventIdCategories[categorySection.Key] = eventIds;
                        }
                    }
                    filterConfig["EventIdCategories"] = eventIdCategories;
                    
                    _logger.LogDebug("Loaded {CategoryCount} event ID categories", eventIdCategories.Count);
                }

                var enabledCategories = _configuration.GetSection("LogProcessing:Filters:EnabledCategories").Get<string[]>();
                if (enabledCategories != null)
                {
                    filterConfig["EnabledCategories"] = enabledCategories;
                    _logger.LogDebug("Enabled categories: {Categories}", string.Join(", ", enabledCategories));
                }

                // Initialize enterprise filters
                var securityRelevanceFilter = new EnterpriseSecurityRelevanceFilter(_loggerFactory.CreateLogger<EnterpriseSecurityRelevanceFilter>());
                securityRelevanceFilter.Initialize(filterConfig);
                _securityFilters.Add(securityRelevanceFilter);

                var eventIdFilter = new EnterpriseWindowsEventIdFilter(_loggerFactory.CreateLogger<EnterpriseWindowsEventIdFilter>());
                eventIdFilter.Initialize(filterConfig);
                _securityFilters.Add(eventIdFilter);

                // Sort filters by priority (descending)
                _securityFilters.Sort((x, y) => y.Priority.CompareTo(x.Priority));

                _logger.LogInformation("Initialized {FilterCount} security filters", _securityFilters.Count);

                await Task.CompletedTask;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error initializing security filters");
                throw;
            }
        }

        /// <summary>
        /// Initializes enrichers with enterprise threat intelligence and asset data.
        /// </summary>
        /// <returns>Task representing the initialization operation.</returns>
        private async Task InitializeEnrichersAsync()
        {
            try
        {
                _logger.LogDebug("Initializing enrichers...");

                // Get enricher configuration
                var enricherConfig = _configuration.GetSection("LogProcessing:Enrichers").Get<Dictionary<string, object>>()
                    ?? new Dictionary<string, object>();

                // Initialize enterprise enrichers
                var geoIpEnricher = new EnterpriseGeoIpEnricher(_loggerFactory.CreateLogger<EnterpriseGeoIpEnricher>());
                await geoIpEnricher.InitializeAsync(enricherConfig);
                _enrichers.Add(geoIpEnricher);

                // Sort enrichers by priority (descending)
                _enrichers.Sort((x, y) => y.Priority.CompareTo(x.Priority));

                _logger.LogInformation("Initialized {EnricherCount} enrichers", _enrichers.Count);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error initializing enrichers");
                throw;
            }
        }

        /// <summary>
        /// Initializes correlators with advanced attack detection capabilities.
        /// </summary>
        /// <returns>Task representing the initialization operation.</returns>
        private async Task InitializeCorrelatorsAsync()
        {
            try
        {
                _logger.LogDebug("Initializing correlators...");

                // Get correlator configuration
                var correlatorConfig = _configuration.GetSection("LogProcessing:Correlators").Get<Dictionary<string, object>>()
                    ?? new Dictionary<string, object>();

                // Merge detection thresholds into correlator config
                foreach (var threshold in _processingConfig.DetectionThresholds)
                {
                    correlatorConfig[threshold.Key] = threshold.Value;
                }

                // Initialize enterprise correlators
                var authCorrelator = new EnterpriseAuthenticationCorrelator(_loggerFactory.CreateLogger<EnterpriseAuthenticationCorrelator>());
                await authCorrelator.InitializeAsync(correlatorConfig);
                _correlators.Add(authCorrelator);

                var privEscCorrelator = new EnterprisePrivilegeEscalationCorrelator(_loggerFactory.CreateLogger<EnterprisePrivilegeEscalationCorrelator>());
                await privEscCorrelator.InitializeAsync(correlatorConfig);
                _correlators.Add(privEscCorrelator);

                _logger.LogInformation("Initialized {CorrelatorCount} correlators", _correlators.Count);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error initializing correlators");
                throw;
            }
        }

        #endregion

        #region Helper Methods

        private Task ParseWindowsEventAsync(WindowsLogEntry log)
        {
            // Extract Windows-specific information
            if (log.Properties.ContainsKey("TargetUserName"))
            {
                log.TargetUserName = log.Properties["TargetUserName"]?.ToString();
                log.Username = log.TargetUserName;
            }

            if (log.Properties.ContainsKey("WorkstationName"))
            {
                log.WorkstationName = log.Properties["WorkstationName"]?.ToString();
                log.ComputerName = log.WorkstationName;
            }

            if (log.Properties.ContainsKey("IpAddress"))
            {
                log.IpAddress = log.Properties["IpAddress"]?.ToString();
            }
            
            return Task.CompletedTask;
        }

        private Task ParseSyslogAsync(SyslogEntry log)
        {
            // Parse syslog structured data
            log.ComputerName = log.Hostname;
            log.ProcessName = log.AppName;
            
            return Task.CompletedTask;
        }

        private Task ParseIISLogAsync(IISLogEntry log)
        {
            // Parse IIS-specific fields
            log.IpAddress = log.ClientIP;
            log.Username = log.Username;
            log.ComputerName = log.Properties.ContainsKey("ServerName") 
                ? log.Properties["ServerName"]?.ToString() 
                : Environment.MachineName;
                
            return Task.CompletedTask;
        }

        private string NormalizeLogLevel(string level)
        {
            return level?.ToUpperInvariant() switch
            {
                "VERBOSE" or "TRACE" or "DEBUG" => "Debug",
                "INFO" or "INFORMATION" => "Information",
                "WARN" or "WARNING" => "Warning",
                "ERR" or "ERROR" => "Error",
                "CRIT" or "CRITICAL" or "FATAL" => "Critical",
                _ => "Information"
            };
        }

        #endregion

        /// <summary>
        /// Gets comprehensive metrics and health information for the log processor.
        /// </summary>
        /// <returns>Dictionary containing processor metrics and statistics.</returns>
        public Dictionary<string, object> GetMetrics()
        {
            var metrics = new Dictionary<string, object>
            {
                ["IsInitialized"] = IsInitialized,
                ["IsProcessing"] = IsProcessing,
                ["ProcessedLogs"] = ProcessedLogs,
                ["FilteredLogs"] = FilteredLogs,
                ["TotalLogs"] = ProcessedLogs + FilteredLogs,
                ["FilterEfficiency"] = ProcessedLogs + FilteredLogs > 0 ? 
                    (double)FilteredLogs / (ProcessedLogs + FilteredLogs) * 100 : 0,
                ["FilterCount"] = _securityFilters.Count,
                ["EnricherCount"] = _enrichers.Count,
                ["CorrelatorCount"] = _correlators.Count,
                ["CorrelationBufferEntities"] = _correlationBuffer.Count,
                ["CorrelationBufferSize"] = _correlationBuffer.Values.Sum(v => v.Count),
                ["Configuration"] = new Dictionary<string, object>
                {
                    ["CorrelationEnabled"] = _processingConfig.EnableCorrelation,
                    ["CorrelationIntervalSeconds"] = _processingConfig.CorrelationIntervalSeconds,
                    ["CorrelationBufferSize"] = _processingConfig.CorrelationBufferSize,
                    ["MinimumSecurityRelevance"] = _processingConfig.MinimumSecurityRelevance,
                    ["ConfigurationSource"] = "Backend Controlled",
                    ["DetectionThresholds"] = _processingConfig.DetectionThresholds
                }
            };

            // Add filter metrics
            var filterMetrics = new Dictionary<string, object>();
            foreach (var filter in _securityFilters)
            {
                try
                {
                    filterMetrics[filter.Name] = filter.GetMetrics();
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Error getting metrics from filter {FilterName}", filter.Name);
                    filterMetrics[filter.Name] = new Dictionary<string, object> { ["Error"] = ex.Message };
                }
            }
            metrics["FilterMetrics"] = filterMetrics;

            // Add enricher metrics
            var enricherMetrics = new Dictionary<string, object>();
            foreach (var enricher in _enrichers)
            {
                try
                {
                    enricherMetrics[enricher.Name] = enricher.GetMetrics();
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Error getting metrics from enricher {EnricherName}", enricher.Name);
                    enricherMetrics[enricher.Name] = new Dictionary<string, object> { ["Error"] = ex.Message };
                }
            }
            metrics["EnricherMetrics"] = enricherMetrics;

            // Add correlator metrics
            var correlatorMetrics = new Dictionary<string, object>();
            foreach (var correlator in _correlators)
            {
                try
                {
                    correlatorMetrics[correlator.Name] = correlator.GetMetrics();
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Error getting metrics from correlator {CorrelatorName}", correlator.Name);
                    correlatorMetrics[correlator.Name] = new Dictionary<string, object> { ["Error"] = ex.Message };
                }
            }
            metrics["CorrelatorMetrics"] = correlatorMetrics;

            return metrics;
        }

        /// <summary>
        /// Disposes of all resources used by the log processor.
        /// </summary>
        /// <returns>ValueTask representing the disposal operation.</returns>
        public ValueTask DisposeAsync()
        {
            try
            {
                _correlationTimer?.Dispose();
                _correlationBuffer.Clear();
            
                // Get configurable disposal timeout
                var disposalTimeoutMs = _configuration.GetValue<int>("Processing:DisposalTimeoutMs", 1000);
            
                // Dispose of all components
                foreach (var enricher in _enrichers.OfType<IAsyncDisposable>())
                {
                    try
                    {
                        enricher.DisposeAsync().AsTask().Wait(disposalTimeoutMs); // Configurable timeout
                    }
                    catch (Exception ex)
                    {
                        _logger.LogWarning(ex, "Error disposing enricher");
                    }
                }
                
                _logger.LogInformation("LogProcessor disposed successfully");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error during LogProcessor disposal");
            }

            return ValueTask.CompletedTask;
        }
    }

    /// <summary>
    /// Event arguments for configuration updates.
    /// </summary>
    public class ConfigurationUpdatedEventArgs : EventArgs
    {
        public string ConfigurationType { get; set; } = "";
        public Dictionary<string, object> Configuration { get; set; } = new();
        public DateTime UpdateTime { get; set; }
        public bool Success { get; set; }
        public string Error { get; set; } = "";
    }
} 
