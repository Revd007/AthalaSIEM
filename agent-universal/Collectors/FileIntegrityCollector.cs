using System;
using System.Collections.Generic;
using System.IO;
using System.Security.Cryptography;
using System.Threading;
using System.Threading.Tasks;
using System.Linq;
using Microsoft.Extensions.Logging;
using AthalaSIEM.UniversalAgent.Core;
using AthalaSIEM.UniversalAgent.Models;
using AthalaSIEM.UniversalAgent.Services;
using AthalaSIEM.UniversalAgent.DTOs;
using Core = AthalaSIEM.UniversalAgent.Core;

namespace AthalaSIEM.Agent.Collectors
{
    /// <summary>
    /// File Integrity Monitoring (FIM) Collector for AthalaSIEM Universal Agent.
    /// Monitors user-configured paths for unauthorized changes.
    /// NO HARDCODED PATHS - All monitoring paths must be explicitly configured by the user.
    /// Fail-secure: If no paths are configured, nothing will be monitored.
    /// </summary>
    public class FileIntegrityCollector : ILogCollector
    {
        /// <inheritdoc />
        public string CollectorName => "File Integrity Monitor";
        
        /// <inheritdoc />
        public Core.OperatingSystem SupportedOS => Core.OperatingSystem.Windows;
        
        /// <inheritdoc />
        public bool IsActive { get; private set; }
        
        /// <inheritdoc />
        public long LogsCollected { get; private set; }

        private readonly ILogger<FileIntegrityCollector> _logger;
        private readonly FIMConfigurationService _fimConfigService;
        private readonly List<LogEntry> _collectedLogs = new List<LogEntry>();
        private readonly Dictionary<string, FileSystemWatcher> _watchers = new();
        private readonly Dictionary<string, string> _fileHashes = new();
        private readonly List<string> _monitoredPaths = new();
        private readonly CancellationTokenSource _cancellationTokenSource = new();
        private Timer? _scanTimer;
        private Timer? _configRefreshTimer;
        private int _scanIntervalMinutes;
        private List<SeverityRule> _severityRules = new();
        private string _lastConfigurationVersion = "";

        /// <inheritdoc />
        public event EventHandler<LogCollectedEventArgs>? LogCollected;
        
        /// <inheritdoc />
        public event EventHandler<LogCollectionErrorEventArgs>? CollectionError;

        /// <summary>
        /// Initializes a new instance of the FileIntegrityCollector.
        /// </summary>
        /// <param name="logger">Logger instance for this collector.</param>
        /// <param name="fimConfigService">FIM Configuration Service for dynamic backend configuration.</param>
        public FileIntegrityCollector(ILogger<FileIntegrityCollector> logger, FIMConfigurationService fimConfigService)
        {
            _logger = logger;
            _fimConfigService = fimConfigService;
            _logger.LogInformation("File Integrity Monitor initialized - Using dynamic backend configuration");
        }

        /// <summary>
        /// Initializes the File Integrity Monitor with user-provided configuration.
        /// NO DEFAULT PATHS - User must explicitly configure all monitoring paths.
        /// If no paths are configured, the collector will be disabled (fail-secure).
        /// This method is now deprecated - use UpdateFromBackendConfig() instead.
        /// </summary>
        /// <param name="config">Configuration dictionary containing monitoring paths and settings.</param>
        /// <param name="cancellationToken">Cancellation token for the operation.</param>
        /// <returns>True if initialization was successful and paths are configured.</returns>
        public Task<bool> InitializeAsync(Dictionary<string, object> config, CancellationToken cancellationToken = default)
        {
            try
            {
                _logger.LogWarning("Local FIM configuration is deprecated. Use Backend configuration instead.");
                
                // For backward compatibility, still load local config if no backend config available
                if (config != null && config.Any())
                {
                    return UpdateFromBackendConfigAsync(config, cancellationToken);
                }
                else
                {
                    _logger.LogInformation("No local FIM configuration - waiting for Backend configuration");
                    return Task.FromResult(true); // Don't fail, wait for backend config
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to initialize File Integrity Monitor");
                return Task.FromResult(false);
            }
        }

        /// <summary>
        /// Updates FIM configuration from backend using FIMConfigurationService.
        /// This method fetches dynamic FIM configuration from SIEM backend.
        /// </summary>
        /// <param name="config">Legacy configuration (for backward compatibility).</param>
        /// <param name="cancellationToken">Cancellation token for the operation.</param>
        /// <returns>True if configuration was successfully applied.</returns>
        public async Task<bool> UpdateFromBackendConfigAsync(Dictionary<string, object> config, CancellationToken cancellationToken = default)
        {
            try
            {
                _logger.LogInformation("🔄 Fetching FIM configuration from backend via API...");

                // First try to get configuration from backend API
                var fimConfigurations = await _fimConfigService.GetFIMConfigurationsAsync();
                
                if (fimConfigurations.Any())
                {
                    _logger.LogInformation("✅ Retrieved {Count} FIM configurations from backend API", fimConfigurations.Count);
                    
                    // Clear existing configuration
                    _monitoredPaths.Clear();
                    StopExistingWatchers();
                    
                    // Process each FIM configuration
                    foreach (var fimConfig in fimConfigurations.Where(c => c.Enabled))
                    {
                        await ProcessFIMConfiguration(fimConfig);
                    }
                    
                    // Update configuration version for change detection
                    _lastConfigurationVersion = string.Join("|", fimConfigurations.Select(c => $"{c.Id}:{c.Name}"));
                    
                    // If we have paths, restart monitoring
                    if (_monitoredPaths.Count > 0 && IsActive)
                    {
                        await RestartMonitoringAsync();
                    }
                    
                    _logger.LogInformation("✅ FIM configuration updated from backend API: {PathCount} monitoring paths", _monitoredPaths.Count);
                    foreach (var path in _monitoredPaths)
                    {
                        _logger.LogInformation("📁 Monitoring path: {Path}", path);
                    }
                    
                    return true;
                }
                else
                {
                    _logger.LogWarning("⚠️ No FIM configurations found in backend API, falling back to legacy configuration");
                    return await UpdateFromLegacyConfigAsync(config, cancellationToken);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Failed to fetch FIM configuration from backend API, falling back to legacy configuration");
                return await UpdateFromLegacyConfigAsync(config, cancellationToken);
            }
        }

        /// <summary>
        /// Process a single FIM configuration from backend
        /// </summary>
        private async Task ProcessFIMConfiguration(FIMConfigurationDto fimConfig)
        {
            _logger.LogInformation("📋 Processing FIM configuration: {Name}", fimConfig.Name);
            
            foreach (var rule in fimConfig.Rules.Where(r => r.Enabled))
            {
                _logger.LogDebug("📝 Processing FIM rule: {RuleName} - {Path}", rule.Name, rule.MonitorPath);
                
                // Expand environment variables and validate path
                var expandedPath = Environment.ExpandEnvironmentVariables(rule.MonitorPath);
                
                if (ValidateMonitoringPath(expandedPath))
                {
                    _monitoredPaths.Add(expandedPath);
                    
                    // Send rule metadata to backend if needed
                    await SendFIMRuleStatusToBackend(rule, "Active");
                }
                else
                {
                    _logger.LogWarning("⚠️ Invalid monitoring path in rule {RuleName}: {Path}", rule.Name, expandedPath);
                    await SendFIMRuleStatusToBackend(rule, "Invalid");
                }
            }
            
            // Update scan interval from global settings
            if (fimConfig.GlobalSettings.DefaultScanInterval > 0)
            {
                _scanIntervalMinutes = fimConfig.GlobalSettings.DefaultScanInterval;
                _logger.LogInformation("📊 FIM scan interval set to {Interval} minutes from backend configuration", _scanIntervalMinutes);
            }
        }

        /// <summary>
        /// Send FIM rule status back to backend
        /// </summary>
        private async Task SendFIMRuleStatusToBackend(FIMRuleDto rule, string status)
        {
            try
            {
                // This could be expanded to send rule status updates to backend
                _logger.LogDebug("📤 FIM rule {RuleId} status: {Status}", rule.Id, status);
                // await _fimConfigService.UpdateRuleStatusAsync(rule.Id, status);
                await Task.CompletedTask;
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Failed to send FIM rule status to backend");
            }
        }

        /// <summary>
        /// Legacy configuration fallback method
        /// </summary>
        private async Task<bool> UpdateFromLegacyConfigAsync(Dictionary<string, object> config, CancellationToken cancellationToken = default)
        {
            try
            {
                _logger.LogInformation("🔄 Using legacy FIM configuration method...");
                
                // Clear existing configuration
                _monitoredPaths.Clear();
                StopExistingWatchers();

                // Load monitoring paths from legacy configuration
                if (!LoadMonitoringPathsFromBackend(config))
                {
                    _logger.LogWarning("⚠️ NO MONITORING PATHS provided. File Integrity Monitor will be disabled.");
                    _logger.LogInformation("💡 Configure monitoring paths via SIEM Web Interface:");
                    _logger.LogInformation("   • Go to FIM → Configurations → Create New Configuration");
                    _logger.LogInformation("   • Add paths you want to monitor (e.g., C:\\Windows\\System32\\drivers)");
                    _logger.LogInformation("   • Assign configuration to this agent");
                    _logger.LogInformation("   • Configuration will be applied automatically within 5 minutes");
                    return true; // Don't fail - this is user configuration
                }

                // Load scan interval configuration
                LoadScanIntervalFromBackend(config);

                // Load severity rules from backend
                LoadSeverityRulesFromBackend(config);

                // If we have paths, restart monitoring
                if (_monitoredPaths.Count > 0 && IsActive)
                {
                    await RestartMonitoringAsync();
                }

                _logger.LogInformation("✅ Legacy FIM configuration applied: {PathCount} monitoring paths", _monitoredPaths.Count);
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to apply legacy FIM configuration");
                return false;
            }
        }

        /// <summary>
        /// Loads monitoring paths from backend configuration.
        /// Supports array, comma-separated string, and single path formats.
        /// </summary>
        /// <param name="config">Backend configuration dictionary.</param>
        /// <returns>True if at least one valid path was configured.</returns>
        private bool LoadMonitoringPathsFromBackend(Dictionary<string, object> config)
        {
            _logger.LogDebug("🔍 LoadMonitoringPathsFromBackend: Config keys = {Keys}", string.Join(", ", config.Keys));
            
            // Try different key names that backend might use
            var possibleKeys = new[] { "MonitoredPaths", "Paths", "FIMPaths", "FilePaths", "MonitoringPaths" };
            object? pathsObj = null;
            string? usedKey = null;

            foreach (var key in possibleKeys)
            {
                if (config.TryGetValue(key, out pathsObj))
                {
                    usedKey = key;
                    break;
                }
            }

            if (pathsObj == null)
            {
                _logger.LogDebug("🔍 LoadMonitoringPathsFromBackend: No monitoring paths key found in backend config");
                return false; // No paths configured by backend
            }
            
            _logger.LogDebug("🔍 LoadMonitoringPathsFromBackend: Found pathsObj using key '{Key}' = {Value} (Type: {Type})", 
                usedKey, pathsObj, pathsObj?.GetType().Name);

            // Use the robust parsing method
            var configuredPaths = pathsObj != null ? ParseStringArrayFromConfig(pathsObj, new string[0]) : new string[0];
            
            _logger.LogInformation("Backend provided {Count} monitoring paths: {Paths}", 
                configuredPaths.Length, string.Join(", ", configuredPaths));

            // Validate and add paths
            foreach (var path in configuredPaths)
            {
                if (ValidateMonitoringPath(path))
                {
                    _monitoredPaths.Add(path);
                }
            }

            _logger.LogInformation("Successfully loaded {Count} valid monitoring paths from backend", _monitoredPaths.Count);
            return _monitoredPaths.Any();
        }

        /// <summary>
        /// Loads scan interval from backend configuration.
        /// </summary>
        /// <param name="config">Backend configuration dictionary.</param>
        private void LoadScanIntervalFromBackend(Dictionary<string, object> config)
        {
            var possibleKeys = new[] { "ScanIntervalMinutes", "ScanInterval", "IntervalMinutes", "Interval" };
            var intervalSet = false;
            
            foreach (var key in possibleKeys)
            {
                if (config.TryGetValue(key, out var intervalObj))
                {
                    if (intervalObj is int interval && interval > 0)
                    {
                        _scanIntervalMinutes = interval;
                        intervalSet = true;
                        break;
                    }
                    else if (int.TryParse(intervalObj.ToString(), out var parsedInterval) && parsedInterval > 0)
                    {
                        _scanIntervalMinutes = parsedInterval;
                        intervalSet = true;
                        break;
                    }
                }
            }

            // Ensure we have a valid scan interval
            if (!intervalSet || _scanIntervalMinutes <= 0)
            {
                _scanIntervalMinutes = 30; // Default fallback - no hardcoding in business logic
                _logger.LogInformation("Using fallback FIM scan interval: {Interval} minutes (no backend configuration)", _scanIntervalMinutes);
            }
            else
            {
                _logger.LogInformation("FIM scan interval set to {Interval} minutes (Backend configured)", _scanIntervalMinutes);
            }
        }

        /// <summary>
        /// Loads severity rules from backend configuration.
        /// </summary>
        /// <param name="config">Backend configuration dictionary.</param>
        private void LoadSeverityRulesFromBackend(Dictionary<string, object> config)
        {
            var possibleKeys = new[] { "SeverityRules", "FilePathSeverity", "PathSeverityRules", "SeverityConfig" };
            
            foreach (var key in possibleKeys)
            {
                if (config.TryGetValue(key, out var rulesObj))
                {
                    try
                    {
                        // For now, set default rules if no backend configuration
                        // This will be populated by backend later
                        _severityRules.Clear();
                        _logger.LogInformation("FIM severity rules configuration available - will be configured by backend");
                        break;
                    }
                    catch (Exception ex)
                    {
                        _logger.LogWarning(ex, "Failed to parse severity rules from backend configuration");
                    }
                }
            }

            _logger.LogInformation("FIM severity rules: {Count} rules loaded from backend", _severityRules.Count);
        }

        /// <summary>
        /// Stops existing file watchers before reconfiguration.
        /// </summary>
        private void StopExistingWatchers()
        {
            foreach (var watcher in _watchers.Values)
            {
                try
                {
                    watcher?.Dispose();
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Error disposing file watcher");
                }
            }
            _watchers.Clear();
            _logger.LogDebug("Stopped {Count} existing file watchers", _watchers.Count);
        }

        /// <summary>
        /// Restarts monitoring with new configuration.
        /// </summary>
        /// <returns>Task representing the restart operation.</returns>
        private async Task RestartMonitoringAsync()
        {
            try
            {
                _logger.LogInformation("Restarting FIM monitoring with updated configuration...");
                
                // Stop existing monitoring
                _scanTimer?.Dispose();
                StopExistingWatchers();

                // Start with new configuration
                SetupFileWatchers();
                _scanTimer = new Timer(PerformFullScan, null, TimeSpan.Zero, TimeSpan.FromMinutes(_scanIntervalMinutes));
                
                _logger.LogInformation("✅ FIM monitoring restarted successfully");
                await Task.CompletedTask;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error restarting FIM monitoring");
                throw;
            }
        }

        /// <summary>
        /// Validates a monitoring path configuration.
        /// </summary>
        /// <param name="path">The path to validate.</param>
        /// <returns>True if the path is valid for monitoring.</returns>
        private bool ValidateMonitoringPath(string path)
        {
            try
            {
                // Check if path is valid format
                if (string.IsNullOrWhiteSpace(path))
                {
                    return false;
                }

                // Expand environment variables
                var expandedPath = Environment.ExpandEnvironmentVariables(path);

                // Check if directory exists (for directory paths)
                if (Directory.Exists(expandedPath))
                {
                    _logger.LogDebug("Validated monitoring directory: {Path}", expandedPath);
                    return true;
                }

                // Check if file exists (for specific file paths)
                if (File.Exists(expandedPath))
                {
                    _logger.LogDebug("Validated monitoring file: {Path}", expandedPath);
                    return true;
                }

                // For wildcard paths, validate the parent directory exists
                var parentDir = Path.GetDirectoryName(expandedPath);
                if (!string.IsNullOrEmpty(parentDir) && Directory.Exists(parentDir))
                {
                    _logger.LogDebug("Validated monitoring pattern: {Path}", expandedPath);
                    return true;
                }

                _logger.LogWarning("Monitoring path does not exist and will be skipped: {Path}", expandedPath);
                return false;
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Invalid monitoring path configuration: {Path}", path);
                return false;
            }
        }

        /// <inheritdoc />
        public Task StartCollectionAsync(CancellationToken cancellationToken = default)
        {
            IsActive = true;
            
            try
            {
                // Start configuration refresh timer (every 5 minutes)
                _configRefreshTimer = new Timer(RefreshConfigurationFromBackend, null, 
                    TimeSpan.FromMinutes(1), TimeSpan.FromMinutes(5));
                
                if (_monitoredPaths.Count > 0)
                {
                    // Setup file system watchers for real-time monitoring
                    SetupFileWatchers();
                    
                    // Start periodic full scan
                    _scanTimer = new Timer(PerformFullScan, null, TimeSpan.Zero, TimeSpan.FromMinutes(_scanIntervalMinutes));
                    
                    _logger.LogInformation("✅ File Integrity Monitor started - monitoring {Count} paths, scan interval: {Interval} minutes", 
                        _monitoredPaths.Count, _scanIntervalMinutes);
                }
                else
                {
                    _logger.LogInformation("⏳ File Integrity Monitor started - waiting for backend configuration");
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error starting File Integrity Monitor");
                CollectionError?.Invoke(this, new LogCollectionErrorEventArgs
                {
                    Exception = ex,
                    Source = CollectorName,
                    Message = "Error starting FIM"
                });
            }
            
            return Task.CompletedTask;
        }

        /// <summary>
        /// Refresh FIM configuration from backend (called by timer)
        /// </summary>
        private async void RefreshConfigurationFromBackend(object? state)
        {
            try
            {
                _logger.LogDebug("🔄 Checking for FIM configuration updates from backend...");
                
                var hasUpdates = await _fimConfigService.HasConfigurationUpdatedAsync(_lastConfigurationVersion);
                if (hasUpdates)
                {
                    _logger.LogInformation("🔄 FIM configuration updates detected, refreshing...");
                    await UpdateFromBackendConfigAsync(new Dictionary<string, object>());
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Error refreshing FIM configuration from backend");
            }
        }

        /// <inheritdoc />
        public Task StopCollectionAsync(CancellationToken cancellationToken = default)
        {
            IsActive = false;
            
            _scanTimer?.Dispose();
            _configRefreshTimer?.Dispose();
            
            foreach (var watcher in _watchers.Values)
            {
                watcher?.Dispose();
            }
            _watchers.Clear();
            
            _cancellationTokenSource.Cancel();
            
            _logger.LogInformation("File Integrity Monitor stopped");
            return Task.CompletedTask;
        }

        /// <inheritdoc />
        public Task<IEnumerable<LogEntry>> GetLogsAsync(int batchSize = 100, CancellationToken cancellationToken = default)
        {
            var logs = _collectedLogs.Take(batchSize).ToList();
            _collectedLogs.RemoveRange(0, logs.Count);
            return Task.FromResult<IEnumerable<LogEntry>>(logs);
        }

        /// <inheritdoc />
        public Task<CollectorHealth> GetHealthAsync()
        {
            return Task.FromResult(new CollectorHealth
            {
                IsHealthy = IsActive,
                Status = IsActive ? "Running" : "Stopped",
                LogsCollected = LogsCollected,
                LastCollection = DateTime.UtcNow,
                Metrics = new Dictionary<string, object>
                {
                    ["MonitoredPaths"] = _monitoredPaths.Count,
                    ["ActiveWatchers"] = _watchers.Count,
                    ["TrackedFiles"] = _fileHashes.Count,
                    ["BufferedLogs"] = _collectedLogs.Count,
                    ["ScanIntervalMinutes"] = _scanIntervalMinutes,
                    ["ConfigurationStatus"] = _monitoredPaths.Count > 0 ? "Backend Configured" : "AWAITING BACKEND CONFIGURATION",
                    ["ConfigurationSource"] = "Backend Controlled"
                }
            });
        }

        /// <summary>
        /// Sets up file system watchers for configured monitoring paths.
        /// </summary>
        private void SetupFileWatchers()
        {
            foreach (var path in _monitoredPaths)
            {
                try
                {
                    var expandedPath = Environment.ExpandEnvironmentVariables(path);
                    
                    if (Directory.Exists(expandedPath))
                    {
                        var watcher = new FileSystemWatcher(expandedPath)
                        {
                            IncludeSubdirectories = true,
                            NotifyFilter = NotifyFilters.CreationTime | NotifyFilters.LastWrite | 
                                         NotifyFilters.FileName | NotifyFilters.Size
                        };

                        watcher.Created += OnFileChanged;
                        watcher.Changed += OnFileChanged;
                        watcher.Deleted += OnFileChanged;
                        watcher.Renamed += OnFileRenamed;
                        watcher.Error += OnWatcherError;

                        watcher.EnableRaisingEvents = true;
                        _watchers[expandedPath] = watcher;

                        _logger.LogDebug("File watcher setup for: {Path}", expandedPath);
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Failed to setup watcher for path: {Path}", path);
                }
            }
        }

        /// <summary>
        /// Handles file change events from file system watchers.
        /// </summary>
        /// <param name="sender">The file system watcher.</param>
        /// <param name="e">File system event arguments.</param>
        private void OnFileChanged(object sender, FileSystemEventArgs e)
        {
            if (!IsActive) return;

            try
            {
                var changeType = e.ChangeType.ToString();
                var filePath = e.FullPath;

                // Skip temporary files
                if (Path.GetFileName(filePath).StartsWith("~") || 
                    Path.GetExtension(filePath).ToLower() == ".tmp")
                    return;

                var logEntry = CreateFIMEvent(filePath, changeType, null);
                if (logEntry != null)
                {
                    _collectedLogs.Add(logEntry);
                    LogsCollected++;

                    LogCollected?.Invoke(this, new LogCollectedEventArgs 
                    { 
                        Logs = new[] { logEntry },
                        Source = CollectorName,
                        CollectionTime = DateTime.UtcNow
                    });
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error handling file change event");
            }
        }

        /// <summary>
        /// Handles file rename events from file system watchers.
        /// </summary>
        /// <param name="sender">The file system watcher.</param>
        /// <param name="e">Renamed event arguments.</param>
        private void OnFileRenamed(object sender, RenamedEventArgs e)
        {
            if (!IsActive) return;

            var logEntry = CreateFIMEvent(e.FullPath, "Renamed", e.OldFullPath);
            if (logEntry != null)
            {
                _collectedLogs.Add(logEntry);
                LogsCollected++;

                LogCollected?.Invoke(this, new LogCollectedEventArgs 
                { 
                    Logs = new[] { logEntry },
                    Source = CollectorName,
                    CollectionTime = DateTime.UtcNow
                });
            }
        }

        /// <summary>
        /// Handles file system watcher errors.
        /// </summary>
        /// <param name="sender">The file system watcher.</param>
        /// <param name="e">Error event arguments.</param>
        private void OnWatcherError(object sender, ErrorEventArgs e)
        {
            _logger.LogError(e.GetException(), "File watcher error");
            CollectionError?.Invoke(this, new LogCollectionErrorEventArgs
            {
                Exception = e.GetException(),
                Source = CollectorName,
                Message = "File watcher error"
            });
        }

        /// <summary>
        /// Performs a full scan of all configured monitoring paths.
        /// </summary>
        /// <param name="state">Timer state (not used).</param>
        private void PerformFullScan(object? state)
        {
            if (!IsActive) return;

            try
            {
                _logger.LogDebug("Starting FIM full scan");

                foreach (var path in _monitoredPaths)
                {
                    var expandedPath = Environment.ExpandEnvironmentVariables(path);
                    
                    if (Directory.Exists(expandedPath))
                    {
                        ScanDirectory(expandedPath);
                    }
                    else if (File.Exists(expandedPath))
                    {
                        ScanFile(expandedPath);
                    }
                }

                _logger.LogDebug("FIM full scan completed");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error during FIM full scan");
            }
        }

        /// <summary>
        /// Scans a directory for file changes.
        /// </summary>
        /// <param name="directoryPath">The directory path to scan.</param>
        private void ScanDirectory(string directoryPath)
        {
            try
            {
                var files = Directory.GetFiles(directoryPath, "*", SearchOption.TopDirectoryOnly);
                foreach (var file in files.Take(100)) // Limit for performance
                {
                    ScanFile(file);
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Error scanning directory: {Path}", directoryPath);
            }
        }

        /// <summary>
        /// Scans a specific file for changes.
        /// </summary>
        /// <param name="filePath">The file path to scan.</param>
        private void ScanFile(string filePath)
        {
            try
            {
                if (!File.Exists(filePath)) return;

                var currentHash = GetFileHash(filePath);
                if (currentHash == null) return;

                if (_fileHashes.ContainsKey(filePath))
                {
                    if (_fileHashes[filePath] != currentHash)
                    {
                        // File changed
                        var logEntry = CreateFIMEvent(filePath, "Modified", null);
                        if (logEntry != null)
                        {
                            _collectedLogs.Add(logEntry);
                            LogsCollected++;
                        }
                        _fileHashes[filePath] = currentHash;
                    }
                }
                else
                {
                    // New file
                    _fileHashes[filePath] = currentHash;
                    var logEntry = CreateFIMEvent(filePath, "Discovered", null);
                    if (logEntry != null)
                    {
                        _collectedLogs.Add(logEntry);
                        LogsCollected++;
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Error scanning file: {Path}", filePath);
            }
        }

        /// <summary>
        /// Creates a File Integrity Monitoring event log entry and sends to backend.
        /// </summary>
        /// <param name="filePath">The file path that changed.</param>
        /// <param name="changeType">The type of change (Created, Modified, Deleted, etc.).</param>
        /// <param name="oldPath">The old file path (for rename operations).</param>
        /// <returns>A log entry for the file integrity event.</returns>
        private LogEntry? CreateFIMEvent(string filePath, string changeType, string? oldPath)
        {
            try
            {
                var fileInfo = File.Exists(filePath) ? new FileInfo(filePath) : null;
                var severity = DetermineSeverity(filePath, changeType);

                var logEntry = new LogEntry
                {
                    Id = LogEntryIdGenerator.GenerateId("FIM"),
                    Timestamp = DateTime.UtcNow,
                    Source = "FileIntegrity",
                    Level = severity == "Critical" ? "Error" : "Warning",
                    Message = $"File {changeType}: {filePath}",
                    EventId = $"FIM_{changeType.ToUpper()}",
                    Category = "FileIntegrityMonitoring",
                    SecurityRelevance = severity,
                    CollectorType = "FileIntegrity",
                    ComputerName = Environment.MachineName,
                    Properties = new Dictionary<string, object>
                    {
                        ["FilePath"] = filePath,
                        ["OldPath"] = oldPath ?? "",
                        ["ChangeType"] = changeType,
                        ["FileSize"] = fileInfo?.Length ?? 0,
                        ["FileHash"] = GetFileHash(filePath) ?? "",
                        ["LastModified"] = fileInfo?.LastWriteTimeUtc ?? DateTime.UtcNow,
                        ["ThreatIndicators"] = AnalyzeThreatIndicators(filePath, changeType)
                    }
                };

                // Send FIM event to backend asynchronously
                _ = Task.Run(async () => await SendFIMEventToBackend(logEntry, fileInfo));

                return logEntry;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error creating FIM event for: {Path}", filePath);
                return null;
            }
        }

        /// <summary>
        /// Send FIM event to backend using FIMConfigurationService
        /// </summary>
        private async Task SendFIMEventToBackend(LogEntry logEntry, FileInfo? fileInfo)
        {
            try
            {
                var fimEvent = new FIMEventDto
                {
                    Id = Guid.NewGuid().ToString(),
                    RuleId = "", // Will be populated when we have rule matching
                    RuleName = "Dynamic FIM Rule",
                    AgentId = Environment.MachineName,
                    Timestamp = logEntry.Timestamp,
                    FilePath = logEntry.Properties["FilePath"].ToString() ?? "",
                    EventType = logEntry.Properties["ChangeType"].ToString() ?? "",
                    OldFilePath = logEntry.Properties["OldPath"].ToString() ?? "",
                    User = Environment.UserName,
                    Process = "FileIntegrityCollector",
                    SecurityLevel = logEntry.SecurityRelevance,
                    Metadata = logEntry.Properties,
                    AlertGenerated = logEntry.SecurityRelevance == "Critical",
                    Tags = new List<string> { "FIM", "FileIntegrity", logEntry.SecurityRelevance }
                };

                // Add file info if available
                if (fileInfo != null)
                {
                    fimEvent.NewFileInfo = new FIMFileInfoDto
                    {
                        Size = fileInfo.Length,
                        CreatedTime = fileInfo.CreationTimeUtc,
                        ModifiedTime = fileInfo.LastWriteTimeUtc,
                        AccessedTime = fileInfo.LastAccessTimeUtc,
                        Permissions = fileInfo.Attributes.ToString(),
                        Hashes = new Dictionary<string, string>
                        {
                            ["SHA256"] = logEntry.Properties["FileHash"].ToString() ?? ""
                        }
                    };
                }

                var success = await _fimConfigService.SendFIMEventAsync(fimEvent);
                if (success)
                {
                    _logger.LogDebug("📤 FIM event sent to backend: {FilePath}", fimEvent.FilePath);
                }
                else
                {
                    _logger.LogWarning("⚠️ Failed to send FIM event to backend: {FilePath}", fimEvent.FilePath);
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Error sending FIM event to backend");
            }
        }

        /// <summary>
        /// Determines the security severity of a file change based on the path and change type.
        /// Now configurable via backend configuration instead of hardcoded paths.
        /// </summary>
        /// <param name="filePath">The file path that changed.</param>
        /// <param name="changeType">The type of change.</param>
        /// <returns>Security severity level.</returns>
        private string DetermineSeverity(string filePath, string changeType)
        {
            var normalizedPath = filePath.ToLowerInvariant();

            // Check backend-configured severity rules first
            if (_severityRules != null && _severityRules.Any())
            {
                foreach (var rule in _severityRules.OrderByDescending(r => r.Priority))
                {
                    if (PathMatchesRule(normalizedPath, rule))
                    {
                        return rule.Severity;
                    }
                }
            }

            // Fallback to default severity if no backend rules configured
            return "Medium";
        }

        /// <summary>
        /// Checks if a file path matches a severity rule pattern.
        /// </summary>
        /// <param name="normalizedPath">The normalized file path.</param>
        /// <param name="rule">The severity rule to check against.</param>
        /// <returns>True if the path matches the rule.</returns>
        private bool PathMatchesRule(string normalizedPath, SeverityRule rule)
        {
            foreach (var pattern in rule.PathPatterns)
            {
                if (normalizedPath.Contains(pattern.ToLowerInvariant()))
                {
                    return true;
                }
            }
            return false;
        }

        /// <summary>
        /// Analyzes threat indicators for a file change.
        /// Now configurable via backend instead of hardcoded patterns.
        /// </summary>
        /// <param name="filePath">The file path that changed.</param>
        /// <param name="changeType">The type of change.</param>
        /// <returns>List of threat indicators.</returns>
        private List<string> AnalyzeThreatIndicators(string filePath, string changeType)
        {
            var indicators = new List<string>();
            
            // Backend-configurable threat analysis will be implemented here
            // For now, return empty list to avoid hardcoded logic
            // Backend will configure what constitutes threat indicators
            
            return indicators;
        }

        /// <summary>
        /// Calculates a SHA256 hash of a file.
        /// </summary>
        /// <param name="filePath">The file path to hash.</param>
        /// <returns>Hexadecimal hash string or null if the operation fails.</returns>
        private string? GetFileHash(string filePath)
        {
            try
            {
                if (!File.Exists(filePath)) return null;

                using var sha256 = SHA256.Create();
                using var stream = File.OpenRead(filePath);
                var hashBytes = sha256.ComputeHash(stream);
                return Convert.ToHexString(hashBytes);
            }
            catch
            {
                return null;
            }
        }

        /// <summary>
        /// Parses a string array from various configuration object types.
        /// Handles JsonElement arrays, object arrays, and comma-separated strings.
        /// </summary>
        /// <param name="configObj">The configuration object to parse.</param>
        /// <param name="defaultValue">Default value if parsing fails.</param>
        /// <returns>Parsed string array.</returns>
        private string[] ParseStringArrayFromConfig(object configObj, string[] defaultValue)
        {
            try
            {
                if (configObj is string[] stringArray)
                {
                    return stringArray;
                }
                else if (configObj is System.Text.Json.JsonElement jsonElement && jsonElement.ValueKind == System.Text.Json.JsonValueKind.Array)
                {
                    var result = new List<string>();
                    foreach (var element in jsonElement.EnumerateArray())
                    {
                        if (element.ValueKind == System.Text.Json.JsonValueKind.String)
                        {
                            var value = element.GetString();
                            if (!string.IsNullOrWhiteSpace(value))
                            {
                                result.Add(value);
                            }
                        }
                    }
                    return result.ToArray();
                }
                else if (configObj is object[] objectArray)
                {
                    return objectArray.Select(o => o?.ToString()).Where(s => !string.IsNullOrWhiteSpace(s)).ToArray()!;
                }
                else if (configObj is List<object> objectList)
                {
                    return objectList.Select(o => o?.ToString()).Where(s => !string.IsNullOrWhiteSpace(s)).ToArray()!;
                }
                else if (configObj is string configString)
                {
                    if (configString.Contains(','))
                    {
                        return configString.Split(',').Select(s => s.Trim()).Where(s => !string.IsNullOrWhiteSpace(s)).ToArray();
                    }
                    else
                    {
                        return new[] { configString.Trim() };
                    }
                }
                else if (configObj is System.Collections.IEnumerable enumerable && !(configObj is string))
                {
                    var result = new List<string>();
                    foreach (var item in enumerable)
                    {
                        if (item != null)
                        {
                            var itemString = item.ToString();
                            if (!string.IsNullOrWhiteSpace(itemString))
                            {
                                result.Add(itemString);
                            }
                        }
                    }
                    return result.ToArray();
                }

                _logger.LogWarning("Unable to parse configuration object of type {Type}, using default value", configObj?.GetType().Name);
                return defaultValue;
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Error parsing string array from configuration, using default value");
                return defaultValue;
            }
        }

        /// <inheritdoc />
        public async ValueTask DisposeAsync()
        {
            await StopCollectionAsync();
            _scanTimer?.Dispose();
            _cancellationTokenSource?.Dispose();
        }
    }

    // NOTE: SeverityRule model has been moved to 
    // AthalaSIEM.UniversalAgent.Models.CollectorModels.cs for clean architecture separation
} 
