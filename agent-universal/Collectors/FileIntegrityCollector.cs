using System;
using System.Collections.Generic;
using System.IO;
using System.Security.Cryptography;
using System.Threading;
using System.Threading.Tasks;
using System.Linq;
using Microsoft.Extensions.Logging;
using AthalaSIEM.Agent.Core;
using AthalaSIEM.UniversalAgent.Models;
using Core = AthalaSIEM.Agent.Core;

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
        private readonly List<LogEntry> _collectedLogs = new List<LogEntry>();
        private readonly Dictionary<string, FileSystemWatcher> _watchers = new();
        private readonly Dictionary<string, string> _fileHashes = new();
        private readonly List<string> _monitoredPaths = new();
        private readonly CancellationTokenSource _cancellationTokenSource = new();
        private Timer? _scanTimer;
        private int _scanIntervalMinutes = 30;

        /// <inheritdoc />
        public event EventHandler<LogCollectedEventArgs>? LogCollected;
        
        /// <inheritdoc />
        public event EventHandler<LogCollectionErrorEventArgs>? CollectionError;

        /// <summary>
        /// Initializes a new instance of the FileIntegrityCollector.
        /// </summary>
        /// <param name="logger">Logger instance for this collector.</param>
        public FileIntegrityCollector(ILogger<FileIntegrityCollector> logger)
        {
            _logger = logger;
        }

        /// <summary>
        /// Initializes the File Integrity Monitor with user-provided configuration.
        /// NO DEFAULT PATHS - User must explicitly configure all monitoring paths.
        /// If no paths are configured, the collector will be disabled (fail-secure).
        /// </summary>
        /// <param name="config">Configuration dictionary containing monitoring paths and settings.</param>
        /// <param name="cancellationToken">Cancellation token for the operation.</param>
        /// <returns>True if initialization was successful and paths are configured.</returns>
        public Task<bool> InitializeAsync(Dictionary<string, object> config, CancellationToken cancellationToken = default)
        {
            try
            {
                _logger.LogInformation("Initializing File Integrity Monitor with user configuration");
                
                // Debug: Log the entire configuration structure
                _logger.LogDebug("🔍 Configuration received with {Count} keys:", config.Count);
                foreach (var kvp in config)
                {
                    _logger.LogDebug("🔍 Config Key: {Key} = {Value} (Type: {Type})", 
                        kvp.Key, kvp.Value, kvp.Value?.GetType().Name);
                    
                    // If it's MonitoredPaths, log more details
                    if (kvp.Key == "MonitoredPaths" && kvp.Value != null)
                    {
                        _logger.LogDebug("🔍 MonitoredPaths detailed analysis:");
                        var pathsObj = kvp.Value;
                        _logger.LogDebug("🔍   Object Type: {Type}", pathsObj.GetType().FullName);
                        _logger.LogDebug("🔍   ToString(): {Value}", pathsObj.ToString());
                        
                        // Try to inspect properties if it's a complex object
                        var type = pathsObj.GetType();
                        if (type.IsArray)
                        {
                            var array = (Array)pathsObj;
                            _logger.LogDebug("🔍   Array Length: {Length}", array.Length);
                            for (int i = 0; i < Math.Min(array.Length, 5); i++)
                            {
                                _logger.LogDebug("🔍   Array[{Index}]: {Value} (Type: {Type})", 
                                    i, array.GetValue(i), array.GetValue(i)?.GetType().Name);
                            }
                        }
                        else if (pathsObj is System.Collections.IEnumerable enumerable && !(pathsObj is string))
                        {
                            _logger.LogDebug("🔍   Is IEnumerable (not string)");
                            int count = 0;
                            foreach (var item in enumerable)
                            {
                                if (count < 5)
                                {
                                    _logger.LogDebug("🔍   Item[{Index}]: {Value} (Type: {Type})", 
                                        count, item, item?.GetType().Name);
                                }
                                count++;
                                if (count >= 10) break; // Limit logging
                            }
                        }
                    }
                }

                // Load monitoring paths from configuration - NO DEFAULTS
                if (!LoadMonitoringPaths(config))
                {
                    _logger.LogWarning("NO MONITORING PATHS CONFIGURED. File Integrity Monitor will be disabled.");
                    _logger.LogInformation("Configure monitoring paths in appsettings.json under Collectors:FileIntegrity:MonitoredPaths");
                    _logger.LogInformation("Example configuration:");
                    _logger.LogInformation("\"MonitoredPaths\": [");
                    _logger.LogInformation("  \"C:\\\\Windows\\\\System32\\\\drivers\",");
                    _logger.LogInformation("  \"C:\\\\Program Files\\\\YourApp\",");
                    _logger.LogInformation("  \"C:\\\\Critical\\\\Files\\\\\"");
                    _logger.LogInformation("]");
                    return Task.FromResult(false);
                }

                // Load scan interval configuration
                LoadScanInterval(config);

                _logger.LogInformation("File Integrity Monitor initialized with {PathCount} monitoring paths", _monitoredPaths.Count);
                foreach (var path in _monitoredPaths)
                {
                    _logger.LogInformation("Monitoring path: {Path}", path);
                }

                return Task.FromResult(true);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to initialize File Integrity Monitor");
                return Task.FromResult(false);
            }
        }

        /// <summary>
        /// Loads monitoring paths from configuration.
        /// Supports array, comma-separated string, and single path formats.
        /// </summary>
        /// <param name="config">Configuration dictionary.</param>
        /// <returns>True if at least one valid path was configured.</returns>
        private bool LoadMonitoringPaths(Dictionary<string, object> config)
        {
            _logger.LogDebug("🔍 LoadMonitoringPaths: Config keys = {Keys}", string.Join(", ", config.Keys));
            
            if (!config.TryGetValue("MonitoredPaths", out var pathsObj))
            {
                _logger.LogDebug("🔍 LoadMonitoringPaths: MonitoredPaths key not found in config");
                return false; // No paths configured
            }
            
            _logger.LogDebug("🔍 LoadMonitoringPaths: Found pathsObj = {Value} (Type: {Type})", 
                pathsObj, pathsObj?.GetType().Name);

            // Use the robust parsing method
            var configuredPaths = ParseStringArrayFromConfig(pathsObj, new string[0]);
            
            _logger.LogDebug("Found {Count} configured paths: {Paths}", 
                configuredPaths.Length, string.Join(", ", configuredPaths));

            // Validate and add paths
            foreach (var path in configuredPaths)
            {
                if (ValidateMonitoringPath(path))
                {
                    _monitoredPaths.Add(path);
                }
            }

            _logger.LogInformation("Successfully loaded {Count} valid monitoring paths", _monitoredPaths.Count);
            return _monitoredPaths.Any();
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

        /// <summary>
        /// Loads scan interval configuration.
        /// </summary>
        /// <param name="config">Configuration dictionary.</param>
        private void LoadScanInterval(Dictionary<string, object> config)
        {
            if (config.TryGetValue("ScanIntervalMinutes", out var intervalObj))
            {
                if (intervalObj is int interval && interval > 0)
                {
                    _scanIntervalMinutes = interval;
                }
                else if (int.TryParse(intervalObj.ToString(), out var parsedInterval) && parsedInterval > 0)
                {
                    _scanIntervalMinutes = parsedInterval;
                }
            }

            _logger.LogDebug("File integrity scan interval set to {Interval} minutes", _scanIntervalMinutes);
        }

        /// <inheritdoc />
        public Task StartCollectionAsync(CancellationToken cancellationToken = default)
        {
            if (_monitoredPaths.Count == 0)
            {
                _logger.LogWarning("Cannot start File Integrity Monitor: No monitoring paths configured");
                return Task.CompletedTask;
            }

            IsActive = true;
            
            try
            {
                // Setup file system watchers for real-time monitoring
                SetupFileWatchers();
                
                // Start periodic full scan
                _scanTimer = new Timer(PerformFullScan, null, TimeSpan.Zero, TimeSpan.FromMinutes(_scanIntervalMinutes));
                
                _logger.LogInformation("File Integrity Monitor started - monitoring {Count} paths, scan interval: {Interval} minutes", 
                    _monitoredPaths.Count, _scanIntervalMinutes);
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

        /// <inheritdoc />
        public Task StopCollectionAsync(CancellationToken cancellationToken = default)
        {
            IsActive = false;
            
            _scanTimer?.Dispose();
            
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
                    ["ConfigurationStatus"] = _monitoredPaths.Count > 0 ? "Configured" : "NOT CONFIGURED"
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
        /// Creates a File Integrity Monitoring event log entry.
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

                return new LogEntry
                {
                    Timestamp = DateTime.UtcNow,
                    Source = "FileIntegrity",
                    Level = severity == "Critical" ? "Error" : "Warning",
                    Message = $"File {changeType}: {filePath}",
                    EventId = $"FIM_{changeType.ToUpper()}",
                    Category = "FileIntegrityMonitoring",
                    SecurityRelevance = severity,
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
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error creating FIM event for: {Path}", filePath);
                return null;
            }
        }

        /// <summary>
        /// Determines the security severity of a file change based on the path and change type.
        /// </summary>
        /// <param name="filePath">The file path that changed.</param>
        /// <param name="changeType">The type of change.</param>
        /// <returns>Security severity level.</returns>
        private string DetermineSeverity(string filePath, string changeType)
        {
            var normalizedPath = filePath.ToLowerInvariant();

            // Critical system paths
            if (normalizedPath.Contains("system32\\drivers") || 
                normalizedPath.Contains("system32\\config") ||
                normalizedPath.Contains("system32") && Path.GetExtension(normalizedPath) == ".exe")
            {
                return "Critical";
            }

            // Important application paths
            if (normalizedPath.Contains("program files") || 
                normalizedPath.Contains("inetpub") ||
                normalizedPath.Contains("windows\\system32"))
            {
                return "High";
            }

            return "Medium";
        }

        /// <summary>
        /// Analyzes threat indicators for a file change.
        /// </summary>
        /// <param name="filePath">The file path that changed.</param>
        /// <param name="changeType">The type of change.</param>
        /// <returns>List of threat indicators.</returns>
        private List<string> AnalyzeThreatIndicators(string filePath, string changeType)
        {
            var indicators = new List<string>();
            var fileName = Path.GetFileName(filePath).ToLowerInvariant();

            // Suspicious extensions
            var suspiciousExts = new[] { ".exe", ".dll", ".scr", ".bat", ".cmd", ".ps1" };
            if (suspiciousExts.Any(ext => fileName.EndsWith(ext)))
            {
                indicators.Add("suspicious_executable");
            }

            // System file modification
            if (filePath.ToLowerInvariant().Contains("system32") && changeType == "Modified")
            {
                indicators.Add("system_file_tampering");
            }

            // Hidden files
            if (fileName.StartsWith("."))
            {
                indicators.Add("hidden_file");
            }

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
} 