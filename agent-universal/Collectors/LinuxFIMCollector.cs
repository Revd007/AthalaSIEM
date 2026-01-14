using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Runtime.Versioning;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using AthalaSIEM.UniversalAgent.Core;
using AthalaSIEM.UniversalAgent.Models;

namespace AthalaSIEM.UniversalAgent.Collectors
{
    /// <summary>
    /// Linux File Integrity Monitoring (FIM) Collector for AthalaSIEM Universal Agent
    /// Uses inotify for real-time file system monitoring on Linux systems
    /// Author: Revian Ravil Athala
    /// Enterprise-grade SIEM file integrity monitoring with comprehensive threat detection
    /// </summary>
    [SupportedOSPlatform("linux")]
    public class LinuxFIMCollector : ILogCollector
    {
        private readonly ILogger<LinuxFIMCollector> _logger;
        private readonly List<LogEntry> _collectedEvents = new();
        private readonly CancellationTokenSource _cancellationTokenSource = new();
        private readonly object _eventsLock = new();
        
        private LinuxFIMConfiguration _config = new();
        private readonly Dictionary<string, FileSystemWatcher> _watchers = new();
        private readonly Dictionary<string, LinuxFileBaseline> _fileBaselines = new();
        private readonly Dictionary<int, Process> _inotifyProcesses = new();
        
        private bool _isActive = false;
        private long _eventsCollected = 0;
        private DateTime _lastCollection = DateTime.MinValue;
        private Timer? _baselineScanTimer;

        #region ILogCollector Implementation

        public string CollectorName => "Linux File Integrity Monitor";
        public Core.OperatingSystem SupportedOS => Core.OperatingSystem.Linux;
        public bool IsActive => _isActive;
        public long LogsCollected => _eventsCollected;

        public event EventHandler<LogCollectedEventArgs>? LogCollected;
        public event EventHandler<LogCollectionErrorEventArgs>? CollectionError;

        #endregion

        public LinuxFIMCollector(ILogger<LinuxFIMCollector> logger)
        {
            _logger = logger;
        }

        public async Task<bool> InitializeAsync(Dictionary<string, object> config, CancellationToken cancellationToken = default)
        {
            try
            {
                if (!RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
                {
                    _logger.LogWarning("LinuxFIMCollector can only run on Linux systems");
                    return false;
                }

                LoadConfiguration(config);
                
                _logger.LogInformation("🐧 Initializing Linux FIM Collector");
                _logger.LogInformation("📁 Monitoring {Count} paths with real-time detection", _config.MonitoredPaths.Count);

                // Validate inotify availability
                if (!await ValidateInotifyAvailabilityAsync())
                {
                    _logger.LogError("❌ inotify is not available on this system");
                    return false;
                }

                // Create initial file baselines
                await CreateFileBaselinesAsync();

                _logger.LogInformation("✅ Linux FIM Collector initialized successfully");
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Failed to initialize Linux FIM Collector");
                CollectionError?.Invoke(this, new LogCollectionErrorEventArgs
                {
                    Exception = ex,
                    Message = "Failed to initialize Linux FIM Collector",
                    Source = CollectorName
                });
                return false;
            }
        }

        public async Task StartCollectionAsync(CancellationToken cancellationToken = default)
        {
            try
            {
                if (_isActive)
                {
                    _logger.LogWarning("Linux FIM Collector is already active");
                    return;
                }

                _logger.LogInformation("🚀 Starting Linux FIM collection for {Count} monitored paths", 
                    _config.MonitoredPaths.Count);

                // Start inotify watchers for each monitored path
                foreach (var path in _config.MonitoredPaths)
                {
                    await StartPathMonitoringAsync(path);
                }

                // Start periodic baseline scan if enabled
                if (_config.EnablePeriodicScan)
                {
                    _baselineScanTimer = new Timer(
                        PerformBaselineScan,
                        null,
                        TimeSpan.FromMinutes(_config.BaselineScanIntervalMinutes),
                        TimeSpan.FromMinutes(_config.BaselineScanIntervalMinutes));
                }

                _isActive = true;
                _logger.LogInformation("✅ Linux FIM Collector started successfully with {Count} active watchers", 
                    _watchers.Count + _inotifyProcesses.Count);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Failed to start Linux FIM Collector");
                CollectionError?.Invoke(this, new LogCollectionErrorEventArgs
                {
                    Exception = ex,
                    Message = "Failed to start Linux FIM Collector",
                    Source = CollectorName
                });
            }
        }

        public async Task StopCollectionAsync(CancellationToken cancellationToken = default)
        {
            try
            {
                _baselineScanTimer?.Dispose();

                // Stop all file system watchers
                foreach (var watcher in _watchers.Values)
                {
                    watcher.EnableRaisingEvents = false;
                    watcher.Dispose();
                }
                _watchers.Clear();

                // Stop all inotify processes
                foreach (var process in _inotifyProcesses.Values)
                {
                    if (!process.HasExited)
                    {
                        process.Kill();
                        await process.WaitForExitAsync();
                    }
                    process.Dispose();
                }
                _inotifyProcesses.Clear();

                _isActive = false;
                _logger.LogInformation("🛑 Linux FIM Collector stopped successfully");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Error stopping Linux FIM Collector");
                CollectionError?.Invoke(this, new LogCollectionErrorEventArgs
                {
                    Exception = ex,
                    Message = "Error stopping Linux FIM Collector",
                    Source = CollectorName
                });
            }
        }

        public Task<IEnumerable<LogEntry>> GetLogsAsync(int batchSize = 100, CancellationToken cancellationToken = default)
        {
            lock (_eventsLock)
            {
                var events = _collectedEvents.Take(batchSize).ToList();
                _collectedEvents.RemoveRange(0, events.Count);
                return Task.FromResult<IEnumerable<LogEntry>>(events);
            }
        }

        public Task<CollectorHealth> GetHealthAsync()
        {
            var health = new CollectorHealth
            {
                IsHealthy = _isActive && !_cancellationTokenSource.Token.IsCancellationRequested,
                Status = _isActive ? "Running" : "Stopped",
                LogsCollected = _eventsCollected,
                LastCollection = _lastCollection,
                Uptime = DateTime.UtcNow - _lastCollection,
                Metrics = new Dictionary<string, object>
                {
                    {"MonitoredPaths", _config.MonitoredPaths.Count},
                    {"ActiveWatchers", _watchers.Count + _inotifyProcesses.Count},
                    {"QueuedEvents", _collectedEvents.Count},
                    {"RealTimeMonitoring", _config.EnableRealTimeMonitoring},
                    {"BaselineFiles", _fileBaselines.Count}
                }
            };

            return Task.FromResult(health);
        }

        #region Configuration Loading

        private void LoadConfiguration(Dictionary<string, object> config)
        {
            _config = new LinuxFIMConfiguration();

            if (config.TryGetValue("MonitoredPaths", out var pathsObj) && pathsObj is List<string> paths)
                _config.MonitoredPaths = paths;

            if (config.TryGetValue("ExcludedPaths", out var excludedObj) && excludedObj is List<string> excluded)
                _config.ExcludedPaths = excluded;

            if (config.TryGetValue("EnableRealTimeMonitoring", out var realtimeObj) && realtimeObj is bool realtime)
                _config.EnableRealTimeMonitoring = realtime;

            if (config.TryGetValue("HashAlgorithm", out var hashObj) && hashObj is string hashAlg)
                _config.HashAlgorithm = hashAlg;

            if (config.TryGetValue("EnablePeriodicScan", out var scanObj) && scanObj is bool enableScan)
                _config.EnablePeriodicScan = enableScan;

            if (config.TryGetValue("BaselineScanIntervalMinutes", out var intervalObj) && intervalObj is int interval)
                _config.BaselineScanIntervalMinutes = Math.Max(5, interval); // Minimum 5 minutes

            if (config.TryGetValue("MonitoredFileExtensions", out var extObj) && extObj is List<string> extensions)
                _config.MonitoredFileExtensions = extensions;

            if (config.TryGetValue("MaxFileSize", out var sizeObj) && sizeObj is long maxSize)
                _config.MaxFileSize = maxSize;

            // Add default critical paths if none specified
            if (!_config.MonitoredPaths.Any())
            {
                _config.MonitoredPaths = new List<string>
                {
                    "/etc/passwd", "/etc/shadow", "/etc/group", "/etc/sudoers",
                    "/etc/ssh/", "/boot/", "/usr/local/bin/", "/opt/"
                };
            }
        }

        #endregion

        #region System Validation

        private async Task<bool> ValidateInotifyAvailabilityAsync()
        {
            try
            {
                // Check if inotifywatch is available
                var process = new Process
                {
                    StartInfo = new ProcessStartInfo
                    {
                        FileName = "which",
                        Arguments = "inotifywatch",
                        UseShellExecute = false,
                        RedirectStandardOutput = true,
                        CreateNoWindow = true
                    }
                };

                process.Start();
                await process.WaitForExitAsync();
                
                if (process.ExitCode == 0)
                {
                    _logger.LogInformation("✅ inotifywatch is available for advanced monitoring");
                    return true;
                }

                // Fallback to basic .NET FileSystemWatcher
                _logger.LogInformation("ℹ️ Using .NET FileSystemWatcher as inotify fallback");
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Error checking inotify availability");
                return true; // Fallback to FileSystemWatcher
            }
        }

        #endregion

        #region File Baseline Management

        private async Task CreateFileBaselinesAsync()
        {
            _logger.LogInformation("📊 Creating file baselines for {Count} monitored paths", _config.MonitoredPaths.Count);

            foreach (var path in _config.MonitoredPaths)
            {
                try
                {
                    await CreateBaselineForPathAsync(path);
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Error creating baseline for path: {Path}", path);
                }
            }

            _logger.LogInformation("📊 Created baselines for {Count} files", _fileBaselines.Count);
        }

        private async Task CreateBaselineForPathAsync(string path)
        {
            try
            {
                if (File.Exists(path))
                {
                    await CreateFileBaselineAsync(path);
                }
                else if (Directory.Exists(path))
                {
                    await CreateDirectoryBaselinesAsync(path);
                }
            }
            catch (Exception ex)
            {
                _logger.LogDebug(ex, "Error creating baseline for path: {Path}", path);
            }
        }

        private async Task CreateFileBaselineAsync(string filePath)
        {
            try
            {
                var fileInfo = new FileInfo(filePath);
                
                // Skip if file is too large
                if (_config.MaxFileSize > 0 && fileInfo.Length > _config.MaxFileSize)
                {
                    _logger.LogDebug("Skipping large file: {FilePath} ({Size} bytes)", filePath, fileInfo.Length);
                    return;
                }

                // Check file extension filter
                if (_config.MonitoredFileExtensions.Any() && 
                    !_config.MonitoredFileExtensions.Contains(fileInfo.Extension.ToLower()))
                {
                    return;
                }

                var baseline = new LinuxFileBaseline
                {
                    FilePath = filePath,
                    Size = fileInfo.Length,
                    LastModified = fileInfo.LastWriteTimeUtc,
                    Permissions = await GetFilePermissionsAsync(filePath),
                    Owner = await GetFileOwnerAsync(filePath),
                    Group = await GetFileGroupAsync(filePath),
                    Hash = await CalculateFileHashAsync(filePath),
                    IsSymlink = fileInfo.Attributes.HasFlag(FileAttributes.ReparsePoint),
                    CreatedTime = fileInfo.CreationTimeUtc,
                    AccessedTime = fileInfo.LastAccessTimeUtc,
                    BaselineCreated = DateTime.UtcNow
                };

                if (baseline.IsSymlink)
                {
                    baseline.SymlinkTarget = await GetSymlinkTargetAsync(filePath);
                }

                // Get extended attributes if available
                baseline.ExtendedAttributes = await GetExtendedAttributesAsync(filePath);

                _fileBaselines[filePath] = baseline;
            }
            catch (Exception ex)
            {
                _logger.LogDebug(ex, "Error creating file baseline: {FilePath}", filePath);
            }
        }

        private async Task CreateDirectoryBaselinesAsync(string directoryPath)
        {
            try
            {
                if (!Directory.Exists(directoryPath))
                    return;

                var files = Directory.GetFiles(directoryPath, "*", SearchOption.AllDirectories);
                var tasks = files.Select(CreateFileBaselineAsync);
                await Task.WhenAll(tasks);
            }
            catch (Exception ex)
            {
                _logger.LogDebug(ex, "Error creating directory baselines: {DirectoryPath}", directoryPath);
            }
        }

        #endregion

        #region Path Monitoring

        private async Task StartPathMonitoringAsync(string path)
        {
            try
            {
                if (IsPathExcluded(path))
                {
                    _logger.LogDebug("Path excluded from monitoring: {Path}", path);
                    return;
                }

                // Try to use inotify if available, otherwise fallback to FileSystemWatcher
                if (await TryStartInotifyMonitoringAsync(path))
                {
                    _logger.LogDebug("✅ Started inotify monitoring for: {Path}", path);
                }
                else
                {
                    StartFileSystemWatcherMonitoring(path);
                    _logger.LogDebug("✅ Started FileSystemWatcher monitoring for: {Path}", path);
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Error starting monitoring for path: {Path}", path);
            }
        }

        private async Task<bool> TryStartInotifyMonitoringAsync(string path)
        {
            try
            {
                if (!File.Exists("/usr/bin/inotifywait") && !File.Exists("/bin/inotifywait"))
                {
                    return false;
                }

                var process = new Process
                {
                    StartInfo = new ProcessStartInfo
                    {
                        FileName = "inotifywait",
                        Arguments = $"-m -r -e create,delete,modify,move,attrib --format '%w%f|%e|%T' --timefmt '%Y-%m-%d %H:%M:%S' \"{path}\"",
                        UseShellExecute = false,
                        RedirectStandardOutput = true,
                        RedirectStandardError = true,
                        CreateNoWindow = true
                    }
                };

                process.OutputDataReceived += (sender, e) => OnInotifyEvent(e.Data, path);
                process.ErrorDataReceived += (sender, e) => 
                {
                    if (!string.IsNullOrEmpty(e.Data))
                        _logger.LogDebug("inotify stderr: {Data}", e.Data);
                };

                process.Start();
                process.BeginOutputReadLine();
                process.BeginErrorReadLine();

                _inotifyProcesses[process.Id] = process;
                
                // Verify process is running
                await Task.Delay(100);
                return !process.HasExited;
            }
            catch (Exception ex)
            {
                _logger.LogDebug(ex, "Failed to start inotify for path: {Path}", path);
                return false;
            }
        }

        private void StartFileSystemWatcherMonitoring(string path)
        {
            try
            {
                var watcher = new FileSystemWatcher();
                
                if (Directory.Exists(path))
                {
                    watcher.Path = path;
                    watcher.IncludeSubdirectories = true;
                }
                else if (File.Exists(path))
                {
                    watcher.Path = Path.GetDirectoryName(path) ?? "/";
                    watcher.Filter = Path.GetFileName(path);
                }
                else
                {
                    return;
                }

                watcher.NotifyFilter = NotifyFilters.CreationTime | NotifyFilters.LastWrite | 
                                     NotifyFilters.FileName | NotifyFilters.DirectoryName | 
                                     NotifyFilters.Size | NotifyFilters.Attributes;

                watcher.Created += OnFileSystemEvent;
                watcher.Changed += OnFileSystemEvent;
                watcher.Deleted += OnFileSystemEvent;
                watcher.Renamed += OnFileSystemRenamed;
                watcher.Error += OnFileSystemError;

                watcher.EnableRaisingEvents = true;
                _watchers[path] = watcher;
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Error starting FileSystemWatcher for path: {Path}", path);
            }
        }

        #endregion

        #region Event Handlers

        private async void OnInotifyEvent(string? eventData, string basePath)
        {
            try
            {
                if (string.IsNullOrEmpty(eventData))
                    return;

                // Parse inotify event: filepath|events|timestamp
                var parts = eventData.Split('|');
                if (parts.Length < 3)
                    return;

                var filePath = parts[0];
                var events = parts[1].Split(',');
                var timestampStr = parts[2];

                if (!DateTime.TryParse(timestampStr, out var timestamp))
                    timestamp = DateTime.UtcNow;

                foreach (var eventType in events)
                {
                    await ProcessFileEventAsync(filePath, MapInotifyEventType(eventType), timestamp);
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Error processing inotify event: {EventData}", eventData);
            }
        }

        private async void OnFileSystemEvent(object sender, FileSystemEventArgs e)
        {
            try
            {
                await ProcessFileEventAsync(e.FullPath, MapFileSystemEventType(e.ChangeType), DateTime.UtcNow);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Error processing file system event: {EventType} {FilePath}", e.ChangeType, e.FullPath);
            }
        }

        private async void OnFileSystemRenamed(object sender, RenamedEventArgs e)
        {
            try
            {
                await ProcessFileEventAsync(e.FullPath, "MOVE", DateTime.UtcNow, e.OldFullPath);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Error processing file rename event: {OldPath} -> {NewPath}", e.OldFullPath, e.FullPath);
            }
        }

        private void OnFileSystemError(object sender, ErrorEventArgs e)
        {
            _logger.LogError(e.GetException(), "FileSystemWatcher error");
            CollectionError?.Invoke(this, new LogCollectionErrorEventArgs
            {
                Exception = e.GetException(),
                Message = "FileSystemWatcher error",
                Source = CollectorName
            });
        }

        #endregion

        #region Event Processing

        private async Task ProcessFileEventAsync(string filePath, string eventType, DateTime timestamp, string? oldPath = null)
        {
            try
            {
                if (IsPathExcluded(filePath))
                    return;

                var fimEvent = await CreateLinuxFIMEventAsync(filePath, eventType, timestamp, oldPath);
                if (fimEvent != null)
                {
                    lock (_eventsLock)
                    {
                        _collectedEvents.Add(fimEvent);
                        _eventsCollected++;
                        _lastCollection = DateTime.UtcNow;

                        // Maintain max queue size
                        if (_collectedEvents.Count > 10000)
                        {
                            _collectedEvents.RemoveRange(0, 1000); // Remove oldest 1000 events
                        }
                    }

                    // Fire event
                    LogCollected?.Invoke(this, new LogCollectedEventArgs
                    {
                        Logs = new[] { fimEvent },
                        Source = CollectorName,
                        CollectionTime = timestamp
                    });

                    // Update baseline for the file
                    if (File.Exists(filePath) && (eventType == "CREATE" || eventType == "MODIFY"))
                    {
                        await CreateFileBaselineAsync(filePath);
                    }
                    else if (eventType == "DELETE" && _fileBaselines.ContainsKey(filePath))
                    {
                        _fileBaselines.Remove(filePath);
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Error processing file event: {EventType} {FilePath}", eventType, filePath);
            }
        }

        private async Task<LogEntry?> CreateLinuxFIMEventAsync(string filePath, string eventType, DateTime timestamp, string? oldPath = null)
        {
            try
            {
                var fimEvent = new LinuxFIMEvent
                {
                    Id = LogEntryIdGenerator.GenerateId("LFIM"),
                    Timestamp = timestamp,
                    Source = "LinuxFIM",
                    Level = DetermineEventLogLevel(eventType, filePath),
                    Message = $"File {eventType.ToLower()}: {filePath}",
                    Category = "FileIntegrity",
                    SecurityRelevance = DetermineSecurityRelevance(filePath, eventType),
                    CollectorType = CollectorName,
                    CollectionTime = DateTime.UtcNow,
                    
                    // FIM-specific properties
                    FilePath = filePath,
                    EventType = eventType,
                    OldFilePath = oldPath,
                    User = await GetCurrentUserAsync(),
                    Process = await GetCurrentProcessAsync(),
                    ProcessId = Environment.ProcessId
                };

                // Get file information
                if (File.Exists(filePath))
                {
                    fimEvent.NewFileInfo = await GetLinuxFileInfoAsync(filePath);
                }

                // Get old file information from baseline
                if (_fileBaselines.TryGetValue(filePath, out var baseline))
                {
                    fimEvent.OldFileInfo = ConvertBaselineToFileInfo(baseline);
                }

                // Analyze threat indicators
                fimEvent.ThreatIndicators = AnalyzeThreatIndicators(filePath, eventType);

                // Add metadata
                fimEvent.Properties = new Dictionary<string, object>
                {
                    ["FilePath"] = filePath,
                    ["EventType"] = eventType,
                    ["SecurityRelevance"] = fimEvent.SecurityRelevance,
                    ["ThreatScore"] = CalculateThreatScore(filePath, eventType),
                    ["IsSystemPath"] = IsSystemPath(filePath),
                    ["IsBinaryFile"] = IsBinaryFile(filePath)
                };

                if (!string.IsNullOrEmpty(oldPath))
                    fimEvent.Properties["OldFilePath"] = oldPath;

                return fimEvent;
            }
            catch (Exception ex)
            {
                _logger.LogDebug(ex, "Error creating FIM event for: {FilePath}", filePath);
                return null;
            }
        }

        #endregion

        #region Helper Methods

        private string MapInotifyEventType(string inotifyEvent)
        {
            return inotifyEvent.ToUpper() switch
            {
                "CREATE" => "CREATE",
                "DELETE" => "DELETE",
                "MODIFY" => "MODIFY",
                "MOVED_FROM" or "MOVED_TO" or "MOVE" => "MOVE",
                "ATTRIB" => "ATTRIB",
                _ => "MODIFY"
            };
        }

        private string MapFileSystemEventType(WatcherChangeTypes changeType)
        {
            return changeType switch
            {
                WatcherChangeTypes.Created => "CREATE",
                WatcherChangeTypes.Deleted => "DELETE",
                WatcherChangeTypes.Changed => "MODIFY",
                WatcherChangeTypes.Renamed => "MOVE",
                _ => "MODIFY"
            };
        }

        private bool IsPathExcluded(string path)
        {
            return _config.ExcludedPaths.Any(excluded => path.StartsWith(excluded, StringComparison.OrdinalIgnoreCase));
        }

        private string DetermineEventLogLevel(string eventType, string filePath)
        {
            return eventType switch
            {
                "DELETE" when IsSystemPath(filePath) => "Critical",
                "DELETE" => "Warning",
                "CREATE" when IsSystemPath(filePath) => "Warning",
                "MODIFY" when filePath.Contains("/etc/passwd") || filePath.Contains("/etc/shadow") => "Critical",
                "MODIFY" when IsSystemPath(filePath) => "Warning",
                _ => "Information"
            };
        }

        private string DetermineSecurityRelevance(string filePath, string eventType)
        {
            // Critical system files
            var criticalPaths = new[]
            {
                "/etc/passwd", "/etc/shadow", "/etc/group", "/etc/sudoers",
                "/etc/ssh/sshd_config", "/boot/", "/usr/bin/", "/sbin/"
            };

            if (criticalPaths.Any(cp => filePath.StartsWith(cp, StringComparison.OrdinalIgnoreCase)))
                return "Critical";

            // High-risk paths
            var highRiskPaths = new[]
            {
                "/etc/", "/usr/local/bin/", "/opt/", "/root/", "/home/"
            };

            if (highRiskPaths.Any(hr => filePath.StartsWith(hr, StringComparison.OrdinalIgnoreCase)))
                return "High";

            // Medium-risk events
            if (eventType == "DELETE" || eventType == "CREATE")
                return "Medium";

            return "Low";
        }

        private bool IsSystemPath(string filePath)
        {
            var systemPaths = new[] { "/etc/", "/boot/", "/usr/bin/", "/sbin/", "/lib/", "/usr/lib/" };
            return systemPaths.Any(sp => filePath.StartsWith(sp, StringComparison.OrdinalIgnoreCase));
        }

        private bool IsBinaryFile(string filePath)
        {
            try
            {
                if (!File.Exists(filePath))
                    return false;

                var buffer = new byte[512];
                using var stream = File.OpenRead(filePath);
                var bytesRead = stream.Read(buffer, 0, buffer.Length);
                
                // Check for null bytes (common in binary files)
                for (int i = 0; i < bytesRead; i++)
                {
                    if (buffer[i] == 0)
                        return true;
                }

                return false;
            }
            catch
            {
                return false;
            }
        }

        private List<string> AnalyzeThreatIndicators(string filePath, string eventType)
        {
            var indicators = new List<string>();

            // Suspicious file extensions
            var suspiciousExtensions = new[] { ".sh", ".py", ".pl", ".php", ".jsp", ".exe" };
            if (suspiciousExtensions.Any(ext => filePath.EndsWith(ext, StringComparison.OrdinalIgnoreCase)))
                indicators.Add("SuspiciousFileExtension");

            // Temporary or hidden files in sensitive locations
            var fileName = Path.GetFileName(filePath);
            if (fileName.StartsWith(".") && IsSystemPath(filePath))
                indicators.Add("HiddenFileInSystemPath");

            if (fileName.StartsWith("tmp") || fileName.Contains("temp"))
                indicators.Add("TemporaryFile");

            // Privilege escalation indicators
            if (filePath.Contains("sudo") || filePath.Contains("passwd") || filePath.Contains("shadow"))
                indicators.Add("PrivilegeEscalationTarget");

            // Persistence mechanisms
            if (filePath.Contains("cron") || filePath.Contains("systemd") || filePath.Contains("init"))
                indicators.Add("PersistenceMechanism");

            return indicators;
        }

        private int CalculateThreatScore(string filePath, string eventType)
        {
            int score = 0;

            // Base score by event type
            score += eventType switch
            {
                "DELETE" => 30,
                "CREATE" => 20,
                "MODIFY" => 10,
                "MOVE" => 15,
                _ => 5
            };

            // Path-based scoring
            if (IsSystemPath(filePath)) score += 40;
            if (filePath.Contains("/etc/")) score += 30;
            if (filePath.Contains("/root/")) score += 25;
            if (filePath.Contains("/home/")) score += 10;

            return Math.Min(100, score); // Cap at 100
        }

        private async void PerformBaselineScan(object? state)
        {
            try
            {
                _logger.LogInformation("🔍 Performing periodic baseline scan");
                
                foreach (var path in _config.MonitoredPaths)
                {
                    await CreateBaselineForPathAsync(path);
                }

                _logger.LogInformation("✅ Baseline scan completed for {Count} files", _fileBaselines.Count);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error during periodic baseline scan");
            }
        }

        #endregion

        #region Linux-Specific File Operations

        private async Task<string> CalculateFileHashAsync(string filePath)
        {
            try
            {
                using var algorithm = _config.HashAlgorithm.ToUpper() switch
                {
                    "SHA256" => (HashAlgorithm)SHA256.Create(),
                    "SHA1" => (HashAlgorithm)SHA1.Create(),
                    "MD5" => (HashAlgorithm)MD5.Create(),
                    _ => (HashAlgorithm)SHA256.Create()
                };

                using var stream = File.OpenRead(filePath);
                var hashBytes = await algorithm.ComputeHashAsync(stream);
                return Convert.ToHexString(hashBytes);
            }
            catch
            {
                return "";
            }
        }

        private async Task<string> GetFilePermissionsAsync(string filePath)
        {
            try
            {
                var process = new Process
                {
                    StartInfo = new ProcessStartInfo
                    {
                        FileName = "stat",
                        Arguments = $"-c %a \"{filePath}\"",
                        UseShellExecute = false,
                        RedirectStandardOutput = true,
                        CreateNoWindow = true
                    }
                };

                process.Start();
                var result = await process.StandardOutput.ReadToEndAsync();
                await process.WaitForExitAsync();

                return result.Trim();
            }
            catch
            {
                return "";
            }
        }

        private async Task<string> GetFileOwnerAsync(string filePath)
        {
            try
            {
                var process = new Process
                {
                    StartInfo = new ProcessStartInfo
                    {
                        FileName = "stat",
                        Arguments = $"-c %U \"{filePath}\"",
                        UseShellExecute = false,
                        RedirectStandardOutput = true,
                        CreateNoWindow = true
                    }
                };

                process.Start();
                var result = await process.StandardOutput.ReadToEndAsync();
                await process.WaitForExitAsync();

                return result.Trim();
            }
            catch
            {
                return "";
            }
        }

        private async Task<string> GetFileGroupAsync(string filePath)
        {
            try
            {
                var process = new Process
                {
                    StartInfo = new ProcessStartInfo
                    {
                        FileName = "stat",
                        Arguments = $"-c %G \"{filePath}\"",
                        UseShellExecute = false,
                        RedirectStandardOutput = true,
                        CreateNoWindow = true
                    }
                };

                process.Start();
                var result = await process.StandardOutput.ReadToEndAsync();
                await process.WaitForExitAsync();

                return result.Trim();
            }
            catch
            {
                return "";
            }
        }

        private async Task<string> GetSymlinkTargetAsync(string filePath)
        {
            try
            {
                var process = new Process
                {
                    StartInfo = new ProcessStartInfo
                    {
                        FileName = "readlink",
                        Arguments = $"\"{filePath}\"",
                        UseShellExecute = false,
                        RedirectStandardOutput = true,
                        CreateNoWindow = true
                    }
                };

                process.Start();
                var result = await process.StandardOutput.ReadToEndAsync();
                await process.WaitForExitAsync();

                return result.Trim();
            }
            catch
            {
                return "";
            }
        }

        private async Task<Dictionary<string, string>> GetExtendedAttributesAsync(string filePath)
        {
            var attributes = new Dictionary<string, string>();
            
            try
            {
                var process = new Process
                {
                    StartInfo = new ProcessStartInfo
                    {
                        FileName = "getfattr",
                        Arguments = $"-d \"{filePath}\"",
                        UseShellExecute = false,
                        RedirectStandardOutput = true,
                        CreateNoWindow = true
                    }
                };

                process.Start();
                var result = await process.StandardOutput.ReadToEndAsync();
                await process.WaitForExitAsync();

                var lines = result.Split('\n', StringSplitOptions.RemoveEmptyEntries);
                foreach (var line in lines)
                {
                    if (line.Contains('='))
                    {
                        var parts = line.Split('=', 2);
                        if (parts.Length == 2)
                        {
                            attributes[parts[0].Trim()] = parts[1].Trim();
                        }
                    }
                }
            }
            catch
            {
                // Extended attributes may not be supported
            }

            return attributes;
        }

        private async Task<LinuxFileInfo> GetLinuxFileInfoAsync(string filePath)
        {
            var fileInfo = new FileInfo(filePath);
            
            return new LinuxFileInfo
            {
                Size = fileInfo.Length,
                CreatedTime = fileInfo.CreationTimeUtc,
                ModifiedTime = fileInfo.LastWriteTimeUtc,
                AccessedTime = fileInfo.LastAccessTimeUtc,
                Permissions = await GetFilePermissionsAsync(filePath),
                Owner = await GetFileOwnerAsync(filePath),
                Group = await GetFileGroupAsync(filePath),
                Hashes = new Dictionary<string, string>
                {
                    [_config.HashAlgorithm] = await CalculateFileHashAsync(filePath)
                },
                IsSymlink = fileInfo.Attributes.HasFlag(FileAttributes.ReparsePoint),
                SymlinkTarget = fileInfo.Attributes.HasFlag(FileAttributes.ReparsePoint) 
                    ? await GetSymlinkTargetAsync(filePath) 
                    : null,
                ExtendedAttributes = await GetExtendedAttributesAsync(filePath)
            };
        }

        private LinuxFileInfo ConvertBaselineToFileInfo(LinuxFileBaseline baseline)
        {
            return new LinuxFileInfo
            {
                Size = baseline.Size,
                CreatedTime = baseline.CreatedTime,
                ModifiedTime = baseline.LastModified,
                AccessedTime = baseline.AccessedTime,
                Permissions = baseline.Permissions,
                Owner = baseline.Owner,
                Group = baseline.Group,
                Hashes = new Dictionary<string, string> { [_config.HashAlgorithm] = baseline.Hash },
                IsSymlink = baseline.IsSymlink,
                SymlinkTarget = baseline.SymlinkTarget,
                ExtendedAttributes = baseline.ExtendedAttributes
            };
        }

        private async Task<string> GetCurrentUserAsync()
        {
            try
            {
                var process = new Process
                {
                    StartInfo = new ProcessStartInfo
                    {
                        FileName = "whoami",
                        UseShellExecute = false,
                        RedirectStandardOutput = true,
                        CreateNoWindow = true
                    }
                };

                process.Start();
                var result = await process.StandardOutput.ReadToEndAsync();
                await process.WaitForExitAsync();

                return result.Trim();
            }
            catch
            {
                return Environment.UserName;
            }
        }

        private async Task<string> GetCurrentProcessAsync()
        {
            try
            {
                var currentProcess = Process.GetCurrentProcess();
                await Task.CompletedTask;
                return currentProcess.ProcessName;
            }
            catch
            {
                await Task.CompletedTask;
                return "unknown";
            }
            finally
            {
                await Task.CompletedTask;
            }
        }

        #endregion

        #region Disposal

        public async ValueTask DisposeAsync()
        {
            await StopCollectionAsync();
            _cancellationTokenSource.Dispose();
        }

        #endregion
    }

    #region Supporting Models

    /// <summary>
    /// Linux FIM configuration
    /// </summary>
    public class LinuxFIMConfiguration
    {
        public List<string> MonitoredPaths { get; set; } = new();
        public List<string> ExcludedPaths { get; set; } = new() { "/tmp/", "/var/tmp/", "/proc/", "/sys/" };
        public bool EnableRealTimeMonitoring { get; set; } = true;
        public string HashAlgorithm { get; set; } = "SHA256";
        public bool EnablePeriodicScan { get; set; } = true;
        public int BaselineScanIntervalMinutes { get; set; } = 60;
        public List<string> MonitoredFileExtensions { get; set; } = new();
        public long MaxFileSize { get; set; } = 100 * 1024 * 1024; // 100MB
    }

    /// <summary>
    /// Linux FIM event extending base LogEntry
    /// </summary>
    public class LinuxFIMEvent : LogEntry
    {
        public string FilePath { get; set; } = "";
        public string EventType { get; set; } = "";
        public string? OldFilePath { get; set; }
        public LinuxFileInfo? OldFileInfo { get; set; }
        public LinuxFileInfo? NewFileInfo { get; set; }
        public string User { get; set; } = "";
        public string Process { get; set; } = "";
        public new int ProcessId { get; set; } // Use 'new' keyword to explicitly hide inherited member
        public List<string> ThreatIndicators { get; set; } = new();
    }

    /// <summary>
    /// Linux file information
    /// </summary>
    public class LinuxFileInfo
    {
        public long Size { get; set; }
        public DateTime CreatedTime { get; set; }
        public DateTime ModifiedTime { get; set; }
        public DateTime AccessedTime { get; set; }
        public string Permissions { get; set; } = "";
        public string Owner { get; set; } = "";
        public string Group { get; set; } = "";
        public Dictionary<string, string> Hashes { get; set; } = new();
        public bool IsSymlink { get; set; }
        public string? SymlinkTarget { get; set; }
        public Dictionary<string, string> ExtendedAttributes { get; set; } = new();
    }

    /// <summary>
    /// File baseline for integrity checking
    /// </summary>
    public class LinuxFileBaseline
    {
        public string FilePath { get; set; } = "";
        public long Size { get; set; }
        public DateTime LastModified { get; set; }
        public DateTime CreatedTime { get; set; }
        public DateTime AccessedTime { get; set; }
        public string Permissions { get; set; } = "";
        public string Owner { get; set; } = "";
        public string Group { get; set; } = "";
        public string Hash { get; set; } = "";
        public bool IsSymlink { get; set; }
        public string? SymlinkTarget { get; set; }
        public Dictionary<string, string> ExtendedAttributes { get; set; } = new();
        public DateTime BaselineCreated { get; set; }
    }

    #endregion
}
