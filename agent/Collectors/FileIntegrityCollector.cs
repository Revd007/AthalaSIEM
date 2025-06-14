using AthalaSIEM.Agent.Models;
using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Text.Json;
using System.Linq;
using System.Collections.Concurrent;
using System.Text.RegularExpressions;

namespace AthalaSIEM.Agent.Collectors
{
    /// <summary>
    /// Enhanced File Integrity Monitoring with multi-collector integration
    /// </summary>
    public class FileIntegrityCollector : ILogCollector
    {
        private readonly ILogger<FileIntegrityCollector> _logger;
        private readonly ILogNormalizer _normalizer;
        private readonly Dictionary<string, FileSystemWatcher> _watchers = new();
        private readonly Dictionary<string, FileIntegrityInfo> _knownFiles = new();
        private readonly object _lockObject = new();
        private bool _isRunning;
        private bool _isPaused;
        private string _errorMessage = string.Empty;
        private CollectorSettings _settings = new();
        
        // Enhanced configuration for multi-collector integration
        private List<string> _monitoredPaths = new();
        private List<string> _excludePatterns = new();
        private List<string> _criticalPaths = new();
        private List<string> _containerPaths = new(); // Container-specific paths
        private List<string> _databasePaths = new(); // Database-specific paths
        private List<string> _iotConfigPaths = new(); // IoT device configuration paths
        private List<string> _cloudConfigPaths = new(); // Cloud service configuration paths
        private bool _realTimeMonitoring = true;
        private int _scanIntervalMinutes = 60;
        private int _maxEventsPerBatch = 50;
        private int _batchIntervalSeconds = 10;
        private int _maxBufferSize = 1000;
        private bool _enableDetailedLogging = false;
        private bool _enablePerformanceOptimization = true;
        private bool _enableThreatIntelligence = true;
        private bool _enableMultiCollectorIntegration = true;
        
        private readonly Queue<NormalizedLogEntry> _eventBuffer = new();
        private Timer? _batchTimer;
        private Timer? _scanTimer;
        private CancellationTokenSource? _cancellationTokenSource;

        /// <summary>
        /// Event raised when a log is collected
        /// </summary>
        public event EventHandler<NormalizedLogEntry>? LogCollected;

        /// <summary>
        /// Gets the type of the collector
        /// </summary>
        public string CollectorType => "FileIntegrity";

        /// <summary>
        /// Gets the status of the collector
        /// </summary>
        public CollectorStatus Status => _isRunning ? (_isPaused ? CollectorStatus.Paused : CollectorStatus.Running) : 
                                        (!string.IsNullOrEmpty(_errorMessage) ? CollectorStatus.Error : CollectorStatus.Stopped);

        /// <summary>
        /// Gets the error message if the collector is in an error state
        /// </summary>
        public string ErrorMessage => _errorMessage;

        /// <summary>
        /// Gets a value indicating whether the collector is running
        /// </summary>
        public bool IsRunning => _isRunning;

        /// <summary>
        /// Gets a value indicating whether the collector is paused
        /// </summary>
        public bool IsPaused => _isPaused;

        /// <summary>
        /// Gets the collector settings
        /// </summary>
        public CollectorSettings Settings => _settings;

        public FileIntegrityCollector(ILogger<FileIntegrityCollector> logger, ILogNormalizer normalizer)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _normalizer = normalizer ?? throw new ArgumentNullException(nameof(normalizer));
        }

        /// <summary>
        /// Initializes the collector with the specified settings
        /// </summary>
        /// <param name="settings">Collector settings</param>
        /// <returns>True if initialization was successful, otherwise false</returns>
        public bool Initialize(CollectorSettings settings)
        {
            _settings = settings ?? throw new ArgumentNullException(nameof(settings));
            _logger.LogInformation("Initializing Enhanced File Integrity Collector");

            try
            {
                ParseSettings();
                InitializePathCategories();
                _logger.LogInformation("Enhanced FIM initialized - Paths: {Count}, Real-time: {RealTime}, Multi-collector: {MultiCollector}", 
                    _monitoredPaths.Count, _realTimeMonitoring, _enableMultiCollectorIntegration);
                return true;
            }
            catch (Exception ex)
            {
                _errorMessage = ex.Message;
                _logger.LogError(ex, "Failed to initialize Enhanced File Integrity Collector");
                return false;
            }
        }

        /// <summary>
        /// Starts the collector
        /// </summary>
        public async Task StartAsync()
        {
            if (_isRunning) return;

            try
            {
                _logger.LogInformation("Starting Enhanced File Integrity Collector");
                _cancellationTokenSource = new CancellationTokenSource();

                if (_realTimeMonitoring)
                {
                    SetupFileSystemWatchers();
                }

                // Start periodic full scan
                _scanTimer = new Timer(async _ => await PerformFullScanAsync(), null, TimeSpan.Zero, TimeSpan.FromMinutes(_scanIntervalMinutes));

                // Start batch processing timer
                _batchTimer = new Timer(ProcessEventBatch, null, TimeSpan.FromSeconds(_batchIntervalSeconds), TimeSpan.FromSeconds(_batchIntervalSeconds));

                _isRunning = true;
                _isPaused = false;
                _errorMessage = string.Empty;

                _logger.LogInformation("Enhanced File Integrity Collector started successfully");
            }
            catch (Exception ex)
            {
                _errorMessage = ex.Message;
                _logger.LogError(ex, "Failed to start Enhanced File Integrity Collector");
                throw;
            }
        }

        /// <summary>
        /// Stops the collector
        /// </summary>
        public async Task StopAsync()
        {
            if (!_isRunning) return;

            try
            {
                _logger.LogInformation("Stopping Enhanced File Integrity Collector");
                
                _isRunning = false;
                _cancellationTokenSource?.Cancel();
                
                // Stop timers
                _scanTimer?.Dispose();
                _batchTimer?.Dispose();
                
                // Stop file watchers
                foreach (var watcher in _watchers.Values)
                {
                    watcher.EnableRaisingEvents = false;
                    watcher.Dispose();
                }
                _watchers.Clear();
                
                // Process remaining events
                ProcessEventBatch(null);
                
                _logger.LogInformation("Enhanced File Integrity Collector stopped");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error stopping Enhanced File Integrity Collector");
            }
        }

        /// <summary>
        /// Pauses the collector
        /// </summary>
        public Task PauseAsync()
        {
            _isPaused = true;
            foreach (var watcher in _watchers.Values)
            {
                watcher.EnableRaisingEvents = false;
            }
            _logger.LogInformation("Enhanced File Integrity Collector paused");
            return Task.CompletedTask;
        }

        /// <summary>
        /// Resumes the collector
        /// </summary>
        public Task ResumeAsync()
        {
            _isPaused = false;
            foreach (var watcher in _watchers.Values)
            {
                watcher.EnableRaisingEvents = true;
            }
            _logger.LogInformation("Enhanced File Integrity Collector resumed");
            return Task.CompletedTask;
        }

        /// <summary>
        /// Collects logs on demand
        /// </summary>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>The number of logs collected</returns>
        public async Task<int> CollectLogsAsync(CancellationToken cancellationToken)
        {
            if (_isPaused || !_isRunning)
                return 0;

            int collectedCount = 0;

            try
            {
                await PerformFullScanAsync();
                collectedCount = _eventBuffer.Count;
                ProcessEventBatch(null);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error collecting file integrity logs");
                _errorMessage = ex.Message;
            }

            return collectedCount;
        }

        /// <summary>
        /// Gets collector statistics
        /// </summary>
        public CollectorStats GetStats()
        {
            return new CollectorStats
            {
                IsRunning = _isRunning,
                IsPaused = _isPaused,
                LastError = _errorMessage,
                FilesMonitored = _knownFiles.Count,
                WatchersActive = _watchers.Count(w => w.Value.EnableRaisingEvents)
            };
        }

        private void ParseSettings()
        {
            if (_settings.Properties.ContainsKey("MonitoredPaths"))
            {
                _monitoredPaths = new List<string>(_settings.Properties["MonitoredPaths"].Split(',').Select(p => p.Trim()));
            }

            if (_settings.Properties.ContainsKey("ExcludePatterns"))
            {
                _excludePatterns = new List<string>(_settings.Properties["ExcludePatterns"].Split(',').Select(p => p.Trim()));
            }

            if (_settings.Properties.ContainsKey("CriticalPaths"))
            {
                _criticalPaths = new List<string>(_settings.Properties["CriticalPaths"].Split(',').Select(p => p.Trim()));
            }

            if (_settings.Properties.ContainsKey("RealTimeMonitoring"))
            {
                bool.TryParse(_settings.Properties["RealTimeMonitoring"], out _realTimeMonitoring);
            }

            if (_settings.Properties.ContainsKey("ScanIntervalMinutes"))
            {
                int.TryParse(_settings.Properties["ScanIntervalMinutes"], out _scanIntervalMinutes);
            }

            if (_settings.Properties.ContainsKey("MaxEventsPerBatch"))
            {
                int.TryParse(_settings.Properties["MaxEventsPerBatch"], out _maxEventsPerBatch);
            }

            if (_settings.Properties.ContainsKey("BatchIntervalSeconds"))
            {
                int.TryParse(_settings.Properties["BatchIntervalSeconds"], out _batchIntervalSeconds);
            }

            if (_settings.Properties.ContainsKey("EnableThreatIntelligence"))
            {
                bool.TryParse(_settings.Properties["EnableThreatIntelligence"], out _enableThreatIntelligence);
            }

            if (_settings.Properties.ContainsKey("EnableMultiCollectorIntegration"))
            {
                bool.TryParse(_settings.Properties["EnableMultiCollectorIntegration"], out _enableMultiCollectorIntegration);
            }
        }

        private void InitializePathCategories()
        {
            // Container-specific paths
            _containerPaths.AddRange(new[]
            {
                "/var/lib/docker",
                "/var/lib/containerd",
                "/etc/docker",
                "/etc/kubernetes",
                "C:\\ProgramData\\Docker",
                "C:\\ProgramData\\containerd"
            });

            // Database-specific paths
            _databasePaths.AddRange(new[]
            {
                "/var/lib/mysql",
                "/var/lib/postgresql",
                "/var/lib/mongodb",
                "/etc/mysql",
                "/etc/postgresql",
                "C:\\Program Files\\Microsoft SQL Server",
                "C:\\Program Files\\MySQL",
                "C:\\Program Files\\PostgreSQL"
            });

            // IoT configuration paths
            _iotConfigPaths.AddRange(new[]
            {
                "/etc/iot",
                "/opt/iot",
                "/usr/local/iot",
                "C:\\Program Files\\IoT",
                "C:\\IoT"
            });

            // Cloud service configuration paths
            _cloudConfigPaths.AddRange(new[]
            {
                "/root/.aws",
                "/root/.azure",
                "/root/.gcp",
                "C:\\Users\\%USERNAME%\\.aws",
                "C:\\Users\\%USERNAME%\\.azure"
            });
        }

        private void SetupFileSystemWatchers()
        {
            foreach (var path in _monitoredPaths)
            {
                try
                {
                    if (!Directory.Exists(path) && !File.Exists(path))
                    {
                        _logger.LogWarning("Monitored path does not exist: {Path}", path);
                        continue;
                    }

                    var watcher = new FileSystemWatcher();
                    
                    if (File.Exists(path))
                    {
                        watcher.Path = Path.GetDirectoryName(path) ?? "";
                        watcher.Filter = Path.GetFileName(path);
                    }
                    else
                    {
                        watcher.Path = path;
                        watcher.Filter = "*.*";
                    }

                    watcher.IncludeSubdirectories = true;
                    watcher.NotifyFilter = NotifyFilters.CreationTime | NotifyFilters.LastWrite | 
                                         NotifyFilters.FileName | NotifyFilters.DirectoryName | 
                                         NotifyFilters.Size | NotifyFilters.Attributes | 
                                         NotifyFilters.Security;

                    watcher.Created += OnFileSystemEvent;
                    watcher.Changed += OnFileSystemEvent;
                    watcher.Deleted += OnFileSystemEvent;
                    watcher.Renamed += OnFileSystemRenamed;
                    watcher.Error += OnFileSystemError;

                    watcher.EnableRaisingEvents = true;
                    _watchers[path] = watcher;

                    _logger.LogInformation("File system watcher setup for: {Path}", path);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Failed to setup watcher for path: {Path}", path);
                }
            }
        }

        private void OnFileSystemEvent(object sender, FileSystemEventArgs e)
        {
            if (_isPaused || ShouldExcludeFile(e.FullPath)) return;

            try
            {
                var fileInfo = new FileInfo(e.FullPath);
                if (!fileInfo.Exists && e.ChangeType != WatcherChangeTypes.Deleted) return;

                var eventEntry = CreateFileIntegrityEvent(e.FullPath, e.ChangeType.ToString(), null, fileInfo);
                
                if (eventEntry != null)
                {
                    lock (_lockObject)
                    {
                        if (_eventBuffer.Count >= _maxBufferSize)
                        {
                            _eventBuffer.Dequeue();
                        }
                        _eventBuffer.Enqueue(eventEntry);
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error processing file system event for: {Path}", e.FullPath);
            }
        }

        private void OnFileSystemRenamed(object sender, RenamedEventArgs e)
        {
            if (_isPaused || ShouldExcludeFile(e.FullPath)) return;

            try
            {
                var fileInfo = new FileInfo(e.FullPath);
                var eventEntry = CreateFileIntegrityEvent(e.FullPath, "Renamed", e.OldFullPath, fileInfo);
                
                if (eventEntry != null)
                {
                    lock (_lockObject)
                    {
                        if (_eventBuffer.Count >= _maxBufferSize)
                        {
                            _eventBuffer.Dequeue();
                        }
                        _eventBuffer.Enqueue(eventEntry);
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error processing file rename event: {OldPath} -> {NewPath}", e.OldFullPath, e.FullPath);
            }
        }

        private void OnFileSystemError(object sender, ErrorEventArgs e)
        {
            _logger.LogError(e.GetException(), "File system watcher error");
        }

        private NormalizedLogEntry? CreateFileIntegrityEvent(string filePath, string changeType, string? oldPath, FileInfo? fileInfo)
        {
            try
            {
                var pathCategory = DeterminePathCategory(filePath);
                var severity = DetermineSeverity(filePath, changeType, pathCategory);
                var threatIndicators = _enableThreatIntelligence ? AnalyzeThreatIndicators(filePath, changeType) : new List<string>();
                
                var details = new
                {
                    file_path = filePath,
                    old_path = oldPath,
                    change_type = changeType,
                    path_category = pathCategory,
                    file_size = fileInfo?.Length ?? 0,
                    creation_time = fileInfo?.CreationTimeUtc.ToString("O"),
                    modification_time = fileInfo?.LastWriteTimeUtc.ToString("O"),
                    attributes = fileInfo?.Attributes.ToString(),
                    file_hash = GetFileHash(filePath),
                    threat_indicators = threatIndicators,
                    collector_integration = _enableMultiCollectorIntegration ? GetCollectorContext(pathCategory) : null
                };

                return new NormalizedLogEntry
                {
                    Id = Guid.NewGuid().ToString(),
                    Timestamp = DateTime.UtcNow,
                    Level = severity == "High" || severity == "Critical" ? "Warning" : "Information",
                    Source = $"FIM/{pathCategory}",
                    Category = "FileIntegrity",
                    EventId = $"FIM_{changeType.ToUpper()}",
                    Message = $"File {changeType.ToLower()}: {filePath}",
                    Details = JsonSerializer.Serialize(details),
                    Tags = CreateTags(pathCategory, changeType, threatIndicators),
                    Severity = severity
                };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error creating file integrity event for: {Path}", filePath);
                return null;
            }
        }

        private string DeterminePathCategory(string filePath)
        {
            if (_containerPaths.Any(cp => filePath.StartsWith(cp, StringComparison.OrdinalIgnoreCase)))
                return "Container";
            
            if (_databasePaths.Any(dp => filePath.StartsWith(dp, StringComparison.OrdinalIgnoreCase)))
                return "Database";
            
            if (_iotConfigPaths.Any(ip => filePath.StartsWith(ip, StringComparison.OrdinalIgnoreCase)))
                return "IoT";
            
            if (_cloudConfigPaths.Any(cp => filePath.StartsWith(cp, StringComparison.OrdinalIgnoreCase)))
                return "CloudServices";
            
            if (_criticalPaths.Any(cp => filePath.StartsWith(cp, StringComparison.OrdinalIgnoreCase)))
                return "Critical";
            
            return "General";
        }

        private string DetermineSeverity(string filePath, string changeType, string pathCategory)
        {
            // Critical paths always get high severity
            if (pathCategory == "Critical") return "Critical";
            
            // Container, Database, IoT, and Cloud paths get elevated severity
            if (pathCategory is "Container" or "Database" or "IoT" or "CloudServices")
            {
                return changeType switch
                {
                    "Deleted" => "High",
                    "Created" => "Medium",
                    "Changed" => "Medium",
                    "Renamed" => "Medium",
                    _ => "Low"
                };
            }
            
            // System files
            if (IsSystemFile(filePath))
            {
                return changeType switch
                {
                    "Deleted" => "High",
                    "Changed" => "Medium",
                    _ => "Low"
                };
            }
            
            return "Low";
        }

        private List<string> AnalyzeThreatIndicators(string filePath, string changeType)
        {
            var indicators = new List<string>();
            
            var fileName = Path.GetFileName(filePath).ToLowerInvariant();
            var extension = Path.GetExtension(filePath).ToLowerInvariant();
            
            // Suspicious file extensions
            var suspiciousExtensions = new[] { ".exe", ".bat", ".cmd", ".ps1", ".vbs", ".js", ".jar", ".scr", ".com", ".pif" };
            if (suspiciousExtensions.Contains(extension))
            {
                indicators.Add("suspicious_extension");
            }
            
            // Suspicious file names
            var suspiciousNames = new[] { "cmd.exe", "powershell.exe", "wscript.exe", "cscript.exe", "regsvr32.exe" };
            if (suspiciousNames.Any(name => fileName.Contains(name)))
            {
                indicators.Add("suspicious_filename");
            }
            
            // Hidden files in suspicious locations
            if (fileName.StartsWith(".") && changeType == "Created")
            {
                indicators.Add("hidden_file_creation");
            }
            
            // Temporary directories
            if (filePath.Contains("temp") || filePath.Contains("tmp"))
            {
                indicators.Add("temp_directory_activity");
            }
            
            return indicators;
        }

        private object? GetCollectorContext(string pathCategory)
        {
            return pathCategory switch
            {
                "Container" => new { collector_type = "Container", integration_level = "high", monitoring_scope = "container_configs" },
                "Database" => new { collector_type = "Database", integration_level = "high", monitoring_scope = "db_configs" },
                "IoT" => new { collector_type = "IoT", integration_level = "medium", monitoring_scope = "device_configs" },
                "CloudServices" => new { collector_type = "CloudServices", integration_level = "high", monitoring_scope = "cloud_configs" },
                _ => null
            };
        }

        private List<string> CreateTags(string pathCategory, string changeType, List<string> threatIndicators)
        {
            var tags = new List<string> { "fim", "file_integrity", pathCategory.ToLower(), changeType.ToLower() };
            
            if (threatIndicators.Any())
            {
                tags.Add("threat_detected");
                tags.AddRange(threatIndicators);
            }
            
            return tags;
        }

        private bool ShouldExcludeFile(string filePath)
        {
            var fileName = Path.GetFileName(filePath);
            return _excludePatterns.Any(pattern => 
                Regex.IsMatch(fileName, pattern.Replace("*", ".*"), RegexOptions.IgnoreCase));
        }

        private bool IsSystemFile(string filePath)
        {
            var systemPaths = new[]
            {
                "/bin", "/sbin", "/usr/bin", "/usr/sbin", "/etc",
                "C:\\Windows\\System32", "C:\\Windows\\SysWOW64", "C:\\Program Files"
            };
            
            return systemPaths.Any(sp => filePath.StartsWith(sp, StringComparison.OrdinalIgnoreCase));
        }

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

        private async Task PerformFullScanAsync()
        {
            if (_isPaused) return;
            
            try
            {
                _logger.LogInformation("Starting full file integrity scan");
                
                foreach (var path in _monitoredPaths)
                {
                    if (!Directory.Exists(path) && !File.Exists(path)) continue;
                    
                    await ScanPathAsync(path);
                }
                
                _logger.LogInformation("Full file integrity scan completed");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error during full file integrity scan");
            }
        }

        private async Task ScanPathAsync(string path)
        {
            try
            {
                if (File.Exists(path))
                {
                    await ScanFileAsync(path);
                }
                else if (Directory.Exists(path))
                {
                    var files = Directory.GetFiles(path, "*", SearchOption.AllDirectories);
                    foreach (var file in files)
                    {
                        if (ShouldExcludeFile(file)) continue;
                        await ScanFileAsync(file);
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error scanning path: {Path}", path);
            }
        }

        private async Task ScanFileAsync(string filePath)
        {
            try
            {
                var fileInfo = new FileInfo(filePath);
                if (!fileInfo.Exists) return;
                
                var currentHash = GetFileHash(filePath);
                if (currentHash == null) return;
                
                lock (_lockObject)
                {
                    if (_knownFiles.TryGetValue(filePath, out var knownInfo))
                    {
                        if (knownInfo.Hash != currentHash || 
                            knownInfo.LastModified != fileInfo.LastWriteTimeUtc ||
                            knownInfo.Size != fileInfo.Length)
                        {
                            var eventEntry = CreateFileIntegrityEvent(filePath, "Changed", null, fileInfo);
                            if (eventEntry != null && _eventBuffer.Count < _maxBufferSize)
                            {
                                _eventBuffer.Enqueue(eventEntry);
                            }
                        }
                        
                        // Update known file info
                        _knownFiles[filePath] = new FileIntegrityInfo
                        {
                            Hash = currentHash,
                            LastModified = fileInfo.LastWriteTimeUtc,
                            Size = fileInfo.Length
                        };
                    }
                    else
                    {
                        // New file discovered
                        _knownFiles[filePath] = new FileIntegrityInfo
                        {
                            Hash = currentHash,
                            LastModified = fileInfo.LastWriteTimeUtc,
                            Size = fileInfo.Length
                        };
                        
                        var eventEntry = CreateFileIntegrityEvent(filePath, "Discovered", null, fileInfo);
                        if (eventEntry != null && _eventBuffer.Count < _maxBufferSize)
                        {
                            _eventBuffer.Enqueue(eventEntry);
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error scanning file: {Path}", filePath);
            }
        }

        private void ProcessEventBatch(object? state)
        {
            if (_eventBuffer.Count == 0) return;
            
            try
            {
                var eventsToProcess = new List<NormalizedLogEntry>();
                
                lock (_lockObject)
                {
                    var count = Math.Min(_maxEventsPerBatch, _eventBuffer.Count);
                    for (int i = 0; i < count; i++)
                    {
                        if (_eventBuffer.Count > 0)
                        {
                            eventsToProcess.Add(_eventBuffer.Dequeue());
                        }
                    }
                }
                
                foreach (var eventEntry in eventsToProcess)
                {
                    LogCollected?.Invoke(this, eventEntry);
                }
                
                if (_enableDetailedLogging && eventsToProcess.Count > 0)
                {
                    _logger.LogInformation("Processed {Count} file integrity events", eventsToProcess.Count);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error processing event batch");
            }
        }

        /// <summary>
        /// Disposes resources
        /// </summary>
        public void Dispose()
        {
            StopAsync().Wait();
            
            _scanTimer?.Dispose();
            _batchTimer?.Dispose();
            _cancellationTokenSource?.Dispose();
            
            foreach (var watcher in _watchers.Values)
            {
                watcher?.Dispose();
            }
        }

        private class FileIntegrityInfo
        {
            public string Hash { get; set; } = string.Empty;
            public DateTime LastModified { get; set; }
            public long Size { get; set; }
        }
    }

    /// <summary>
    /// Collector statistics
    /// </summary>
    public class CollectorStats
    {
        public bool IsRunning { get; set; }
        public bool IsPaused { get; set; }
        public string LastError { get; set; } = string.Empty;
        public int FilesMonitored { get; set; }
        public int WatchersActive { get; set; }
    }
} 