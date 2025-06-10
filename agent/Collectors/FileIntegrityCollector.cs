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

namespace AthalaSIEM.Agent.Collectors
{
    /// <summary>
    /// File Integrity Monitoring collector that monitors file changes
    /// </summary>
    public class FileIntegrityCollector : ILogCollector
    {
        private readonly ILogger<FileIntegrityCollector> _logger;
        private readonly ILogNormalizer _normalizer;
        private Timer? _scanTimer;
        private readonly List<FileSystemWatcher> _watchers = new();
        private readonly Dictionary<string, FileIntegrityData> _fileBaseline = new();
        private readonly object _lockObject = new();
        private bool _isRunning;
        private bool _isPaused;
        private string _errorMessage = string.Empty;
        private CollectorSettings _settings = new();
        private readonly List<string> _monitoredPaths = new();
        private readonly List<string> _excludePatterns = new();
        private bool _realTimeMonitoring = true;
        private int _scanIntervalMinutes = 60;
        
        /// <summary>
        /// Event raised when a log is collected
        /// </summary>
        public event EventHandler<NormalizedLogEntry>? LogCollected;

        /// <summary>
        /// Gets the type of the collector
        /// </summary>
        public string CollectorType => "FileIntegrity";

        /// <summary>
        /// Gets a value indicating whether the collector is running
        /// </summary>
        public bool IsRunning => _isRunning;

        /// <summary>
        /// Gets a value indicating whether the collector is paused
        /// </summary>
        public bool IsPaused => _isPaused;

        /// <summary>
        /// Gets the last error message
        /// </summary>
        public string LastError => _errorMessage;

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
        public void Initialize(CollectorSettings settings)
        {
            _settings = settings ?? throw new ArgumentNullException(nameof(settings));
            _logger.LogInformation("Initializing File Integrity Collector");

            try
            {
                // Parse settings
                ParseSettings();
                
                // Create baseline if it doesn't exist
                Task.Run(CreateInitialBaseline);
                
                _logger.LogInformation("File Integrity Collector initialized successfully");
            }
            catch (Exception ex)
            {
                _errorMessage = ex.Message;
                _logger.LogError(ex, "Failed to initialize File Integrity Collector");
                throw;
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
                _logger.LogInformation("Starting File Integrity Collector");
                
                if (_realTimeMonitoring)
                {
                    StartFileSystemWatchers();
                }
                
                // Start periodic full scan
                int intervalMs = _scanIntervalMinutes * 60 * 1000;
                _scanTimer = new Timer(async _ => await PerformFullScan(), null, intervalMs, intervalMs);
                
                _isRunning = true;
                _isPaused = false;
                _errorMessage = string.Empty;
                
                _logger.LogInformation("File Integrity Collector started successfully");
            }
            catch (Exception ex)
            {
                _errorMessage = ex.Message;
                _logger.LogError(ex, "Failed to start File Integrity Collector");
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
                _logger.LogInformation("Stopping File Integrity Collector");
                
                _scanTimer?.Dispose();
                _scanTimer = null;
                
                // Stop file system watchers
                foreach (var watcher in _watchers)
                {
                    watcher.EnableRaisingEvents = false;
                    watcher.Dispose();
                }
                _watchers.Clear();
                
                _isRunning = false;
                _logger.LogInformation("File Integrity Collector stopped");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error stopping File Integrity Collector");
            }
        }

        /// <summary>
        /// Pauses the collector
        /// </summary>
        public void Pause()
        {
            if (!_isRunning || _isPaused) return;
            
            _isPaused = true;
            foreach (var watcher in _watchers)
            {
                watcher.EnableRaisingEvents = false;
            }
            
            _logger.LogInformation("File Integrity Collector paused");
        }

        /// <summary>
        /// Resumes the collector
        /// </summary>
        public void Resume()
        {
            if (!_isRunning || !_isPaused) return;
            
            _isPaused = false;
            foreach (var watcher in _watchers)
            {
                watcher.EnableRaisingEvents = true;
            }
            
            _logger.LogInformation("File Integrity Collector resumed");
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
                FilesMonitored = _fileBaseline.Count,
                WatchersActive = _watchers.Count(w => w.EnableRaisingEvents)
            };
        }

        private void ParseSettings()
        {
            if (_settings.Properties.ContainsKey("MonitoredPaths"))
            {
                var paths = _settings.Properties["MonitoredPaths"].Split(',', StringSplitOptions.RemoveEmptyEntries);
                _monitoredPaths.AddRange(paths.Select(p => p.Trim()));
            }
            else
            {
                // Default critical system paths
                if (OperatingSystem.IsWindows())
                {
                    _monitoredPaths.AddRange(new[]
                    {
                        @"C:\Windows\System32",
                        @"C:\Program Files",
                        @"C:\Program Files (x86)"
                    });
                }
                else if (OperatingSystem.IsLinux())
                {
                    _monitoredPaths.AddRange(new[]
                    {
                        "/etc",
                        "/bin",
                        "/sbin",
                        "/usr/bin",
                        "/usr/sbin"
                    });
                }
            }

            if (_settings.Properties.ContainsKey("ExcludePatterns"))
            {
                var patterns = _settings.Properties["ExcludePatterns"].Split(',', StringSplitOptions.RemoveEmptyEntries);
                _excludePatterns.AddRange(patterns.Select(p => p.Trim()));
            }
            else
            {
                // Default exclude patterns
                _excludePatterns.AddRange(new[]
                {
                    "*.tmp", "*.log", "*.swp", "*.lock", "*~"
                });
            }

            if (_settings.Properties.ContainsKey("RealTimeMonitoring"))
            {
                bool.TryParse(_settings.Properties["RealTimeMonitoring"], out _realTimeMonitoring);
            }

            if (_settings.Properties.ContainsKey("ScanIntervalMinutes"))
            {
                int.TryParse(_settings.Properties["ScanIntervalMinutes"], out _scanIntervalMinutes);
            }
        }

        private async Task CreateInitialBaseline()
        {
            _logger.LogInformation("Creating initial file integrity baseline");
            
            try
            {
                foreach (var path in _monitoredPaths)
                {
                    if (Directory.Exists(path))
                    {
                        await ScanDirectory(path, isBaseline: true);
                    }
                    else if (File.Exists(path))
                    {
                        await ScanFile(path, isBaseline: true);
                    }
                }
                
                _logger.LogInformation("Initial baseline created with {Count} files", _fileBaseline.Count);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error creating initial baseline");
            }
        }

        private void StartFileSystemWatchers()
        {
            foreach (var path in _monitoredPaths)
            {
                if (!Directory.Exists(path)) continue;

                try
                {
                    var watcher = new FileSystemWatcher(path)
                    {
                        IncludeSubdirectories = true,
                        NotifyFilter = NotifyFilters.LastWrite | NotifyFilters.CreationTime | 
                                     NotifyFilters.FileName | NotifyFilters.DirectoryName | 
                                     NotifyFilters.Size | NotifyFilters.Attributes
                    };

                    watcher.Changed += OnFileChanged;
                    watcher.Created += OnFileCreated;
                    watcher.Deleted += OnFileDeleted;
                    watcher.Renamed += OnFileRenamed;
                    watcher.Error += OnWatcherError;

                    watcher.EnableRaisingEvents = true;
                    _watchers.Add(watcher);
                    
                    _logger.LogDebug("Started file system watcher for path: {Path}", path);
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Failed to start watcher for path: {Path}", path);
                }
            }
        }

        private async void OnFileChanged(object sender, FileSystemEventArgs e)
        {
            if (_isPaused || ShouldExcludeFile(e.FullPath)) return;
            
            try
            {
                await ProcessFileChange(e.FullPath, "Modified");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error processing file change: {Path}", e.FullPath);
            }
        }

        private async void OnFileCreated(object sender, FileSystemEventArgs e)
        {
            if (_isPaused || ShouldExcludeFile(e.FullPath)) return;
            
            try
            {
                await ProcessFileChange(e.FullPath, "Created");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error processing file creation: {Path}", e.FullPath);
            }
        }

        private void OnFileDeleted(object sender, FileSystemEventArgs e)
        {
            if (_isPaused || ShouldExcludeFile(e.FullPath)) return;
            
            try
            {
                ProcessFileDelete(e.FullPath);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error processing file deletion: {Path}", e.FullPath);
            }
        }

        private async void OnFileRenamed(object sender, RenamedEventArgs e)
        {
            if (_isPaused || ShouldExcludeFile(e.FullPath)) return;
            
            try
            {
                ProcessFileDelete(e.OldFullPath);
                await ProcessFileChange(e.FullPath, "Renamed");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error processing file rename: {OldPath} -> {NewPath}", e.OldFullPath, e.FullPath);
            }
        }

        private void OnWatcherError(object sender, ErrorEventArgs e)
        {
            _logger.LogError(e.GetException(), "File system watcher error");
        }

        private async Task ProcessFileChange(string filePath, string changeType)
        {
            try
            {
                var fileInfo = new FileInfo(filePath);
                if (!fileInfo.Exists) return;

                var currentHash = await CalculateFileHash(filePath);
                var currentData = new FileIntegrityData
                {
                    FilePath = filePath,
                    Hash = currentHash,
                    Size = fileInfo.Length,
                    LastModified = fileInfo.LastWriteTimeUtc,
                    Created = fileInfo.CreationTimeUtc,
                    Attributes = fileInfo.Attributes.ToString()
                };

                lock (_lockObject)
                {
                    if (_fileBaseline.TryGetValue(filePath, out var baselineData))
                    {
                        if (baselineData.Hash != currentHash)
                        {
                            // File has been modified
                            GenerateIntegrityAlert(filePath, changeType, baselineData, currentData);
                            _fileBaseline[filePath] = currentData;
                        }
                    }
                    else
                    {
                        // New file
                        GenerateIntegrityAlert(filePath, changeType, null, currentData);
                        _fileBaseline[filePath] = currentData;
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error processing file change for: {Path}", filePath);
            }
        }

        private void ProcessFileDelete(string filePath)
        {
            lock (_lockObject)
            {
                if (_fileBaseline.TryGetValue(filePath, out var baselineData))
                {
                    GenerateIntegrityAlert(filePath, "Deleted", baselineData, null);
                    _fileBaseline.Remove(filePath);
                }
            }
        }

        private async Task PerformFullScan()
        {
            if (_isPaused) return;
            
            _logger.LogInformation("Starting full file integrity scan");
            
            try
            {
                foreach (var path in _monitoredPaths)
                {
                    if (Directory.Exists(path))
                    {
                        await ScanDirectory(path, isBaseline: false);
                    }
                    else if (File.Exists(path))
                    {
                        await ScanFile(path, isBaseline: false);
                    }
                }
                
                _logger.LogInformation("Full file integrity scan completed");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error during full scan");
            }
        }

        private async Task ScanDirectory(string directoryPath, bool isBaseline)
        {
            try
            {
                var files = Directory.GetFiles(directoryPath, "*", SearchOption.AllDirectories);
                
                foreach (var file in files)
                {
                    if (ShouldExcludeFile(file)) continue;
                    await ScanFile(file, isBaseline);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error scanning directory: {Path}", directoryPath);
            }
        }

        private async Task ScanFile(string filePath, bool isBaseline)
        {
            try
            {
                var fileInfo = new FileInfo(filePath);
                if (!fileInfo.Exists) return;

                var hash = await CalculateFileHash(filePath);
                var currentData = new FileIntegrityData
                {
                    FilePath = filePath,
                    Hash = hash,
                    Size = fileInfo.Length,
                    LastModified = fileInfo.LastWriteTimeUtc,
                    Created = fileInfo.CreationTimeUtc,
                    Attributes = fileInfo.Attributes.ToString()
                };

                lock (_lockObject)
                {
                    if (isBaseline)
                    {
                        _fileBaseline[filePath] = currentData;
                    }
                    else
                    {
                        if (_fileBaseline.TryGetValue(filePath, out var baselineData))
                        {
                            if (baselineData.Hash != hash)
                            {
                                GenerateIntegrityAlert(filePath, "Modified", baselineData, currentData);
                                _fileBaseline[filePath] = currentData;
                            }
                        }
                        else
                        {
                            GenerateIntegrityAlert(filePath, "Created", null, currentData);
                            _fileBaseline[filePath] = currentData;
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error scanning file: {Path}", filePath);
            }
        }

        private async Task<string> CalculateFileHash(string filePath)
        {
            try
            {
                using var stream = File.OpenRead(filePath);
                using var sha256 = SHA256.Create();
                var hashBytes = await sha256.ComputeHashAsync(stream);
                return Convert.ToHexString(hashBytes);
            }
            catch
            {
                return string.Empty;
            }
        }

        private bool ShouldExcludeFile(string filePath)
        {
            var fileName = Path.GetFileName(filePath);
            
            foreach (var pattern in _excludePatterns)
            {
                if (IsMatch(fileName, pattern))
                {
                    return true;
                }
            }
            
            return false;
        }

        private static bool IsMatch(string fileName, string pattern)
        {
            // Simple wildcard matching
            if (pattern.Contains('*'))
            {
                var regex = pattern.Replace("*", ".*").Replace("?", ".");
                return System.Text.RegularExpressions.Regex.IsMatch(fileName, $"^{regex}$", System.Text.RegularExpressions.RegexOptions.IgnoreCase);
            }
            
            return string.Equals(fileName, pattern, StringComparison.OrdinalIgnoreCase);
        }

        private void GenerateIntegrityAlert(string filePath, string changeType, FileIntegrityData? baseline, FileIntegrityData? current)
        {
            try
            {
                var alertData = new Dictionary<string, object>
                {
                    ["change_type"] = changeType,
                    ["file_path"] = filePath,
                    ["timestamp"] = DateTime.UtcNow
                };

                if (baseline != null)
                {
                    alertData["baseline_hash"] = baseline.Hash;
                    alertData["baseline_size"] = baseline.Size;
                    alertData["baseline_modified"] = baseline.LastModified;
                }

                if (current != null)
                {
                    alertData["current_hash"] = current.Hash;
                    alertData["current_size"] = current.Size;
                    alertData["current_modified"] = current.LastModified;
                    alertData["file_attributes"] = current.Attributes;
                }

                var logEntry = new NormalizedLogEntry
                {
                    Id = Guid.NewGuid().ToString(),
                    Timestamp = DateTime.UtcNow,
                    Level = LogLevel.Warning.ToString(),
                    Source = "FileIntegrityMonitoring",
                    Category = "Security",
                    EventId = "FIM001",
                    Message = $"File integrity change detected: {changeType} - {filePath}",
                    Details = JsonSerializer.Serialize(alertData),
                    Tags = new List<string> { "FIM", "FileIntegrity", changeType },
                    Severity = changeType == "Deleted" ? "High" : "Medium"
                };

                // Normalize the log entry
                var normalizedEntry = _normalizer.NormalizeLogEntry(logEntry);
                
                // Raise the LogCollected event
                LogCollected?.Invoke(this, normalizedEntry);
                
                _logger.LogWarning("File integrity alert: {ChangeType} - {FilePath}", changeType, filePath);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error generating integrity alert for: {Path}", filePath);
            }
        }

        /// <summary>
        /// Disposes resources
        /// </summary>
        public void Dispose()
        {
            StopAsync().Wait();
            foreach (var watcher in _watchers)
            {
                watcher?.Dispose();
            }
            _scanTimer?.Dispose();
        }
    }

    /// <summary>
    /// File integrity data structure
    /// </summary>
    internal class FileIntegrityData
    {
        public string FilePath { get; set; } = string.Empty;
        public string Hash { get; set; } = string.Empty;
        public long Size { get; set; }
        public DateTime LastModified { get; set; }
        public DateTime Created { get; set; }
        public string Attributes { get; set; } = string.Empty;
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