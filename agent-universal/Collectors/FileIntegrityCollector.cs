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
    /// File Integrity Monitoring (FIM) Collector for AthalaSIEM Universal Agent
    /// Monitors critical system files for unauthorized changes
    /// </summary>
    public class FileIntegrityCollector : ILogCollector
    {
        public string CollectorName => "File Integrity Monitor";
        public Core.OperatingSystem SupportedOS => Core.OperatingSystem.Windows;
        public bool IsActive { get; private set; }
        public long LogsCollected { get; private set; }

        private readonly ILogger<FileIntegrityCollector> _logger;
        private readonly List<LogEntry> _collectedLogs = new List<LogEntry>();
        private readonly Dictionary<string, FileSystemWatcher> _watchers = new();
        private readonly Dictionary<string, string> _fileHashes = new();
        private readonly List<string> _monitoredPaths = new();
        private readonly CancellationTokenSource _cancellationTokenSource = new();
        private Timer? _scanTimer;

        public event EventHandler<LogCollectedEventArgs>? LogCollected;
        public event EventHandler<LogCollectionErrorEventArgs>? CollectionError;

        public FileIntegrityCollector(ILogger<FileIntegrityCollector> logger)
        {
            _logger = logger;
        }

        public Task<bool> InitializeAsync(Dictionary<string, object> config, CancellationToken cancellationToken = default)
        {
            try
            {
                _logger.LogInformation("Initializing File Integrity Monitor");

                // Default critical paths to monitor
                var defaultPaths = new[]
                {
                    @"C:\Windows\System32\drivers",
                    @"C:\Windows\System32\config", 
                    @"C:\Windows\System32\*.exe",
                    @"C:\Program Files\AthalaSIEM",
                    @"C:\inetpub\wwwroot"
                };

                if (config.ContainsKey("MonitoredPaths"))
                {
                    var configPaths = config["MonitoredPaths"].ToString()?.Split(',') ?? defaultPaths;
                    _monitoredPaths.AddRange(configPaths.Select(p => p.Trim()));
                }
                else
                {
                    _monitoredPaths.AddRange(defaultPaths);
                }

                return Task.FromResult(true);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to initialize File Integrity Monitor");
                return Task.FromResult(false);
            }
        }

        public Task StartCollectionAsync(CancellationToken cancellationToken = default)
        {
            IsActive = true;
            
            try
            {
                // Setup file system watchers for real-time monitoring
                SetupFileWatchers();
                
                // Start periodic full scan (every 30 minutes)
                _scanTimer = new Timer(PerformFullScan, null, TimeSpan.Zero, TimeSpan.FromMinutes(30));
                
                _logger.LogInformation("File Integrity Monitor started - monitoring {Count} paths", _monitoredPaths.Count);
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
            
            return Task.CompletedTask;
        }

        public Task<IEnumerable<LogEntry>> GetLogsAsync(int batchSize = 100, CancellationToken cancellationToken = default)
        {
            var logs = _collectedLogs.Take(batchSize).ToList();
            _collectedLogs.RemoveRange(0, logs.Count);
            return Task.FromResult<IEnumerable<LogEntry>>(logs);
        }

        public Task<CollectorHealth> GetHealthAsync()
        {
            return Task.FromResult(new CollectorHealth
            {
                IsHealthy = true,
                Status = IsActive ? "Running" : "Stopped",
                LogsCollected = LogsCollected,
                LastCollection = DateTime.UtcNow,
                Metrics = new Dictionary<string, object>
                {
                    ["MonitoredPaths"] = _monitoredPaths.Count,
                    ["ActiveWatchers"] = _watchers.Count,
                    ["TrackedFiles"] = _fileHashes.Count,
                    ["BufferedLogs"] = _collectedLogs.Count
                }
            });
        }

        private void SetupFileWatchers()
        {
            foreach (var path in _monitoredPaths)
            {
                try
                {
                    if (Directory.Exists(path))
                    {
                        var watcher = new FileSystemWatcher(path)
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
                        _watchers[path] = watcher;

                        _logger.LogDebug("File watcher setup for: {Path}", path);
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Failed to setup watcher for path: {Path}", path);
                }
            }
        }

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

        private void PerformFullScan(object? state)
        {
            if (!IsActive) return;

            try
            {
                _logger.LogInformation("Starting FIM full scan");

                foreach (var path in _monitoredPaths)
                {
                    if (Directory.Exists(path))
                    {
                        ScanDirectory(path);
                    }
                    else if (File.Exists(path))
                    {
                        ScanFile(path);
                    }
                }

                _logger.LogInformation("FIM full scan completed");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error during FIM full scan");
            }
        }

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
                    Properties = new Dictionary<string, object>
                    {
                        ["FilePath"] = filePath,
                        ["OldPath"] = oldPath ?? "",
                        ["ChangeType"] = changeType,
                        ["FileSize"] = fileInfo?.Length ?? 0,
                        ["FileHash"] = GetFileHash(filePath) ?? "",
                        ["LastModified"] = fileInfo?.LastWriteTimeUtc ?? DateTime.UtcNow,
                        ["ComputerName"] = Environment.MachineName,
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

        private string DetermineSeverity(string filePath, string changeType)
        {
            // Critical system paths
            if (filePath.Contains(@"System32\drivers") || 
                filePath.Contains(@"System32\config") ||
                filePath.Contains(@"System32\*.exe"))
            {
                return "Critical";
            }

            // Important application paths
            if (filePath.Contains("Program Files") || filePath.Contains("inetpub"))
            {
                return "High";
            }

            return "Medium";
        }

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
            if (filePath.Contains("System32") && changeType == "Modified")
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

        public async ValueTask DisposeAsync()
        {
            await StopCollectionAsync();
            _scanTimer?.Dispose();
            _cancellationTokenSource?.Dispose();
        }
    }
} 