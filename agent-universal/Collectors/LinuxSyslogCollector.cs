using AthalaSIEM.UniversalAgent.Core;
using AthalaSIEM.UniversalAgent.Models;
using Microsoft.Extensions.Logging;
using System.Runtime.InteropServices;
using System.Runtime.Versioning;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System;

namespace AthalaSIEM.UniversalAgent.Core.Collectors
{
    /// <summary>
    /// Linux Syslog Collector - Cross-distribution compatibility
    /// Integrates with existing AthalaSIEM collector pipeline
    /// </summary>
    [SupportedOSPlatform("linux")]
    public class LinuxSyslogCollector : ILogCollector
    {
        private readonly ILogger<LinuxSyslogCollector> _logger;
        private readonly List<FileSystemWatcher> _watchers = new();
        private readonly List<LogEntry> _collectedLogs = new();
        private readonly CancellationTokenSource _cancellationTokenSource = new();
        private LinuxSyslogConfiguration _config = new();
        private SystemdJournalReader? _journalReader;
        private SyslogParser _parser;
        private bool _isActive = false;
        private long _logsCollected = 0;

        public string CollectorName => "Linux Syslog";
        public Core.OperatingSystem SupportedOS => Core.OperatingSystem.Linux;
        public bool IsActive => _isActive;
        public long LogsCollected => _logsCollected;

        public event EventHandler<LogCollectedEventArgs>? LogCollected;
        public event EventHandler<LogCollectionErrorEventArgs>? CollectionError;

        public LinuxSyslogCollector(ILogger<LinuxSyslogCollector> logger)
        {
            _logger = logger;
            _parser = new SyslogParser(_logger);
        }

        public async Task<bool> InitializeAsync(Dictionary<string, object> config, CancellationToken cancellationToken = default)
        {
            try
            {
                if (!RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
                {
                    _logger.LogWarning("LinuxSyslogCollector can only run on Linux systems");
                    return false;
                }

                _config = DetectLinuxConfiguration();
                _logger.LogInformation("Initializing Linux Syslog Collector for {Distribution} {Version}", 
                    _config.DistributionName, _config.Version);

                // Load configuration from backend
                LoadConfigurationFromBackend(config);

                // Initialize traditional syslog file watchers
                await InitializeSyslogFileWatchers();

                // Initialize systemd journal monitoring if available
                if (_config.HasSystemd && File.Exists("/usr/bin/journalctl"))
                {
                    _journalReader = new SystemdJournalReader(_logger);
                    await _journalReader.InitializeAsync();
                    _journalReader.JournalEntryReceived += OnJournalEntryReceived;
                }

                _logger.LogInformation("Linux Syslog Collector initialized with {WatcherCount} file watchers and {JournalStatus} journal reader",
                    _watchers.Count, _journalReader != null ? "active" : "disabled");

                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to initialize Linux Syslog Collector");
                CollectionError?.Invoke(this, new LogCollectionErrorEventArgs 
                { 
                    Exception = ex, 
                    Message = "Failed to initialize Linux Syslog Collector",
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
                    _logger.LogWarning("Linux Syslog Collector is already active");
                    return;
                }

                // Start file watchers
                foreach (var watcher in _watchers)
                {
                    watcher.EnableRaisingEvents = true;
                }

                // Start systemd journal monitoring
                if (_journalReader != null)
                {
                    await _journalReader.StartAsync();
                }

                _isActive = true;
                _logger.LogInformation("Linux Syslog Collector started successfully");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to start Linux Syslog Collector");
                CollectionError?.Invoke(this, new LogCollectionErrorEventArgs 
                { 
                    Exception = ex, 
                    Message = "Failed to start Linux Syslog Collector",
                    Source = CollectorName
                });
            }
        }

        public async Task StopCollectionAsync(CancellationToken cancellationToken = default)
        {
            try
            {
                // Stop file watchers
                foreach (var watcher in _watchers)
                {
                    watcher.EnableRaisingEvents = false;
                }

                // Stop systemd journal monitoring
                if (_journalReader != null)
                {
                    await _journalReader.StopAsync();
                }

                _isActive = false;
                _logger.LogInformation("Linux Syslog Collector stopped successfully");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error stopping Linux Syslog Collector");
                CollectionError?.Invoke(this, new LogCollectionErrorEventArgs 
                { 
                    Exception = ex, 
                    Message = "Error stopping Linux Syslog Collector",
                    Source = CollectorName
                });
            }
        }

        public Task<IEnumerable<LogEntry>> GetLogsAsync(int batchSize = 100, CancellationToken cancellationToken = default)
        {
            lock (_collectedLogs)
            {
                var logs = _collectedLogs.Take(batchSize).ToList();
                _collectedLogs.RemoveRange(0, logs.Count);
                return Task.FromResult<IEnumerable<LogEntry>>(logs);
            }
        }

        public Task<CollectorHealth> GetHealthAsync()
        {
            var health = new CollectorHealth
            {
                IsHealthy = _isActive && !_cancellationTokenSource.Token.IsCancellationRequested,
                Status = _isActive ? "Running" : "Stopped",
                LogsCollected = _logsCollected,
                LastCollection = DateTime.UtcNow,
                Uptime = DateTime.UtcNow - DateTime.UtcNow.AddMinutes(-5), // Placeholder
                Metrics = new Dictionary<string, object>
                {
                    {"ActiveWatchers", _watchers.Count},
                    {"HasJournalReader", _journalReader != null},
                    {"QueuedLogs", _collectedLogs.Count}
                }
            };

            return Task.FromResult(health);
        }

        private void LoadConfigurationFromBackend(Dictionary<string, object> config)
        {
            // Load syslog paths from backend configuration
            if (config.TryGetValue("SyslogPaths", out var pathsObj) && pathsObj is List<string> paths)
            {
                _config.CustomSyslogPaths = paths;
            }

            // Load other configuration settings
            if (config.TryGetValue("EnableSystemdJournal", out var enableJournalObj) && enableJournalObj is bool enableJournal)
            {
                _config.EnableSystemdJournal = enableJournal;
            }
        }

        private Task InitializeSyslogFileWatchers()
        {
            var syslogPaths = _config.GetSyslogPaths();
            
            foreach (var path in syslogPaths)
            {
                if (File.Exists(path.FilePath))
                {
                    var watcher = new FileSystemWatcher(
                        Path.GetDirectoryName(path.FilePath)!, 
                        Path.GetFileName(path.FilePath))
                    {
                        NotifyFilter = NotifyFilters.LastWrite | NotifyFilters.Size,
                        EnableRaisingEvents = false
                    };

                    watcher.Changed += async (sender, e) => await OnSyslogFileChanged(path, e.FullPath);
                    _watchers.Add(watcher);

                    _logger.LogDebug("File watcher configured for {LogPath} ({LogType})", 
                        path.FilePath, path.LogType);
                }
                else
                {
                    _logger.LogDebug("Syslog file not found: {LogPath}", path.FilePath);
                }
            }
            
            return Task.CompletedTask;
        }

        private async Task OnSyslogFileChanged(SyslogPath path, string filePath)
        {
            try
            {
                var newLines = await ReadNewLogLines(filePath);
                
                foreach (var line in newLines)
                {
                    var logEntry = _parser.ParseSyslogLine(line, path.LogType, _config);
                    if (logEntry != null)
                    {
                        lock (_collectedLogs)
                        {
                            _collectedLogs.Add(logEntry);
                            _logsCollected++;
                        }
                        
                        LogCollected?.Invoke(this, new LogCollectedEventArgs 
                        { 
                            Logs = new[] { logEntry },
                            Source = CollectorName,
                            CollectionTime = DateTime.UtcNow
                        });
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Error processing syslog file changes: {FilePath}", filePath);
                CollectionError?.Invoke(this, new LogCollectionErrorEventArgs 
                { 
                    Exception = ex, 
                    Message = $"Error processing syslog file changes: {filePath}",
                    Source = CollectorName
                });
            }
        }

        private void OnJournalEntryReceived(object? sender, JournalEntryEventArgs e)
        {
            try
            {
                var logEntry = _parser.ParseSystemdJournalEntry(e.Entry, _config);
                if (logEntry != null)
                {
                    lock (_collectedLogs)
                    {
                        _collectedLogs.Add(logEntry);
                        _logsCollected++;
                    }
                    
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
                _logger.LogWarning(ex, "Error processing systemd journal entry");
                CollectionError?.Invoke(this, new LogCollectionErrorEventArgs 
                { 
                    Exception = ex, 
                    Message = "Error processing systemd journal entry",
                    Source = CollectorName
                });
            }
        }

        private LinuxSyslogConfiguration DetectLinuxConfiguration()
        {
            return new LinuxSyslogConfiguration
            {
                DistributionName = DetectDistribution(),
                Version = DetectVersion(),
                HasSystemd = Directory.Exists("/run/systemd/system"),
                SyslogDaemon = DetectSyslogDaemon(),
                EnableSystemdJournal = Directory.Exists("/run/systemd/system")
            };
        }

        private string DetectDistribution()
        {
            try
            {
                if (File.Exists("/etc/os-release"))
                {
                    var content = File.ReadAllText("/etc/os-release");
                    var idMatch = System.Text.RegularExpressions.Regex.Match(content, @"^ID=""?([^""\n]+)""?", 
                        System.Text.RegularExpressions.RegexOptions.Multiline);
                    return idMatch.Success ? idMatch.Groups[1].Value : "unknown";
                }
                return "unknown";
            }
            catch
            {
                return "unknown";
            }
        }

        private string DetectVersion()
        {
            try
            {
                if (File.Exists("/etc/os-release"))
                {
                    var content = File.ReadAllText("/etc/os-release");
                    var versionMatch = System.Text.RegularExpressions.Regex.Match(content, @"^VERSION_ID=""?([^""\n]+)""?",
                        System.Text.RegularExpressions.RegexOptions.Multiline);
                    return versionMatch.Success ? versionMatch.Groups[1].Value : "unknown";
                }
                return "unknown";
            }
            catch
            {
                return "unknown";
            }
        }

        private SyslogDaemonType DetectSyslogDaemon()
        {
            try
            {
                var processes = System.Diagnostics.Process.GetProcesses();
                
                if (processes.Any(p => p.ProcessName.Contains("rsyslog")))
                    return SyslogDaemonType.Rsyslog;
                if (processes.Any(p => p.ProcessName.Contains("syslog-ng")))
                    return SyslogDaemonType.SyslogNg;
                if (Directory.Exists("/run/systemd/system"))
                    return SyslogDaemonType.SystemdJournal;
                if (processes.Any(p => p.ProcessName.Contains("busybox")))
                    return SyslogDaemonType.Busybox;
                
                return SyslogDaemonType.Unknown;
            }
            catch
            {
                return SyslogDaemonType.Unknown;
            }
        }

        private async Task<IEnumerable<string>> ReadNewLogLines(string filePath)
        {
            try
            {
                // Simple implementation - read last 100 lines
                // In production, this should track file position
                var lines = await File.ReadAllLinesAsync(filePath);
                return lines.TakeLast(100);
            }
            catch
            {
                return new List<string>();
            }
        }

        public async ValueTask DisposeAsync()
        {
            await StopCollectionAsync();
            
            foreach (var watcher in _watchers)
            {
                watcher.Dispose();
            }
            _watchers.Clear();
            
            if (_journalReader != null)
            {
                await _journalReader.DisposeAsync();
            }
            
            _cancellationTokenSource.Dispose();
        }
    }

    // Supporting classes for Linux Syslog functionality
    public class LinuxSyslogConfiguration
    {
        public string DistributionName { get; set; } = "";
        public string Version { get; set; } = "";
        public bool HasSystemd { get; set; }
        public SyslogDaemonType SyslogDaemon { get; set; }
        public bool EnableSystemdJournal { get; set; } = true;
        public List<string> CustomSyslogPaths { get; set; } = new();

        public List<SyslogPath> GetSyslogPaths()
        {
            var paths = new List<SyslogPath>();

            // Add custom paths from backend
            foreach (var customPath in CustomSyslogPaths)
            {
                paths.Add(new SyslogPath { FilePath = customPath, LogType = "Custom" });
            }

            // Add standard paths based on distribution
            switch (DistributionName.ToLower())
            {
                case "ubuntu":
                case "debian":
                    paths.AddRange(new[]
                    {
                        new SyslogPath { FilePath = "/var/log/syslog", LogType = "System" },
                        new SyslogPath { FilePath = "/var/log/auth.log", LogType = "Authentication" },
                        new SyslogPath { FilePath = "/var/log/kern.log", LogType = "Kernel" }
                    });
                    break;
                case "centos":
                case "rhel":
                case "fedora":
                    paths.AddRange(new[]
                    {
                        new SyslogPath { FilePath = "/var/log/messages", LogType = "System" },
                        new SyslogPath { FilePath = "/var/log/secure", LogType = "Authentication" },
                        new SyslogPath { FilePath = "/var/log/maillog", LogType = "Mail" }
                    });
                    break;
                default:
                    // Generic paths that should work on most systems
                    paths.AddRange(new[]
                    {
                        new SyslogPath { FilePath = "/var/log/messages", LogType = "System" },
                        new SyslogPath { FilePath = "/var/log/syslog", LogType = "System" }
                    });
                    break;
            }

            return paths.Where(p => !string.IsNullOrEmpty(p.FilePath)).ToList();
        }
    }

    public class SyslogPath
    {
        public string FilePath { get; set; } = "";
        public string LogType { get; set; } = "";
    }

    public enum SyslogDaemonType
    {
        Unknown,
        Rsyslog,
        SyslogNg,
        SystemdJournal,
        Busybox
    }

    /// <summary>
    /// Enhanced Systemd Journal Reader with production-ready journal integration
    /// Author: Revian Ravil Athala
    /// </summary>
    public class SystemdJournalReader : IAsyncDisposable
    {
        private readonly ILogger _logger;
        private readonly CancellationTokenSource _cancellationTokenSource = new();
        private Process? _journalProcess;
        private Task? _readingTask;
        private bool _isRunning = false;
        private readonly object _lock = new();

        public event EventHandler<JournalEntryEventArgs>? JournalEntryReceived;

        public SystemdJournalReader(ILogger logger)
        {
            _logger = logger;
        }

        public async Task<bool> InitializeAsync()
        {
            try
            {
                // Check if journalctl is available
                if (!await IsJournalctlAvailableAsync())
                {
                    _logger.LogWarning("🚫 journalctl is not available on this system");
                    return false;
                }

                // Check if systemd journal is accessible
                if (!await IsJournalAccessibleAsync())
                {
                    _logger.LogWarning("🚫 systemd journal is not accessible (may need elevated privileges)");
                    return false;
                }

                _logger.LogInformation("✅ Systemd journal reader initialized successfully");
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Failed to initialize systemd journal reader");
                return false;
            }
        }

        public async Task StartAsync()
        {
            lock (_lock)
            {
                if (_isRunning)
                {
                    _logger.LogWarning("Systemd journal reader is already running");
                    return;
                }
                _isRunning = true;
            }

            try
            {
                _logger.LogInformation("🚀 Starting systemd journal reader");

                // Start journalctl in follow mode
                _journalProcess = new Process
                {
                    StartInfo = new ProcessStartInfo
                    {
                        FileName = "journalctl",
                        Arguments = "--follow --output=json --no-pager --lines=0", // Follow new entries only, JSON output
                        UseShellExecute = false,
                        RedirectStandardOutput = true,
                        RedirectStandardError = true,
                        CreateNoWindow = true
                    }
                };

                _journalProcess.Start();
                _logger.LogInformation("📖 Started journalctl process (PID: {ProcessId})", _journalProcess.Id);

                // Start reading task
                _readingTask = Task.Run(ReadJournalEntriesAsync, _cancellationTokenSource.Token);

                _logger.LogInformation("✅ Systemd journal reader started successfully");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Failed to start systemd journal reader");
                lock (_lock)
                {
                    _isRunning = false;
                }
                throw;
            }
        }

        public async Task StopAsync()
        {
            lock (_lock)
            {
                if (!_isRunning)
                    return;
                _isRunning = false;
            }

            try
            {
                _logger.LogInformation("🛑 Stopping systemd journal reader");

                // Cancel the reading task
                _cancellationTokenSource.Cancel();

                // Stop the journalctl process
                if (_journalProcess != null && !_journalProcess.HasExited)
                {
                    _journalProcess.Kill();
                    await _journalProcess.WaitForExitAsync();
                }

                // Wait for reading task to complete
                if (_readingTask != null)
                {
                    try
                    {
                        await _readingTask;
                    }
                    catch (OperationCanceledException)
                    {
                        // Expected when cancelling
                    }
                }

                _logger.LogInformation("✅ Systemd journal reader stopped successfully");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Error stopping systemd journal reader");
            }
        }

        private async Task ReadJournalEntriesAsync()
        {
            try
            {
                if (_journalProcess?.StandardOutput == null)
                {
                    _logger.LogError("Journal process standard output is not available");
                    return;
                }

                _logger.LogInformation("📚 Started reading systemd journal entries");

                while (!_cancellationTokenSource.Token.IsCancellationRequested && 
                       _journalProcess != null && !_journalProcess.HasExited)
                {
                    try
                    {
                        var line = await _journalProcess.StandardOutput.ReadLineAsync();
                        
                        if (line == null)
                        {
                            // End of stream
                            break;
                        }

                        if (string.IsNullOrWhiteSpace(line))
                            continue;

                        // Parse JSON journal entry
                        var journalEntry = ParseJournalJsonEntry(line);
                        if (journalEntry != null)
                        {
                            // Fire event
                            JournalEntryReceived?.Invoke(this, new JournalEntryEventArgs
                            {
                                Entry = journalEntry
                            });
                        }
                    }
                    catch (Exception ex) when (!(ex is OperationCanceledException))
                    {
                        _logger.LogWarning(ex, "Error reading journal entry");
                        await Task.Delay(1000, _cancellationTokenSource.Token); // Brief delay before retrying
                    }
                }

                _logger.LogInformation("📚 Finished reading systemd journal entries");
            }
            catch (OperationCanceledException)
            {
                _logger.LogInformation("📚 Journal reading was cancelled");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Error in journal reading task");
            }
        }

        private JournalEntry? ParseJournalJsonEntry(string jsonLine)
        {
            try
            {
                using var document = System.Text.Json.JsonDocument.Parse(jsonLine);
                var root = document.RootElement;

                var entry = new JournalEntry
                {
                    Message = GetStringProperty(root, "MESSAGE") ?? "",
                    Unit = GetStringProperty(root, "_SYSTEMD_UNIT") ?? GetStringProperty(root, "UNIT") ?? "",
                    Priority = GetIntProperty(root, "PRIORITY") ?? 6, // Default to info
                    Timestamp = ParseJournalTimestamp(root)
                };

                // Add all journal fields as additional data
                foreach (var property in root.EnumerateObject())
                {
                    try
                    {
                        var value = property.Value.ValueKind switch
                        {
                            System.Text.Json.JsonValueKind.String => property.Value.GetString() ?? "",
                            System.Text.Json.JsonValueKind.Number => property.Value.GetDouble().ToString(),
                            System.Text.Json.JsonValueKind.True => "true",
                            System.Text.Json.JsonValueKind.False => "false",
                            _ => property.Value.ToString()
                        };
                        entry.Fields[property.Name] = value;
                    }
                    catch
                    {
                        entry.Fields[property.Name] = property.Value.ToString();
                    }
                }

                return entry;
            }
            catch (Exception ex)
            {
                _logger.LogDebug(ex, "Error parsing journal JSON entry: {JsonLine}", jsonLine);
                return null;
            }
        }

        private DateTime ParseJournalTimestamp(System.Text.Json.JsonElement root)
        {
            // Try different timestamp fields
            var timestampFields = new[] { "__REALTIME_TIMESTAMP", "_SOURCE_REALTIME_TIMESTAMP", "TIMESTAMP" };
            
            foreach (var field in timestampFields)
            {
                if (root.TryGetProperty(field, out var timestampElement))
                {
                    if (timestampElement.ValueKind == System.Text.Json.JsonValueKind.String)
                    {
                        var timestampStr = timestampElement.GetString();
                        
                        // Journal timestamps are often in microseconds since epoch
                        if (long.TryParse(timestampStr, out var microseconds))
                        {
                            try
                            {
                                return DateTimeOffset.FromUnixTimeMilliseconds(microseconds / 1000).DateTime;
                            }
                            catch
                            {
                                // Fallback to current time
                            }
                        }
                    }
                    else if (timestampElement.ValueKind == System.Text.Json.JsonValueKind.Number)
                    {
                        var microseconds = timestampElement.GetInt64();
                        try
                        {
                            return DateTimeOffset.FromUnixTimeMilliseconds(microseconds / 1000).DateTime;
                        }
                        catch
                        {
                            // Fallback to current time
                        }
                    }
                }
            }

            return DateTime.UtcNow;
        }

        private string? GetStringProperty(System.Text.Json.JsonElement root, string propertyName)
        {
            return root.TryGetProperty(propertyName, out var element) && 
                   element.ValueKind == System.Text.Json.JsonValueKind.String 
                   ? element.GetString() 
                   : null;
        }

        private int? GetIntProperty(System.Text.Json.JsonElement root, string propertyName)
        {
            if (root.TryGetProperty(propertyName, out var element))
            {
                if (element.ValueKind == System.Text.Json.JsonValueKind.Number)
                    return element.GetInt32();
                else if (element.ValueKind == System.Text.Json.JsonValueKind.String)
                {
                    var str = element.GetString();
                    if (int.TryParse(str, out var intValue))
                        return intValue;
                }
            }
            return null;
        }

        private async Task<bool> IsJournalctlAvailableAsync()
        {
            try
            {
                var process = new Process
                {
                    StartInfo = new ProcessStartInfo
                    {
                        FileName = "which",
                        Arguments = "journalctl",
                        UseShellExecute = false,
                        RedirectStandardOutput = true,
                        CreateNoWindow = true
                    }
                };

                process.Start();
                await process.WaitForExitAsync();
                return process.ExitCode == 0;
            }
            catch
            {
                return false;
            }
        }

        private async Task<bool> IsJournalAccessibleAsync()
        {
            try
            {
                var process = new Process
                {
                    StartInfo = new ProcessStartInfo
                    {
                        FileName = "journalctl",
                        Arguments = "--lines=1 --no-pager",
                        UseShellExecute = false,
                        RedirectStandardOutput = true,
                        RedirectStandardError = true,
                        CreateNoWindow = true
                    }
                };

                process.Start();
                await process.WaitForExitAsync();
                return process.ExitCode == 0;
            }
            catch
            {
                return false;
            }
        }

        public ValueTask DisposeAsync()
        {
            try
            {
                _cancellationTokenSource.Cancel();
                
                if (_journalProcess != null && !_journalProcess.HasExited)
                {
                    _journalProcess.Kill();
                    _journalProcess.Dispose();
                }

                _cancellationTokenSource.Dispose();
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Error disposing systemd journal reader");
            }

            return ValueTask.CompletedTask;
        }
    }

    public class JournalEntryEventArgs : EventArgs
    {
        public JournalEntry Entry { get; set; } = new();
    }

    public class JournalEntry
    {
        public string Message { get; set; } = "";
        public DateTime Timestamp { get; set; }
        public string Unit { get; set; } = "";
        public int Priority { get; set; }
        public Dictionary<string, string> Fields { get; set; } = new();
    }

    /// <summary>
    /// Enhanced Syslog Entry extending base LogEntry with syslog-specific properties
    /// </summary>
    public class SyslogEntry : LogEntry
    {
        // RFC 5424 specific properties
        public int Facility { get; set; }
        public int Severity { get; set; }
        public string AppName { get; set; } = "";
        public string ProcId { get; set; } = "";
        public string MsgId { get; set; } = "";
        public string Hostname { get; set; } = "";
        
        // Additional syslog metadata
        public string SyslogFormat { get; set; } = "RFC3164"; // RFC3164, RFC5424, CEF, JSON, KeyValue, Traditional
        public Dictionary<string, string> StructuredData { get; set; } = new();
        public Dictionary<string, object> ExtensionData { get; set; } = new();
    }

    public class SyslogParser
    {
        private readonly ILogger _logger;

        public SyslogParser(ILogger logger)
        {
            _logger = logger;
        }

        public LogEntry? ParseSyslogLine(string line, string logType, LinuxSyslogConfiguration config)
        {
            try
            {
                if (string.IsNullOrWhiteSpace(line))
                    return null;

                // Auto-detect syslog format and parse accordingly
                if (line.StartsWith("<") && line.Contains(">"))
                {
                    // RFC 5424 or RFC 3164 with priority
                    return ParseRFC5424OrRFC3164(line, logType, config);
                }
                else if (line.StartsWith("CEF:"))
                {
                    // Common Event Format (CEF)
                    return ParseCEF(line, logType, config);
                }
                else if (line.TrimStart().StartsWith("{") && line.TrimEnd().EndsWith("}"))
                {
                    // JSON structured log
                    return ParseJSONLog(line, logType, config);
                }
                else if (line.Contains("=") && (line.Contains("src=") || line.Contains("dst=") || line.Contains("proto=")))
                {
                    // Key-value pairs (LEEF-like)
                    return ParseKeyValueLog(line, logType, config);
                }
                else
                {
                    // Traditional syslog or unstructured
                    return ParseTraditionalSyslog(line, logType, config);
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Failed to parse syslog line: {Line}", line);
                return null;
            }
        }

        /// <summary>
        /// Parse RFC 5424 or RFC 3164 format with priority
        /// </summary>
        private LogEntry? ParseRFC5424OrRFC3164(string line, string logType, LinuxSyslogConfiguration config)
        {
            try
            {
                // Extract priority
                var priorityEnd = line.IndexOf('>');
                if (priorityEnd == -1) return null;

                var priorityStr = line.Substring(1, priorityEnd - 1);
                if (!int.TryParse(priorityStr, out var priority))
                    return null;

                var facility = priority / 8;
                var severity = priority % 8;
                var remainder = line.Substring(priorityEnd + 1);

                // Check if RFC 5424 (has version number after priority)
                if (remainder.Length > 0 && char.IsDigit(remainder[0]))
                {
                    return ParseRFC5424(remainder, logType, config, facility, severity);
                }
                else
                {
                    return ParseRFC3164(remainder, logType, config, facility, severity);
                }
            }
            catch (Exception ex)
            {
                _logger.LogDebug(ex, "Error parsing RFC syslog format");
                return null;
            }
        }

        /// <summary>
        /// Parse RFC 5424 format
        /// Format: <priority>version timestamp hostname app-name procid msgid [structured-data] msg
        /// </summary>
        private LogEntry? ParseRFC5424(string line, string logType, LinuxSyslogConfiguration config, int facility, int severity)
        {
            try
            {
                var parts = line.Split(' ', 7);
                if (parts.Length < 6) return null;

                var version = parts[0];
                var timestamp = ParseISO8601Timestamp(parts[1]);
                var hostname = parts[2];
                var appName = parts[3];
                var procId = parts[4];
                var msgId = parts[5];
                
                var structuredData = "";
                var message = "";
                
                if (parts.Length >= 7)
                {
                    var remainder = parts[6];
                    if (remainder.StartsWith("["))
                    {
                        var sdEnd = remainder.LastIndexOf(']');
                        if (sdEnd != -1)
                        {
                            structuredData = remainder.Substring(0, sdEnd + 1);
                            message = remainder.Substring(sdEnd + 1).Trim();
                        }
                        else
                        {
                            message = remainder;
                        }
                    }
                    else
                    {
                        message = remainder;
                    }
                }

                var syslogEntry = new SyslogEntry
                {
                    Id = LogEntryIdGenerator.GenerateId("SYSL"),
                    Timestamp = timestamp,
                    Hostname = hostname != "-" ? hostname : "",
                    Source = appName != "-" ? appName : "syslog",
                    Message = message,
                    Level = MapSeverityToLevel(severity),
                    Category = logType,
                    CollectorType = "LinuxSyslog",
                    SecurityRelevance = DetermineSecurityRelevance(message, logType),
                    CollectionTime = DateTime.UtcNow,
                    Facility = facility,
                    Severity = severity,
                    AppName = appName != "-" ? appName : "",
                    ProcId = procId != "-" ? procId : "",
                    MsgId = msgId != "-" ? msgId : "",
                    Properties = new Dictionary<string, object>
                    {
                        ["RFC5424Version"] = version,
                        ["Facility"] = facility,
                        ["Severity"] = severity,
                        ["StructuredData"] = structuredData
                    }
                };

                return syslogEntry;
            }
            catch (Exception ex)
            {
                _logger.LogDebug(ex, "Error parsing RFC 5424 format");
                return null;
            }
        }

        /// <summary>
        /// Parse RFC 3164 format
        /// Format: <priority>timestamp hostname program[pid]: message
        /// </summary>
        private LogEntry? ParseRFC3164(string line, string logType, LinuxSyslogConfiguration config, int facility, int severity)
        {
            try
            {
                // Traditional syslog parsing - RFC 3164 format
                // Example: Mar 10 10:15:56 hostname program[pid]: message
                var parts = line.Split(' ', 6);
                if (parts.Length < 6)
                    return null;

                var syslogEntry = new SyslogEntry
                {
                    Id = LogEntryIdGenerator.GenerateId("SYSL"),
                    Timestamp = ParseSyslogTimestamp(parts[0], parts[1], parts[2]),
                    Hostname = parts[3],
                    Source = ExtractProgramName(parts[4]),
                    Message = parts[5],
                    Level = MapSeverityToLevel(severity),
                    Category = logType,
                    CollectorType = "LinuxSyslog",
                    SecurityRelevance = DetermineSecurityRelevance(line, logType),
                    CollectionTime = DateTime.UtcNow,
                    Facility = facility,
                    Severity = severity,
                    Properties = new Dictionary<string, object>
                    {
                        ["Facility"] = facility,
                        ["Severity"] = severity,
                        ["Format"] = "RFC3164"
                    }
                };

                return syslogEntry;
            }
            catch (Exception ex)
            {
                _logger.LogDebug(ex, "Error parsing RFC 3164 format");
                return null;
            }
        }

        /// <summary>
        /// Parse Common Event Format (CEF)
        /// Format: CEF:version|device_vendor|device_product|device_version|signature_id|name|severity|extension
        /// </summary>
        private LogEntry? ParseCEF(string line, string logType, LinuxSyslogConfiguration config)
        {
            try
            {
                if (!line.StartsWith("CEF:"))
                    return null;

                var cefData = line.Substring(4); // Remove "CEF:" prefix
                var parts = cefData.Split('|');
                
                if (parts.Length < 7)
                    return null;

                var version = parts[0];
                var deviceVendor = parts[1];
                var deviceProduct = parts[2];
                var deviceVersion = parts[3];
                var signatureId = parts[4];
                var name = parts[5];
                var severity = parts[6];
                var extension = parts.Length > 7 ? parts[7] : "";

                // Parse extension key-value pairs
                var extensionData = ParseCEFExtension(extension);

                var syslogEntry = new SyslogEntry
                {
                    Id = LogEntryIdGenerator.GenerateId("CEFL"),
                    Timestamp = DateTime.UtcNow, // CEF might have timestamp in extension
                    Source = $"{deviceVendor} {deviceProduct}",
                    Message = name,
                    Level = MapCEFSeverityToLevel(severity),
                    Category = "CEF",
                    CollectorType = "LinuxSyslog",
                    SecurityRelevance = DetermineCEFSecurityRelevance(name, severity),
                    CollectionTime = DateTime.UtcNow,
                    Properties = new Dictionary<string, object>
                    {
                        ["CEFVersion"] = version,
                        ["DeviceVendor"] = deviceVendor,
                        ["DeviceProduct"] = deviceProduct,
                        ["DeviceVersion"] = deviceVersion,
                        ["SignatureId"] = signatureId,
                        ["CEFSeverity"] = severity,
                        ["Format"] = "CEF"
                    }
                };

                // Add extension data to properties
                foreach (var kvp in extensionData)
                {
                    syslogEntry.Properties[$"CEF_{kvp.Key}"] = kvp.Value;
                }

                // Extract timestamp from extension if available
                if (extensionData.TryGetValue("rt", out var rtValue) && DateTime.TryParse(rtValue, out var timestamp))
                {
                    syslogEntry.Timestamp = timestamp;
                }

                return syslogEntry;
            }
            catch (Exception ex)
            {
                _logger.LogDebug(ex, "Error parsing CEF format");
                return null;
            }
        }

        /// <summary>
        /// Parse JSON structured log
        /// </summary>
        private LogEntry? ParseJSONLog(string line, string logType, LinuxSyslogConfiguration config)
        {
            try
            {
                using var document = System.Text.Json.JsonDocument.Parse(line);
                var root = document.RootElement;

                var syslogEntry = new SyslogEntry
                {
                    Id = LogEntryIdGenerator.GenerateId("JSON"),
                    Timestamp = ExtractJSONTimestamp(root),
                    Source = ExtractJSONString(root, "source", "logger", "service", "app") ?? "json-log",
                    Message = ExtractJSONString(root, "message", "msg", "text") ?? line,
                    Level = MapJSONLevelToLevel(ExtractJSONString(root, "level", "severity", "priority") ?? "info"),
                    Category = logType,
                    CollectorType = "LinuxSyslog",
                    SecurityRelevance = DetermineJSONSecurityRelevance(root),
                    CollectionTime = DateTime.UtcNow,
                    Properties = new Dictionary<string, object> { ["Format"] = "JSON" }
                };

                // Add all JSON fields as properties
                foreach (var property in root.EnumerateObject())
                {
                    try
                    {
                        var value = property.Value.ValueKind switch
                        {
                            System.Text.Json.JsonValueKind.String => property.Value.GetString(),
                            System.Text.Json.JsonValueKind.Number => property.Value.GetDouble(),
                            System.Text.Json.JsonValueKind.True => true,
                            System.Text.Json.JsonValueKind.False => false,
                            _ => property.Value.ToString()
                        };
                        syslogEntry.Properties[$"JSON_{property.Name}"] = value ?? "";
                    }
                    catch
                    {
                        syslogEntry.Properties[$"JSON_{property.Name}"] = property.Value.ToString();
                    }
                }

                return syslogEntry;
            }
            catch (Exception ex)
            {
                _logger.LogDebug(ex, "Error parsing JSON log format");
                return null;
            }
        }

        /// <summary>
        /// Parse key-value structured log (LEEF-like)
        /// </summary>
        private LogEntry? ParseKeyValueLog(string line, string logType, LinuxSyslogConfiguration config)
        {
            try
            {
                var kvPairs = new Dictionary<string, string>();
                var parts = line.Split(' ');

                foreach (var part in parts)
                {
                    var equalIndex = part.IndexOf('=');
                    if (equalIndex > 0 && equalIndex < part.Length - 1)
                    {
                        var key = part.Substring(0, equalIndex);
                        var value = part.Substring(equalIndex + 1);
                        kvPairs[key] = value;
                    }
                }

                var syslogEntry = new SyslogEntry
                {
                    Id = LogEntryIdGenerator.GenerateId("KVPL"),
                    Timestamp = DateTime.UtcNow,
                    Source = kvPairs.GetValueOrDefault("src", kvPairs.GetValueOrDefault("source", "keyvalue-log")),
                    Message = kvPairs.GetValueOrDefault("msg", kvPairs.GetValueOrDefault("message", line)),
                    Level = MapKeyValueLevelToLevel(kvPairs.GetValueOrDefault("level", kvPairs.GetValueOrDefault("severity", "info"))),
                    Category = logType,
                    CollectorType = "LinuxSyslog",
                    SecurityRelevance = DetermineKeyValueSecurityRelevance(kvPairs),
                    CollectionTime = DateTime.UtcNow,
                    Properties = new Dictionary<string, object> { ["Format"] = "KeyValue" }
                };

                // Add all key-value pairs as properties
                foreach (var kvp in kvPairs)
                {
                    syslogEntry.Properties[$"KV_{kvp.Key}"] = kvp.Value;
                }

                return syslogEntry;
            }
            catch (Exception ex)
            {
                _logger.LogDebug(ex, "Error parsing key-value log format");
                return null;
            }
        }

        /// <summary>
        /// Parse traditional/unstructured syslog
        /// </summary>
        private LogEntry? ParseTraditionalSyslog(string line, string logType, LinuxSyslogConfiguration config)
        {
            try
            {
                // Fallback to basic parsing
                var parts = line.Split(' ', 6);
                if (parts.Length < 3)
                {
                    // Very basic log - just timestamp and message
                    return new SyslogEntry
                    {
                        Id = LogEntryIdGenerator.GenerateId("TRAD"),
                        Timestamp = DateTime.UtcNow,
                        Source = "traditional-syslog",
                        Message = line,
                        Level = "Information",
                        Category = logType,
                        CollectorType = "LinuxSyslog",
                        SecurityRelevance = DetermineSecurityRelevance(line, logType),
                        CollectionTime = DateTime.UtcNow,
                        Properties = new Dictionary<string, object> { ["Format"] = "Traditional" }
                    };
                }

                var syslogEntry = new SyslogEntry
                {
                    Id = LogEntryIdGenerator.GenerateId("TRAD"),
                    Timestamp = parts.Length >= 3 ? ParseSyslogTimestamp(parts[0], parts[1], parts[2]) : DateTime.UtcNow,
                    Hostname = parts.Length >= 4 ? parts[3] : "",
                    Source = parts.Length >= 5 ? ExtractProgramName(parts[4]) : "syslog",
                    Message = parts.Length >= 6 ? parts[5] : (parts.Length >= 4 ? string.Join(" ", parts.Skip(3)) : line),
                    Level = "Information",
                    Category = logType,
                    CollectorType = "LinuxSyslog",
                    SecurityRelevance = DetermineSecurityRelevance(line, logType),
                    CollectionTime = DateTime.UtcNow,
                    Properties = new Dictionary<string, object> { ["Format"] = "Traditional" }
                };

                return syslogEntry;
            }
            catch (Exception ex)
            {
                _logger.LogDebug(ex, "Error parsing traditional syslog format");
                return null;
            }
        }

        public LogEntry? ParseSystemdJournalEntry(JournalEntry entry, LinuxSyslogConfiguration config)
        {
            try
            {
                var syslogEntry = new SyslogEntry
                {
                    Timestamp = entry.Timestamp,
                    Source = entry.Unit,
                    Message = entry.Message,
                    Level = MapJournalPriorityToLevel(entry.Priority),
                    Category = "SystemdJournal",
                    CollectorType = "LinuxSyslog",
                    SecurityRelevance = DetermineSecurityRelevance(entry.Message, "SystemdJournal"),
                    CollectionTime = DateTime.UtcNow,
                    StructuredData = entry.Fields
                };

                return syslogEntry;
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Failed to parse systemd journal entry");
                return null;
            }
        }

        private DateTime ParseSyslogTimestamp(string month, string day, string time)
        {
            try
            {
                var currentYear = DateTime.Now.Year;
                var dateString = $"{month} {day} {currentYear} {time}";
                return DateTime.ParseExact(dateString, "MMM d yyyy HH:mm:ss", null);
            }
            catch
            {
                return DateTime.UtcNow;
            }
        }

        private string ExtractProgramName(string programField)
        {
            // Extract program name from "program[pid]:" format
            var colonIndex = programField.IndexOf(':');
            if (colonIndex > 0)
                programField = programField.Substring(0, colonIndex);

            var bracketIndex = programField.IndexOf('[');
            if (bracketIndex > 0)
                programField = programField.Substring(0, bracketIndex);

            return programField;
        }

        private string MapJournalPriorityToLevel(int priority)
        {
            return priority switch
            {
                0 or 1 or 2 => "Critical",
                3 => "Error", 
                4 => "Warning",
                5 or 6 => "Information",
                7 => "Debug",
                _ => "Information"
            };
        }

        private string DetermineSecurityRelevance(string message, string logType)
        {
            var lowerMessage = message.ToLower();
            
            // Critical security events
            if (lowerMessage.Contains("authentication failure") || 
                lowerMessage.Contains("login failed") ||
                lowerMessage.Contains("access denied") ||
                lowerMessage.Contains("privilege escalation") ||
                lowerMessage.Contains("malware") ||
                lowerMessage.Contains("intrusion") ||
                lowerMessage.Contains("breach"))
                return "Critical";
            
            // High security relevance patterns
            if (lowerMessage.Contains("failed") && lowerMessage.Contains("login") ||
                lowerMessage.Contains("invalid user") ||
                lowerMessage.Contains("sudo") ||
                lowerMessage.Contains("su:") ||
                lowerMessage.Contains("unauthorized") ||
                lowerMessage.Contains("suspicious") ||
                lowerMessage.Contains("attack"))
            {
                return "High";
            }

            // Medium security relevance patterns
            if (lowerMessage.Contains("ssh") ||
                lowerMessage.Contains("firewall") ||
                lowerMessage.Contains("iptables") ||
                lowerMessage.Contains("warning") ||
                lowerMessage.Contains("timeout") ||
                logType == "Authentication")
            {
                return "Medium";
            }

            return "Low";
        }

        #region Enhanced Parsing Helper Methods

        /// <summary>
        /// Parse ISO 8601 timestamp for RFC 5424
        /// </summary>
        private DateTime ParseISO8601Timestamp(string timestamp)
        {
            try
            {
                if (DateTime.TryParse(timestamp, out var result))
                    return result;
                
                // Try parsing various ISO 8601 formats
                var formats = new[]
                {
                    "yyyy-MM-ddTHH:mm:ss.fffZ",
                    "yyyy-MM-ddTHH:mm:ssZ",
                    "yyyy-MM-ddTHH:mm:ss.fff",
                    "yyyy-MM-ddTHH:mm:ss"
                };
                
                foreach (var format in formats)
                {
                    if (DateTime.TryParseExact(timestamp, format, null, System.Globalization.DateTimeStyles.None, out result))
                        return result;
                }
                
                return DateTime.UtcNow;
            }
            catch
            {
                return DateTime.UtcNow;
            }
        }

        /// <summary>
        /// Map syslog severity to log level
        /// </summary>
        private string MapSeverityToLevel(int severity)
        {
            return severity switch
            {
                0 => "Critical",    // Emergency
                1 => "Critical",    // Alert
                2 => "Critical",    // Critical
                3 => "Error",       // Error
                4 => "Warning",     // Warning
                5 => "Warning",     // Notice
                6 => "Information", // Informational
                7 => "Debug",       // Debug
                _ => "Information"
            };
        }

        /// <summary>
        /// Parse CEF extension key-value pairs
        /// </summary>
        private Dictionary<string, string> ParseCEFExtension(string extension)
        {
            var result = new Dictionary<string, string>();
            
            if (string.IsNullOrWhiteSpace(extension))
                return result;

            try
            {
                var parts = extension.Split(' ');
                foreach (var part in parts)
                {
                    var equalIndex = part.IndexOf('=');
                    if (equalIndex > 0 && equalIndex < part.Length - 1)
                    {
                        var key = part.Substring(0, equalIndex);
                        var value = part.Substring(equalIndex + 1);
                        result[key] = value;
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogDebug(ex, "Error parsing CEF extension");
            }
            
            return result;
        }

        /// <summary>
        /// Map CEF severity to log level
        /// </summary>
        private string MapCEFSeverityToLevel(string severity)
        {
            return severity switch
            {
                "0" => "Information",
                "1" or "2" or "3" => "Information",
                "4" or "5" or "6" => "Warning",
                "7" or "8" or "9" => "Error",
                "10" => "Critical",
                _ => "Information"
            };
        }

        /// <summary>
        /// Determine CEF security relevance
        /// </summary>
        private string DetermineCEFSecurityRelevance(string name, string severity)
        {
            var severityInt = int.TryParse(severity, out var sev) ? sev : 0;
            
            if (severityInt >= 8)
                return "Critical";
            if (severityInt >= 6)
                return "High";
            if (severityInt >= 4)
                return "Medium";
                
            return "Low";
        }

        /// <summary>
        /// Extract timestamp from JSON log
        /// </summary>
        private DateTime ExtractJSONTimestamp(System.Text.Json.JsonElement root)
        {
            var timestampFields = new[] { "timestamp", "@timestamp", "time", "datetime", "date" };
            
            foreach (var field in timestampFields)
            {
                if (root.TryGetProperty(field, out var timestampElement))
                {
                    if (timestampElement.ValueKind == System.Text.Json.JsonValueKind.String)
                    {
                        var timestampStr = timestampElement.GetString();
                        if (!string.IsNullOrEmpty(timestampStr) && DateTime.TryParse(timestampStr, out var timestamp))
                            return timestamp;
                    }
                }
            }
            
            return DateTime.UtcNow;
        }

        /// <summary>
        /// Extract string value from JSON using multiple possible field names
        /// </summary>
        private string? ExtractJSONString(System.Text.Json.JsonElement root, params string[] fieldNames)
        {
            foreach (var fieldName in fieldNames)
            {
                if (root.TryGetProperty(fieldName, out var element) && 
                    element.ValueKind == System.Text.Json.JsonValueKind.String)
                {
                    return element.GetString();
                }
            }
            return null;
        }

        /// <summary>
        /// Map JSON log level to standard level
        /// </summary>
        private string MapJSONLevelToLevel(string level)
        {
            return level.ToLower() switch
            {
                "trace" or "debug" => "Debug",
                "info" or "information" => "Information",
                "warn" or "warning" => "Warning",
                "error" or "err" => "Error",
                "fatal" or "critical" => "Critical",
                _ => "Information"
            };
        }

        /// <summary>
        /// Determine JSON security relevance based on content
        /// </summary>
        private string DetermineJSONSecurityRelevance(System.Text.Json.JsonElement root)
        {
            // Check for security-related fields
            var securityFields = new[] { "security", "auth", "login", "error", "threat", "alert" };
            
            foreach (var field in securityFields)
            {
                if (root.TryGetProperty(field, out _))
                {
                    return "Medium";
                }
            }
            
            // Check log level
            var level = ExtractJSONString(root, "level", "severity", "priority");
            if (!string.IsNullOrEmpty(level))
            {
                return level.ToLower() switch
                {
                    "error" or "critical" or "fatal" => "High",
                    "warning" or "warn" => "Medium",
                    _ => "Low"
                };
            }
            
            return "Low";
        }

        /// <summary>
        /// Map key-value log level to standard level
        /// </summary>
        private string MapKeyValueLevelToLevel(string level)
        {
            return level.ToLower() switch
            {
                "debug" or "trace" => "Debug",
                "info" or "information" => "Information",
                "warn" or "warning" => "Warning",
                "error" or "err" => "Error",
                "critical" or "fatal" => "Critical",
                _ => "Information"
            };
        }

        /// <summary>
        /// Determine key-value security relevance
        /// </summary>
        private string DetermineKeyValueSecurityRelevance(Dictionary<string, string> kvPairs)
        {
            // Check for security-related keys
            var securityKeys = new[] { "threat", "attack", "malware", "intrusion", "auth", "login" };
            
            foreach (var key in securityKeys)
            {
                if (kvPairs.ContainsKey(key))
                    return "High";
            }
            
            // Check severity/level
            if (kvPairs.TryGetValue("severity", out var severity) || kvPairs.TryGetValue("level", out severity))
            {
                return severity.ToLower() switch
                {
                    "high" or "critical" or "error" => "High",
                    "medium" or "warning" => "Medium",
                    _ => "Low"
                };
            }
            
            return "Low";
        }

        #endregion
    }
}
