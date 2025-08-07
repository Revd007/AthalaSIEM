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

    public class SystemdJournalReader : IAsyncDisposable
    {
        private readonly ILogger _logger;
        
#pragma warning disable CS0067 // Event is never used - it's for future implementation
        public event EventHandler<JournalEntryEventArgs>? JournalEntryReceived;
#pragma warning restore CS0067

        public SystemdJournalReader(ILogger logger)
        {
            _logger = logger;
        }

        public Task<bool> InitializeAsync()
        {
            _logger.LogInformation("Systemd Journal Reader initialized");
            return Task.FromResult(true);
        }

        public Task StartAsync()
        {
            _logger.LogInformation("Systemd Journal Reader started");
            return Task.CompletedTask;
        }

        public Task StopAsync()
        {
            _logger.LogInformation("Systemd Journal Reader stopped");
            return Task.CompletedTask;
        }

        public ValueTask DisposeAsync()
        {
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

                // Basic syslog parsing - RFC 3164 format
                // Example: Mar 10 10:15:56 hostname program[pid]: message
                var parts = line.Split(' ', 6);
                if (parts.Length < 6)
                    return null;

                var syslogEntry = new SyslogEntry
                {
                    Timestamp = ParseSyslogTimestamp(parts[0], parts[1], parts[2]),
                    Hostname = parts[3],
                    Source = ExtractProgramName(parts[4]),
                    Message = parts[5],
                    Level = "Information",
                    Category = logType,
                    CollectorType = "LinuxSyslog",
                    SecurityRelevance = DetermineSecurityRelevance(line, logType),
                    CollectionTime = DateTime.UtcNow
                };

                return syslogEntry;
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Failed to parse syslog line: {Line}", line);
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
            
            // High security relevance patterns
            if (lowerMessage.Contains("failed") && lowerMessage.Contains("login") ||
                lowerMessage.Contains("authentication failure") ||
                lowerMessage.Contains("invalid user") ||
                lowerMessage.Contains("sudo") ||
                lowerMessage.Contains("su:"))
            {
                return "High";
            }

            // Medium security relevance patterns
            if (lowerMessage.Contains("ssh") ||
                lowerMessage.Contains("firewall") ||
                lowerMessage.Contains("iptables") ||
                logType == "Authentication")
            {
                return "Medium";
            }

            return "Low";
        }
    }
}
