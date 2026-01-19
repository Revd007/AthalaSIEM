using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Runtime.InteropServices;
using Microsoft.Extensions.Logging;
using AthalaSIEM.UniversalAgent.Core;
using AthalaSIEM.UniversalAgent.Models;
using System.Diagnostics;
using System.Text.RegularExpressions;
using LocalLogEntry = AthalaSIEM.UniversalAgent.Models.LogEntry;

namespace AthalaSIEM.Agent.Collectors
{
    /// <summary>
    /// Enterprise Firewall Collector - Cross-platform firewall log monitoring
    /// Supports Windows Firewall, Linux iptables, and other firewall solutions
    /// Configuration managed via SIEM Web Interface - NO hardcoded rules
    /// </summary>
    public class FirewallCollector : ILogCollector
    {
        /// <inheritdoc />
        public string CollectorName => "Firewall Monitor";
        
        /// <inheritdoc />
        public AthalaSIEM.UniversalAgent.Core.OperatingSystem SupportedOS => AthalaSIEM.UniversalAgent.Core.OperatingSystem.Universal;
        
        /// <inheritdoc />
        public bool IsActive { get; private set; }
        
        /// <inheritdoc />
        public long LogsCollected { get; private set; }

        private readonly ILogger<FirewallCollector> _logger;
        private readonly List<LocalLogEntry> _collectedLogs = new List<LocalLogEntry>();
        private readonly List<FileSystemWatcher> _watchers = new();
        private readonly CancellationTokenSource _cancellationTokenSource = new();
        private Timer? _scanTimer;
        
        // Configuration from backend
        private bool _monitorInbound = true;
        private bool _monitorOutbound = true;
        private bool _monitorBlocked = true;
        private bool _monitorAllowed = false;
        private List<string> _firewallLogPaths = new();
        private FirewallType _detectedFirewallType;

        /// <inheritdoc />
        public event EventHandler<LogCollectedEventArgs>? LogCollected;
        
        /// <inheritdoc />
        public event EventHandler<LogCollectionErrorEventArgs>? CollectionError;

        /// <summary>
        /// Initializes a new instance of the FirewallCollector.
        /// </summary>
        /// <param name="logger">Logger instance for this collector.</param>
        public FirewallCollector(ILogger<FirewallCollector> logger)
        {
            _logger = logger;
            _logger.LogInformation("Firewall Monitor initialized - Cross-platform firewall monitoring");
        }

        /// <inheritdoc />
        public async Task<bool> InitializeAsync(Dictionary<string, object> config, CancellationToken cancellationToken = default)
        {
            try
            {
                _logger.LogInformation("🔥 Initializing Firewall Monitor...");

                // Detect firewall type
                _detectedFirewallType = DetectFirewallType();
                _logger.LogInformation("🔍 Detected firewall type: {FirewallType}", _detectedFirewallType);

                // Load configuration from backend
                LoadFirewallConfiguration(config);

                // Get firewall log paths based on detected firewall
                _firewallLogPaths = GetFirewallLogPaths();
                
                if (_firewallLogPaths.Count == 0)
                {
                    _logger.LogWarning("No firewall log paths configured. Firewall monitoring disabled.");
                    _logger.LogInformation("💡 Configure firewall monitoring via SIEM Web Interface:");
                    _logger.LogInformation("   • Go to Collectors → Firewall → Configure Paths");
                    _logger.LogInformation("   • Enable the log sources you want to monitor");
                    return true; // Don't fail - this is configuration dependent
                }

                // Setup file watchers for firewall logs
                await SetupFirewallWatchersAsync();

                _logger.LogInformation(" Firewall Monitor initialized: {PathCount} log paths, Type: {Type}", 
                    _firewallLogPaths.Count, _detectedFirewallType);
                
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to initialize Firewall Monitor");
                return false;
            }
        }

        /// <summary>
        /// Load firewall configuration from backend
        /// </summary>
        private void LoadFirewallConfiguration(Dictionary<string, object> config)
        {
            // Load monitoring preferences from backend
            if (config.TryGetValue("MonitorInbound", out var inboundObj) && inboundObj is bool inbound)
                _monitorInbound = inbound;

            if (config.TryGetValue("MonitorOutbound", out var outboundObj) && outboundObj is bool outbound)
                _monitorOutbound = outbound;

            if (config.TryGetValue("MonitorBlocked", out var blockedObj) && blockedObj is bool blocked)
                _monitorBlocked = blocked;

            if (config.TryGetValue("MonitorAllowed", out var allowedObj) && allowedObj is bool allowed)
                _monitorAllowed = allowed;

            // Load custom firewall log paths from backend
            if (config.TryGetValue("CustomLogPaths", out var pathsObj) && pathsObj is List<string> customPaths)
            {
                _firewallLogPaths.AddRange(customPaths);
            }

            _logger.LogInformation(" Firewall configuration loaded: Inbound={Inbound}, Outbound={Outbound}, Blocked={Blocked}, Allowed={Allowed}",
                _monitorInbound, _monitorOutbound, _monitorBlocked, _monitorAllowed);
        }

        /// <summary>
        /// Detect the type of firewall running on the system
        /// </summary>
        private FirewallType DetectFirewallType()
        {
            try
            {
                if (System.OperatingSystem.IsWindows())
                {
                    return DetectWindowsFirewall();
                }
                else if (System.OperatingSystem.IsLinux())
                {
                    return DetectLinuxFirewall();
                }
                else
                {
                    return FirewallType.Unknown;
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Failed to detect firewall type");
                return FirewallType.Unknown;
            }
        }

        /// <summary>
        /// Detect Windows firewall type
        /// </summary>
        private FirewallType DetectWindowsFirewall()
        {
            try
            {
                // Check if Windows Defender Firewall is running
                var process = Process.Start(new ProcessStartInfo
                {
                    FileName = "netsh",
                    Arguments = "advfirewall show allprofiles",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    CreateNoWindow = true
                });

                if (process != null)
                {
                    var output = process.StandardOutput.ReadToEnd();
                    process.WaitForExit();

                    if (output.Contains("Windows Defender Firewall"))
                    {
                        return FirewallType.WindowsDefender;
                    }
                }

                // Check for third-party firewalls
                var processes = Process.GetProcesses();
                if (processes.Any(p => p.ProcessName.ToLower().Contains("zonealarm")))
                    return FirewallType.ZoneAlarm;
                if (processes.Any(p => p.ProcessName.ToLower().Contains("comodo")))
                    return FirewallType.ComodoFirewall;

                return FirewallType.WindowsDefender; // Default for Windows
            }
            catch
            {
                return FirewallType.WindowsDefender;
            }
        }

        /// <summary>
        /// Detect Linux firewall type
        /// </summary>
        private FirewallType DetectLinuxFirewall()
        {
            try
            {
                // Check for iptables
                if (File.Exists("/usr/sbin/iptables") || File.Exists("/sbin/iptables"))
                {
                    return FirewallType.Iptables;
                }

                // Check for ufw
                if (File.Exists("/usr/sbin/ufw"))
                {
                    return FirewallType.UFW;
                }

                // Check for firewalld
                if (File.Exists("/usr/bin/firewall-cmd"))
                {
                    return FirewallType.Firewalld;
                }

                // Check for nftables
                if (File.Exists("/usr/sbin/nft"))
                {
                    return FirewallType.Nftables;
                }

                return FirewallType.Iptables; // Default for Linux
            }
            catch
            {
                return FirewallType.Iptables;
            }
        }

        /// <summary>
        /// Get firewall log paths based on detected firewall type
        /// </summary>
        private List<string> GetFirewallLogPaths()
        {
            var paths = new List<string>();

            try
            {
                switch (_detectedFirewallType)
                {
                    case FirewallType.WindowsDefender:
                        paths.AddRange(GetWindowsFirewallLogPaths());
                        break;
                    case FirewallType.Iptables:
                    case FirewallType.UFW:
                    case FirewallType.Firewalld:
                    case FirewallType.Nftables:
                        paths.AddRange(GetLinuxFirewallLogPaths());
                        break;
                    case FirewallType.ZoneAlarm:
                        paths.Add(@"C:\Program Files\Zone Labs\ZoneAlarm\zlclient.log");
                        break;
                    case FirewallType.ComodoFirewall:
                        paths.Add(@"C:\ProgramData\Comodo\Firewall Pro\Logs\cfplogvw.log");
                        break;
                }

                // Filter paths that actually exist
                return paths.Where(File.Exists).ToList();
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Error getting firewall log paths");
                return new List<string>();
            }
        }

        /// <summary>
        /// Get Windows firewall log paths
        /// </summary>
        private List<string> GetWindowsFirewallLogPaths()
        {
            var paths = new List<string>();

            try
            {
                // Default Windows Firewall log locations
                var defaultPaths = new[]
                {
                    @"C:\Windows\System32\LogFiles\Firewall\pfirewall.log",
                    @"C:\Windows\pfirewall.log",
                    @"C:\Windows\System32\LogFiles\Firewall\domainfw.log",
                    @"C:\Windows\System32\LogFiles\Firewall\privatefw.log",
                    @"C:\Windows\System32\LogFiles\Firewall\publicfw.log"
                };

                paths.AddRange(defaultPaths);

                // Try to get configured log path from registry (Windows only)
                if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                {
                    try
                    {
                        using var key = Microsoft.Win32.Registry.LocalMachine.OpenSubKey(
                            @"SYSTEM\CurrentControlSet\Services\SharedAccess\Parameters\FirewallPolicy\StandardProfile\Logging");
                        
                        if (key != null)
                        {
                            var logPath = key.GetValue("LogFilePath")?.ToString();
                            if (!string.IsNullOrEmpty(logPath))
                            {
                                paths.Add(logPath);
                            }
                        }
                    }
                    catch (Exception ex)
                    {
                        _logger.LogWarning(ex, "Failed to read firewall log path from registry");
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Error getting Windows firewall log paths");
            }

            return paths;
        }

        /// <summary>
        /// Get Linux firewall log paths
        /// </summary>
        private List<string> GetLinuxFirewallLogPaths()
        {
            var paths = new List<string>();

            try
            {
                // Common Linux firewall log locations
                var defaultPaths = new[]
                {
                    "/var/log/ufw.log",           // UFW
                    "/var/log/iptables.log",      // iptables
                    "/var/log/firewalld",         // firewalld
                    "/var/log/kern.log",          // Kernel messages (includes firewall)
                    "/var/log/messages",          // System messages
                    "/var/log/syslog",            // Syslog
                    "/var/log/secure",            // Security messages (RHEL/CentOS)
                    "/var/log/auth.log"           // Authentication messages (Debian/Ubuntu)
                };

                paths.AddRange(defaultPaths);

                // Check for custom iptables log configuration
                if (File.Exists("/etc/rsyslog.conf"))
                {
                    var rsyslogConfig = File.ReadAllText("/etc/rsyslog.conf");
                    var matches = Regex.Matches(rsyslogConfig, @"kern\.\*\s+(.+\.log)");
                    foreach (Match match in matches)
                    {
                        paths.Add(match.Groups[1].Value);
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Error getting Linux firewall log paths");
            }

            return paths;
        }

        /// <summary>
        /// Setup file watchers for firewall logs
        /// </summary>
        private async Task SetupFirewallWatchersAsync()
        {
            foreach (var logPath in _firewallLogPaths)
            {
                try
                {
                    if (File.Exists(logPath))
                    {
                        var directory = Path.GetDirectoryName(logPath);
                        var fileName = Path.GetFileName(logPath);

                        if (!string.IsNullOrEmpty(directory) && !string.IsNullOrEmpty(fileName))
                        {
                            var watcher = new FileSystemWatcher(directory, fileName)
                            {
                                NotifyFilter = NotifyFilters.LastWrite | NotifyFilters.Size,
                                EnableRaisingEvents = false
                            };

                            watcher.Changed += async (sender, e) => await OnFirewallLogChangedAsync(e.FullPath);
                            watcher.Error += OnWatcherError;

                            _watchers.Add(watcher);
                            _logger.LogDebug("📁 File watcher setup for firewall log: {LogPath}", logPath);
                        }
                    }
                    else
                    {
                        _logger.LogDebug("Firewall log file not found: {LogPath}", logPath);
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Failed to setup watcher for firewall log: {LogPath}", logPath);
                }
            }

            await Task.CompletedTask;
        }

        /// <inheritdoc />
        public Task StartCollectionAsync(CancellationToken cancellationToken = default)
        {
            IsActive = true;
            
            try
            {
                // Start file watchers
                foreach (var watcher in _watchers)
                {
                    watcher.EnableRaisingEvents = true;
                }

                // Start periodic scan timer
                _scanTimer = new Timer(PerformFirewallScan, null, TimeSpan.Zero, TimeSpan.FromMinutes(5));
                
                _logger.LogInformation(" Firewall Monitor started - monitoring {Count} log files", _watchers.Count);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error starting Firewall Monitor");
                CollectionError?.Invoke(this, new LogCollectionErrorEventArgs
                {
                    Exception = ex,
                    Source = CollectorName,
                    Message = "Error starting Firewall Monitor"
                });
            }
            
            return Task.CompletedTask;
        }

        /// <inheritdoc />
        public Task StopCollectionAsync(CancellationToken cancellationToken = default)
        {
            IsActive = false;
            
            _scanTimer?.Dispose();
            
            foreach (var watcher in _watchers)
            {
                watcher?.Dispose();
            }
            _watchers.Clear();
            
            _cancellationTokenSource.Cancel();
            
            _logger.LogInformation("Firewall Monitor stopped");
            return Task.CompletedTask;
        }

        /// <inheritdoc />
        public Task<IEnumerable<LocalLogEntry>> GetLogsAsync(int batchSize = 100, CancellationToken cancellationToken = default)
        {
            var logs = _collectedLogs.Take(batchSize).ToList();
            _collectedLogs.RemoveRange(0, logs.Count);
            return Task.FromResult<IEnumerable<LocalLogEntry>>(logs);
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
                    ["FirewallType"] = _detectedFirewallType.ToString(),
                    ["MonitoredLogFiles"] = _firewallLogPaths.Count,
                    ["ActiveWatchers"] = _watchers.Count,
                    ["BufferedLogs"] = _collectedLogs.Count,
                    ["MonitorInbound"] = _monitorInbound,
                    ["MonitorOutbound"] = _monitorOutbound,
                    ["MonitorBlocked"] = _monitorBlocked,
                    ["MonitorAllowed"] = _monitorAllowed
                }
            });
        }

        /// <summary>
        /// Handle firewall log file changes
        /// </summary>
        private async Task OnFirewallLogChangedAsync(string logPath)
        {
            if (!IsActive) return;

            try
            {
                var newLines = await ReadNewLogLinesAsync(logPath);
                
                foreach (var line in newLines)
                {
                    var logEntry = ParseFirewallLogLine(line, logPath);
                    if (logEntry != null && ShouldCollectLogEntry(logEntry))
                    {
                        _collectedLogs.Add(logEntry);
                        LogsCollected++;

                        LogCollected?.Invoke(this, new LogCollectedEventArgs 
                        { 
                            Logs = new List<LocalLogEntry> { logEntry },
                            Source = CollectorName,
                            CollectionTime = DateTime.UtcNow
                        });
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Error processing firewall log changes: {LogPath}", logPath);
            }
        }

        /// <summary>
        /// Parse a firewall log line into a LogEntry
        /// </summary>
        private LocalLogEntry? ParseFirewallLogLine(string line, string logPath)
        {
            try
            {
                if (string.IsNullOrWhiteSpace(line))
                    return null;

                return _detectedFirewallType switch
                {
                    FirewallType.WindowsDefender => ParseWindowsFirewallLog(line, logPath),
                    FirewallType.Iptables or FirewallType.UFW or FirewallType.Firewalld => ParseLinuxFirewallLog(line, logPath),
                    _ => ParseGenericFirewallLog(line, logPath)
                };
            }
            catch (Exception ex)
            {
                _logger.LogDebug("Failed to parse firewall log line: {Error}", ex.Message);
                return null;
            }
        }

        /// <summary>
        /// Parse Windows Firewall log format
        /// </summary>
        private LocalLogEntry? ParseWindowsFirewallLog(string line, string logPath)
        {
            // Windows Firewall log format: date time action protocol src-ip dst-ip src-port dst-port size tcpflags tcpsyn tcpack tcpwin icmptype icmpcode info path
            var parts = line.Split(' ');
            if (parts.Length < 8) return null;

            var logEntry = new LocalLogEntry
            {
                Timestamp = DateTime.TryParse($"{parts[0]} {parts[1]}", out var timestamp) ? timestamp : DateTime.UtcNow,
                Source = "WindowsFirewall",
                Level = parts[2].ToLower() == "drop" ? "Warning" : "Information",
                Message = $"Firewall {parts[2]}: {parts[3]} from {parts[4]}:{parts[6]} to {parts[5]}:{parts[7]}",
                Category = "FirewallActivity",
                SecurityRelevance = parts[2].ToLower() == "drop" ? "High" : "Medium",
                CollectorType = "Firewall",
                Properties = new Dictionary<string, object>
                {
                    ["Action"] = parts[2],
                    ["Protocol"] = parts[3],
                    ["SourceIP"] = parts[4],
                    ["DestinationIP"] = parts[5],
                    ["SourcePort"] = parts[6],
                    ["DestinationPort"] = parts[7],
                    ["LogFile"] = logPath,
                    ["FirewallType"] = "Windows Defender"
                }
            };

            return logEntry;
        }

        /// <summary>
        /// Parse Linux firewall log format
        /// </summary>
        private LocalLogEntry? ParseLinuxFirewallLog(string line, string logPath)
        {
            // Linux iptables/UFW log format varies, but typically includes: timestamp hostname kernel: [UFW BLOCK] ...
            var logEntry = new LocalLogEntry
            {
                Timestamp = DateTime.UtcNow,
                Source = "LinuxFirewall",
                Level = line.Contains("BLOCK") || line.Contains("DROP") ? "Warning" : "Information",
                Message = line,
                Category = "FirewallActivity",
                SecurityRelevance = line.Contains("BLOCK") || line.Contains("DROP") ? "High" : "Medium",
                CollectorType = "Firewall",
                Properties = new Dictionary<string, object>
                {
                    ["LogFile"] = logPath,
                    ["FirewallType"] = _detectedFirewallType.ToString(),
                    ["RawMessage"] = line
                }
            };

            // Extract IP addresses and ports using regex
            var ipRegex = new Regex(@"SRC=(\d+\.\d+\.\d+\.\d+).*DST=(\d+\.\d+\.\d+\.\d+)");
            var portRegex = new Regex(@"SPT=(\d+).*DPT=(\d+)");
            var protoRegex = new Regex(@"PROTO=(\w+)");

            var ipMatch = ipRegex.Match(line);
            if (ipMatch.Success)
            {
                logEntry.Properties["SourceIP"] = ipMatch.Groups[1].Value;
                logEntry.Properties["DestinationIP"] = ipMatch.Groups[2].Value;
            }

            var portMatch = portRegex.Match(line);
            if (portMatch.Success)
            {
                logEntry.Properties["SourcePort"] = portMatch.Groups[1].Value;
                logEntry.Properties["DestinationPort"] = portMatch.Groups[2].Value;
            }

            var protoMatch = protoRegex.Match(line);
            if (protoMatch.Success)
            {
                logEntry.Properties["Protocol"] = protoMatch.Groups[1].Value;
            }

            // Determine action
            if (line.Contains("BLOCK") || line.Contains("DROP"))
                logEntry.Properties["Action"] = "BLOCK";
            else if (line.Contains("ALLOW") || line.Contains("ACCEPT"))
                logEntry.Properties["Action"] = "ALLOW";

            return logEntry;
        }

        /// <summary>
        /// Parse generic firewall log format
        /// </summary>
        private LocalLogEntry? ParseGenericFirewallLog(string line, string logPath)
        {
            return new LocalLogEntry
            {
                Timestamp = DateTime.UtcNow,
                Source = "Firewall",
                Level = "Information",
                Message = line,
                Category = "FirewallActivity",
                SecurityRelevance = "Medium",
                CollectorType = "Firewall",
                Properties = new Dictionary<string, object>
                {
                    ["LogFile"] = logPath,
                    ["FirewallType"] = _detectedFirewallType.ToString(),
                    ["RawMessage"] = line
                }
            };
        }

        /// <summary>
        /// Check if log entry should be collected based on configuration
        /// </summary>
        private bool ShouldCollectLogEntry(LocalLogEntry logEntry)
        {
            var action = logEntry.Properties.GetValueOrDefault("Action", "").ToString()?.ToLower();
            
            return action switch
            {
                "block" or "drop" => _monitorBlocked,
                "allow" or "accept" => _monitorAllowed,
                _ => true // Collect unknown actions by default
            };
        }

        /// <summary>
        /// Read new lines from log file
        /// </summary>
        private async Task<IEnumerable<string>> ReadNewLogLinesAsync(string logPath)
        {
            try
            {
                // Simple implementation - read last 50 lines
                // In production, this should track file position
                var lines = await File.ReadAllLinesAsync(logPath);
                return lines.TakeLast(50);
            }
            catch
            {
                return new List<string>();
            }
        }

        /// <summary>
        /// Periodic firewall scan
        /// </summary>
        private void PerformFirewallScan(object? state)
        {
            if (!IsActive) return;

            try
            {
                _logger.LogDebug("🔍 Performing firewall log scan...");
                
                foreach (var logPath in _firewallLogPaths)
                {
                    _ = Task.Run(async () => await OnFirewallLogChangedAsync(logPath));
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error during firewall scan");
            }
        }

        /// <summary>
        /// Handle file watcher errors
        /// </summary>
        private void OnWatcherError(object sender, ErrorEventArgs e)
        {
            _logger.LogError(e.GetException(), "Firewall log watcher error");
            CollectionError?.Invoke(this, new LogCollectionErrorEventArgs
            {
                Exception = e.GetException(),
                Source = CollectorName,
                Message = "Firewall log watcher error"
            });
        }

        /// <inheritdoc />
        public async ValueTask DisposeAsync()
        {
            await StopCollectionAsync();
            _scanTimer?.Dispose();
            _cancellationTokenSource?.Dispose();
        }
    }

    /// <summary>
    /// Supported firewall types
    /// </summary>
    public enum FirewallType
    {
        Unknown,
        WindowsDefender,
        ZoneAlarm,
        ComodoFirewall,
        Iptables,
        UFW,
        Firewalld,
        Nftables,
        PfSense,
        OPNsense
    }
} 
