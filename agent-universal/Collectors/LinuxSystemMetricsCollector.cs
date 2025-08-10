using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Runtime.Versioning;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using AthalaSIEM.UniversalAgent.Core;
using AthalaSIEM.UniversalAgent.Models;

namespace AthalaSIEM.UniversalAgent.Collectors
{
    /// <summary>
    /// Linux System Metrics Collector for AthalaSIEM Universal Agent
    /// Comprehensive system monitoring for CPU, Memory, Disk, Network, and Process metrics
    /// Author: Revian Ravil Athala
    /// Enterprise-grade SIEM system monitoring following security best practices
    /// </summary>
    [SupportedOSPlatform("linux")]
    public class LinuxSystemMetricsCollector : ILogCollector
    {
        private readonly ILogger<LinuxSystemMetricsCollector> _logger;
        private readonly List<LinuxSystemMetricsEntry> _collectedMetrics = new();
        private readonly CancellationTokenSource _cancellationTokenSource = new();
        private readonly object _metricsLock = new();
        
        private LinuxSystemMetricsConfiguration _config = new();
        private Timer? _collectionTimer;
        private bool _isActive = false;
        private long _metricsCollected = 0;
        private DateTime _lastCollection = DateTime.MinValue;
        
        // Performance tracking
        private readonly Dictionary<string, double> _previousCpuTimes = new();
        private readonly Dictionary<string, NetworkMetrics> _previousNetworkStats = new();
        private readonly Dictionary<string, DiskIOMetrics> _previousDiskIOStats = new();

        #region ILogCollector Implementation

        public string CollectorName => "Linux System Metrics";
        public Core.OperatingSystem SupportedOS => Core.OperatingSystem.Linux;
        public bool IsActive => _isActive;
        public long LogsCollected => _metricsCollected;

        public event EventHandler<LogCollectedEventArgs>? LogCollected;
        public event EventHandler<LogCollectionErrorEventArgs>? CollectionError;

        #endregion

        public LinuxSystemMetricsCollector(ILogger<LinuxSystemMetricsCollector> logger)
        {
            _logger = logger;
        }

        public async Task<bool> InitializeAsync(Dictionary<string, object> config, CancellationToken cancellationToken = default)
        {
            try
            {
                if (!RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
                {
                    _logger.LogWarning("LinuxSystemMetricsCollector can only run on Linux systems");
                    return false;
                }

                LoadConfiguration(config);
                
                _logger.LogInformation("🐧 Initializing Linux System Metrics Collector");
                _logger.LogInformation("📊 Monitoring - CPU: {CPU}, Memory: {Memory}, Disk: {Disk}, Network: {Network}",
                    _config.EnableCPUMonitoring, _config.EnableMemoryMonitoring, 
                    _config.EnableDiskMonitoring, _config.EnableNetworkMonitoring);

                // Validate system capabilities
                await ValidateSystemCapabilitiesAsync();

                _logger.LogInformation("✅ Linux System Metrics Collector initialized successfully");
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Failed to initialize Linux System Metrics Collector");
                CollectionError?.Invoke(this, new LogCollectionErrorEventArgs
                {
                    Exception = ex,
                    Message = "Failed to initialize Linux System Metrics Collector",
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
                    _logger.LogWarning("Linux System Metrics Collector is already active");
                    return;
                }

                _logger.LogInformation("🚀 Starting Linux System Metrics collection every {Interval} seconds", 
                    _config.CollectionIntervalSeconds);

                _collectionTimer = new Timer(
                    CollectSystemMetrics,
                    null,
                    TimeSpan.Zero,
                    TimeSpan.FromSeconds(_config.CollectionIntervalSeconds));

                _isActive = true;
                _logger.LogInformation("✅ Linux System Metrics Collector started successfully");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Failed to start Linux System Metrics Collector");
                CollectionError?.Invoke(this, new LogCollectionErrorEventArgs
                {
                    Exception = ex,
                    Message = "Failed to start Linux System Metrics Collector",
                    Source = CollectorName
                });
            }
        }

        public async Task StopCollectionAsync(CancellationToken cancellationToken = default)
        {
            try
            {
                _collectionTimer?.Dispose();
                _isActive = false;
                _logger.LogInformation("🛑 Linux System Metrics Collector stopped successfully");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Error stopping Linux System Metrics Collector");
                CollectionError?.Invoke(this, new LogCollectionErrorEventArgs
                {
                    Exception = ex,
                    Message = "Error stopping Linux System Metrics Collector",
                    Source = CollectorName
                });
            }
        }

        public Task<IEnumerable<LogEntry>> GetLogsAsync(int batchSize = 100, CancellationToken cancellationToken = default)
        {
            lock (_metricsLock)
            {
                var metrics = _collectedMetrics.Take(batchSize).Cast<LogEntry>().ToList();
                _collectedMetrics.RemoveRange(0, metrics.Count);
                return Task.FromResult<IEnumerable<LogEntry>>(metrics);
            }
        }

        public Task<CollectorHealth> GetHealthAsync()
        {
            var health = new CollectorHealth
            {
                IsHealthy = _isActive && !_cancellationTokenSource.Token.IsCancellationRequested,
                Status = _isActive ? "Running" : "Stopped",
                LogsCollected = _metricsCollected,
                LastCollection = _lastCollection,
                Uptime = DateTime.UtcNow - _lastCollection,
                Metrics = new Dictionary<string, object>
                {
                    {"CollectionInterval", _config.CollectionIntervalSeconds},
                    {"QueuedMetrics", _collectedMetrics.Count},
                    {"CPUMonitoring", _config.EnableCPUMonitoring},
                    {"MemoryMonitoring", _config.EnableMemoryMonitoring},
                    {"DiskMonitoring", _config.EnableDiskMonitoring},
                    {"NetworkMonitoring", _config.EnableNetworkMonitoring}
                }
            };

            return Task.FromResult(health);
        }

        #region Configuration Loading

        private void LoadConfiguration(Dictionary<string, object> config)
        {
            _config = new LinuxSystemMetricsConfiguration();

            if (config.TryGetValue("EnableCPUMonitoring", out var cpuObj) && cpuObj is bool enableCpu)
                _config.EnableCPUMonitoring = enableCpu;

            if (config.TryGetValue("EnableMemoryMonitoring", out var memObj) && memObj is bool enableMem)
                _config.EnableMemoryMonitoring = enableMem;

            if (config.TryGetValue("EnableDiskMonitoring", out var diskObj) && diskObj is bool enableDisk)
                _config.EnableDiskMonitoring = enableDisk;

            if (config.TryGetValue("EnableNetworkMonitoring", out var netObj) && netObj is bool enableNet)
                _config.EnableNetworkMonitoring = enableNet;

            if (config.TryGetValue("CollectionIntervalSeconds", out var intervalObj) && intervalObj is int interval)
                _config.CollectionIntervalSeconds = Math.Max(10, interval); // Minimum 10 seconds

            if (config.TryGetValue("MonitoredNetworkInterfaces", out var netIfObj) && netIfObj is List<string> netIfs)
                _config.MonitoredNetworkInterfaces = netIfs;

            if (config.TryGetValue("MonitoredFilesystems", out var fsObj) && fsObj is List<string> filesystems)
                _config.MonitoredFilesystems = filesystems;
        }

        #endregion

        #region System Validation

        private async Task ValidateSystemCapabilitiesAsync()
        {
            var capabilities = new List<string>();

            // Check /proc filesystem
            if (Directory.Exists("/proc"))
                capabilities.Add("/proc filesystem");

            // Check specific files
            var requiredFiles = new[]
            {
                "/proc/stat", "/proc/meminfo", "/proc/loadavg",
                "/proc/diskstats", "/proc/net/dev"
            };

            foreach (var file in requiredFiles)
            {
                if (File.Exists(file))
                    capabilities.Add(Path.GetFileName(file));
            }

            _logger.LogInformation("🔍 System capabilities detected: {Capabilities}", 
                string.Join(", ", capabilities));
        }

        #endregion

        #region Metrics Collection

        private async void CollectSystemMetrics(object? state)
        {
            var stopwatch = Stopwatch.StartNew();
            
            try
            {
                var metrics = new LinuxSystemMetricsEntry
                {
                    Id = LogEntryIdGenerator.GenerateId("LSYS"),
                    Timestamp = DateTime.UtcNow,
                    Source = "LinuxSystemMetrics",
                    Level = "Information",
                    Category = "SystemMetrics",
                    CollectorType = CollectorName,
                    SecurityRelevance = "Low",
                    CollectionTime = DateTime.UtcNow,
                    AgentId = Environment.MachineName
                };

                // Collect all metrics in parallel for better performance
                var tasks = new List<Task>();

                if (_config.EnableCPUMonitoring)
                    tasks.Add(CollectCPUMetricsAsync(metrics));

                if (_config.EnableMemoryMonitoring)
                    tasks.Add(CollectMemoryMetricsAsync(metrics));

                if (_config.EnableDiskMonitoring)
                    tasks.Add(CollectDiskMetricsAsync(metrics));

                if (_config.EnableNetworkMonitoring)
                    tasks.Add(CollectNetworkMetricsAsync(metrics));

                if (_config.EnableProcessMonitoring)
                    tasks.Add(CollectProcessMetricsAsync(metrics));

                // Collect system information
                await CollectSystemInformationAsync(metrics);

                // Wait for all metrics collection to complete
                await Task.WhenAll(tasks);

                stopwatch.Stop();
                metrics.Message = $"System metrics collected in {stopwatch.ElapsedMilliseconds}ms";
                metrics.Properties["CollectionDurationMs"] = stopwatch.ElapsedMilliseconds;

                // Add to collection
                lock (_metricsLock)
                {
                    _collectedMetrics.Add(metrics);
                    _metricsCollected++;
                    _lastCollection = DateTime.UtcNow;

                    // Maintain max history
                    if (_collectedMetrics.Count > _config.MaxHistoryEntries)
                    {
                        _collectedMetrics.RemoveRange(0, _collectedMetrics.Count - _config.MaxHistoryEntries);
                    }
                }

                // Fire event
                LogCollected?.Invoke(this, new LogCollectedEventArgs
                {
                    Logs = new[] { metrics },
                    Source = CollectorName,
                    CollectionTime = DateTime.UtcNow
                });

                _logger.LogDebug("📊 System metrics collected successfully in {Duration}ms", 
                    stopwatch.ElapsedMilliseconds);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Error collecting system metrics");
                CollectionError?.Invoke(this, new LogCollectionErrorEventArgs
                {
                    Exception = ex,
                    Message = "Error collecting system metrics",
                    Source = CollectorName
                });
            }
        }

        #endregion

        #region CPU Metrics Collection

        private async Task CollectCPUMetricsAsync(LinuxSystemMetricsEntry metrics)
        {
            try
            {
                // Read /proc/stat for CPU usage
                if (File.Exists("/proc/stat"))
                {
                    var statLines = await File.ReadAllLinesAsync("/proc/stat");
                    var cpuLine = statLines.FirstOrDefault(l => l.StartsWith("cpu "));
                    
                    if (cpuLine != null)
                    {
                        var cpuTimes = ParseCpuTimes(cpuLine);
                        metrics.CpuUsagePercent = CalculateCpuUsage(cpuTimes);
                        
                        // Per-core CPU usage
                        var coreLines = statLines.Where(l => l.StartsWith("cpu") && l != cpuLine).ToList();
                        metrics.CpuCoreCount = coreLines.Count;
                        
                        foreach (var coreLine in coreLines)
                        {
                            var parts = coreLine.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                            if (parts.Length > 0)
                            {
                                var coreId = parts[0];
                                var coreTimes = ParseCpuTimes(coreLine);
                                var coreUsage = CalculateCpuUsage(coreTimes, coreId);
                                metrics.PerCoreCpuUsage[coreId] = coreUsage;
                            }
                        }
                    }
                }

                // Read /proc/loadavg for load averages
                if (File.Exists("/proc/loadavg"))
                {
                    var loadavg = await File.ReadAllTextAsync("/proc/loadavg");
                    var parts = loadavg.Split(' ');
                    if (parts.Length >= 3)
                    {
                        if (double.TryParse(parts[0], out var load1))
                            metrics.LoadAverage1Min = load1;
                        if (double.TryParse(parts[1], out var load5))
                            metrics.LoadAverage5Min = load5;
                        if (double.TryParse(parts[2], out var load15))
                            metrics.LoadAverage15Min = load15;
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Error collecting CPU metrics");
            }
        }

        private Dictionary<string, long> ParseCpuTimes(string cpuLine)
        {
            var parts = cpuLine.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            var times = new Dictionary<string, long>();
            
            if (parts.Length >= 8)
            {
                times["user"] = long.Parse(parts[1]);
                times["nice"] = long.Parse(parts[2]);
                times["system"] = long.Parse(parts[3]);
                times["idle"] = long.Parse(parts[4]);
                times["iowait"] = long.Parse(parts[5]);
                times["irq"] = long.Parse(parts[6]);
                times["softirq"] = long.Parse(parts[7]);
            }
            
            return times;
        }

        private double CalculateCpuUsage(Dictionary<string, long> currentTimes, string? coreId = null)
        {
            var key = coreId ?? "cpu";
            
            if (!_previousCpuTimes.ContainsKey(key))
            {
                _previousCpuTimes[key] = currentTimes.Values.Sum();
                return 0.0;
            }

            var totalCurrent = currentTimes.Values.Sum();
            var totalPrevious = _previousCpuTimes[key];
            var totalDiff = totalCurrent - totalPrevious;

            if (totalDiff <= 0) return 0.0;

            var idleCurrent = currentTimes.GetValueOrDefault("idle", 0);
            var idlePrevious = _previousCpuTimes.ContainsKey($"{key}_idle") ? _previousCpuTimes[$"{key}_idle"] : idleCurrent;
            var idleDiff = idleCurrent - idlePrevious;

            _previousCpuTimes[key] = totalCurrent;
            _previousCpuTimes[$"{key}_idle"] = idleCurrent;

            return Math.Max(0, Math.Min(100, (1.0 - (double)idleDiff / totalDiff) * 100.0));
        }

        #endregion

        #region Memory Metrics Collection

        private async Task CollectMemoryMetricsAsync(LinuxSystemMetricsEntry metrics)
        {
            try
            {
                if (!File.Exists("/proc/meminfo")) return;

                var memLines = await File.ReadAllLinesAsync("/proc/meminfo");
                var memInfo = new Dictionary<string, long>();

                foreach (var line in memLines)
                {
                    var parts = line.Split(':', StringSplitOptions.RemoveEmptyEntries);
                    if (parts.Length >= 2)
                    {
                        var key = parts[0].Trim();
                        var valueStr = parts[1].Trim().Replace(" kB", "");
                        if (long.TryParse(valueStr, out var value))
                        {
                            memInfo[key] = value * 1024; // Convert kB to bytes
                        }
                    }
                }

                // Memory metrics
                metrics.MemoryTotalBytes = memInfo.GetValueOrDefault("MemTotal", 0);
                metrics.MemoryFreeBytes = memInfo.GetValueOrDefault("MemFree", 0);
                metrics.MemoryAvailableBytes = memInfo.GetValueOrDefault("MemAvailable", 0);
                metrics.MemoryCachedBytes = memInfo.GetValueOrDefault("Cached", 0);
                metrics.MemoryBuffersBytes = memInfo.GetValueOrDefault("Buffers", 0);
                
                metrics.MemoryUsedBytes = metrics.MemoryTotalBytes - metrics.MemoryFreeBytes;
                metrics.MemoryUsagePercent = metrics.MemoryTotalBytes > 0 
                    ? (double)metrics.MemoryUsedBytes / metrics.MemoryTotalBytes * 100.0 
                    : 0.0;

                // Swap metrics
                metrics.SwapTotalBytes = memInfo.GetValueOrDefault("SwapTotal", 0);
                metrics.SwapFreeBytes = memInfo.GetValueOrDefault("SwapFree", 0);
                metrics.SwapUsedBytes = metrics.SwapTotalBytes - metrics.SwapFreeBytes;
                metrics.SwapUsagePercent = metrics.SwapTotalBytes > 0 
                    ? (double)metrics.SwapUsedBytes / metrics.SwapTotalBytes * 100.0 
                    : 0.0;
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Error collecting memory metrics");
            }
        }

        #endregion

        #region Disk Metrics Collection

        private async Task CollectDiskMetricsAsync(LinuxSystemMetricsEntry metrics)
        {
            try
            {
                // Get filesystem usage using df command
                await CollectFilesystemUsageAsync(metrics);
                
                // Get disk I/O statistics from /proc/diskstats
                await CollectDiskIOStatsAsync(metrics);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Error collecting disk metrics");
            }
        }

        private async Task CollectFilesystemUsageAsync(LinuxSystemMetricsEntry metrics)
        {
            try
            {
                var process = new Process
                {
                    StartInfo = new ProcessStartInfo
                    {
                        FileName = "df",
                        Arguments = "-B1 -T", // Byte units, show filesystem type
                        UseShellExecute = false,
                        RedirectStandardOutput = true,
                        CreateNoWindow = true
                    }
                };

                process.Start();
                var output = await process.StandardOutput.ReadToEndAsync();
                await process.WaitForExitAsync();

                var lines = output.Split('\n', StringSplitOptions.RemoveEmptyEntries).Skip(1); // Skip header

                foreach (var line in lines)
                {
                    var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                    if (parts.Length >= 7)
                    {
                        var filesystem = parts[0];
                        var fsType = parts[1];
                        var mountPoint = parts[6];

                        // Skip excluded filesystems
                        if (_config.ExcludedFilesystems.Contains(fsType))
                            continue;

                        // Only monitor specified filesystems if configured
                        if (_config.MonitoredFilesystems.Any() && !_config.MonitoredFilesystems.Contains(mountPoint))
                            continue;

                        if (long.TryParse(parts[2], out var total) &&
                            long.TryParse(parts[3], out var used) &&
                            long.TryParse(parts[4], out var available))
                        {
                            var diskMetric = new DiskMetrics
                            {
                                Filesystem = filesystem,
                                MountPoint = mountPoint,
                                FilesystemType = fsType,
                                TotalBytes = total,
                                UsedBytes = used,
                                AvailableBytes = available,
                                UsagePercent = total > 0 ? (double)used / total * 100.0 : 0.0
                            };

                            metrics.DiskUsage[mountPoint] = diskMetric;
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Error collecting filesystem usage");
            }
        }

        private async Task CollectDiskIOStatsAsync(LinuxSystemMetricsEntry metrics)
        {
            try
            {
                if (!File.Exists("/proc/diskstats")) return;

                var diskLines = await File.ReadAllLinesAsync("/proc/diskstats");

                foreach (var line in diskLines)
                {
                    var parts = line.Trim().Split(' ', StringSplitOptions.RemoveEmptyEntries);
                    if (parts.Length >= 14)
                    {
                        var deviceName = parts[2];
                        
                        // Skip loop devices and ram devices
                        if (deviceName.StartsWith("loop") || deviceName.StartsWith("ram"))
                            continue;

                        if (long.TryParse(parts[3], out var readOps) &&
                            long.TryParse(parts[5], out var readSectors) &&
                            long.TryParse(parts[7], out var writeOps) &&
                            long.TryParse(parts[9], out var writeSectors))
                        {
                            var diskIO = new DiskIOMetrics
                            {
                                DeviceName = deviceName,
                                ReadOperations = readOps,
                                WriteOperations = writeOps,
                                ReadBytes = readSectors * 512, // Sectors are typically 512 bytes
                                WriteBytes = writeSectors * 512
                            };

                            // Calculate rates if we have previous data
                            if (_previousDiskIOStats.ContainsKey(deviceName))
                            {
                                var previous = _previousDiskIOStats[deviceName];
                                var timeDiff = _config.CollectionIntervalSeconds;
                                
                                diskIO.ReadBytesPerSecond = (diskIO.ReadBytes - previous.ReadBytes) / timeDiff;
                                diskIO.WriteBytesPerSecond = (diskIO.WriteBytes - previous.WriteBytes) / timeDiff;
                            }

                            _previousDiskIOStats[deviceName] = diskIO;
                            metrics.DiskIOStats[deviceName] = diskIO;
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Error collecting disk I/O statistics");
            }
        }

        #endregion

        #region Network Metrics Collection

        private async Task CollectNetworkMetricsAsync(LinuxSystemMetricsEntry metrics)
        {
            try
            {
                if (!File.Exists("/proc/net/dev")) return;

                var netLines = await File.ReadAllLinesAsync("/proc/net/dev");
                
                // Skip header lines
                foreach (var line in netLines.Skip(2))
                {
                    var colonIndex = line.IndexOf(':');
                    if (colonIndex == -1) continue;

                    var interfaceName = line.Substring(0, colonIndex).Trim();
                    var stats = line.Substring(colonIndex + 1).Trim().Split(' ', StringSplitOptions.RemoveEmptyEntries);

                    if (stats.Length >= 16)
                    {
                        // Only monitor specified interfaces if configured
                        if (_config.MonitoredNetworkInterfaces.Any() && 
                            !_config.MonitoredNetworkInterfaces.Contains(interfaceName))
                            continue;

                        if (long.TryParse(stats[0], out var bytesReceived) &&
                            long.TryParse(stats[1], out var packetsReceived) &&
                            long.TryParse(stats[2], out var receiveErrors) &&
                            long.TryParse(stats[3], out var receiveDropped) &&
                            long.TryParse(stats[8], out var bytesSent) &&
                            long.TryParse(stats[9], out var packetsSent) &&
                            long.TryParse(stats[10], out var transmitErrors) &&
                            long.TryParse(stats[11], out var transmitDropped))
                        {
                            var networkMetric = new NetworkMetrics
                            {
                                InterfaceName = interfaceName,
                                BytesReceived = bytesReceived,
                                PacketsReceived = packetsReceived,
                                ReceiveErrors = receiveErrors,
                                ReceiveDropped = receiveDropped,
                                BytesSent = bytesSent,
                                PacketsSent = packetsSent,
                                TransmitErrors = transmitErrors,
                                TransmitDropped = transmitDropped
                            };

                            // Get additional interface information
                            await EnrichNetworkInterfaceInfo(networkMetric);

                            // Calculate rates if we have previous data
                            if (_previousNetworkStats.ContainsKey(interfaceName))
                            {
                                var previous = _previousNetworkStats[interfaceName];
                                var timeDiff = _config.CollectionIntervalSeconds;

                                networkMetric.BytesReceivedPerSecond = (networkMetric.BytesReceived - previous.BytesReceived) / timeDiff;
                                networkMetric.BytesSentPerSecond = (networkMetric.BytesSent - previous.BytesSent) / timeDiff;
                                networkMetric.PacketsReceivedPerSecond = (networkMetric.PacketsReceived - previous.PacketsReceived) / timeDiff;
                                networkMetric.PacketsSentPerSecond = (networkMetric.PacketsSent - previous.PacketsSent) / timeDiff;
                            }

                            _previousNetworkStats[interfaceName] = networkMetric;
                            metrics.NetworkStats[interfaceName] = networkMetric;
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Error collecting network metrics");
            }
        }

        private async Task EnrichNetworkInterfaceInfo(NetworkMetrics networkMetric)
        {
            try
            {
                var interfacePath = $"/sys/class/net/{networkMetric.InterfaceName}";
                
                // Check if interface is up
                var operstatePath = Path.Combine(interfacePath, "operstate");
                if (File.Exists(operstatePath))
                {
                    var operstate = (await File.ReadAllTextAsync(operstatePath)).Trim();
                    networkMetric.IsUp = operstate == "up";
                }

                // Get MTU
                var mtuPath = Path.Combine(interfacePath, "mtu");
                if (File.Exists(mtuPath))
                {
                    var mtuStr = (await File.ReadAllTextAsync(mtuPath)).Trim();
                    if (long.TryParse(mtuStr, out var mtu))
                        networkMetric.MTU = mtu;
                }

                // Get MAC address
                var addressPath = Path.Combine(interfacePath, "address");
                if (File.Exists(addressPath))
                {
                    networkMetric.MACAddress = (await File.ReadAllTextAsync(addressPath)).Trim();
                }

                // Get interface type
                var typePath = Path.Combine(interfacePath, "type");
                if (File.Exists(typePath))
                {
                    var typeStr = (await File.ReadAllTextAsync(typePath)).Trim();
                    networkMetric.InterfaceType = typeStr switch
                    {
                        "1" => "Ethernet",
                        "24" => "Loopback",
                        "772" => "Loopback",
                        _ => "Unknown"
                    };
                }
            }
            catch (Exception ex)
            {
                _logger.LogDebug(ex, "Error enriching network interface info for {Interface}", networkMetric.InterfaceName);
            }
        }

        #endregion

        #region Process Metrics Collection

        private async Task CollectProcessMetricsAsync(LinuxSystemMetricsEntry metrics)
        {
            try
            {
                if (!File.Exists("/proc/stat")) return;

                var statLines = await File.ReadAllLinesAsync("/proc/stat");
                var processLine = statLines.FirstOrDefault(l => l.StartsWith("processes "));
                var procsRunningLine = statLines.FirstOrDefault(l => l.StartsWith("procs_running "));
                var procsBlockedLine = statLines.FirstOrDefault(l => l.StartsWith("procs_blocked "));

                // Get total processes from /proc directory
                var procDirs = Directory.GetDirectories("/proc")
                    .Where(d => Regex.IsMatch(Path.GetFileName(d), @"^\d+$"))
                    .Count();

                metrics.TotalProcesses = procDirs;

                if (procsRunningLine != null)
                {
                    var parts = procsRunningLine.Split(' ');
                    if (parts.Length >= 2 && int.TryParse(parts[1], out var running))
                        metrics.RunningProcesses = running;
                }

                // Calculate sleeping and zombie processes
                metrics.SleepingProcesses = metrics.TotalProcesses - metrics.RunningProcesses;
                
                // Count zombie processes
                var zombieCount = 0;
                try
                {
                    var procDirNames = Directory.GetDirectories("/proc")
                        .Where(d => Regex.IsMatch(Path.GetFileName(d), @"^\d+$"));

                    foreach (var procDir in procDirNames.Take(100)) // Limit to avoid performance issues
                    {
                        var statFile = Path.Combine(procDir, "stat");
                        if (File.Exists(statFile))
                        {
                            var statContent = await File.ReadAllTextAsync(statFile);
                            var statParts = statContent.Split(' ');
                            if (statParts.Length >= 3 && statParts[2] == "Z")
                                zombieCount++;
                        }
                    }
                }
                catch
                {
                    // Ignore errors when reading process stats
                }

                metrics.ZombieProcesses = zombieCount;
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Error collecting process metrics");
            }
        }

        #endregion

        #region System Information Collection

        private async Task CollectSystemInformationAsync(LinuxSystemMetricsEntry metrics)
        {
            try
            {
                // Get kernel version
                if (File.Exists("/proc/version"))
                {
                    var version = await File.ReadAllTextAsync("/proc/version");
                    metrics.KernelVersion = version.Trim();
                }

                // Get distribution information
                if (File.Exists("/etc/os-release"))
                {
                    var osRelease = await File.ReadAllTextAsync("/etc/os-release");
                    var idMatch = Regex.Match(osRelease, @"^ID=""?([^""\n]+)""?", RegexOptions.Multiline);
                    var versionMatch = Regex.Match(osRelease, @"^VERSION_ID=""?([^""\n]+)""?", RegexOptions.Multiline);

                    if (idMatch.Success)
                        metrics.Distribution = idMatch.Groups[1].Value;
                    if (versionMatch.Success)
                        metrics.DistributionVersion = versionMatch.Groups[1].Value;
                }

                // Get system uptime
                if (File.Exists("/proc/uptime"))
                {
                    var uptimeStr = await File.ReadAllTextAsync("/proc/uptime");
                    var uptimeParts = uptimeStr.Split(' ');
                    if (uptimeParts.Length >= 1 && double.TryParse(uptimeParts[0], out var uptimeSeconds))
                    {
                        metrics.SystemUptime = TimeSpan.FromSeconds(uptimeSeconds);
                        metrics.BootTime = DateTime.UtcNow - metrics.SystemUptime;
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Error collecting system information");
            }
        }

        #endregion

        #region Disposal

        public async ValueTask DisposeAsync()
        {
            await StopCollectionAsync();
            _collectionTimer?.Dispose();
            _cancellationTokenSource.Dispose();
        }

        #endregion
    }
}
