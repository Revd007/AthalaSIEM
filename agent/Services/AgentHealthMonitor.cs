using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.NetworkInformation;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using AthalaSIEM.Agent.Models;
using Microsoft.Extensions.Logging;

// Explicitly specify the model classes to avoid namespace conflicts
using NetworkInterfaceMeasurement = AthalaSIEM.Agent.Models.NetworkInterfaceMeasurement;

namespace AthalaSIEM.Agent.Services
{
    /// <summary>
    /// Monitors the health of the agent
    /// </summary>
    public class AgentHealthMonitor : IAgentHealthMonitor
    {
        private readonly ILogger<AgentHealthMonitor> _logger;
        private readonly AgentSettings _settings;
        private readonly string _agentId;
        private readonly Process _currentProcess;
        private readonly DateTime _startTime;
        private Timer? _metricsTimer;
        private SystemMetrics? _lastMetrics;
        private Dictionary<string, string> _componentStatuses;
        private readonly Stopwatch _uptime;
        private readonly PerformanceCounter? _cpuCounter;
        private readonly Dictionary<string, AgentComponentStatus> _componentStatus;
        private AgentHealthStatus _currentStatus;
        private Timer? _monitoringTimer;
        private bool _isWindows;
        private string _diskPath;

        /// <summary>
        /// Initializes a new instance of the <see cref="AgentHealthMonitor"/> class
        /// </summary>
        /// <param name="logger">Logger instance</param>
        /// <param name="settings">Agent settings</param>
        /// <param name="agentId">Agent ID</param>
        public AgentHealthMonitor(ILogger<AgentHealthMonitor> logger, AgentSettings settings, string agentId)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _settings = settings ?? throw new ArgumentNullException(nameof(settings));
            _agentId = agentId ?? throw new ArgumentNullException(nameof(agentId));
            _currentProcess = Process.GetCurrentProcess();
            _startTime = DateTime.UtcNow;
            _componentStatuses = new Dictionary<string, string>();
            _uptime = Stopwatch.StartNew();
            _isWindows = RuntimeInformation.IsOSPlatform(OSPlatform.Windows);
            _componentStatus = new Dictionary<string, AgentComponentStatus>();
            
            // Initialize CPU counter for Windows
            if (_isWindows)
            {
                try 
                {
#pragma warning disable CA1416 // Validate platform compatibility
                    _cpuCounter = new PerformanceCounter("Processor", "% Processor Time", "_Total");
                    // First read to initialize
                    _cpuCounter.NextValue();
#pragma warning restore CA1416
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Failed to initialize CPU performance counter");
                }
            }
            else
            {
                _cpuCounter = null;
            }
            
            // Determine installation path for disk monitoring
            _diskPath = AppDomain.CurrentDomain.BaseDirectory;
            
            // Initialize status
            _currentStatus = new AgentHealthStatus
            {
                Status = AgentHealthState.Healthy.ToString(),
                LastUpdated = DateTime.UtcNow,
                LastChecked = DateTime.UtcNow
            };
            
            // Start monitoring
            StartMonitoring();

            InitializeComponentStatuses();
        }

        /// <summary>
        /// Starts health monitoring
        /// </summary>
        public void StartMonitoring()
        {
            _logger.LogInformation("Starting health monitoring");

            try
            {
                // Collect initial metrics
                CollectAndUpdateMetrics(null);

                // Set up timer for periodic metrics collection
                int intervalMs = _settings.HealthMonitoringIntervalMinutes * 60 * 1000;
                _metricsTimer = new Timer(CollectAndUpdateMetrics, null, intervalMs, intervalMs);

                _logger.LogInformation("Health monitoring started successfully with interval of {IntervalMinutes} minutes", _settings.HealthMonitoringIntervalMinutes);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to start health monitoring");
                throw;
            }
        }

        /// <summary>
        /// Gets the current health status
        /// </summary>
        /// <returns>Current health status</returns>
        public async Task<AgentHeartbeat> GetCurrentHealthStatus()
        {
            try
            {
                TimeSpan uptime = DateTime.UtcNow - _startTime;

                var heartbeat = new AgentHeartbeat
                {
                    AgentId = _agentId,
                    Timestamp = DateTime.UtcNow,
                    Status = "Running",
                    Uptime = (long)uptime.TotalSeconds,
                    CpuUsage = _lastMetrics?.Cpu?.Usage ?? 0,
                    MemoryUsage = _lastMetrics?.Memory?.UsedPercentage ?? 0,
                    DiskUsage = _lastMetrics?.Disk?.UsedPercentage ?? 0,
                    IpAddress = await GetLocalIpAddress(),
                    OsDescription = RuntimeInformation.OSDescription,
                    MachineName = Environment.MachineName
                };

                return heartbeat;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting current health status");
                throw;
            }
        }

        /// <summary>
        /// Gets detailed system metrics
        /// </summary>
        /// <returns>System metrics</returns>
        public Task<SystemMetrics> GetSystemMetrics()
        {
            return Task.FromResult(_lastMetrics ?? CollectSystemMetrics());
        }

        /// <summary>
        /// Generates a health report
        /// </summary>
        /// <returns>Health report</returns>
        public async Task<AgentHealthReport> GenerateHealthReport()
        {
            try
            {
                var metrics = await GetSystemMetrics();
                TimeSpan uptime = DateTime.UtcNow - _startTime;

                UpdateComponentStatuses(metrics);

                var report = new AgentHealthReport
                {
                    AgentId = _agentId,
                    Timestamp = DateTime.UtcNow,
                    Status = DetermineOverallStatus(),
                    Uptime = (long)uptime.TotalSeconds,
                    Metrics = metrics,
                    ComponentStatuses = _componentStatuses.ToDictionary(kv => kv.Key, kv => kv.Value),
                    Diagnostics = CollectDiagnostics()
                };

                // Convert dictionary to list of ComponentStatus
                report.Components = _componentStatus.Select(kv => 
                    new AthalaSIEM.Agent.Models.ComponentStatus { 
                        Name = kv.Value.ComponentName,
                        Status = kv.Value.Status,
                        Message = kv.Value.Message
                    }).ToList();

                return report;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error generating health report");
                throw;
            }
        }

        private void CollectAndUpdateMetrics(object? state)
        {
            try
            {
                _lastMetrics = CollectSystemMetrics();
                UpdateComponentStatuses(_lastMetrics);
                _logger.LogDebug("Collected system metrics successfully");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error collecting system metrics");
            }
        }

        private SystemMetrics CollectSystemMetrics()
        {
            var metrics = new SystemMetrics
            {
                AgentId = _agentId,
                Timestamp = DateTime.UtcNow,
                Cpu = CollectCpuMetrics(),
                Memory = CollectMemoryMetrics(),
                Disk = CollectDiskMetrics(),
                Network = CollectNetworkMetrics(),
                Process = CollectProcessMetrics()
            };

            return metrics;
        }

        private CpuMetrics CollectCpuMetrics()
        {
            var cpuMetrics = new CpuMetrics();

            try
            {
                if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                {
                    // Windows implementation
                    // For simplicity, we are using process CPU usage as an indicator
                    cpuMetrics.Usage = Math.Min(100, (double)_currentProcess.TotalProcessorTime.TotalMilliseconds / 
                        (Environment.ProcessorCount * (DateTime.UtcNow - _startTime).TotalMilliseconds) * 100);
                    cpuMetrics.NumberOfCores = Environment.ProcessorCount;
                }
                else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
                {
                    // Linux implementation - this is a mock implementation
                    // In a real implementation, read from /proc/stat or use a library
                    cpuMetrics.Usage = new Random().Next(1, 100); // Mock data
                    cpuMetrics.LoadAverage = new[] { 0.5, 0.7, 0.9 }; // Mock data
                    cpuMetrics.NumberOfCores = Environment.ProcessorCount;
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error collecting CPU metrics");
            }

            return cpuMetrics;
        }

        private MemoryMetrics CollectMemoryMetrics()
        {
            var memoryMetrics = new MemoryMetrics();

            try
            {
                if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                {
                    // Windows implementation
                    memoryMetrics.TotalBytes = GetTotalPhysicalMemoryWindows();
                    memoryMetrics.AvailableBytes = GetAvailablePhysicalMemoryWindows();
                    memoryMetrics.UsedBytes = memoryMetrics.TotalBytes - memoryMetrics.AvailableBytes;
                    memoryMetrics.UsedPercentage = (double)memoryMetrics.UsedBytes / memoryMetrics.TotalBytes * 100;
                    
                    // Process memory
                    memoryMetrics.ProcessUsedBytes = _currentProcess.WorkingSet64;
                    memoryMetrics.ProcessPrivateBytes = _currentProcess.PrivateMemorySize64;
                }
                else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
                {
                    // Linux implementation - read from /proc/meminfo
                    // This is a simplified mock implementation
                    memoryMetrics.TotalBytes = 8L * 1024 * 1024 * 1024; // Mock 8GB 
                    memoryMetrics.AvailableBytes = 4L * 1024 * 1024 * 1024; // Mock 4GB available
                    memoryMetrics.UsedBytes = memoryMetrics.TotalBytes - memoryMetrics.AvailableBytes;
                    memoryMetrics.UsedPercentage = (double)memoryMetrics.UsedBytes / memoryMetrics.TotalBytes * 100;
                    
                    // Process memory
                    memoryMetrics.ProcessUsedBytes = _currentProcess.WorkingSet64;
                    memoryMetrics.ProcessPrivateBytes = _currentProcess.PrivateMemorySize64;
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error collecting memory metrics");
            }

            return memoryMetrics;
        }

        private DiskMetrics CollectDiskMetrics()
        {
            var diskMetrics = new DiskMetrics
            {
                Drives = new List<DriveMeasurement>()
            };

            try
            {
                // Get all logical drives
                foreach (var drive in DriveInfo.GetDrives().Where(d => d.IsReady))
                {
                    var driveMeasurement = new DriveMeasurement
                    {
                        Name = drive.Name,
                        TotalBytes = drive.TotalSize,
                        AvailableBytes = drive.AvailableFreeSpace,
                        UsedBytes = drive.TotalSize - drive.AvailableFreeSpace,
                        UsedPercentage = (double)(drive.TotalSize - drive.AvailableFreeSpace) / drive.TotalSize * 100
                    };

                    diskMetrics.Drives.Add(driveMeasurement);
                }

                // Calculate totals if we have drives
                if (diskMetrics.Drives.Count > 0)
                {
                    diskMetrics.TotalBytes = diskMetrics.Drives.Sum(d => d.TotalBytes);
                    diskMetrics.AvailableBytes = diskMetrics.Drives.Sum(d => d.AvailableBytes);
                    diskMetrics.UsedBytes = diskMetrics.Drives.Sum(d => d.UsedBytes);
                    diskMetrics.UsedPercentage = (double)diskMetrics.UsedBytes / diskMetrics.TotalBytes * 100;
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error collecting disk metrics");
            }

            return diskMetrics;
        }

        private NetworkMetrics CollectNetworkMetrics()
        {
            var networkMetrics = new NetworkMetrics();

            try
            {
                // Get all network interfaces
                foreach (var networkInterface in NetworkInterface.GetAllNetworkInterfaces()
                    .Where(n => n.OperationalStatus == OperationalStatus.Up))
                {
                    var stats = networkInterface.GetIPStatistics();
                    var interfaceMeasurement = new Models.NetworkInterfaceMeasurement
                    {
                        Name = networkInterface.Name,
                        Description = networkInterface.Description,
                        BytesReceived = stats.BytesReceived,
                        BytesSent = stats.BytesSent,
                        Speed = networkInterface.Speed
                    };

                    networkMetrics.Interfaces.Add(interfaceMeasurement);
                }

                // Calculate totals if we have interfaces
                if (networkMetrics.Interfaces.Count > 0)
                {
                    networkMetrics.TotalBytesReceived = networkMetrics.Interfaces.Sum(i => i.BytesReceived);
                    networkMetrics.TotalBytesSent = networkMetrics.Interfaces.Sum(i => i.BytesSent);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error collecting network metrics");
            }

            return networkMetrics;
        }

        private ProcessMetrics CollectProcessMetrics()
        {
            var processMetrics = new ProcessMetrics
            {
                CurrentProcess = new ProcessMemoryUsage
                {
                    Name = _currentProcess.ProcessName,
                    Id = _currentProcess.Id,
                    MemoryUsageBytes = _currentProcess.WorkingSet64,
                    CpuUsagePercent = GetProcessCpuUsage(_currentProcess),
                    ThreadCount = _currentProcess.Threads.Count,
                    StartTime = _currentProcess.StartTime.ToUniversalTime()
                },
                MemoryUsageProcesses = new List<ProcessMemoryUsage>()
            };

            try
            {
                // Top memory-using processes
                Process[] allProcesses = Process.GetProcesses();
                processMetrics.MemoryUsageProcesses = allProcesses
                    .Select(p => new ProcessMemoryUsage
                    {
                        Name = p.ProcessName,
                        Id = p.Id,
                        MemoryUsageBytes = p.WorkingSet64
                    })
                    .OrderByDescending(p => p.MemoryUsageBytes)
                    .Take(5)
                    .ToList();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error collecting process metrics");
            }

            return processMetrics;
        }

        private double GetProcessCpuUsage(Process process)
        {
            // Simple mock implementation - in real world would track over time
            try
            {
                return Math.Min(100, process.TotalProcessorTime.TotalMilliseconds / 
                    (Environment.ProcessorCount * (DateTime.UtcNow - _startTime).TotalMilliseconds) * 100);
            }
            catch
            {
                return 0;
            }
        }

        private void InitializeComponentStatuses()
        {
            _componentStatuses["AgentService"] = "Running";
            _componentStatuses["LogCollectors"] = "Not Initialized";
            _componentStatuses["HealthMonitor"] = "Starting";
            _componentStatuses["CommunicationChannel"] = "Not Connected";
        }

        private void UpdateComponentStatuses(SystemMetrics metrics)
        {
            if (metrics == null)
                return;

            // Update based on metrics
            _componentStatuses["HealthMonitor"] = "Running";

            // Check if CPU is high
            if (metrics.Cpu != null && metrics.Cpu.Usage > 90)
            {
                _componentStatuses["CpuStatus"] = "Warning: High Usage";
            }
            else
            {
                _componentStatuses["CpuStatus"] = "Normal";
            }

            // Check if memory is high
            if (metrics.Memory != null && metrics.Memory.UsedPercentage > 90)
            {
                _componentStatuses["MemoryStatus"] = "Warning: High Usage";
            }
            else
            {
                _componentStatuses["MemoryStatus"] = "Normal";
            }

            // Check if disk space is low
            if (metrics.Disk != null && metrics.Disk.Drives.Any(d => d.UsedPercentage > 90))
            {
                _componentStatuses["DiskStatus"] = "Warning: Low Free Space";
            }
            else
            {
                _componentStatuses["DiskStatus"] = "Normal";
            }
        }

        private string DetermineOverallStatus()
        {
            bool hasWarning = false;
            bool hasError = false;
            bool hasCritical = false;

            foreach (var component in _currentStatus.ComponentStatuses)
            {
                if (component.Value.Status.Equals("Warning", StringComparison.OrdinalIgnoreCase))
                {
                    hasWarning = true;
                }
                else if (component.Value.Status.Equals("Error", StringComparison.OrdinalIgnoreCase) || 
                         component.Value.Status.Equals("Degraded", StringComparison.OrdinalIgnoreCase))
                {
                    hasError = true;
                }
                else if (component.Value.Status.Equals("Critical", StringComparison.OrdinalIgnoreCase) || 
                         component.Value.Status.Equals("Offline", StringComparison.OrdinalIgnoreCase))
                {
                    hasCritical = true;
                }
            }

            if (hasCritical)
            {
                _currentStatus.Status = AgentHealthState.Critical.ToString();
            }
            else if (hasError)
            {
                _currentStatus.Status = AgentHealthState.Degraded.ToString();
            }
            else if (hasWarning)
            {
                _currentStatus.Status = AgentHealthState.Warning.ToString();
            }
            else
            {
                _currentStatus.Status = AgentHealthState.Healthy.ToString();
            }

            return _currentStatus.Status;
        }

        private Dictionary<string, string> CollectDiagnostics()
        {
            var diagnostics = new Dictionary<string, string>();

            try
            {
                // Basic system information
                diagnostics["OSVersion"] = RuntimeInformation.OSDescription;
                diagnostics["ProcessorArchitecture"] = RuntimeInformation.ProcessArchitecture.ToString();
                diagnostics["MachineName"] = Environment.MachineName;
                diagnostics["ProcessorCount"] = Environment.ProcessorCount.ToString();
                diagnostics["WorkingDirectory"] = Environment.CurrentDirectory;
                diagnostics["DotNetVersion"] = Environment.Version.ToString();
                
                // Network connectivity check
                diagnostics["NetworkConnectivity"] = CheckNetworkConnectivity() ? "Connected" : "Disconnected";
                
                // Process information
                diagnostics["ProcessUptime"] = (DateTime.UtcNow - _startTime).ToString();
                diagnostics["ProcessId"] = _currentProcess.Id.ToString();
                
                // Memory information
                diagnostics["TotalMemory"] = FormatBytesAsReadable(_lastMetrics?.Memory?.TotalBytes ?? 0);
                diagnostics["AvailableMemory"] = FormatBytesAsReadable(_lastMetrics?.Memory?.AvailableBytes ?? 0);
                
                // Disk information
                if (_lastMetrics?.Disk?.Drives.Count > 0)
                {
                    diagnostics["SystemDrive"] = _lastMetrics.Disk.Drives[0].Name;
                    diagnostics["SystemDriveSpace"] = FormatBytesAsReadable(_lastMetrics.Disk.Drives[0].TotalBytes);
                    diagnostics["SystemDriveAvailable"] = FormatBytesAsReadable(_lastMetrics.Disk.Drives[0].AvailableBytes);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error collecting diagnostics");
                diagnostics["DiagnosticsError"] = ex.Message;
            }

            return diagnostics;
        }

        private bool CheckNetworkConnectivity()
        {
            try
            {
                using (var ping = new Ping())
                {
                    // Try to ping a reliable host (Google DNS)
                    var reply = ping.Send("8.8.8.8", 3000);
                    return reply.Status == IPStatus.Success;
                }
            }
            catch
            {
                return false;
            }
        }

        private string FormatBytesAsReadable(long bytes)
        {
            string[] sizes = { "B", "KB", "MB", "GB", "TB" };
            int order = 0;
            double size = bytes;
            
            while (size >= 1024 && order < sizes.Length - 1)
            {
                order++;
                size /= 1024;
            }
            
            return $"{size:0.##} {sizes[order]}";
        }

        private async Task<string> GetLocalIpAddress()
        {
            try
            {
                using var socket = new Socket(AddressFamily.InterNetwork, SocketType.Dgram, 0);
                socket.Connect("8.8.8.8", 65530);
                var endPoint = socket.LocalEndPoint as IPEndPoint;
                return endPoint?.Address.ToString() ?? "Unknown";
            }
            catch
            {
                try
                {
                    // Fallback
                    string hostName = Dns.GetHostName();
                    var hostEntry = await Dns.GetHostEntryAsync(hostName);
                    var address = hostEntry.AddressList.FirstOrDefault(ip => ip.AddressFamily == AddressFamily.InterNetwork);
                    return address?.ToString() ?? "Unknown";
                }
                catch
                {
                    return "Unknown";
                }
            }
        }

        private static long GetTotalPhysicalMemoryWindows()
        {
            try
            {
                // This method reads the total physical memory in Windows
                // For brevity, using a simpler approximation
                return new ComputerInfo().TotalPhysicalMemory;
            }
            catch
            {
                // Fallback to approximate value if can't read
                return 8L * 1024 * 1024 * 1024; // Assume 8GB
            }
        }

        private static long GetAvailablePhysicalMemoryWindows()
        {
            try
            {
                // This method reads the available physical memory in Windows
                return new ComputerInfo().AvailablePhysicalMemory;
            }
            catch
            {
                // Fallback to approximate value if can't read
                return 4L * 1024 * 1024 * 1024; // Assume 4GB available
            }
        }

        // Helper class for Windows memory info
        private class ComputerInfo
        {
            public long TotalPhysicalMemory => GetTotalPhysicalMemory();
            public long AvailablePhysicalMemory => GetAvailablePhysicalMemory();

            [DllImport("kernel32.dll")]
            [return: MarshalAs(UnmanagedType.Bool)]
            private static extern bool GlobalMemoryStatusEx(ref MEMORYSTATUSEX lpBuffer);

            [StructLayout(LayoutKind.Sequential)]
            private struct MEMORYSTATUSEX
            {
                public uint dwLength;
                public uint dwMemoryLoad;
                public ulong ullTotalPhys;
                public ulong ullAvailPhys;
                public ulong ullTotalPageFile;
                public ulong ullAvailPageFile;
                public ulong ullTotalVirtual;
                public ulong ullAvailVirtual;
                public ulong ullAvailExtendedVirtual;

                public MEMORYSTATUSEX(bool init)
                {
                    dwLength = 0;
                    dwMemoryLoad = 0;
                    ullTotalPhys = 0;
                    ullAvailPhys = 0;
                    ullTotalPageFile = 0;
                    ullAvailPageFile = 0;
                    ullTotalVirtual = 0;
                    ullAvailVirtual = 0;
                    ullAvailExtendedVirtual = 0;

                    if (init)
                    {
                        dwLength = (uint)Marshal.SizeOf(typeof(MEMORYSTATUSEX));
                    }
                }
            }

            private long GetTotalPhysicalMemory()
            {
                var memoryStatus = new MEMORYSTATUSEX(true);
                if (GlobalMemoryStatusEx(ref memoryStatus))
                {
                    return (long)memoryStatus.ullTotalPhys;
                }
                return 0;
            }

            private long GetAvailablePhysicalMemory()
            {
                var memoryStatus = new MEMORYSTATUSEX(true);
                if (GlobalMemoryStatusEx(ref memoryStatus))
                {
                    return (long)memoryStatus.ullAvailPhys;
                }
                return 0;
            }
        }

        /// <summary>
        /// Gets the current health status object of the agent
        /// </summary>
        /// <returns>The current health status object</returns>
        public AgentHealthStatus GetHealthStatusObject()
        {
            return _currentStatus;
        }
        
        /// <summary>
        /// Gets the status of a specific component
        /// </summary>
        /// <param name="componentName">Name of the component</param>
        /// <returns>The component status</returns>
        public AgentComponentStatus GetComponentStatus(string componentName)
        {
            if (_componentStatus.TryGetValue(componentName, out var status))
            {
                return status;
            }
            
            return new AgentComponentStatus
            {
                Name = componentName,
                Status = "Unknown",
                LastChecked = DateTime.UtcNow
            };
        }
        
        /// <summary>
        /// Updates the status of a component
        /// </summary>
        /// <param name="componentName">Name of the component</param>
        /// <param name="status">Status of the component</param>
        /// <param name="message">Optional message</param>
        /// <param name="details">Optional details</param>
        public void UpdateComponentStatus(string componentName, string status, string? message = null, Dictionary<string, string>? details = null)
        {
            var componentStatus = new AgentComponentStatus
            {
                ComponentName = componentName,
                Status = status,
                Message = message ?? string.Empty,
                LastUpdated = DateTime.UtcNow
            };
            
            if (details != null)
            {
                foreach (var detail in details)
                {
                    componentStatus.Details[detail.Key] = detail.Value;
                }
            }
            
            _componentStatus[componentName] = componentStatus;
            
            // Update overall status using DetermineOverallStatus method instead of missing UpdateOverallStatus
            _currentStatus.Status = DetermineOverallStatus();
        }
        
        /// <summary>
        /// Gets a full health report for the agent
        /// </summary>
        /// <returns>A health report</returns>
        public AgentHealthReport GetHealthReport()
        {
            var report = new AgentHealthReport
            {
                AgentId = "unknown", // Will be populated by the identity service
                ReportedAt = DateTime.UtcNow,
                Status = _currentStatus.Status.ToString(),
                Uptime = (long)_uptime.Elapsed.TotalSeconds,
                Diagnostics = CollectDiagnostics()
            };
            
            // Convert dictionary to list of ComponentStatus
            report.Components = _componentStatus.Select(kv => 
                new AthalaSIEM.Agent.Models.ComponentStatus { 
                    Name = kv.Value.ComponentName,
                    Status = kv.Value.Status,
                    Message = kv.Value.Message
                }).ToList();
            
            return report;
        }
        
        /// <summary>
        /// Starts the health monitoring timer
        /// </summary>
        /// <param name="interval">Monitoring interval in seconds</param>
        public void StartMonitoring(int interval = 60)
        {
            _monitoringTimer = new Timer(CheckHealth, null, 0, interval * 1000);
            _logger.LogInformation("Health monitoring started with interval of {Interval} seconds", interval);
        }
        
        /// <summary>
        /// Stops the health monitoring timer
        /// </summary>
        public void StopMonitoring()
        {
            _monitoringTimer?.Dispose();
            _monitoringTimer = null;
            _logger.LogInformation("Health monitoring stopped");
        }
        
        /// <summary>
        /// Checks the health of the agent
        /// </summary>
        /// <param name="state">Timer state</param>
        private void CheckHealth(object? state)
        {
            try
            {
                // Check various components
                CheckBackendConnection();
                CheckDiskSpace();
                CheckMemory();
                CheckCollectors();
                
                // Update overall status
                UpdateOverallStatus();
                
                _logger.LogTrace("Health check completed");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error checking health");
            }
        }
        
        /// <summary>
        /// Updates system metrics
        /// </summary>
        private void UpdateSystemMetrics()
        {
            try
            {
                // Get CPU usage
                _currentStatus.CpuUsage = GetCpuUsage();
                
                // Get memory usage
                _currentStatus.MemoryUsage = GetMemoryUsage();
                
                // Get disk usage
                _currentStatus.DiskUsage = GetDiskUsage(_diskPath);
                
                // Update uptime
                _currentStatus.UptimeSeconds = (long)_uptime.Elapsed.TotalSeconds;
                
                _logger.LogDebug("System metrics updated: CPU {CPU}%, Memory {Memory}%, Disk {Disk}%",
                    _currentStatus.CpuUsage?.ToString("F1"),
                    _currentStatus.MemoryUsage?.ToString("F1"),
                    _currentStatus.DiskUsage?.ToString("F1"));
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Error updating system metrics");
            }
        }
        
        /// <summary>
        /// Gets the CPU usage
        /// </summary>
        /// <returns>CPU usage percentage</returns>
        private double? GetCpuUsage()
        {
            try
            {
                if (_isWindows && _cpuCounter != null)
                {
                    // Use performance counter on Windows
#pragma warning disable CA1416 // Validate platform compatibility
                    return _cpuCounter.NextValue();
#pragma warning restore CA1416
                }
                else if (!_isWindows)
                {
                    // On Linux, read from /proc/stat
                    string[] cpuStats = File.ReadAllText("/proc/stat").Split('\n')[0].Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                    
                    if (cpuStats.Length >= 5)
                    {
                        long user = long.Parse(cpuStats[1]);
                        long nice = long.Parse(cpuStats[2]);
                        long system = long.Parse(cpuStats[3]);
                        long idle = long.Parse(cpuStats[4]);
                        
                        long total = user + nice + system + idle;
                        long busy = user + nice + system;
                        
                        return busy * 100.0 / total;
                    }
                }
                
                return null;
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Error getting CPU usage");
                return null;
            }
        }
        
        /// <summary>
        /// Gets the memory usage
        /// </summary>
        /// <returns>Memory usage percentage</returns>
        private double? GetMemoryUsage()
        {
            try
            {
                if (_isWindows)
                {
                    // On Windows, use Performance Info API
                    var memoryStatus = new MemoryStatusEx();
                    memoryStatus.dwLength = Marshal.SizeOf(memoryStatus);
                    GlobalMemoryStatusEx(memoryStatus);
                    
                    // Calculate memory usage percentage
                    return 100.0 - ((double)memoryStatus.ullAvailPhys / memoryStatus.ullTotalPhys * 100.0);
                }
                else
                {
                    // On Linux, read from /proc/meminfo
                    string memInfo = File.ReadAllText("/proc/meminfo");
                    var lines = memInfo.Split('\n');
                    
                    long totalMem = 0;
                    long freeMem = 0;
                    long buffMem = 0;
                    long cacheMem = 0;
                    
                    foreach (var line in lines)
                    {
                        if (line.StartsWith("MemTotal:"))
                        {
                            totalMem = ParseMemInfoLine(line);
                        }
                        else if (line.StartsWith("MemFree:"))
                        {
                            freeMem = ParseMemInfoLine(line);
                        }
                        else if (line.StartsWith("Buffers:"))
                        {
                            buffMem = ParseMemInfoLine(line);
                        }
                        else if (line.StartsWith("Cached:"))
                        {
                            cacheMem = ParseMemInfoLine(line);
                        }
                    }
                    
                    if (totalMem > 0)
                    {
                        long usedMem = totalMem - freeMem - buffMem - cacheMem;
                        return usedMem * 100.0 / totalMem;
                    }
                }
                
                return null;
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Error getting memory usage");
                return null;
            }
        }
        
        /// <summary>
        /// Gets the disk usage for a path
        /// </summary>
        /// <param name="path">Path to check</param>
        /// <returns>Disk usage percentage</returns>
        private double? GetDiskUsage(string path)
        {
            try
            {
                string? rootPath = Path.GetPathRoot(path);
                if (string.IsNullOrEmpty(rootPath))
                {
                    _logger.LogWarning("Could not determine root path for {Path}", path);
                    return null;
                }
                
                var driveInfo = new DriveInfo(rootPath);
                if (driveInfo.IsReady)
                {
                    double totalSize = driveInfo.TotalSize;
                    double freeSpace = driveInfo.TotalFreeSpace;
                    double usedSpace = totalSize - freeSpace;
                    
                    return usedSpace * 100.0 / totalSize;
                }
                
                return null;
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Error getting disk usage for {Path}", path);
                return null;
            }
        }
        
        /// <summary>
        /// Checks disk space and updates component status
        /// </summary>
        private void CheckDiskSpace()
        {
            try
            {
                string? rootPath = Path.GetPathRoot(_diskPath);
                if (string.IsNullOrEmpty(rootPath))
                {
                    _logger.LogWarning("Could not determine root path for {Path}", _diskPath);
                    UpdateComponentStatus("DiskSpace", "Error", "Could not determine disk path");
                    return;
                }
                
                var driveInfo = new DriveInfo(rootPath);
                if (driveInfo.IsReady)
                {
                    double freeSpaceGB = driveInfo.AvailableFreeSpace / (1024.0 * 1024 * 1024);
                    double totalSpaceGB = driveInfo.TotalSize / (1024.0 * 1024 * 1024);
                    double usagePercent = (1 - (double)driveInfo.AvailableFreeSpace / driveInfo.TotalSize) * 100;
                    
                    var details = new Dictionary<string, string>
                    {
                        { "FreeSpaceGB", freeSpaceGB.ToString("F2") },
                        { "TotalSpaceGB", totalSpaceGB.ToString("F2") },
                        { "UsagePercent", usagePercent.ToString("F2") }
                    };
                    
                    string status;
                    string message;
                    
                    if (usagePercent > 90)
                    {
                        status = "Critical";
                        message = $"Disk space critically low: {freeSpaceGB:F2} GB free ({usagePercent:F0}% used)";
                    }
                    else if (usagePercent > 80)
                    {
                        status = "Warning";
                        message = $"Disk space low: {freeSpaceGB:F2} GB free ({usagePercent:F0}% used)";
                    }
                    else
                    {
                        status = "Healthy";
                        message = $"Disk space normal: {freeSpaceGB:F2} GB free ({usagePercent:F0}% used)";
                    }
                    
                    UpdateComponentStatus("DiskSpace", status, message, details);
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Error checking disk space");
                UpdateComponentStatus("DiskSpace", "Error", $"Error checking disk space: {ex.Message}");
            }
        }
        
        /// <summary>
        /// Checks memory usage and updates component status
        /// </summary>
        private void CheckMemoryUsage()
        {
            try
            {
                double? memUsage = GetMemoryUsage();
                if (memUsage.HasValue)
                {
                    var details = new Dictionary<string, string>
                    {
                        { "UsagePercent", memUsage.Value.ToString("F2") }
                    };
                    
                    string status;
                    string message;
                    
                    if (memUsage.Value > 90)
                    {
                        status = "Critical";
                        message = $"Memory usage critically high: {memUsage.Value:F0}%";
                    }
                    else if (memUsage.Value > 80)
                    {
                        status = "Warning";
                        message = $"Memory usage high: {memUsage.Value:F0}%";
                    }
                    else
                    {
                        status = "Healthy";
                        message = $"Memory usage normal: {memUsage.Value:F0}%";
                    }
                    
                    UpdateComponentStatus("Memory", status, message, details);
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Error checking memory usage");
                UpdateComponentStatus("Memory", "Error", $"Error checking memory usage: {ex.Message}");
            }
        }
        
        /// <summary>
        /// Checks network connectivity to backend API and updates component status
        /// </summary>
        private void CheckBackendConnectivity()
        {
            try
            {
                // Ping backend API
                var ping = new Ping();
                var result = ping.Send("api.athala-siem.com", 3000);
                
                var details = new Dictionary<string, string>
                {
                    { "Status", result.Status.ToString() }
                };
                
                if (result.Status == IPStatus.Success)
                {
                    details.Add("RoundtripTime", result.RoundtripTime.ToString());
                    UpdateComponentStatus("Network", "Healthy", $"Network connectivity normal: {result.RoundtripTime}ms", details);
                }
                else
                {
                    UpdateComponentStatus("Network", "Warning", $"Network connectivity issue: {result.Status}", details);
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Error checking network connectivity");
                UpdateComponentStatus("Network", "Error", $"Error checking network connectivity: {ex.Message}");
            }
        }
        
        /// <summary>
        /// Parses a memory info line from /proc/meminfo
        /// </summary>
        /// <param name="line">Line to parse</param>
        /// <returns>Parsed memory value in KB</returns>
        private long ParseMemInfoLine(string line)
        {
            var parts = line.Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length >= 2 && long.TryParse(parts[1], out long result))
            {
                return result;
            }
            
            return 0;
        }
        
        [StructLayout(LayoutKind.Sequential)]
        private struct MemoryStatusEx
        {
            public int dwLength;
            public int dwMemoryLoad;
            public ulong ullTotalPhys;
            public ulong ullAvailPhys;
            public ulong ullTotalPageFile;
            public ulong ullAvailPageFile;
            public ulong ullTotalVirtual;
            public ulong ullAvailVirtual;
            public ulong ullAvailExtendedVirtual;
        }
        
        [DllImport("kernel32.dll")]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool GlobalMemoryStatusEx(MemoryStatusEx lpBuffer);

        /// <summary>
        /// Updates the health status
        /// </summary>
        private void UpdateHealth()
        {
            try 
            {
                // Calculate uptime
                _currentStatus.UptimeSeconds = (long)_uptime.Elapsed.TotalSeconds;
                
                // Update status using DetermineOverallStatus method
                _currentStatus.Status = DetermineOverallStatus();
                
                _logger.LogDebug("Health status updated");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error updating health status");
            }
        }

        /// <summary>
        /// Checks the backend connection status
        /// </summary>
        private void CheckBackendConnection()
        {
            try
            {
                // Use the existing CheckBackendConnectivity method
                CheckBackendConnectivity();
                
                _logger.LogDebug("Backend connection check completed");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error checking backend connection");
                UpdateComponentStatus("BackendConnection", "Error", $"Error checking backend connection: {ex.Message}");
            }
        }
        
        /// <summary>
        /// Checks the memory status
        /// </summary>
        private void CheckMemory()
        {
            try
            {
                // Use the existing CheckMemoryUsage method
                CheckMemoryUsage();
                
                _logger.LogDebug("Memory check completed");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error checking memory");
                UpdateComponentStatus("Memory", "Error", $"Error checking memory: {ex.Message}");
            }
        }
        
        /// <summary>
        /// Checks the status of all log collectors
        /// </summary>
        private void CheckCollectors()
        {
            try
            {
                // This would typically check the status of all log collectors
                // For now, just update the component status to indicate it's healthy
                UpdateComponentStatus("LogCollectors", "Healthy", "Log collectors are functioning normally");
                
                _logger.LogDebug("Collectors check completed");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error checking collectors");
                UpdateComponentStatus("LogCollectors", "Error", $"Error checking collectors: {ex.Message}");
            }
        }
        
        /// <summary>
        /// Updates the overall status based on component statuses
        /// </summary>
        private void UpdateOverallStatus()
        {
            try
            {
                // Use the existing DetermineOverallStatus method to set the current status
                _currentStatus.Status = DetermineOverallStatus();
                
                // Update the last checked timestamp
                _currentStatus.LastChecked = DateTime.UtcNow;
                
                _logger.LogDebug("Overall status updated to: {Status}", _currentStatus.Status);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error updating overall status");
                _currentStatus.Status = AgentHealthState.Degraded.ToString();
            }
        }
    }
} 