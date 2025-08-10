using System;
using System.Collections.Generic;

namespace AthalaSIEM.UniversalAgent.Models
{
    /// <summary>
    /// Linux System Metrics Models for AthalaSIEM Universal Agent
    /// Comprehensive system monitoring for enterprise SIEM
    /// Author: Revian Ravil Athala
    /// </summary>

    /// <summary>
    /// Linux system metrics log entry extending base LogEntry
    /// </summary>
    public class LinuxSystemMetricsEntry : LogEntry
    {
        // CPU Metrics
        public double CpuUsagePercent { get; set; }
        public double LoadAverage1Min { get; set; }
        public double LoadAverage5Min { get; set; }
        public double LoadAverage15Min { get; set; }
        public int CpuCoreCount { get; set; }
        public Dictionary<string, double> PerCoreCpuUsage { get; set; } = new();

        // Memory Metrics
        public long MemoryTotalBytes { get; set; }
        public long MemoryUsedBytes { get; set; }
        public long MemoryFreeBytes { get; set; }
        public long MemoryAvailableBytes { get; set; }
        public long MemoryCachedBytes { get; set; }
        public long MemoryBuffersBytes { get; set; }
        public double MemoryUsagePercent { get; set; }
        
        // Swap Metrics
        public long SwapTotalBytes { get; set; }
        public long SwapUsedBytes { get; set; }
        public long SwapFreeBytes { get; set; }
        public double SwapUsagePercent { get; set; }

        // Disk Metrics
        public Dictionary<string, DiskMetrics> DiskUsage { get; set; } = new();
        public Dictionary<string, DiskIOMetrics> DiskIOStats { get; set; } = new();

        // Network Metrics
        public Dictionary<string, NetworkMetrics> NetworkStats { get; set; } = new();

        // Process Metrics
        public int TotalProcesses { get; set; }
        public int RunningProcesses { get; set; }
        public int SleepingProcesses { get; set; }
        public int ZombieProcesses { get; set; }

        // System Information
        public string KernelVersion { get; set; } = "";
        public string Distribution { get; set; } = "";
        public string DistributionVersion { get; set; } = "";
        public TimeSpan SystemUptime { get; set; }
        public DateTime BootTime { get; set; }
    }

    /// <summary>
    /// Disk usage metrics for filesystem monitoring
    /// </summary>
    public class DiskMetrics
    {
        public string Filesystem { get; set; } = "";
        public string MountPoint { get; set; } = "";
        public string FilesystemType { get; set; } = "";
        public long TotalBytes { get; set; }
        public long UsedBytes { get; set; }
        public long AvailableBytes { get; set; }
        public double UsagePercent { get; set; }
        public long TotalInodes { get; set; }
        public long UsedInodes { get; set; }
        public long FreeInodes { get; set; }
        public double InodeUsagePercent { get; set; }
    }

    /// <summary>
    /// Disk I/O statistics for performance monitoring
    /// </summary>
    public class DiskIOMetrics
    {
        public string DeviceName { get; set; } = "";
        public long ReadOperations { get; set; }
        public long WriteOperations { get; set; }
        public long ReadBytes { get; set; }
        public long WriteBytes { get; set; }
        public double ReadBytesPerSecond { get; set; }
        public double WriteBytesPerSecond { get; set; }
        public double IOUtilizationPercent { get; set; }
        public double AverageIOWaitTime { get; set; }
    }

    /// <summary>
    /// Network interface statistics
    /// </summary>
    public class NetworkMetrics
    {
        public string InterfaceName { get; set; } = "";
        public string InterfaceType { get; set; } = "";
        public bool IsUp { get; set; }
        public string IPAddress { get; set; } = "";
        public string MACAddress { get; set; } = "";
        public long MTU { get; set; }
        
        // Traffic Statistics
        public long BytesReceived { get; set; }
        public long BytesSent { get; set; }
        public long PacketsReceived { get; set; }
        public long PacketsSent { get; set; }
        
        // Error Statistics
        public long ReceiveErrors { get; set; }
        public long TransmitErrors { get; set; }
        public long ReceiveDropped { get; set; }
        public long TransmitDropped { get; set; }
        
        // Rate Statistics
        public double BytesReceivedPerSecond { get; set; }
        public double BytesSentPerSecond { get; set; }
        public double PacketsReceivedPerSecond { get; set; }
        public double PacketsSentPerSecond { get; set; }
    }

    /// <summary>
    /// System metrics collection configuration
    /// </summary>
    public class LinuxSystemMetricsConfiguration
    {
        public bool EnableCPUMonitoring { get; set; } = true;
        public bool EnableMemoryMonitoring { get; set; } = true;
        public bool EnableDiskMonitoring { get; set; } = true;
        public bool EnableNetworkMonitoring { get; set; } = true;
        public bool EnableProcessMonitoring { get; set; } = true;
        
        public int CollectionIntervalSeconds { get; set; } = 30;
        public int MaxHistoryEntries { get; set; } = 1000;
        
        // Thresholds for alerting
        public double CpuAlertThreshold { get; set; } = 80.0;
        public double MemoryAlertThreshold { get; set; } = 85.0;
        public double DiskAlertThreshold { get; set; } = 90.0;
        
        // Specific interfaces/filesystems to monitor
        public List<string> MonitoredNetworkInterfaces { get; set; } = new();
        public List<string> MonitoredFilesystems { get; set; } = new();
        public List<string> ExcludedFilesystems { get; set; } = new() { "tmpfs", "devtmpfs", "sysfs", "proc" };
    }

    /// <summary>
    /// Linux system metrics collector health status
    /// </summary>
    public class LinuxSystemMetricsHealth
    {
        public bool IsHealthy { get; set; }
        public string Status { get; set; } = "Unknown";
        public long MetricsCollected { get; set; }
        public DateTime LastCollection { get; set; }
        public TimeSpan Uptime { get; set; }
        
        public Dictionary<string, object> SystemStatus { get; set; } = new();
        public List<string> Errors { get; set; } = new();
        public List<string> Warnings { get; set; } = new();
        
        // Performance metrics
        public double CollectionLatencyMs { get; set; }
        public double AverageCpuUsage { get; set; }
        public double AverageMemoryUsage { get; set; }
        public int ActiveNetworkInterfaces { get; set; }
        public int MonitoredFilesystems { get; set; }
    }

    /// <summary>
    /// System metrics collection event arguments
    /// </summary>
    public class SystemMetricsCollectedEventArgs : EventArgs
    {
        public LinuxSystemMetricsEntry Metrics { get; set; } = new();
        public DateTime CollectionTime { get; set; } = DateTime.UtcNow;
        public TimeSpan CollectionDuration { get; set; }
        public string CollectorName { get; set; } = "LinuxSystemMetrics";
    }

    /// <summary>
    /// System metrics collection error event arguments
    /// </summary>
    public class SystemMetricsErrorEventArgs : EventArgs
    {
        public Exception Exception { get; set; } = new();
        public string ErrorMessage { get; set; } = "";
        public string MetricType { get; set; } = "";
        public DateTime ErrorTime { get; set; } = DateTime.UtcNow;
        public string Source { get; set; } = "LinuxSystemMetrics";
    }
}
