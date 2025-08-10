using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;

namespace Backend.DTOs
{
    /// <summary>
    /// Linux Agent Data Transfer Objects for AthalaSIEM Backend
    /// Author: Revian Ravil Athala
    /// Enterprise SIEM Linux agent communication DTOs
    /// </summary>

    /// <summary>
    /// Linux system metrics DTO for backend communication
    /// </summary>
    public class LinuxSystemMetricsDto
    {
        [Required]
        public string AgentId { get; set; } = "";

        [Required]
        public DateTime Timestamp { get; set; }

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
        public Dictionary<string, DiskMetricsDto> DiskUsage { get; set; } = new();
        public Dictionary<string, DiskIOMetricsDto> DiskIOStats { get; set; } = new();

        // Network Metrics
        public Dictionary<string, NetworkMetricsDto> NetworkStats { get; set; } = new();

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

        // Collection metadata
        public double CollectionDurationMs { get; set; }
        public string CollectorVersion { get; set; } = "1.0.0";
    }

    /// <summary>
    /// Disk usage metrics DTO
    /// </summary>
    public class DiskMetricsDto
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
    /// Disk I/O statistics DTO
    /// </summary>
    public class DiskIOMetricsDto
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
    /// Network interface statistics DTO
    /// </summary>
    public class NetworkMetricsDto
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
    /// Linux FIM (File Integrity Monitoring) event DTO
    /// </summary>
    public class LinuxFIMEventDto
    {
        [Required]
        public string AgentId { get; set; } = "";

        [Required]
        public DateTime Timestamp { get; set; }

        [Required]
        public string FilePath { get; set; } = "";

        [Required]
        public string EventType { get; set; } = ""; // CREATE, MODIFY, DELETE, MOVE, ATTRIB

        public string? OldFilePath { get; set; }
        public LinuxFileInfoDto? OldFileInfo { get; set; }
        public LinuxFileInfoDto? NewFileInfo { get; set; }

        public string User { get; set; } = "";
        public string Process { get; set; } = "";
        public int? ProcessId { get; set; }

        public string SecurityLevel { get; set; } = "Medium";
        public List<string> ThreatIndicators { get; set; } = new();
        public Dictionary<string, object> Metadata { get; set; } = new();
    }

    /// <summary>
    /// Linux file information DTO
    /// </summary>
    public class LinuxFileInfoDto
    {
        public long Size { get; set; }
        public DateTime CreatedTime { get; set; }
        public DateTime ModifiedTime { get; set; }
        public DateTime AccessedTime { get; set; }
        public string Permissions { get; set; } = "";
        public string Owner { get; set; } = "";
        public string Group { get; set; } = "";
        public string Attributes { get; set; } = "";
        public Dictionary<string, string> Hashes { get; set; } = new();
        public string MimeType { get; set; } = "";
        public bool IsSymlink { get; set; }
        public string? SymlinkTarget { get; set; }
        public Dictionary<string, string> ExtendedAttributes { get; set; } = new();
    }

    /// <summary>
    /// Linux syslog configuration DTO
    /// </summary>
    public class LinuxSyslogConfigDto
    {
        public bool EnableSystemdJournal { get; set; } = true;
        public List<string> CustomLogPaths { get; set; } = new();
        public List<string> SupportedFormats { get; set; } = new() { "RFC3164", "RFC5424", "CEF", "JSON" };
        public bool ParseStructuredLogs { get; set; } = true;
        public Dictionary<string, string> LogSourceMappings { get; set; } = new();
        public int MaxLogLineLength { get; set; } = 8192;
        public string DefaultLogLevel { get; set; } = "Information";
    }

    /// <summary>
    /// Linux agent configuration DTO
    /// </summary>
    public class LinuxAgentConfigDto
    {
        // System Metrics Configuration
        public bool EnableSystemMetrics { get; set; } = true;
        public int SystemMetricsIntervalSeconds { get; set; } = 30;
        public bool EnableCPUMonitoring { get; set; } = true;
        public bool EnableMemoryMonitoring { get; set; } = true;
        public bool EnableDiskMonitoring { get; set; } = true;
        public bool EnableNetworkMonitoring { get; set; } = true;
        public List<string> MonitoredNetworkInterfaces { get; set; } = new();
        public List<string> MonitoredFilesystems { get; set; } = new();

        // Syslog Configuration
        public LinuxSyslogConfigDto SyslogConfig { get; set; } = new();

        // FIM Configuration
        public bool EnableFIM { get; set; } = true;
        public List<string> FIMMonitoredPaths { get; set; } = new();
        public List<string> FIMExcludedPaths { get; set; } = new();
        public bool FIMRealTimeMonitoring { get; set; } = true;
        public string FIMHashAlgorithm { get; set; } = "SHA256";
        public int FIMScanIntervalMinutes { get; set; } = 60;

        // Communication Configuration
        public int HeartbeatIntervalSeconds { get; set; } = 60;
        public int LogBatchSize { get; set; } = 100;
        public int LogBatchIntervalSeconds { get; set; } = 30;
        public bool EnableCompression { get; set; } = true;
        public int MaxRetryAttempts { get; set; } = 3;

        // Security Configuration
        public bool EnableThreatDetection { get; set; } = true;
        public bool EnableBehaviorAnalysis { get; set; } = false;
        public string SecurityLogLevel { get; set; } = "Information";
    }

    /// <summary>
    /// Linux agent health status DTO
    /// </summary>
    public class LinuxAgentHealthDto
    {
        [Required]
        public string AgentId { get; set; } = "";

        [Required]
        public DateTime Timestamp { get; set; }

        public string Status { get; set; } = "Unknown"; // Running, Stopped, Error, Degraded
        public bool IsHealthy { get; set; }
        public TimeSpan Uptime { get; set; }

        // Resource Usage
        public double CpuUsage { get; set; }
        public double MemoryUsage { get; set; }
        public double DiskUsage { get; set; }

        // Collector Status
        public Dictionary<string, CollectorHealthDto> CollectorHealth { get; set; } = new();

        // Performance Metrics
        public long TotalLogsCollected { get; set; }
        public long TotalLogsForwarded { get; set; }
        public long TotalErrors { get; set; }
        public double AverageLogProcessingTime { get; set; }

        // System Information
        public string KernelVersion { get; set; } = "";
        public string Distribution { get; set; } = "";
        public string AgentVersion { get; set; } = "";
        
        public List<string> Errors { get; set; } = new();
        public List<string> Warnings { get; set; } = new();
        public Dictionary<string, object> AdditionalMetrics { get; set; } = new();
    }

    /// <summary>
    /// Collector health status DTO
    /// </summary>
    public class CollectorHealthDto
    {
        public string Name { get; set; } = "";
        public bool IsHealthy { get; set; }
        public string Status { get; set; } = "Unknown";
        public long LogsCollected { get; set; }
        public DateTime LastCollection { get; set; }
        public TimeSpan Uptime { get; set; }
        public List<string> Errors { get; set; } = new();
        public Dictionary<string, object> Metrics { get; set; } = new();
    }

    /// <summary>
    /// Linux deployment configuration DTO
    /// </summary>
    public class LinuxDeploymentConfigDto
    {
        [Required]
        public string ManagerUrl { get; set; } = "";

        public string RegistrationServer { get; set; } = "";
        public string RegistrationPassword { get; set; } = "";
        public string AgentName { get; set; } = "Revian Ravil Athala";
        public string AgentGroup { get; set; } = "";
        public string Protocol { get; set; } = "tcp";
        public int KeepAliveInterval { get; set; } = 60;
        public int TimeReconnect { get; set; } = 60;
        
        // SSL/TLS Configuration
        public string? RegistrationCA { get; set; }
        public string? RegistrationCertificate { get; set; }
        public string? RegistrationKey { get; set; }
        
        // Advanced Configuration
        public Dictionary<string, object> CustomProperties { get; set; } = new();
        public DateTime ConfigurationTime { get; set; } = DateTime.UtcNow;
        public string ConfigurationVersion { get; set; } = "1.0.0";
    }

    /// <summary>
    /// Linux agent registration response DTO
    /// </summary>
    public class LinuxAgentRegistrationResponseDto
    {
        public bool Success { get; set; }
        public string Message { get; set; } = "";
        public string AgentId { get; set; } = "";
        public string ApiKey { get; set; } = "";
        public LinuxAgentConfigDto Configuration { get; set; } = new();
        public DateTime ExpirationTime { get; set; }
        public Dictionary<string, object> AdditionalData { get; set; } = new();
    }

    /// <summary>
    /// Batch metrics submission DTO for performance
    /// </summary>
    public class LinuxMetricsBatchDto
    {
        [Required]
        public string AgentId { get; set; } = "";

        [Required]
        public List<LinuxSystemMetricsDto> SystemMetrics { get; set; } = new();

        public List<LinuxFIMEventDto> FIMEvents { get; set; } = new();
        
        [Required]
        public DateTime BatchTimestamp { get; set; }

        public int BatchSize { get; set; }
        public string CompressionType { get; set; } = "none";
        public string BatchId { get; set; } = Guid.NewGuid().ToString();
    }
}
