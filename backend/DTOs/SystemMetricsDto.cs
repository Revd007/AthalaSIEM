// This file is being kept as a reference but the class is defined elsewhere
// to avoid duplicate definitions.

/*
using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;

namespace Backend.DTOs
{
    /// <summary>
    /// Data transfer object for system metrics
    /// </summary>
    public class SystemMetricsDto
    {
        /// <summary>
        /// Gets or sets the timestamp
        /// </summary>
        public DateTime Timestamp { get; set; } = DateTime.UtcNow;
        
        /// <summary>
        /// Gets or sets the CPU usage percentage
        /// </summary>
        [Range(0, 100)]
        public double CpuUsage { get; set; }
        
        /// <summary>
        /// Gets or sets the memory usage percentage
        /// </summary>
        [Range(0, 100)]
        public double MemoryUsage { get; set; }
        
        /// <summary>
        /// Gets or sets the disk usage percentage
        /// </summary>
        [Range(0, 100)]
        public double DiskUsage { get; set; }
        
        /// <summary>
        /// Gets or sets the network throughput in Mbps
        /// </summary>
        public double NetworkThroughput { get; set; }
        
        /// <summary>
        /// Gets or sets the process count
        /// </summary>
        public int ProcessCount { get; set; }
        
        /// <summary>
        /// Gets or sets the system uptime in seconds
        /// </summary>
        public long SystemUptime { get; set; }
        
        /// <summary>
        /// Gets or sets the top processes by CPU usage
        /// </summary>
        public List<ProcessInfoDto> TopProcessesByCpu { get; set; } = new List<ProcessInfoDto>();
        
        /// <summary>
        /// Gets or sets the top processes by memory usage
        /// </summary>
        public List<ProcessInfoDto> TopProcessesByMemory { get; set; } = new List<ProcessInfoDto>();
        
        /// <summary>
        /// Gets or sets the disk information
        /// </summary>
        public List<DiskInfoDto> Disks { get; set; } = new List<DiskInfoDto>();
        
        /// <summary>
        /// Gets or sets the network information
        /// </summary>
        public List<NetworkInfoDto> Networks { get; set; } = new List<NetworkInfoDto>();
    }
    
    /// <summary>
    /// Data transfer object for process information
    /// </summary>
    public class ProcessInfoDto
    {
        /// <summary>
        /// Gets or sets the process ID
        /// </summary>
        public int ProcessId { get; set; }
        
        /// <summary>
        /// Gets or sets the process name
        /// </summary>
        public string Name { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the CPU usage percentage
        /// </summary>
        public double CpuUsage { get; set; }
        
        /// <summary>
        /// Gets or sets the memory usage in MB
        /// </summary>
        public double MemoryUsageMB { get; set; }
        
        /// <summary>
        /// Gets or sets the start time
        /// </summary>
        public DateTime? StartTime { get; set; }
    }
    
    /// <summary>
    /// Data transfer object for disk information
    /// </summary>
    public class DiskInfoDto
    {
        /// <summary>
        /// Gets or sets the disk name
        /// </summary>
        public string Name { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the total size in GB
        /// </summary>
        public double TotalSizeGB { get; set; }
        
        /// <summary>
        /// Gets or sets the free space in GB
        /// </summary>
        public double FreeSpaceGB { get; set; }
        
        /// <summary>
        /// Gets or sets the usage percentage
        /// </summary>
        public double UsagePercent { get; set; }
    }
    
    /// <summary>
    /// Data transfer object for network information
    /// </summary>
    public class NetworkInfoDto
    {
        /// <summary>
        /// Gets or sets the network interface name
        /// </summary>
        public string Name { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the IP address
        /// </summary>
        public string IpAddress { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the bytes sent
        /// </summary>
        public long BytesSent { get; set; }
        
        /// <summary>
        /// Gets or sets the bytes received
        /// </summary>
        public long BytesReceived { get; set; }
        
        /// <summary>
        /// Gets or sets the send speed in Mbps
        /// </summary>
        public double SendSpeedMbps { get; set; }
        
        /// <summary>
        /// Gets or sets the receive speed in Mbps
        /// </summary>
        public double ReceiveSpeedMbps { get; set; }
    }
}
*/ 