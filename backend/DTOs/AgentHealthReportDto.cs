using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using Backend.Models;

namespace Backend.DTOs
{
    /// <summary>
    /// Data transfer object for agent health reports
    /// </summary>
    public class AgentHealthReportDto
    {
        [Required]
        public required string Id { get; set; }

        [Required]
        public required string AgentId { get; set; }

        /// <summary>
        /// Gets or sets the timestamp
        /// </summary>
        public DateTime Timestamp { get; set; } = DateTime.UtcNow;
        
        /// <summary>
        /// Gets or sets the overall status
        /// </summary>
        public string OverallStatus { get; set; } = "Healthy";
        
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
        /// Gets or sets the metrics
        /// </summary>
        public Dictionary<string, MetricDto> Metrics { get; set; } = new Dictionary<string, MetricDto>();

        /// <summary>
        /// Gets or sets the creation timestamp
        /// </summary>
        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    }
    
    /// <summary>
    /// Data transfer object for a metric
    /// </summary>
    public class MetricDto
    {
        /// <summary>
        /// Gets or sets the metric name
        /// </summary>
        public string Name { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the metric value
        /// </summary>
        public double Value { get; set; }
        
        /// <summary>
        /// Gets or sets the metric unit
        /// </summary>
        public string Unit { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the metric status
        /// </summary>
        public string Status { get; set; } = "Healthy";
    }

    public class HealthReportDTO
    {
        /// <summary>
        /// Gets or sets the timestamp of the health report
        /// </summary>
        public DateTime Timestamp { get; set; } = DateTime.UtcNow;

        /// <summary>
        /// Gets or sets the CPU usage percentage
        /// </summary>
        public double CpuUsagePercent { get; set; }

        /// <summary>
        /// Gets or sets the memory usage percentage
        /// </summary>
        public double MemoryUsagePercent { get; set; }

        /// <summary>
        /// Gets or sets the disk usage percentage
        /// </summary>
        public double DiskUsagePercent { get; set; }

        /// <summary>
        /// Gets or sets the available disk space in bytes
        /// </summary>
        public long AvailableDiskSpaceBytes { get; set; }

        /// <summary>
        /// Gets or sets the total disk space in bytes
        /// </summary>
        public long TotalDiskSpaceBytes { get; set; }

        /// <summary>
        /// Gets or sets the available memory in bytes
        /// </summary>
        public long AvailableMemoryBytes { get; set; }

        /// <summary>
        /// Gets or sets the total memory in bytes
        /// </summary>
        public long TotalMemoryBytes { get; set; }

        /// <summary>
        /// Gets or sets the system uptime in seconds
        /// </summary>
        public long SystemUptimeSeconds { get; set; }

        /// <summary>
        /// Gets or sets the agent uptime in seconds
        /// </summary>
        public long AgentUptimeSeconds { get; set; }

        /// <summary>
        /// Gets or sets the number of active processes
        /// </summary>
        public int ActiveProcesses { get; set; }

        /// <summary>
        /// Gets or sets the network usage in bytes per second
        /// </summary>
        public long NetworkUsageBytesPerSecond { get; set; }

        /// <summary>
        /// Gets or sets the process details if enabled in configuration
        /// </summary>
        public List<ProcessInfoDTO> TopProcesses { get; set; } = new List<ProcessInfoDTO>();

        /// <summary>
        /// Gets or sets any alerts generated from this health report
        /// </summary>
        public List<string> Alerts { get; set; } = new List<string>();

        /// <summary>
        /// Gets or sets additional health information
        /// </summary>
        public Dictionary<string, string> AdditionalMetrics { get; set; } = new Dictionary<string, string>();
    }

    /// <summary>
    /// Data transfer object for process information
    /// </summary>
    public class ProcessInfoDTO
    {
        /// <summary>
        /// Gets or sets the process ID
        /// </summary>
        public int ProcessId { get; set; }

        /// <summary>
        /// Gets or sets the process name
        /// </summary>
        public string ProcessName { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the CPU usage percentage of the process
        /// </summary>
        public double CpuUsagePercent { get; set; }

        /// <summary>
        /// Gets or sets the memory usage in bytes of the process
        /// </summary>
        public long MemoryUsageBytes { get; set; }

        /// <summary>
        /// Gets or sets the start time of the process
        /// </summary>
        public DateTime StartTime { get; set; }

        /// <summary>
        /// Gets or sets the user running the process
        /// </summary>
        public string Username { get; set; } = string.Empty;
    }
} 