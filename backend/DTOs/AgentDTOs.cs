using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using Backend.Models;

namespace Backend.DTOs
{
    /// <summary>
    /// DTO for agent registration requests
    /// </summary>
    public class AgentRegistrationDto
    {
        [Required]
        [MaxLength(255)]
        public string Hostname { get; set; } = string.Empty;
        
        [Required]
        [MaxLength(45)]
        public string IPAddress { get; set; } = string.Empty;
        
        [Required]
        [MaxLength(255)]
        public string OperatingSystem { get; set; } = string.Empty;
        
        [MaxLength(50)]
        public string Version { get; set; } = string.Empty;
        
        // Registration key can be provided in the body (optional as it can also be in headers)
        public string? RegistrationKey { get; set; }
        
        // Token-based registration
        public string? DeploymentToken { get; set; }
    }

    /// <summary>
    /// Result of an agent registration operation
    /// </summary>
    public class AgentRegistrationResultDto
    {
        public bool Success { get; set; }
        public string AgentId { get; set; } = string.Empty;
        public string ApiKey { get; set; } = string.Empty;
        public string ErrorMessage { get; set; } = string.Empty;
    }

    /// <summary>
    /// DTO for agent data
    /// </summary>
    public class AgentDto
    {
        /// <summary>
        /// Gets or sets the agent ID
        /// </summary>
        public string Id { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the agent name
        /// </summary>
        public string Name { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the agent hostname
        /// </summary>
        public string Hostname { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the agent IP address
        /// </summary>
        public string IpAddress { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the agent IP address (alias for IpAddress)
        /// </summary>
        public string IPAddress { get => IpAddress; set => IpAddress = value; }
        
        /// <summary>
        /// Gets or sets the agent version
        /// </summary>
        public string Version { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the agent status
        /// </summary>
        public string Status { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the agent type
        /// </summary>
        public string Type { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the last connected time
        /// </summary>
        public DateTime? LastConnected { get; set; }
        
        /// <summary>
        /// Gets or sets the installation date
        /// </summary>
        public DateTime InstallDate { get; set; }
        
        /// <summary>
        /// Gets or sets whether the agent is enabled
        /// </summary>
        public bool IsEnabled { get; set; }
        
        /// <summary>
        /// Gets or sets the operating system
        /// </summary>
        public string OperatingSystem { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the CPU usage
        /// </summary>
        public double? CpuUsage { get; set; }
        
        /// <summary>
        /// Gets or sets the memory usage
        /// </summary>
        public double? MemoryUsage { get; set; }
        
        /// <summary>
        /// Gets or sets the disk usage
        /// </summary>
        public double? DiskUsage { get; set; }
        
        /// <summary>
        /// Gets or sets whether to collect event logs
        /// </summary>
        public bool CollectEventLogs { get; set; }
        
        /// <summary>
        /// Gets or sets whether to collect system metrics
        /// </summary>
        public bool CollectSystemMetrics { get; set; }
        
        /// <summary>
        /// Gets or sets the event logs to monitor
        /// </summary>
        public List<string> EventLogsToMonitor { get; set; } = new List<string>();
        
        /// <summary>
        /// Gets or sets the agent configuration
        /// </summary>
        public Dictionary<string, string> Configuration { get; set; } = new Dictionary<string, string>();
        
        /// <summary>
        /// Gets or sets the last health status
        /// </summary>
        public string HealthStatus { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the agent tags
        /// </summary>
        public List<string> Tags { get; set; } = new List<string>();
    }

    /// <summary>
    /// DTO for log ingestion
    /// </summary>
    public class LogIngestDto
    {
        [Required]
        public DateTime Timestamp { get; set; } = DateTime.UtcNow;
        
        [Required]
        public string LogSource { get; set; } = string.Empty;
        
        [Required]
        public SeverityModels Severity { get; set; } = SeverityModels.Low;
        
        [Required]
        public string RawLog { get; set; } = string.Empty;
    }
    
    /// <summary>
    /// DTO for system metrics data
    /// </summary>
    public class SystemMetricsDto
    {
        /// <summary>
        /// Gets or sets the agent ID
        /// </summary>
        public string AgentId { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the timestamp
        /// </summary>
        public DateTime Timestamp { get; set; }
        
        /// <summary>
        /// Gets or sets the CPU metrics
        /// </summary>
        public CpuMetricsDto Cpu { get; set; } = new();
        
        /// <summary>
        /// Gets or sets the memory metrics
        /// </summary>
        public MemoryMetricsDto Memory { get; set; } = new();
        
        /// <summary>
        /// Gets or sets the disk metrics
        /// </summary>
        public DiskMetricsDto Disk { get; set; } = new();
        
        /// <summary>
        /// Gets or sets the network metrics
        /// </summary>
        public NetworkMetricsDto Network { get; set; } = new();
        
        /// <summary>
        /// Gets or sets the process metrics
        /// </summary>
        public ProcessMetricsDto Process { get; set; } = new();
    }

    /// <summary>
    /// CPU metrics data transfer object
    /// </summary>
    public class CpuMetricsDto
    {
        /// <summary>
        /// Gets or sets the CPU usage
        /// </summary>
        public double Usage { get; set; }
        
        /// <summary>
        /// Gets or sets the number of cores
        /// </summary>
        public int Cores { get; set; }
        
        /// <summary>
        /// Gets or sets the system usage
        /// </summary>
        public double SystemUsage { get; set; }
        
        /// <summary>
        /// Gets or sets the user usage
        /// </summary>
        public double UserUsage { get; set; }
        
        /// <summary>
        /// Gets or sets the idle usage
        /// </summary>
        public double IdleUsage { get; set; }
    }

    /// <summary>
    /// Memory metrics data transfer object
    /// </summary>
    public class MemoryMetricsDto
    {
        /// <summary>
        /// Gets or sets the total memory
        /// </summary>
        public long TotalBytes { get; set; }
        
        /// <summary>
        /// Gets or sets the used memory
        /// </summary>
        public long UsedBytes { get; set; }
        
        /// <summary>
        /// Gets or sets the free memory
        /// </summary>
        public long FreeBytes { get; set; }
        
        /// <summary>
        /// Gets or sets the used percentage
        /// </summary>
        public double UsedPercentage { get; set; }
    }

    /// <summary>
    /// Disk metrics data transfer object
    /// </summary>
    public class DiskMetricsDto
    {
        /// <summary>
        /// Gets or sets the total disk space
        /// </summary>
        public long TotalBytes { get; set; }
        
        /// <summary>
        /// Gets or sets the used disk space
        /// </summary>
        public long UsedBytes { get; set; }
        
        /// <summary>
        /// Gets or sets the free disk space
        /// </summary>
        public long FreeBytes { get; set; }
        
        /// <summary>
        /// Gets or sets the used percentage
        /// </summary>
        public double UsedPercentage { get; set; }
        
        /// <summary>
        /// Gets or sets the disk read rate
        /// </summary>
        public double ReadBytesPerSecond { get; set; }
        
        /// <summary>
        /// Gets or sets the disk write rate
        /// </summary>
        public double WriteBytesPerSecond { get; set; }
    }

    /// <summary>
    /// Network metrics data transfer object
    /// </summary>
    public class NetworkMetricsDto
    {
        /// <summary>
        /// Gets or sets the received bytes
        /// </summary>
        public long ReceivedBytes { get; set; }
        
        /// <summary>
        /// Gets or sets the sent bytes
        /// </summary>
        public long SentBytes { get; set; }
        
        /// <summary>
        /// Gets or sets the receive rate
        /// </summary>
        public double ReceiveBytesPerSecond { get; set; }
        
        /// <summary>
        /// Gets or sets the send rate
        /// </summary>
        public double SendBytesPerSecond { get; set; }
        
        /// <summary>
        /// Gets or sets the number of connections
        /// </summary>
        public int ConnectionCount { get; set; }
    }

    /// <summary>
    /// Process metrics data transfer object
    /// </summary>
    public class ProcessMetricsDto
    {
        /// <summary>
        /// Gets or sets the CPU usage
        /// </summary>
        public double CpuUsage { get; set; }
        
        /// <summary>
        /// Gets or sets the memory usage
        /// </summary>
        public long MemoryUsage { get; set; }
        
        /// <summary>
        /// Gets or sets the thread count
        /// </summary>
        public int ThreadCount { get; set; }
        
        /// <summary>
        /// Gets or sets the handle count
        /// </summary>
        public int HandleCount { get; set; }
        
        /// <summary>
        /// Gets or sets the uptime
        /// </summary>
        public TimeSpan Uptime { get; set; }
    }

    /// <summary>
    /// DTO for updating agent status
    /// </summary>
    public class UpdateStatusDto
    {
        [Required]
        public string Status { get; set; } = string.Empty;
        
        public string Reason { get; set; } = string.Empty;
    }
    
    /// <summary>
    /// DTO for validating API key
    /// </summary>
    public class ValidateApiKeyDto
    {
        [Required]
        public string ApiKey { get; set; } = string.Empty;
    }

    /// <summary>
    /// DTO for agent pre-configuration for deployment
    /// </summary>
    public class AgentPreConfigDto
    {
        /// <summary>
        /// Gets or sets the server URL for the agent to connect to
        /// </summary>
        public string IpAddress { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the server port
        /// </summary>
        public int Port { get; set; } = 443;
        
        /// <summary>
        /// Gets or sets the agent name
        /// </summary>
        public string Name { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets whether to use SSL for communication
        /// </summary>
        public bool UseSSL { get; set; } = true;
        
        /// <summary>
        /// Gets or sets the collectors to enable
        /// </summary>
        public List<string> Collectors { get; set; } = new List<string>();
    }
    
    /// <summary>
    /// DTO for generating a deployment token
    /// </summary>
    public class GenerateTokenRequestDto
    {
        /// <summary>
        /// Gets or sets the installer type
        /// </summary>
        [Required]
        public string InstallerType { get; set; } = "windows";
        
        /// <summary>
        /// Gets or sets the agent pre-configuration
        /// </summary>
        public AgentPreConfigDto? Configuration { get; set; }
    }
    
    /// <summary>
    /// DTO for deployment token response
    /// </summary>
    public class AgentTokenDto
    {
        /// <summary>
        /// Gets or sets the token
        /// </summary>
        public string Token { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the token expiration time
        /// </summary>
        public DateTime ExpiresAt { get; set; }
        
        /// <summary>
        /// Gets or sets the download URL
        /// </summary>
        public string DownloadUrl { get; set; } = string.Empty;
    }
    
    /// <summary>
    /// DTO for agent collectors configuration
    /// </summary>
    public class CollectorsConfigDto
    {
        /// <summary>
        /// Gets or sets whether to collect event logs
        /// </summary>
        public bool CollectEventLogs { get; set; } = true;
        
        /// <summary>
        /// Gets or sets whether to collect system metrics
        /// </summary>
        public bool CollectSystemMetrics { get; set; } = true;
        
        /// <summary>
        /// Gets or sets whether to enable file integrity monitoring
        /// </summary>
        public bool EnableFileIntegrityMonitoring { get; set; } = false;
        
        /// <summary>
        /// Gets or sets whether to enable network monitoring
        /// </summary>
        public bool EnableNetworkMonitoring { get; set; } = false;
    }
} 