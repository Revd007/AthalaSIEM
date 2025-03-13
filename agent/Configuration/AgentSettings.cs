using System;
using System.Collections.Generic;
using System.IO;

namespace Agent.Configuration
{
    /// <summary>
    /// Agent settings class to hold configuration
    /// </summary>
    public class AgentSettings
    {
        /// <summary>
        /// Gets or sets the agent ID (GUID)
        /// </summary>
        public string? AgentId { get; set; }
        
        /// <summary>
        /// Gets or sets the agent name
        /// </summary>
        public string AgentName { get; set; } = Environment.MachineName;
        
        /// <summary>
        /// Gets or sets the backend URL
        /// </summary>
        public string BackendUrl { get; set; } = "https://localhost:5135";
        
        /// <summary>
        /// Gets or sets the API key for authentication
        /// </summary>
        public string? ApiKey { get; set; }
        
        /// <summary>
        /// Gets or sets the agent type (Windows, Linux, etc.)
        /// </summary>
        public string AgentType { get; set; } = GetDefaultAgentType();
        
        /// <summary>
        /// Gets or sets the heartbeat interval in minutes
        /// </summary>
        public int HeartbeatIntervalMinutes { get; set; } = 5;
        
        /// <summary>
        /// Gets or sets the configuration refresh interval in minutes
        /// </summary>
        public int ConfigRefreshIntervalMinutes { get; set; } = 15;
        
        /// <summary>
        /// Gets or sets the log batch size (number of logs to batch before sending)
        /// </summary>
        public int LogBatchSize { get; set; } = 100;
        
        /// <summary>
        /// Gets or sets the maximum log batch interval in seconds
        /// </summary>
        public int MaxLogBatchIntervalSeconds { get; set; } = 60;
        
        /// <summary>
        /// Gets or sets whether to collect system metrics
        /// </summary>
        public bool CollectSystemMetrics { get; set; } = true;
        
        /// <summary>
        /// Gets or sets the system metrics collection interval in minutes
        /// </summary>
        public int SystemMetricsIntervalMinutes { get; set; } = 5;
        
        /// <summary>
        /// Gets or sets whether to use compression for log data
        /// </summary>
        public bool UseCompression { get; set; } = true;
        
        /// <summary>
        /// Gets or sets whether to encrypt logs
        /// </summary>
        public bool EncryptLogs { get; set; } = true;
        
        /// <summary>
        /// Gets or sets the log collectors configuration
        /// </summary>
        public List<CollectorSettings> Collectors { get; set; } = new List<CollectorSettings>();
        
        /// <summary>
        /// Gets or sets the local log buffer path
        /// </summary>
        public string LogBufferPath { get; set; } = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData),
            "AthalaSIEM", "LogBuffer");
        
        /// <summary>
        /// Gets or sets the proxy settings
        /// </summary>
        public ProxySettings? Proxy { get; set; }
        
        /// <summary>
        /// Updates settings from a remote configuration
        /// </summary>
        public void UpdateFrom(AgentSettings remoteConfig)
        {
            if (remoteConfig == null)
                return;

            // Only update non-identity properties
            HeartbeatIntervalMinutes = remoteConfig.HeartbeatIntervalMinutes;
            ConfigRefreshIntervalMinutes = remoteConfig.ConfigRefreshIntervalMinutes;
            LogBatchSize = remoteConfig.LogBatchSize;
            MaxLogBatchIntervalSeconds = remoteConfig.MaxLogBatchIntervalSeconds;
            CollectSystemMetrics = remoteConfig.CollectSystemMetrics;
            SystemMetricsIntervalMinutes = remoteConfig.SystemMetricsIntervalMinutes;
            UseCompression = remoteConfig.UseCompression;
            EncryptLogs = remoteConfig.EncryptLogs;
            
            // Update collectors if provided
            if (remoteConfig.Collectors != null && remoteConfig.Collectors.Count > 0)
                Collectors = remoteConfig.Collectors;
            
            // Update proxy settings if provided
            if (remoteConfig.Proxy != null)
                Proxy = remoteConfig.Proxy;
        }
        
        private static string GetDefaultAgentType()
        {
            if (OperatingSystem.IsWindows())
                return "Windows";
            else if (OperatingSystem.IsLinux())
                return "Linux";
            else if (OperatingSystem.IsMacOS())
                return "MacOS";
            else
                return "Unknown";
        }
    }
    
    /// <summary>
    /// Configuration for a specific log collector
    /// </summary>
    public class CollectorSettings
    {
        /// <summary>
        /// Gets or sets the collector type
        /// </summary>
        public string Type { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets whether this collector is enabled
        /// </summary>
        public bool Enabled { get; set; } = true;
        
        /// <summary>
        /// Gets or sets the collection interval in seconds (if applicable)
        /// </summary>
        public int IntervalSeconds { get; set; } = 60;
        
        /// <summary>
        /// Gets or sets collector-specific settings as a dictionary
        /// </summary>
        public Dictionary<string, string> Settings { get; set; } = new Dictionary<string, string>();
    }
    
    /// <summary>
    /// Proxy settings for the agent
    /// </summary>
    public class ProxySettings
    {
        /// <summary>
        /// Gets or sets whether to use a proxy
        /// </summary>
        public bool Enabled { get; set; } = false;
        
        /// <summary>
        /// Gets or sets the proxy address
        /// </summary>
        public string Address { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the proxy port
        /// </summary>
        public int Port { get; set; } = 8080;
        
        /// <summary>
        /// Gets or sets the proxy username
        /// </summary>
        public string? Username { get; set; }
        
        /// <summary>
        /// Gets or sets the proxy password
        /// </summary>
        public string? Password { get; set; }
    }
} 