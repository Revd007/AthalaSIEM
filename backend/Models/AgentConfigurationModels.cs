using System;
using System.ComponentModel.DataAnnotations;

namespace Backend.Models
{
    public class AgentConfigurationModels
    {
        public string Id { get; set; } = string.Empty;
        public string AgentId { get; set; } = string.Empty;
        public bool Enabled { get; set; } = true;
        public bool CollectEventLogs { get; set; } = true;
        public bool CollectSystemMetrics { get; set; } = true;
        public string EventLogsToMonitor { get; set; } = "Application,System,Security";
        public int LogCollectionIntervalSeconds { get; set; } = 60;
        public int MaxLogBufferCount { get; set; } = 1000;
        public int MaxLogBufferTimeSeconds { get; set; } = 300;
        public bool EnableRealTimeMonitoring { get; set; } = false;
        public bool EnableAlerting { get; set; } = true;
        public int CpuAlertThresholdPercent { get; set; } = 90;
        public int MemoryAlertThresholdPercent { get; set; } = 90;
        public int DiskAlertThresholdPercent { get; set; } = 90;
        public DateTime CreatedAt { get; set; }
        public DateTime UpdatedAt { get; set; }

        // Navigation property
        public AgentModels Agent { get; set; } = null!;
    }
} 