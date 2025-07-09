using System;
using System.Collections.Generic;

namespace Backend.DTOs
{
    /// <summary>
    /// Enum for time intervals
    /// </summary>
    public enum TimeInterval
    {
        Minute,
        Hour,
        Day,
        Week,
        Month
    }

    /// <summary>
    /// DTO for dashboard overview data
    /// </summary>
    public class DashboardOverviewDto
    {
        public AgentStatisticsDto AgentStatistics { get; set; } = new();
        public LogStatisticsDto LogStatistics { get; set; } = new();
        public SystemHealthDto SystemHealth { get; set; } = new();
        public List<LogEntryDto> RecentCriticalLogs { get; set; } = new();
        public DateTime LastUpdated { get; set; }
    }

    /// <summary>
    /// DTO for agent statistics
    /// </summary>
    public class AgentStatisticsDto
    {
        public int TotalAgents { get; set; }
        public int OnlineAgents { get; set; }
        public int OfflineAgents { get; set; }
        public int WarningAgents { get; set; }
        public DateTime LastUpdated { get; set; }
        public double OnlinePercentage => TotalAgents > 0 ? (double)OnlineAgents / TotalAgents * 100 : 0;
    }

    /// <summary>
    /// DTO for log statistics
    /// </summary>
    public class LogStatisticsDto
    {
        public int TotalLogs { get; set; }
        public int CriticalLogs { get; set; }
        public int ErrorLogs { get; set; }
        public int WarningLogs { get; set; }
        public int InfoLogs { get; set; }
        public double LogsPerHour { get; set; }
        public TimeRangeDto TimeRange { get; set; } = new();
    }

    /// <summary>
    /// DTO for system health status
    /// </summary>
    public class SystemHealthDto
    {
        public string OverallStatus { get; set; } = string.Empty;
        public string AgentConnectivity { get; set; } = string.Empty;
        public string LogIngestionRate { get; set; } = string.Empty;
        public string ErrorRate { get; set; } = string.Empty;
        public DateTime LastUpdated { get; set; }
    }

    /// <summary>
    /// DTO for time range
    /// </summary>
    public class TimeRangeDto
    {
        public DateTime StartTime { get; set; }
        public DateTime EndTime { get; set; }
        public TimeSpan Duration => EndTime - StartTime;
    }

    /// <summary>
    /// DTO for real-time metrics
    /// </summary>
    public class RealTimeMetricsDto
    {
        public double LogIngestionRate { get; set; }
        public double ErrorRate { get; set; }
        public int OnlineAgents { get; set; }
        public int TotalAgents { get; set; }
        public DateTime Timestamp { get; set; }
        public int LogsLast5Minutes { get; set; }
        public int ErrorsLast5Minutes { get; set; }
    }

    /// <summary>
    /// DTO for agent health summary
    /// </summary>
    public class AgentHealthSummaryDto
    {
        public int TotalAgents { get; set; }
        public int HealthyAgents { get; set; }
        public int UnhealthyAgents { get; set; }
        public int OfflineAgents { get; set; }
        public Dictionary<string, int> AgentsByType { get; set; } = new();
        public Dictionary<string, int> AgentsByOperatingSystem { get; set; } = new();
        public ResourceUsageDto AverageResourceUsage { get; set; } = new();
        public DateTime LastUpdated { get; set; }
    }

    /// <summary>
    /// DTO for resource usage
    /// </summary>
    public class ResourceUsageDto
    {
        public double CpuUsage { get; set; }
        public double MemoryUsage { get; set; }
        public double DiskUsage { get; set; }
    }

    /// <summary>
    /// DTO for log source statistics
    /// </summary>
    public class LogSourceStatDto
    {
        public string Source { get; set; } = string.Empty;
        public int LogCount { get; set; }
        public double Percentage { get; set; }
    }

    /// <summary>
    /// DTO for security events summary
    /// </summary>
    public class SecurityEventsSummaryDto
    {
        public int TotalSecurityEvents { get; set; }
        public int FailedLoginAttempts { get; set; }
        public int SuspiciousActivities { get; set; }
        public TimeRangeDto TimeRange { get; set; } = new();
        public DateTime LastUpdated { get; set; }
    }

    /// <summary>
    /// DTO for system performance metrics
    /// </summary>
    public class SystemPerformanceDto
    {
        public double LogProcessingRate { get; set; }
        public double AverageResponseTime { get; set; }
        public TimeSpan SystemUptime { get; set; }
        public double MemoryUsage { get; set; }
        public double CpuUsage { get; set; }
        public double DiskUsage { get; set; }
        public int ActiveConnections { get; set; }
        public DateTime LastUpdated { get; set; }
    }
} 