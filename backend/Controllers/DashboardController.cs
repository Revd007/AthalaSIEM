using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using Backend.DTOs;
using Backend.Services;
using Backend.Models;

namespace Backend.Controllers
{
    /// <summary>
    /// Controller for dashboard and monitoring APIs
    /// </summary>
    [ApiController]
    [Route("api/[controller]")]
    [Authorize]
    public class DashboardController : ControllerBase
    {
        private readonly ILogService _logService;
        private readonly ILogAnalysisService _logAnalysisService;
        private readonly IAgentService _agentService;
        private readonly ILogger<DashboardController> _logger;

        /// <summary>
        /// Initializes a new instance of the <see cref="DashboardController"/> class
        /// </summary>
        /// <param name="logService">The log service</param>
        /// <param name="logAnalysisService">The log analysis service</param>
        /// <param name="agentService">The agent service</param>
        /// <param name="logger">The logger</param>
        public DashboardController(
            ILogService logService,
            ILogAnalysisService logAnalysisService,
            IAgentService agentService,
            ILogger<DashboardController> logger)
        {
            _logService = logService ?? throw new ArgumentNullException(nameof(logService));
            _logAnalysisService = logAnalysisService ?? throw new ArgumentNullException(nameof(logAnalysisService));
            _agentService = agentService ?? throw new ArgumentNullException(nameof(agentService));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }

        /// <summary>
        /// Gets the main dashboard overview data
        /// </summary>
        /// <returns>Dashboard overview data</returns>
        [HttpGet("overview")]
        [ProducesResponseType(StatusCodes.Status200OK)]
        public async Task<ActionResult<DashboardOverviewDto>> GetDashboardOverview()
        {
            try
            {
                var endTime = DateTime.UtcNow;
                var startTime = endTime.AddHours(-24); // Last 24 hours

                // Get agent statistics
                var allAgents = await _agentService.GetAllAgentsAsync();
                var agentStats = new AgentStatisticsDto
                {
                    TotalAgents = allAgents.Count(),
                    OnlineAgents = allAgents.Count(a => a.Status == AgentStatus.Online),
                    OfflineAgents = allAgents.Count(a => a.Status == AgentStatus.Offline),
                    WarningAgents = allAgents.Count(a => a.Status == AgentStatus.Warning),
                    LastUpdated = DateTime.UtcNow
                };

                // Get log statistics
                var logSummary = await _logService.GetLogSummaryAsync(startTime, endTime);
                var logStats = new LogStatisticsDto
                {
                    TotalLogs = logSummary.TotalLogs,
                    CriticalLogs = logSummary.CriticalCount,
                    ErrorLogs = logSummary.ErrorCount,
                    WarningLogs = logSummary.WarningCount,
                    InfoLogs = logSummary.InfoCount,
                    LogsPerHour = logSummary.LogsPerHour,
                    TimeRange = new TimeRangeDto
                    {
                        StartTime = startTime,
                        EndTime = endTime
                    }
                };

                // Get recent critical logs
                var recentCriticalLogs = await _logService.GetCriticalLogsAsync(10);

                // Get system health indicators
                var systemHealth = new SystemHealthDto
                {
                    OverallStatus = CalculateOverallSystemHealth(agentStats, logStats),
                    AgentConnectivity = CalculateAgentConnectivityHealth(agentStats),
                    LogIngestionRate = CalculateLogIngestionHealth(logStats),
                    ErrorRate = CalculateErrorRateHealth(logStats),
                    LastUpdated = DateTime.UtcNow
                };

                var overview = new DashboardOverviewDto
                {
                    AgentStatistics = agentStats,
                    LogStatistics = logStats,
                    SystemHealth = systemHealth,
                    RecentCriticalLogs = recentCriticalLogs.Take(5).ToList(),
                    LastUpdated = DateTime.UtcNow
                };

                return Ok(overview);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting dashboard overview");
                return StatusCode(500, new { Error = "Internal server error while getting dashboard overview" });
            }
        }

        /// <summary>
        /// Gets real-time system metrics
        /// </summary>
        /// <returns>Real-time system metrics</returns>
        [HttpGet("metrics/realtime")]
        [ProducesResponseType(StatusCodes.Status200OK)]
        public async Task<ActionResult<RealTimeMetricsDto>> GetRealTimeMetrics()
        {
            try
            {
                var endTime = DateTime.UtcNow;
                var startTime = endTime.AddMinutes(-5); // Last 5 minutes

                // Get recent logs count
                var recentLogsCount = await _logService.GetLogCountBySeverityAsync(startTime, endTime);
                var totalRecentLogs = recentLogsCount.Values.Sum();

                // Get agent status
                var allAgents = await _agentService.GetAllAgentsAsync();
                var onlineAgents = allAgents.Count(a => a.Status == AgentStatus.Online);

                // Calculate rates
                var logIngestionRate = totalRecentLogs / 5.0; // logs per minute
                var errorRate = (recentLogsCount.GetValueOrDefault("Error", 0) + 
                               recentLogsCount.GetValueOrDefault("Critical", 0)) / 5.0;

                var metrics = new RealTimeMetricsDto
                {
                    LogIngestionRate = logIngestionRate,
                    ErrorRate = errorRate,
                    OnlineAgents = onlineAgents,
                    TotalAgents = allAgents.Count(),
                    Timestamp = DateTime.UtcNow,
                    LogsLast5Minutes = totalRecentLogs,
                    ErrorsLast5Minutes = recentLogsCount.GetValueOrDefault("Error", 0) + 
                                       recentLogsCount.GetValueOrDefault("Critical", 0)
                };

                return Ok(metrics);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting real-time metrics");
                return StatusCode(500, new { Error = "Internal server error while getting real-time metrics" });
            }
        }

        /// <summary>
        /// Gets log trend data for charts
        /// </summary>
        /// <param name="hours">Number of hours to look back (default: 24)</param>
        /// <param name="interval">Time interval (hour, day, default: hour)</param>
        /// <returns>Log trend data</returns>
        [HttpGet("trends/logs")]
        [ProducesResponseType(StatusCodes.Status200OK)]
        public async Task<ActionResult<LogTrendsDto>> GetLogTrends(
            [FromQuery] int hours = 24,
            [FromQuery] string interval = "hour")
        {
            try
            {
                var endTime = DateTime.UtcNow;
                var startTime = endTime.AddHours(-hours);

                Backend.Models.TimeInterval timeInterval;
                if (!Enum.TryParse<Backend.Models.TimeInterval>(interval, true, out timeInterval))
                {
                    timeInterval = Backend.Models.TimeInterval.Hour;
                }

                var trends = await _logAnalysisService.GetLogTrendsAsync(startTime, endTime, timeInterval);
                return Ok(trends);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting log trends");
                return StatusCode(500, new { Error = "Internal server error while getting log trends" });
            }
        }

        /// <summary>
        /// Gets agent health summary
        /// </summary>
        /// <returns>Agent health summary</returns>
        [HttpGet("agents/health")]
        [ProducesResponseType(StatusCodes.Status200OK)]
        public async Task<ActionResult<AgentHealthSummaryDto>> GetAgentHealthSummary()
        {
            try
            {
                var agents = await _agentService.GetAllAgentsAsync();
                var agentsList = agents.ToList();

                var healthSummary = new AgentHealthSummaryDto
                {
                    TotalAgents = agentsList.Count,
                    HealthyAgents = agentsList.Count(a => a.Status == AgentStatus.Online),
                    UnhealthyAgents = agentsList.Count(a => a.Status == AgentStatus.Offline || a.Status == AgentStatus.Warning),
                    OfflineAgents = agentsList.Count(a => a.Status == AgentStatus.Offline),
                    AgentsByType = agentsList.GroupBy(a => a.Type.ToString())
                        .ToDictionary(g => g.Key, g => g.Count()),
                    AgentsByOperatingSystem = agentsList.GroupBy(a => a.OperatingSystem ?? "Unknown")
                        .ToDictionary(g => g.Key, g => g.Count()),
                    AverageResourceUsage = new ResourceUsageDto
                    {
                        CpuUsage = agentsList.Where(a => a.CpuUsage.HasValue).Average(a => a.CpuUsage ?? 0),
                        MemoryUsage = agentsList.Where(a => a.MemoryUsage.HasValue).Average(a => a.MemoryUsage ?? 0),
                        DiskUsage = agentsList.Where(a => a.DiskUsage.HasValue).Average(a => a.DiskUsage ?? 0)
                    },
                    LastUpdated = DateTime.UtcNow
                };

                return Ok(healthSummary);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting agent health summary");
                return StatusCode(500, new { Error = "Internal server error while getting agent health summary" });
            }
        }

        /// <summary>
        /// Gets top log sources
        /// </summary>
        /// <param name="hours">Number of hours to look back (default: 24)</param>
        /// <param name="limit">Maximum number of sources to return (default: 10)</param>
        /// <returns>Top log sources</returns>
        [HttpGet("logs/top-sources")]
        [ProducesResponseType(StatusCodes.Status200OK)]
        public async Task<ActionResult<List<LogSourceStatDto>>> GetTopLogSources(
            [FromQuery] int hours = 24,
            [FromQuery] int limit = 10)
        {
            try
            {
                var endTime = DateTime.UtcNow;
                var startTime = endTime.AddHours(-hours);

                var logsBySource = await _logService.GetLogCountBySourceAsync(startTime, endTime);
                
                var topSources = logsBySource
                    .OrderByDescending(kvp => kvp.Value)
                    .Take(limit)
                    .Select(kvp => new LogSourceStatDto
                    {
                        Source = kvp.Key,
                        LogCount = kvp.Value,
                        Percentage = logsBySource.Values.Sum() > 0 
                            ? (double)kvp.Value / logsBySource.Values.Sum() * 100 
                            : 0
                    })
                    .ToList();

                return Ok(topSources);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting top log sources");
                return StatusCode(500, new { Error = "Internal server error while getting top log sources" });
            }
        }

        /// <summary>
        /// Gets security events summary
        /// </summary>
        /// <param name="hours">Number of hours to look back (default: 24)</param>
        /// <returns>Security events summary</returns>
        [HttpGet("security/events")]
        [ProducesResponseType(StatusCodes.Status200OK)]
        public async Task<ActionResult<SecurityEventsSummaryDto>> GetSecurityEventsSummary(
            [FromQuery] int hours = 24)
        {
            try
            {
                var endTime = DateTime.UtcNow;
                var startTime = endTime.AddHours(-hours);

                // Get security-related logs (this is a simplified version)
                var query = new LogQueryDto
                {
                    StartTime = startTime,
                    EndTime = endTime,
                    Categories = new List<string> { "Security", "Authentication", "Authorization" },
                    Limit = 1000
                };

                var securityLogs = await _logService.SearchLogsAsync(query);
                
                var summary = new SecurityEventsSummaryDto
                {
                    TotalSecurityEvents = securityLogs.TotalCount,
                    FailedLoginAttempts = securityLogs.Items.Count(l => 
                        l.Message.Contains("failed", StringComparison.OrdinalIgnoreCase) && 
                        l.Message.Contains("login", StringComparison.OrdinalIgnoreCase)),
                    SuspiciousActivities = securityLogs.Items.Count(l => 
                        l.Level == "Warning" || l.Level == "Error"),
                    TimeRange = new TimeRangeDto
                    {
                        StartTime = startTime,
                        EndTime = endTime
                    },
                    LastUpdated = DateTime.UtcNow
                };

                return Ok(summary);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting security events summary");
                return StatusCode(500, new { Error = "Internal server error while getting security events summary" });
            }
        }

        /// <summary>
        /// Gets system performance metrics
        /// </summary>
        /// <returns>System performance metrics</returns>
        [HttpGet("performance")]
        [ProducesResponseType(StatusCodes.Status200OK)]
        public async Task<ActionResult<SystemPerformanceDto>> GetSystemPerformance()
        {
            try
            {
                var endTime = DateTime.UtcNow;
                var startTime = endTime.AddHours(-1); // Last hour

                // Get log processing performance
                var recentLogs = await _logService.GetLogCountBySeverityAsync(startTime, endTime);
                var totalRecentLogs = recentLogs.Values.Sum();

                // Get agent performance
                var agents = await _agentService.GetAllAgentsAsync();
                var agentsList = agents.ToList();

                var performance = new SystemPerformanceDto
                {
                    LogProcessingRate = totalRecentLogs / 60.0, // logs per minute
                    AverageResponseTime = 150, // This would come from actual metrics
                    SystemUptime = TimeSpan.FromDays(30), // This would come from actual system metrics
                    MemoryUsage = 65.2, // This would come from actual system metrics
                    CpuUsage = 45.8, // This would come from actual system metrics
                    DiskUsage = 78.3, // This would come from actual system metrics
                    ActiveConnections = agentsList.Count(a => a.Status == AgentStatus.Online),
                    LastUpdated = DateTime.UtcNow
                };

                return Ok(performance);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting system performance");
                return StatusCode(500, new { Error = "Internal server error while getting system performance" });
            }
        }

        // Private helper methods

        /// <summary>
        /// Calculates overall system health status
        /// </summary>
        /// <param name="agentStats">Agent statistics</param>
        /// <param name="logStats">Log statistics</param>
        /// <returns>Overall health status</returns>
        private string CalculateOverallSystemHealth(AgentStatisticsDto agentStats, LogStatisticsDto logStats)
        {
            var agentHealthScore = agentStats.TotalAgents > 0 
                ? (double)agentStats.OnlineAgents / agentStats.TotalAgents 
                : 1.0;

            var errorRate = logStats.TotalLogs > 0 
                ? (double)(logStats.CriticalLogs + logStats.ErrorLogs) / logStats.TotalLogs 
                : 0.0;

            if (agentHealthScore >= 0.9 && errorRate <= 0.05)
                return "Healthy";
            else if (agentHealthScore >= 0.7 && errorRate <= 0.15)
                return "Warning";
            else
                return "Critical";
        }

        /// <summary>
        /// Calculates agent connectivity health
        /// </summary>
        /// <param name="agentStats">Agent statistics</param>
        /// <returns>Agent connectivity health status</returns>
        private string CalculateAgentConnectivityHealth(AgentStatisticsDto agentStats)
        {
            if (agentStats.TotalAgents == 0) return "Unknown";

            var onlinePercentage = (double)agentStats.OnlineAgents / agentStats.TotalAgents;

            if (onlinePercentage >= 0.95) return "Excellent";
            if (onlinePercentage >= 0.85) return "Good";
            if (onlinePercentage >= 0.70) return "Fair";
            return "Poor";
        }

        /// <summary>
        /// Calculates log ingestion health
        /// </summary>
        /// <param name="logStats">Log statistics</param>
        /// <returns>Log ingestion health status</returns>
        private string CalculateLogIngestionHealth(LogStatisticsDto logStats)
        {
            if (logStats.LogsPerHour >= 1000) return "High";
            if (logStats.LogsPerHour >= 100) return "Normal";
            if (logStats.LogsPerHour >= 10) return "Low";
            return "Very Low";
        }

        /// <summary>
        /// Calculates error rate health
        /// </summary>
        /// <param name="logStats">Log statistics</param>
        /// <returns>Error rate health status</returns>
        private string CalculateErrorRateHealth(LogStatisticsDto logStats)
        {
            if (logStats.TotalLogs == 0) return "Unknown";

            var errorRate = (double)(logStats.CriticalLogs + logStats.ErrorLogs) / logStats.TotalLogs;

            if (errorRate <= 0.01) return "Excellent";
            if (errorRate <= 0.05) return "Good";
            if (errorRate <= 0.10) return "Fair";
            return "Poor";
        }
    }
} 