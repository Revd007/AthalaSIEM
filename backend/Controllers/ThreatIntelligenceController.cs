using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Authorization;
using Microsoft.EntityFrameworkCore;
using Backend.Data;
using Backend.Models;
using System.Text.Json;
using Microsoft.Extensions.Logging;
using System.Text.RegularExpressions;
using System.Security.Cryptography;
using System.Text;
using Backend.Services;
using System;
using System.Threading.Tasks;
using System.Collections.Generic;

namespace Backend.Controllers
{
    /// <summary>
    /// API controller for threat intelligence operations with multi-collector support
    /// </summary>
    [Authorize]
    [ApiController]
    [Route("api/[controller]")]
    public class ThreatIntelligenceController : ControllerBase
    {
        private readonly IThreatIntelligenceService _threatIntelligenceService;
        private readonly ILogger<ThreatIntelligenceController> _logger;

        public ThreatIntelligenceController(
            IThreatIntelligenceService threatIntelligenceService,
            ILogger<ThreatIntelligenceController> logger)
        {
            _threatIntelligenceService = threatIntelligenceService ?? throw new ArgumentNullException(nameof(threatIntelligenceService));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }

        /// <summary>
        /// Analyzes a log entry for threats
        /// </summary>
        /// <param name="logEntry">The log entry to analyze</param>
        /// <returns>Threat analysis result</returns>
        [HttpPost("analyze")]
        public async Task<ActionResult<ThreatAnalysisResult>> AnalyzeLogEntry([FromBody] LogEntryModels logEntry)
        {
            try
            {
                if (logEntry == null)
                {
                    return BadRequest("Log entry is required");
                }

                var result = await _threatIntelligenceService.AnalyzeLogEntryAsync(logEntry);
                return Ok(result);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error analyzing log entry {LogId}", logEntry?.Id);
                return StatusCode(500, "Internal server error during threat analysis");
            }
        }

        /// <summary>
        /// Gets threat summary for a specific collector type
        /// </summary>
        /// <param name="collectorType">The collector type (Container, CloudServices, Database, IoT, FileIntegrity)</param>
        /// <param name="since">Optional start date for analysis period</param>
        /// <returns>Collector threat summary</returns>
        [HttpGet("summary/{collectorType}")]
        public async Task<ActionResult<CollectorThreatSummary>> GetCollectorThreatSummary(
            string collectorType, 
            [FromQuery] DateTime? since = null)
        {
            try
            {
                if (string.IsNullOrEmpty(collectorType))
                {
                    return BadRequest("Collector type is required");
                }

                var validCollectorTypes = new[] { "Container", "CloudServices", "Database", "IoT", "FileIntegrity", "General" };
                if (!validCollectorTypes.Contains(collectorType, StringComparer.OrdinalIgnoreCase))
                {
                    return BadRequest($"Invalid collector type. Valid types: {string.Join(", ", validCollectorTypes)}");
                }

                var summary = await _threatIntelligenceService.GetCollectorThreatSummaryAsync(collectorType, since);
                return Ok(summary);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting threat summary for collector {CollectorType}", collectorType);
                return StatusCode(500, "Internal server error while retrieving threat summary");
            }
        }

        /// <summary>
        /// Gets threat summaries for all collector types
        /// </summary>
        /// <param name="since">Optional start date for analysis period</param>
        /// <returns>List of collector threat summaries</returns>
        [HttpGet("summary")]
        public async Task<ActionResult<List<CollectorThreatSummary>>> GetAllCollectorThreatSummaries(
            [FromQuery] DateTime? since = null)
        {
            try
            {
                var collectorTypes = new[] { "Container", "CloudServices", "Database", "IoT", "FileIntegrity", "General" };
                var summaries = new List<CollectorThreatSummary>();

                foreach (var collectorType in collectorTypes)
                {
                    try
                    {
                        var summary = await _threatIntelligenceService.GetCollectorThreatSummaryAsync(collectorType, since);
                        summaries.Add(summary);
                    }
                    catch (Exception ex)
                    {
                        _logger.LogWarning(ex, "Failed to get threat summary for collector {CollectorType}", collectorType);
                    }
                }

                return Ok(summaries);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting all threat summaries");
                return StatusCode(500, "Internal server error while retrieving threat summaries");
            }
        }

        /// <summary>
        /// Finds threat correlations across collectors
        /// </summary>
        /// <param name="timeWindowHours">Time window in hours for correlation analysis</param>
        /// <param name="minimumOccurrences">Minimum number of occurrences to consider as correlation</param>
        /// <returns>List of threat correlations</returns>
        [HttpGet("correlations")]
        public async Task<ActionResult<List<ThreatCorrelation>>> FindThreatCorrelations(
            [FromQuery] int timeWindowHours = 24,
            [FromQuery] int minimumOccurrences = 3)
        {
            try
            {
                if (timeWindowHours <= 0 || timeWindowHours > 168) // Max 1 week
                {
                    return BadRequest("Time window must be between 1 and 168 hours");
                }

                if (minimumOccurrences < 2)
                {
                    return BadRequest("Minimum occurrences must be at least 2");
                }

                var timeWindow = TimeSpan.FromHours(timeWindowHours);
                var correlations = await _threatIntelligenceService.FindThreatCorrelationsAsync(timeWindow, minimumOccurrences);
                
                return Ok(correlations);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error finding threat correlations");
                return StatusCode(500, "Internal server error while finding threat correlations");
            }
        }

        /// <summary>
        /// Gets threat intelligence statistics
        /// </summary>
        /// <returns>Threat intelligence statistics</returns>
        [HttpGet("statistics")]
        public async Task<ActionResult<object>> GetThreatIntelligenceStatistics()
        {
            try
            {
                var collectorTypes = new[] { "Container", "CloudServices", "Database", "IoT", "FileIntegrity", "General" };
                var since = DateTime.UtcNow.AddDays(-30); // Last 30 days
                
                var statistics = new
                {
                    CollectorSummaries = new List<object>(),
                    OverallStats = new
                    {
                        TotalThreats = 0,
                        CriticalThreats = 0,
                        HighThreats = 0,
                        ActiveCorrelations = 0
                    },
                    TrendData = new
                    {
                        ThreatsByDay = new Dictionary<string, int>(),
                        ThreatsByCollector = new Dictionary<string, int>()
                    }
                };

                var summaries = new List<object>();
                var totalThreats = 0;
                var criticalThreats = 0;
                var highThreats = 0;
                var threatsByCollector = new Dictionary<string, int>();

                foreach (var collectorType in collectorTypes)
                {
                    try
                    {
                        var summary = await _threatIntelligenceService.GetCollectorThreatSummaryAsync(collectorType, since);
                        
                        var collectorSummary = new
                        {
                            CollectorType = summary.CollectorType,
                            TotalLogs = summary.TotalLogs,
                            TopThreatIndicators = summary.TopThreatIndicators.Take(3),
                            RecommendedActions = summary.RecommendedActions.Take(3)
                        };

                        summaries.Add(collectorSummary);
                        
                        totalThreats += summary.TotalLogs;
                        threatsByCollector[collectorType] = summary.TotalLogs;
                    }
                    catch (Exception ex)
                    {
                        _logger.LogWarning(ex, "Failed to get statistics for collector {CollectorType}", collectorType);
                    }
                }

                // Get correlations for active count
                var correlations = await _threatIntelligenceService.FindThreatCorrelationsAsync(TimeSpan.FromHours(24), 2);

                var result = new
                {
                    CollectorSummaries = summaries,
                    OverallStats = new
                    {
                        TotalThreats = totalThreats,
                        CriticalThreats = criticalThreats,
                        HighThreats = highThreats,
                        ActiveCorrelations = correlations.Count
                    },
                    TrendData = new
                    {
                        ThreatsByCollector = threatsByCollector,
                        AnalysisPeriod = since.ToString("yyyy-MM-dd")
                    }
                };

                return Ok(result);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting threat intelligence statistics");
                return StatusCode(500, "Internal server error while retrieving statistics");
            }
        }

        /// <summary>
        /// Gets threat indicators for a specific collector type
        /// </summary>
        /// <param name="collectorType">The collector type</param>
        /// <param name="limit">Maximum number of indicators to return</param>
        /// <returns>List of threat indicators</returns>
        [HttpGet("indicators/{collectorType}")]
        public async Task<ActionResult<List<string>>> GetCollectorThreatIndicators(
            string collectorType,
            [FromQuery] int limit = 10)
        {
            try
            {
                if (string.IsNullOrEmpty(collectorType))
                {
                    return BadRequest("Collector type is required");
                }

                if (limit <= 0 || limit > 100)
                {
                    return BadRequest("Limit must be between 1 and 100");
                }

                var summary = await _threatIntelligenceService.GetCollectorThreatSummaryAsync(collectorType);
                var indicators = summary.TopThreatIndicators.Take(limit).ToList();
                
                return Ok(indicators);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting threat indicators for collector {CollectorType}", collectorType);
                return StatusCode(500, "Internal server error while retrieving threat indicators");
            }
        }

        /// <summary>
        /// Gets recommended actions for threat mitigation by collector type
        /// </summary>
        /// <param name="collectorType">The collector type</param>
        /// <returns>List of recommended actions</returns>
        [HttpGet("recommendations/{collectorType}")]
        public async Task<ActionResult<List<string>>> GetCollectorRecommendations(string collectorType)
        {
            try
            {
                if (string.IsNullOrEmpty(collectorType))
                {
                    return BadRequest("Collector type is required");
                }

                var summary = await _threatIntelligenceService.GetCollectorThreatSummaryAsync(collectorType);
                return Ok(summary.RecommendedActions);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting recommendations for collector {CollectorType}", collectorType);
                return StatusCode(500, "Internal server error while retrieving recommendations");
            }
        }

        /// <summary>
        /// Health check endpoint for threat intelligence service
        /// </summary>
        /// <returns>Health status</returns>
        [HttpGet("health")]
        public Task<ActionResult<object>> GetHealthStatus()
        {
            try
            {
                // Basic health checks
                var healthStatus = new
                {
                    Status = "Healthy",
                    Timestamp = DateTime.UtcNow,
                    Services = new
                    {
                        ThreatIntelligence = "Operational",
                        CollectorProfiles = "Active",
                        CorrelationEngine = "Running"
                    },
                    Version = "1.0.0"
                };

                return Task.FromResult<ActionResult<object>>(Ok(healthStatus));
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error checking threat intelligence health");
                return Task.FromResult<ActionResult<object>>(StatusCode(500, "Internal server error during health check"));
            }
        }
    }

    // Request/Response DTOs
    public class CreateThreatIndicatorRequest
    {
        public string Type { get; set; } = string.Empty;
        public string Value { get; set; } = string.Empty;
        public string Confidence { get; set; } = "Medium";
        public string Severity { get; set; } = "Medium";
        public string? ThreatType { get; set; }
        public string? MalwareFamily { get; set; }
        public string? Description { get; set; }
        public List<string>? Tags { get; set; }
        public DateTime? ExpiresAt { get; set; }
    }

    public class ThreatSearchRequest
    {
        public string SearchValue { get; set; } = string.Empty;
        public string? IndicatorType { get; set; }
        public DateTime? StartDate { get; set; }
        public DateTime? EndDate { get; set; }
        public bool IncludeEnrichment { get; set; } = true;
    }

    public class CreateWhitelistRequest
    {
        public string Type { get; set; } = string.Empty;
        public string Value { get; set; } = string.Empty;
        public string? Reason { get; set; }
        public DateTime? ExpiresAt { get; set; }
    }
} 
