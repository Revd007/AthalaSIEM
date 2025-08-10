using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Authorization;
using Microsoft.Extensions.Logging;
using Backend.DTOs;
using Backend.Services;
using Backend.Data;
using System.Security.Claims;
using Microsoft.EntityFrameworkCore;

namespace Backend.Controllers
{
    /// <summary>
    /// Linux Agent Controller for AthalaSIEM Backend
    /// Handles Linux-specific agent communication and metrics
    /// Author: Revian Ravil Athala
    /// Enterprise SIEM Linux agent management
    /// </summary>
    [ApiController]
    [Route("api/linux-agent")]
    [Authorize]
    public class LinuxAgentController : ControllerBase
    {
        private readonly ApplicationDbContext _context;
        private readonly ILogger<LinuxAgentController> _logger;
        private readonly IAgentService _agentService;
        private readonly ILogService _logService;

        public LinuxAgentController(
            ApplicationDbContext context,
            ILogger<LinuxAgentController> logger,
            IAgentService agentService,
            ILogService logService)
        {
            _context = context;
            _logger = logger;
            _agentService = agentService;
            _logService = logService;
        }

        #region System Metrics Endpoints

        /// <summary>
        /// Receives system metrics from Linux agents
        /// </summary>
        /// <param name="agentId">Linux agent ID</param>
        /// <param name="metricsDto">System metrics data</param>
        /// <returns>Success or failure response</returns>
        [HttpPost("{agentId}/system-metrics")]
        [ProducesResponseType(StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status400BadRequest)]
        [ProducesResponseType(StatusCodes.Status401Unauthorized)]
        [ProducesResponseType(StatusCodes.Status404NotFound)]
        public async Task<IActionResult> ReceiveSystemMetrics(
            [FromRoute] string agentId,
            [FromBody] LinuxSystemMetricsDto metricsDto)
        {
            try
            {
                if (!ModelState.IsValid)
                {
                    return BadRequest(new { Errors = ModelState.Values.SelectMany(v => v.Errors).Select(e => e.ErrorMessage) });
                }

                // Validate API key
                if (!Request.Headers.TryGetValue("X-API-Key", out var apiKey))
                {
                    return Unauthorized(new { Error = "API key is required" });
                }

                var isValidApiKey = await _agentService.ValidateApiKeyAsync(Guid.Parse(agentId), apiKey.ToString());
                if (!isValidApiKey)
                {
                    return Unauthorized(new { Error = "Invalid API key" });
                }

                _logger.LogInformation("🐧 Received Linux system metrics from agent {AgentId} - CPU: {CPU}%, Memory: {Memory}%, Processes: {Processes}",
                    agentId, metricsDto.CpuUsagePercent.ToString("F1"), metricsDto.MemoryUsagePercent.ToString("F1"), metricsDto.TotalProcesses);

                // Convert to log entry for storage
                var logEntry = new Backend.DTOs.LogBatchDto
                {
                    AgentId = agentId,
                    Logs = new List<Backend.DTOs.LogEntryDto>
                    {
                        new Backend.DTOs.LogEntryDto
                        {
                            Id = Guid.NewGuid().ToString(),
                            Timestamp = metricsDto.Timestamp,
                            Source = "LinuxSystemMetrics",
                            Level = "Information",
                            Message = $"System metrics: CPU {metricsDto.CpuUsagePercent:F1}%, Memory {metricsDto.MemoryUsagePercent:F1}%, Load {metricsDto.LoadAverage1Min:F2}",
                            Category = "SystemMetrics",
                            SecurityRelevance = DetermineSecurityRelevance(metricsDto),
                            CollectorType = "LinuxSystemMetrics",
                            Properties = ConvertMetricsToProperties(metricsDto)
                        }
                    }
                };

                // Process through log service
                await _logService.ProcessLogBatchAsync(agentId, logEntry);

                return Ok(new 
                { 
                    Success = true, 
                    Message = "Linux system metrics received successfully",
                    ProcessedAt = DateTime.UtcNow,
                    MetricsCount = 1
                });
            }
            catch (FormatException)
            {
                return BadRequest(new { Error = "Invalid agent ID format" });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Error processing Linux system metrics from agent {AgentId}", agentId);
                return StatusCode(500, new { Error = "Internal server error processing system metrics" });
            }
        }

        /// <summary>
        /// Receives batch system metrics from Linux agents for better performance
        /// </summary>
        /// <param name="agentId">Linux agent ID</param>
        /// <param name="batchDto">Batch metrics data</param>
        /// <returns>Success or failure response</returns>
        [HttpPost("{agentId}/system-metrics/batch")]
        [ProducesResponseType(StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status400BadRequest)]
        [ProducesResponseType(StatusCodes.Status401Unauthorized)]
        public async Task<IActionResult> ReceiveSystemMetricsBatch(
            [FromRoute] string agentId,
            [FromBody] LinuxMetricsBatchDto batchDto)
        {
            try
            {
                if (!ModelState.IsValid)
                {
                    return BadRequest(new { Errors = ModelState.Values.SelectMany(v => v.Errors).Select(e => e.ErrorMessage) });
                }

                // Validate API key
                if (!Request.Headers.TryGetValue("X-API-Key", out var apiKey))
                {
                    return Unauthorized(new { Error = "API key is required" });
                }

                var isValidApiKey = await _agentService.ValidateApiKeyAsync(Guid.Parse(agentId), apiKey.ToString());
                if (!isValidApiKey)
                {
                    return Unauthorized(new { Error = "Invalid API key" });
                }

                _logger.LogInformation("🐧 Received Linux metrics batch from agent {AgentId} - {Count} system metrics, {FIMCount} FIM events",
                    agentId, batchDto.SystemMetrics.Count, batchDto.FIMEvents.Count);

                // Convert metrics to log entries
                var logEntries = new List<Backend.DTOs.LogEntryDto>();

                // Process system metrics
                foreach (var metric in batchDto.SystemMetrics)
                {
                    logEntries.Add(new Backend.DTOs.LogEntryDto
                    {
                        Id = Guid.NewGuid().ToString(),
                        Timestamp = metric.Timestamp,
                        Source = "LinuxSystemMetrics",
                        Level = "Information",
                        Message = $"System metrics: CPU {metric.CpuUsagePercent:F1}%, Memory {metric.MemoryUsagePercent:F1}%",
                        Category = "SystemMetrics",
                        SecurityRelevance = DetermineSecurityRelevance(metric),
                        CollectorType = "LinuxSystemMetrics",
                        Properties = ConvertMetricsToProperties(metric)
                    });
                }

                // Process FIM events
                foreach (var fimEvent in batchDto.FIMEvents)
                {
                    logEntries.Add(new Backend.DTOs.LogEntryDto
                    {
                        Id = Guid.NewGuid().ToString(),
                        Timestamp = fimEvent.Timestamp,
                        Source = "LinuxFIM",
                        Level = DetermineFIMLogLevel(fimEvent),
                        Message = $"File {fimEvent.EventType}: {fimEvent.FilePath}",
                        Category = "FileIntegrity",
                        SecurityRelevance = fimEvent.SecurityLevel,
                        CollectorType = "LinuxFIM",
                        Properties = ConvertFIMEventToProperties(fimEvent)
                    });
                }

                // Process through log service
                var logBatch = new Backend.DTOs.LogBatchDto
                {
                    AgentId = agentId,
                    Logs = logEntries
                };

                await _logService.ProcessLogBatchAsync(agentId, logBatch);

                return Ok(new 
                { 
                    Success = true, 
                    Message = "Linux metrics batch processed successfully",
                    ProcessedAt = DateTime.UtcNow,
                    SystemMetricsCount = batchDto.SystemMetrics.Count,
                    FIMEventsCount = batchDto.FIMEvents.Count,
                    BatchId = batchDto.BatchId
                });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Error processing Linux metrics batch from agent {AgentId}", agentId);
                return StatusCode(500, new { Error = "Internal server error processing metrics batch" });
            }
        }

        #endregion

        #region FIM Events Endpoints

        /// <summary>
        /// Receives FIM events from Linux agents
        /// </summary>
        /// <param name="agentId">Linux agent ID</param>
        /// <param name="fimEvents">FIM events array</param>
        /// <returns>Success or failure response</returns>
        [HttpPost("{agentId}/fim-events")]
        [ProducesResponseType(StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status400BadRequest)]
        [ProducesResponseType(StatusCodes.Status401Unauthorized)]
        public async Task<IActionResult> ReceiveFIMEvents(
            [FromRoute] string agentId,
            [FromBody] LinuxFIMEventDto[] fimEvents)
        {
            try
            {
                if (!ModelState.IsValid)
                {
                    return BadRequest(new { Errors = ModelState.Values.SelectMany(v => v.Errors).Select(e => e.ErrorMessage) });
                }

                // Validate API key
                if (!Request.Headers.TryGetValue("X-API-Key", out var apiKey))
                {
                    return Unauthorized(new { Error = "API key is required" });
                }

                var isValidApiKey = await _agentService.ValidateApiKeyAsync(Guid.Parse(agentId), apiKey.ToString());
                if (!isValidApiKey)
                {
                    return Unauthorized(new { Error = "Invalid API key" });
                }

                _logger.LogInformation("🐧 Received {Count} Linux FIM events from agent {AgentId}", fimEvents.Length, agentId);

                // Convert to log entries
                var logEntries = fimEvents.Select(fimEvent => new Backend.DTOs.LogEntryDto
                {
                    Id = Guid.NewGuid().ToString(),
                    Timestamp = fimEvent.Timestamp,
                    Source = "LinuxFIM",
                    Level = DetermineFIMLogLevel(fimEvent),
                    Message = $"File {fimEvent.EventType}: {fimEvent.FilePath}",
                    Category = "FileIntegrity",
                    SecurityRelevance = fimEvent.SecurityLevel,
                    CollectorType = "LinuxFIM",
                    Properties = ConvertFIMEventToProperties(fimEvent)
                }).ToList();

                // Process through log service
                var logBatch = new Backend.DTOs.LogBatchDto
                {
                    AgentId = agentId,
                    Logs = logEntries
                };

                await _logService.ProcessLogBatchAsync(agentId, logBatch);

                return Ok(new 
                { 
                    Success = true, 
                    Message = "Linux FIM events processed successfully",
                    ProcessedAt = DateTime.UtcNow,
                    EventsCount = fimEvents.Length
                });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Error processing Linux FIM events from agent {AgentId}", agentId);
                return StatusCode(500, new { Error = "Internal server error processing FIM events" });
            }
        }

        #endregion

        #region Configuration Endpoints

        /// <summary>
        /// Gets Linux agent configuration
        /// </summary>
        /// <param name="agentId">Linux agent ID</param>
        /// <returns>Linux agent configuration</returns>
        [HttpGet("{agentId}/configuration")]
        [ProducesResponseType(StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status404NotFound)]
        [ProducesResponseType(StatusCodes.Status401Unauthorized)]
        public async Task<ActionResult<LinuxAgentConfigDto>> GetLinuxAgentConfiguration([FromRoute] string agentId)
        {
            try
            {
                // Validate API key
                if (!Request.Headers.TryGetValue("X-API-Key", out var apiKey))
                {
                    return Unauthorized(new { Error = "API key is required" });
                }

                var isValidApiKey = await _agentService.ValidateApiKeyAsync(Guid.Parse(agentId), apiKey.ToString());
                if (!isValidApiKey)
                {
                    return Unauthorized(new { Error = "Invalid API key" });
                }

                var agent = await _agentService.GetAgentByIdAsync(agentId);
                if (agent == null)
                {
                    return NotFound(new { Error = "Linux agent not found" });
                }

                // Generate Linux-specific configuration
                var config = new LinuxAgentConfigDto
                {
                    EnableSystemMetrics = true,
                    SystemMetricsIntervalSeconds = 30,
                    EnableCPUMonitoring = true,
                    EnableMemoryMonitoring = true,
                    EnableDiskMonitoring = true,
                    EnableNetworkMonitoring = true,
                    
                    SyslogConfig = new LinuxSyslogConfigDto
                    {
                        EnableSystemdJournal = true,
                        SupportedFormats = new List<string> { "RFC3164", "RFC5424", "CEF", "JSON" },
                        ParseStructuredLogs = true,
                        MaxLogLineLength = 8192
                    },
                    
                    EnableFIM = true,
                    FIMMonitoredPaths = new List<string>
                    {
                        "/etc/passwd", "/etc/shadow", "/etc/group", "/etc/sudoers",
                        "/etc/ssh/", "/boot/", "/opt/", "/usr/local/bin/"
                    },
                    FIMRealTimeMonitoring = true,
                    FIMHashAlgorithm = "SHA256",
                    
                    HeartbeatIntervalSeconds = 60,
                    LogBatchSize = 100,
                    EnableCompression = true,
                    EnableThreatDetection = true
                };

                _logger.LogInformation("🐧 Provided Linux configuration to agent {AgentId}", agentId);

                return Ok(config);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Error getting Linux agent configuration for {AgentId}", agentId);
                return StatusCode(500, new { Error = "Internal server error getting configuration" });
            }
        }

        /// <summary>
        /// Updates Linux agent configuration
        /// </summary>
        /// <param name="agentId">Linux agent ID</param>
        /// <param name="configDto">New configuration</param>
        /// <returns>Success or failure response</returns>
        [HttpPut("{agentId}/configuration")]
        [ProducesResponseType(StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status400BadRequest)]
        [ProducesResponseType(StatusCodes.Status404NotFound)]
        [ProducesResponseType(StatusCodes.Status401Unauthorized)]
        public async Task<IActionResult> UpdateLinuxAgentConfiguration(
            [FromRoute] string agentId,
            [FromBody] LinuxAgentConfigDto configDto)
        {
            try
            {
                if (!ModelState.IsValid)
                {
                    return BadRequest(new { Errors = ModelState.Values.SelectMany(v => v.Errors).Select(e => e.ErrorMessage) });
                }

                // Validate API key
                if (!Request.Headers.TryGetValue("X-API-Key", out var apiKey))
                {
                    return Unauthorized(new { Error = "API key is required" });
                }

                var isValidApiKey = await _agentService.ValidateApiKeyAsync(Guid.Parse(agentId), apiKey.ToString());
                if (!isValidApiKey)
                {
                    return Unauthorized(new { Error = "Invalid API key" });
                }

                var agent = await _agentService.GetAgentByIdAsync(agentId);
                if (agent == null)
                {
                    return NotFound(new { Error = "Linux agent not found" });
                }

                // Update agent configuration (implementation depends on your agent model structure)
                // This is a placeholder - you'll need to implement based on your agent model

                _logger.LogInformation("🐧 Updated Linux configuration for agent {AgentId}", agentId);

                return Ok(new 
                { 
                    Success = true, 
                    Message = "Linux agent configuration updated successfully",
                    UpdatedAt = DateTime.UtcNow
                });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Error updating Linux agent configuration for {AgentId}", agentId);
                return StatusCode(500, new { Error = "Internal server error updating configuration" });
            }
        }

        #endregion

        #region Health Endpoints

        /// <summary>
        /// Receives health status from Linux agents
        /// </summary>
        /// <param name="agentId">Linux agent ID</param>
        /// <param name="healthDto">Health status data</param>
        /// <returns>Success or failure response</returns>
        [HttpPost("{agentId}/health")]
        [ProducesResponseType(StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status400BadRequest)]
        [ProducesResponseType(StatusCodes.Status401Unauthorized)]
        public async Task<IActionResult> ReceiveLinuxAgentHealth(
            [FromRoute] string agentId,
            [FromBody] LinuxAgentHealthDto healthDto)
        {
            try
            {
                if (!ModelState.IsValid)
                {
                    return BadRequest(new { Errors = ModelState.Values.SelectMany(v => v.Errors).Select(e => e.ErrorMessage) });
                }

                // Validate API key
                if (!Request.Headers.TryGetValue("X-API-Key", out var apiKey))
                {
                    return Unauthorized(new { Error = "API key is required" });
                }

                var isValidApiKey = await _agentService.ValidateApiKeyAsync(Guid.Parse(agentId), apiKey.ToString());
                if (!isValidApiKey)
                {
                    return Unauthorized(new { Error = "Invalid API key" });
                }

                _logger.LogInformation("🐧 Received Linux agent health from {AgentId} - Status: {Status}, Healthy: {Healthy}",
                    agentId, healthDto.Status, healthDto.IsHealthy);

                // Process health data (store in database, trigger alerts, etc.)
                // Implementation depends on your health monitoring requirements

                return Ok(new 
                { 
                    Success = true, 
                    Message = "Linux agent health received successfully",
                    ReceivedAt = DateTime.UtcNow
                });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Error processing Linux agent health from {AgentId}", agentId);
                return StatusCode(500, new { Error = "Internal server error processing health data" });
            }
        }

        #endregion

        #region Helper Methods

        private string DetermineSecurityRelevance(LinuxSystemMetricsDto metrics)
        {
            // High security relevance for resource exhaustion indicators
            if (metrics.CpuUsagePercent > 90 || metrics.MemoryUsagePercent > 95)
                return "High";

            // Medium security relevance for elevated resource usage
            if (metrics.CpuUsagePercent > 80 || metrics.MemoryUsagePercent > 85 || 
                metrics.LoadAverage1Min > 5.0 || metrics.ZombieProcesses > 10)
                return "Medium";

            return "Low";
        }

        private string DetermineFIMLogLevel(LinuxFIMEventDto fimEvent)
        {
            return fimEvent.EventType.ToLower() switch
            {
                "delete" => "Warning",
                "create" when fimEvent.FilePath.Contains("/etc/") => "Warning",
                "modify" when fimEvent.FilePath.Contains("/etc/passwd") || fimEvent.FilePath.Contains("/etc/shadow") => "Critical",
                "modify" when fimEvent.FilePath.Contains("/etc/") => "Warning",
                _ => "Information"
            };
        }

        private Dictionary<string, object> ConvertMetricsToProperties(LinuxSystemMetricsDto metrics)
        {
            return new Dictionary<string, object>
            {
                ["CpuUsagePercent"] = metrics.CpuUsagePercent,
                ["MemoryUsagePercent"] = metrics.MemoryUsagePercent,
                ["LoadAverage1Min"] = metrics.LoadAverage1Min,
                ["LoadAverage5Min"] = metrics.LoadAverage5Min,
                ["LoadAverage15Min"] = metrics.LoadAverage15Min,
                ["TotalProcesses"] = metrics.TotalProcesses,
                ["RunningProcesses"] = metrics.RunningProcesses,
                ["ZombieProcesses"] = metrics.ZombieProcesses,
                ["SystemUptime"] = metrics.SystemUptime.ToString(),
                ["KernelVersion"] = metrics.KernelVersion,
                ["Distribution"] = metrics.Distribution,
                ["DistributionVersion"] = metrics.DistributionVersion,
                ["CollectionDurationMs"] = metrics.CollectionDurationMs,
                ["DiskUsageCount"] = metrics.DiskUsage.Count,
                ["NetworkInterfacesCount"] = metrics.NetworkStats.Count
            };
        }

        private Dictionary<string, object> ConvertFIMEventToProperties(LinuxFIMEventDto fimEvent)
        {
            var properties = new Dictionary<string, object>
            {
                ["FilePath"] = fimEvent.FilePath,
                ["EventType"] = fimEvent.EventType,
                ["User"] = fimEvent.User,
                ["Process"] = fimEvent.Process,
                ["SecurityLevel"] = fimEvent.SecurityLevel
            };

            if (fimEvent.ProcessId.HasValue)
                properties["ProcessId"] = fimEvent.ProcessId.Value;

            if (!string.IsNullOrEmpty(fimEvent.OldFilePath))
                properties["OldFilePath"] = fimEvent.OldFilePath;

            if (fimEvent.ThreatIndicators.Any())
                properties["ThreatIndicators"] = string.Join(", ", fimEvent.ThreatIndicators);

            // Add file information if available
            if (fimEvent.NewFileInfo != null)
            {
                properties["FileSize"] = fimEvent.NewFileInfo.Size;
                properties["FilePermissions"] = fimEvent.NewFileInfo.Permissions;
                properties["FileOwner"] = fimEvent.NewFileInfo.Owner;
                properties["FileGroup"] = fimEvent.NewFileInfo.Group;
                
                if (fimEvent.NewFileInfo.Hashes.Any())
                    properties["FileHashes"] = string.Join(", ", fimEvent.NewFileInfo.Hashes.Select(h => $"{h.Key}:{h.Value}"));
            }

            // Add metadata
            foreach (var kvp in fimEvent.Metadata)
                properties[$"Metadata_{kvp.Key}"] = kvp.Value;

            return properties;
        }

        #endregion
    }
}
