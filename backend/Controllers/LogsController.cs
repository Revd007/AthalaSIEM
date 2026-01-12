using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using Backend.DTOs;
using Backend.Services;
using Backend.Models;
using System.Text.Json;
using System.Security.Claims;
using MediatR;
using Backend.Application.Commands;

namespace Backend.Controllers
{
    /// <summary>
    /// Controller for managing logs
    /// </summary>
    [ApiController]
    [Route("api/[controller]")]
    [Authorize]
    public class LogsController : ControllerBase
    {
        private readonly ILogService _logService;
        private readonly ILogAnalysisService _logAnalysisService;
        private readonly IAgentService _agentService;
        private readonly IMediator _mediator;
        private readonly ILogger<LogsController> _logger;
        
        /// <summary>
        /// Initializes a new instance of the <see cref="LogsController"/> class
        /// </summary>
        /// <param name="logService">The log service</param>
        /// <param name="logAnalysisService">The log analysis service</param>
        /// <param name="agentService">The agent service</param>
        /// <param name="mediator">The mediator for CQRS</param>
        /// <param name="logger">The logger</param>
        public LogsController(
            ILogService logService,
            ILogAnalysisService logAnalysisService,
            IAgentService agentService,
            IMediator mediator,
            ILogger<LogsController> logger)
        {
            _logService = logService ?? throw new ArgumentNullException(nameof(logService));
            _logAnalysisService = logAnalysisService ?? throw new ArgumentNullException(nameof(logAnalysisService));
            _agentService = agentService ?? throw new ArgumentNullException(nameof(agentService));
            _mediator = mediator ?? throw new ArgumentNullException(nameof(mediator));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }
        
        /// <summary>
        /// Receives a batch of logs from an agent (Agent endpoint - no UI authorization required)
        /// </summary>
        /// <param name="logBatch">The batch of logs to process</param>
        /// <returns>Success or failure response</returns>
        [HttpPost("batch")]
        [AllowAnonymous]
        [ProducesResponseType(StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status400BadRequest)]
        [ProducesResponseType(StatusCodes.Status401Unauthorized)]
        [ProducesResponseType(StatusCodes.Status500InternalServerError)]
        public async Task<ActionResult<LogBatchResponseDto>> SubmitLogBatch([FromBody] LogBatchDto logBatch)
        {
            try
            {
                if (logBatch == null || logBatch.Logs == null || logBatch.Logs.Count == 0)
                {
                    return BadRequest(new { Error = "Log batch cannot be empty" });
                }

                // Validate agent authentication
                if (!Request.Headers.TryGetValue("X-API-Key", out var apiKey) || string.IsNullOrEmpty(apiKey))
                {
                    _logger.LogWarning("Log batch submission without API key from {IP}", HttpContext.Connection.RemoteIpAddress);
                    return Unauthorized(new { Error = "API key is required" });
                }

                // Validate agent ID from header or body
                string? agentId = null;
                if (Request.Headers.TryGetValue("X-Agent-Id", out var headerAgentId))
                {
                    agentId = headerAgentId;
                }
                else if (!string.IsNullOrEmpty(logBatch.AgentId))
                {
                    agentId = logBatch.AgentId;
                }

                if (string.IsNullOrEmpty(agentId))
                {
                    return BadRequest(new { Error = "Agent ID is required" });
                }

                // Validate API key for the agent
                var isValidApiKey = await _agentService.ValidateApiKeyAsync(agentId, apiKey.ToString());
                if (!isValidApiKey)
                {
                    _logger.LogWarning("Invalid API key for agent {AgentId} from {IP}", agentId, HttpContext.Connection.RemoteIpAddress);
                    return Unauthorized(new { Error = "Invalid API key for agent" });
                }

                // Process the log batch using new CQRS architecture
                var processedCount = 0;
                var failedCount = 0;
                
                foreach (var logDto in logBatch.Logs)
                {
                    try
                    {
                        var command = new IngestLogCommand
                        {
                            AgentId = agentId,
                            Message = logDto.Message ?? string.Empty,
                            Source = logDto.Source ?? string.Empty,
                            Category = logDto.Category,
                            EventId = logDto.EventId,
                            Timestamp = logDto.Timestamp,
                            Properties = logDto.Properties
                        };
                        
                        var ingestResult = await _mediator.Send(command);
                        if (ingestResult.Success)
                        {
                            processedCount++;
                        }
                        else
                        {
                            failedCount++;
                        }
                    }
                    catch (Exception ex)
                    {
                        failedCount++;
                        _logger.LogWarning(ex, "Failed to ingest log from agent {AgentId}", agentId);
                    }
                }
                
                // Also process via legacy service for backward compatibility
                var result = await _logService.ProcessLogBatchAsync(agentId, logBatch);
                
                _logger.LogInformation("Successfully processed {LogCount} logs from agent {AgentId}", 
                    logBatch.Logs.Count, agentId);

                return Ok(new LogBatchResponseDto
                {
                    Success = true,
                    ProcessedCount = result.ProcessedCount,
                    FailedCount = result.FailedCount,
                    Message = $"Successfully processed {result.ProcessedCount} logs",
                    BatchId = result.BatchId,
                    ProcessingTimeMs = result.ProcessingTimeMs
                });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error processing log batch from agent");
                return StatusCode(500, new { Error = "Internal server error while processing log batch" });
            }
        }
        
        /// <summary>
        /// Gets logs with optional filtering
        /// </summary>
        /// <param name="query">Query parameters</param>
        /// <returns>Logs matching the query</returns>
        [HttpGet]
        [ProducesResponseType(StatusCodes.Status200OK)]
        public async Task<ActionResult<PaginatedResult<LogEntryDto>>> GetLogs([FromQuery] LogQueryDto query)
        {
            _logger.LogInformation("Getting logs with query: {Query}", query);
            
            // Set default values if not provided
            if (query.Limit <= 0)
            {
                query.Limit = 100;
            }
            
            if (query.Limit > 1000)
            {
                query.Limit = 1000;
            }
            
            // Get logs from service
            var logs = await _logService.SearchLogsAsync(query);
            
            _logger.LogInformation("Found {Count} logs matching query", logs.TotalCount);
            return Ok(logs);
        }
        
        /// <summary>
        /// Gets a specific log entry by ID
        /// </summary>
        /// <param name="id">The log entry ID</param>
        /// <returns>The log entry</returns>
        [HttpGet("{id}")]
        [ProducesResponseType(StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status404NotFound)]
        public async Task<ActionResult<LogEntryDto>> GetLogById(string id)
        {
            var log = await _logService.GetLogByIdAsync(id);
            if (log == null)
            {
                return NotFound(new { Error = $"Log with ID {id} not found" });
            }
            
            return Ok(log);
        }
        
        /// <summary>
        /// Gets logs by severity
        /// </summary>
        /// <param name="severity">The log severity</param>
        /// <param name="limit">Maximum number of logs to return</param>
        /// <param name="offset">Offset for pagination</param>
        /// <returns>Logs with the specified severity</returns>
        [HttpGet("severity/{severity}")]
        [ProducesResponseType(StatusCodes.Status200OK)]
        public async Task<ActionResult<PaginatedResult<LogEntryDto>>> GetLogsBySeverity(
            string severity, 
            [FromQuery] int limit = 100, 
            [FromQuery] int offset = 0)
        {
            var query = new LogQueryDto
            {
                Severity = severity,
                Limit = limit,
                Offset = offset
            };
            
            var logs = await _logService.SearchLogsAsync(query);
            return Ok(logs);
        }
        
        /// <summary>
        /// Gets logs by time range
        /// </summary>
        /// <param name="start">Start time (UTC)</param>
        /// <param name="end">End time (UTC)</param>
        /// <param name="limit">Maximum number of logs to return</param>
        /// <param name="offset">Offset for pagination</param>
        /// <returns>Logs within the specified time range</returns>
        [HttpGet("timerange")]
        [ProducesResponseType(StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status400BadRequest)]
        public async Task<ActionResult<PaginatedResult<LogEntryDto>>> GetLogsByTimeRange(
            [FromQuery] DateTime start, 
            [FromQuery] DateTime? end = null, 
            [FromQuery] int limit = 100, 
            [FromQuery] int offset = 0)
        {
            if (end == null)
            {
                end = DateTime.UtcNow;
            }
            
            if (start > end)
            {
                return BadRequest(new { Error = "Start time must be before end time" });
            }
            
            var query = new LogQueryDto
            {
                StartTime = start,
                EndTime = end.Value,
                Limit = limit,
                Offset = offset
            };
            
            var logs = await _logService.SearchLogsAsync(query);
            return Ok(logs);
        }
        
        /// <summary>
        /// Searches logs with full-text search
        /// </summary>
        /// <param name="searchTerm">Search term</param>
        /// <param name="limit">Maximum number of logs to return</param>
        /// <param name="offset">Offset for pagination</param>
        /// <returns>Logs matching the search term</returns>
        [HttpGet("search")]
        [ProducesResponseType(StatusCodes.Status200OK)]
        public async Task<ActionResult<PaginatedResult<LogEntryDto>>> SearchLogs(
            [FromQuery] string searchTerm, 
            [FromQuery] int limit = 100, 
            [FromQuery] int offset = 0)
        {
            var query = new LogQueryDto
            {
                SearchTerm = searchTerm,
                Limit = limit,
                Offset = offset
            };
            
            var logs = await _logService.SearchLogsAsync(query);
            return Ok(logs);
        }
        
        /// <summary>
        /// Gets log summary statistics
        /// </summary>
        /// <param name="start">Start time (UTC)</param>
        /// <param name="end">End time (UTC)</param>
        /// <returns>Log summary statistics</returns>
        [HttpGet("summary")]
        [ProducesResponseType(StatusCodes.Status200OK)]
        public async Task<ActionResult<LogSummaryDto>> GetLogSummary(
            [FromQuery] DateTime? start = null, 
            [FromQuery] DateTime? end = null)
        {
            if (start == null)
            {
                start = DateTime.UtcNow.AddDays(-7);
            }
            
            if (end == null)
            {
                end = DateTime.UtcNow;
            }
            
            var summary = await _logService.GetLogSummaryAsync(start.Value, end.Value);
            return Ok(summary);
        }
        
        /// <summary>
        /// Gets log trend data for charting
        /// </summary>
        /// <param name="start">Start time (UTC)</param>
        /// <param name="end">End time (UTC)</param>
        /// <param name="interval">Time interval (hour, day, week, month)</param>
        /// <returns>Log trend data</returns>
        [HttpGet("trends")]
        [ProducesResponseType(StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status400BadRequest)]
        public async Task<ActionResult<LogTrendsDto>> GetLogTrends(
            [FromQuery] DateTime? start = null, 
            [FromQuery] DateTime? end = null,
            [FromQuery] string interval = "hour")
        {
            if (start == null)
            {
                start = DateTime.UtcNow.AddDays(-7);
            }
            
            if (end == null)
            {
                end = DateTime.UtcNow;
            }
            
            Backend.Models.TimeInterval timeInterval;
            if (!Enum.TryParse<Backend.Models.TimeInterval>(interval, true, out timeInterval))
            {
                return BadRequest(new { Error = $"Invalid time interval: {interval}" });
            }
            
            var trends = await _logAnalysisService.GetLogTrendsAsync(start.Value, end.Value, timeInterval);
            return Ok(trends);
        }
        
        /// <summary>
        /// Gets the top anomalies detected in logs
        /// </summary>
        /// <param name="start">Start time (UTC)</param>
        /// <param name="end">End time (UTC)</param>
        /// <param name="limit">Maximum number of anomalies to return</param>
        /// <returns>Top log anomalies</returns>
        [HttpGet("anomalies")]
        [ProducesResponseType(StatusCodes.Status200OK)]
        public async Task<ActionResult<IEnumerable<LogAnomalyDto>>> GetLogAnomalies(
            [FromQuery] DateTime? start = null, 
            [FromQuery] DateTime? end = null,
            [FromQuery] int limit = 10)
        {
            if (start == null)
            {
                start = DateTime.UtcNow.AddDays(-7);
            }
            
            if (end == null)
            {
                end = DateTime.UtcNow;
            }
            
            var anomalies = await _logAnalysisService.GetLogAnomaliesAsync(start.Value, end.Value, limit);
            return Ok(anomalies);
        }
        
        /// <summary>
        /// Gets the top patterns detected in logs
        /// </summary>
        /// <param name="start">Start time (UTC)</param>
        /// <param name="end">End time (UTC)</param>
        /// <param name="limit">Maximum number of patterns to return</param>
        /// <returns>Top log patterns</returns>
        [HttpGet("patterns")]
        [ProducesResponseType(StatusCodes.Status200OK)]
        public async Task<ActionResult<IEnumerable<LogPatternDto>>> GetLogPatterns(
            [FromQuery] DateTime? start = null, 
            [FromQuery] DateTime? end = null,
            [FromQuery] int limit = 10)
        {
            if (start == null)
            {
                start = DateTime.UtcNow.AddDays(-7);
            }
            
            if (end == null)
            {
                end = DateTime.UtcNow;
            }
            
            var patterns = await _logAnalysisService.GetLogPatternsAsync(start.Value, end.Value, limit);
            return Ok(patterns);
        }
        
        /// <summary>
        /// Gets correlation between logs and alerts
        /// </summary>
        /// <param name="logId">The log entry ID</param>
        /// <param name="timeWindowMinutes">Time window in minutes</param>
        /// <returns>Correlated logs and alerts</returns>
        [HttpGet("{logId}/correlation")]
        [ProducesResponseType(StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status404NotFound)]
        public async Task<ActionResult<LogCorrelationDto>> GetLogCorrelation(
            string logId, 
            [FromQuery] int timeWindowMinutes = 15)
        {
            var log = await _logService.GetLogByIdAsync(logId);
            if (log == null)
            {
                return NotFound(new { Error = $"Log with ID {logId} not found" });
            }
            
            var correlation = await _logAnalysisService.GetLogCorrelationAsync(logId, TimeSpan.FromMinutes(timeWindowMinutes));
            return Ok(correlation);
        }
        
        /// <summary>
        /// Exports logs to a file
        /// </summary>
        /// <param name="query">Query parameters</param>
        /// <param name="format">Export format (csv, json)</param>
        /// <returns>File with exported logs</returns>
        [HttpGet("export")]
        [ProducesResponseType(StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status400BadRequest)]
        public async Task<ActionResult> ExportLogs(
            [FromQuery] LogQueryDto query, 
            [FromQuery] string format = "csv")
        {
            if (string.IsNullOrEmpty(format) || (format.ToLower() != "csv" && format.ToLower() != "json"))
            {
                return BadRequest(new { Error = "Format must be 'csv' or 'json'" });
            }
            
            // Set a reasonable limit for exports
            if (query.Limit <= 0 || query.Limit > 10000)
            {
                query.Limit = 10000;
            }
            
            byte[] fileContent;
            string contentType;
            string fileName;
            
            if (format.ToLower() == "csv")
            {
                fileContent = await _logService.ExportLogsToCsvAsync(query);
                contentType = "text/csv";
                fileName = $"logs-export-{DateTime.UtcNow:yyyyMMdd-HHmmss}.csv";
            }
            else
            {
                fileContent = await _logService.ExportLogsToJsonAsync(query);
                contentType = "application/json";
                fileName = $"logs-export-{DateTime.UtcNow:yyyyMMdd-HHmmss}.json";
            }
            
            return File(fileContent, contentType, fileName);
        }
    }
} 