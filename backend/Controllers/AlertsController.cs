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

namespace Backend.Controllers
{
    /// <summary>
    /// Controller for managing alerts
    /// </summary>
    [ApiController]
    [Route("api/[controller]")]
    [Authorize]
    public class AlertsController : ControllerBase
    {
        private readonly IAlertService _alertService;
        private readonly IAgentService _agentService;
        private readonly ILogger<AlertsController> _logger;
        
        /// <summary>
        /// Initializes a new instance of the <see cref="AlertsController"/> class
        /// </summary>
        /// <param name="alertService">The alert service</param>
        /// <param name="agentService">The agent service</param>
        /// <param name="logger">The logger</param>
        public AlertsController(
            IAlertService alertService,
            IAgentService agentService,
            ILogger<AlertsController> logger)
        {
            _alertService = alertService ?? throw new ArgumentNullException(nameof(alertService));
            _agentService = agentService ?? throw new ArgumentNullException(nameof(agentService));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }
        
        /// <summary>
        /// Gets all alerts with optional filtering
        /// </summary>
        /// <param name="query">Query parameters</param>
        /// <returns>Alerts matching the query</returns>
        [HttpGet]
        [ProducesResponseType(StatusCodes.Status200OK)]
        public async Task<ActionResult<PaginatedResult<AlertDto>>> GetAlerts([FromQuery] AlertQueryDto query)
        {
            _logger.LogInformation("Getting alerts with query: {Query}", query);
            
            // Set default values if not provided
            if (query.Limit <= 0)
            {
                query.Limit = 100;
            }
            
            if (query.Limit > 1000)
            {
                query.Limit = 1000;
            }
            
            var alerts = await _alertService.SearchAlertsAsync(query);
            
            _logger.LogInformation("Found {Count} alerts matching query", alerts.TotalCount);
            return Ok(alerts);
        }
        
        /// <summary>
        /// Gets a specific alert by ID
        /// </summary>
        /// <param name="id">The alert ID</param>
        /// <returns>The alert</returns>
        [HttpGet("{id}")]
        [ProducesResponseType(StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status404NotFound)]
        public async Task<ActionResult<AlertDto>> GetAlertById(string id)
        {
            var alert = await _alertService.GetAlertByIdAsync(id);
            if (alert == null)
            {
                return NotFound(new { Error = $"Alert with ID {id} not found" });
            }
            
            return Ok(alert);
        }
        
        /// <summary>
        /// Gets alerts by severity
        /// </summary>
        /// <param name="severity">The alert severity</param>
        /// <param name="limit">Maximum number of alerts to return</param>
        /// <param name="offset">Offset for pagination</param>
        /// <returns>Alerts with the specified severity</returns>
        [HttpGet("severity/{severity}")]
        [ProducesResponseType(StatusCodes.Status200OK)]
        public async Task<ActionResult<PaginatedResult<AlertDto>>> GetAlertsBySeverity(
            string severity, 
            [FromQuery] int limit = 100, 
            [FromQuery] int offset = 0)
        {
            var query = new AlertQueryDto
            {
                Severity = severity,
                Limit = limit,
                Offset = offset
            };
            
            var alerts = await _alertService.SearchAlertsAsync(query);
            return Ok(alerts);
        }
        
        /// <summary>
        /// Gets alerts by status
        /// </summary>
        /// <param name="status">The alert status</param>
        /// <param name="limit">Maximum number of alerts to return</param>
        /// <param name="offset">Offset for pagination</param>
        /// <returns>Alerts with the specified status</returns>
        [HttpGet("status/{status}")]
        [ProducesResponseType(StatusCodes.Status200OK)]
        public async Task<ActionResult<PaginatedResult<AlertDto>>> GetAlertsByStatus(
            string status, 
            [FromQuery] int limit = 100, 
            [FromQuery] int offset = 0)
        {
            var query = new AlertQueryDto
            {
                Status = status,
                Limit = limit,
                Offset = offset
            };
            
            var alerts = await _alertService.SearchAlertsAsync(query);
            return Ok(alerts);
        }
        
        /// <summary>
        /// Gets alerts by agent
        /// </summary>
        /// <param name="agentId">The agent ID</param>
        /// <param name="limit">Maximum number of alerts to return</param>
        /// <param name="offset">Offset for pagination</param>
        /// <returns>Alerts for the specified agent</returns>
        [HttpGet("agent/{agentId}")]
        [ProducesResponseType(StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status404NotFound)]
        public async Task<ActionResult<PaginatedResult<AlertDto>>> GetAlertsByAgent(
            string agentId, 
            [FromQuery] int limit = 100, 
            [FromQuery] int offset = 0)
        {
            var agent = await _agentService.GetAgentByIdAsync(agentId);
            if (agent == null)
            {
                return NotFound(new { Error = $"Agent with ID {agentId} not found" });
            }
            
            var query = new AlertQueryDto
            {
                AgentId = agentId,
                Limit = limit,
                Offset = offset
            };
            
            var alerts = await _alertService.SearchAlertsAsync(query);
            return Ok(alerts);
        }
        
        /// <summary>
        /// Creates a new alert
        /// </summary>
        /// <param name="createAlertDto">The alert data</param>
        /// <returns>The created alert</returns>
        [HttpPost]
        [ProducesResponseType(typeof(AlertDto), StatusCodes.Status201Created)]
        [ProducesResponseType(StatusCodes.Status400BadRequest)]
        public async Task<IActionResult> CreateAlert([FromBody] CreateAlertDto createAlertDto)
        {
            if (!ModelState.IsValid)
            {
                return BadRequest(ModelState);
            }
            
            // Add audit information
            createAlertDto.GeneratedBy = User.Identity?.Name ?? "System";
            
            var alert = await _alertService.CreateAlertAsync(createAlertDto);
            return CreatedAtAction(nameof(GetAlertById), new { id = alert.Id }, alert);
        }
        
        /// <summary>
        /// Updates an alert status
        /// </summary>
        /// <param name="id">The alert ID</param>
        /// <param name="updateStatusDto">The status update data</param>
        /// <returns>The updated alert</returns>
        [HttpPut("{id}/status")]
        [ProducesResponseType(typeof(AlertDto), StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status404NotFound)]
        public async Task<IActionResult> UpdateAlertStatus(string id, [FromBody] UpdateAlertStatusDto updateStatusDto)
        {
            var alert = await _alertService.GetAlertByIdAsync(id);
            if (alert == null)
            {
                return NotFound(new { Error = $"Alert with ID {id} not found" });
            }
            
            // Add audit information
            updateStatusDto.UpdatedBy = User.Identity?.Name ?? "System";
            updateStatusDto.UpdatedAt = DateTime.UtcNow;
            
            var updatedAlert = await _alertService.UpdateAlertStatusAsync(id, updateStatusDto);
            return Ok(updatedAlert);
        }
        
        /// <summary>
        /// Adds a comment to an alert
        /// </summary>
        /// <param name="id">The alert ID</param>
        /// <param name="commentDto">Comment data</param>
        /// <returns>The updated alert</returns>
        [HttpPost("{id}/comments")]
        [Authorize(Roles = "Admin,Operator")]
        [ProducesResponseType(StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status400BadRequest)]
        [ProducesResponseType(StatusCodes.Status404NotFound)]
        public async Task<ActionResult<AlertDto>> AddAlertComment(
            string id, 
            [FromBody] AddAlertCommentDto commentDto)
        {
            if (commentDto == null || string.IsNullOrEmpty(commentDto.Comment))
            {
                return BadRequest(new { Error = "Comment is required" });
            }
            
            var alert = await _alertService.GetAlertByIdAsync(id);
            if (alert == null)
            {
                return NotFound(new { Error = $"Alert with ID {id} not found" });
            }
            
            // Add author information
            commentDto.Author = User.Identity?.Name ?? "System";
            commentDto.CreatedAt = DateTime.UtcNow;
            
            var updatedAlert = await _alertService.AddAlertCommentAsync(id, commentDto);
            return Ok(updatedAlert);
        }
        
        /// <summary>
        /// Bulk updates alert statuses
        /// </summary>
        /// <param name="bulkUpdateDto">Bulk update data</param>
        /// <returns>Result of the bulk update</returns>
        [HttpPut("bulk-update")]
        [Authorize(Roles = "Admin,Operator")]
        [ProducesResponseType(StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status400BadRequest)]
        public async Task<ActionResult<BulkUpdateResultDto>> BulkUpdateAlerts(
            [FromBody] BulkUpdateAlertsDto bulkUpdateDto)
        {
            if (bulkUpdateDto == null || bulkUpdateDto.AlertIds == null || bulkUpdateDto.AlertIds.Count == 0)
            {
                return BadRequest(new { Error = "Alert IDs are required" });
            }
            
            if (string.IsNullOrEmpty(bulkUpdateDto.Status))
            {
                return BadRequest(new { Error = "Status is required" });
            }
            
            // Add audit information
            bulkUpdateDto.UpdatedBy = User.Identity?.Name ?? "System";
            bulkUpdateDto.UpdatedAt = DateTime.UtcNow;
            
            var result = await _alertService.BulkUpdateAlertStatusAsync(bulkUpdateDto);
            return Ok(result);
        }
        
        /// <summary>
        /// Gets alert summary statistics
        /// </summary>
        /// <param name="start">Start time (UTC)</param>
        /// <param name="end">End time (UTC)</param>
        /// <returns>Alert summary statistics</returns>
        [HttpGet("summary")]
        [ProducesResponseType(StatusCodes.Status200OK)]
        public async Task<ActionResult<AlertSummaryDto>> GetAlertSummary(
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
            
            var summary = await _alertService.GetAlertSummaryAsync(start.Value, end.Value);
            return Ok(summary);
        }
        
        /// <summary>
        /// Gets alert trends
        /// </summary>
        /// <param name="start">Start time</param>
        /// <param name="end">End time</param>
        /// <param name="interval">Time interval (hour, day, week, month)</param>
        /// <returns>Alert trends</returns>
        [HttpGet("trends")]
        [ProducesResponseType(StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status400BadRequest)]
        public async Task<ActionResult<AlertTrendsDto>> GetAlertTrends(
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
            
            var trends = await _alertService.GetAlertTrendsAsync(start.Value, end.Value, timeInterval.ToString().ToLower());
            return Ok(trends);
        }
        
        /// <summary>
        /// Gets related alerts for a specific alert
        /// </summary>
        /// <param name="id">The alert ID</param>
        /// <param name="maxResults">Maximum number of related alerts to return</param>
        /// <returns>Related alerts</returns>
        [HttpGet("{id}/related")]
        [ProducesResponseType(StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status404NotFound)]
        public async Task<ActionResult<IEnumerable<AlertDto>>> GetRelatedAlerts(
            string id, 
            [FromQuery] int maxResults = 10)
        {
            var alert = await _alertService.GetAlertByIdAsync(id);
            if (alert == null)
            {
                return NotFound(new { Error = $"Alert with ID {id} not found" });
            }
            
            var relatedAlerts = await _alertService.GetRelatedAlertsAsync(id, maxResults);
            return Ok(relatedAlerts);
        }
        
        /// <summary>
        /// Exports alerts to a file
        /// </summary>
        /// <param name="query">Query parameters</param>
        /// <param name="format">Export format (csv, json)</param>
        /// <returns>File with exported alerts</returns>
        [HttpGet("export")]
        [ProducesResponseType(StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status400BadRequest)]
        public async Task<ActionResult> ExportAlerts(
            [FromQuery] AlertQueryDto query, 
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
                fileContent = await _alertService.ExportAlertsToCsvAsync(query);
                contentType = "text/csv";
                fileName = $"alerts-export-{DateTime.UtcNow:yyyyMMdd-HHmmss}.csv";
            }
            else
            {
                fileContent = await _alertService.ExportAlertsToJsonAsync(query);
                contentType = "application/json";
                fileName = $"alerts-export-{DateTime.UtcNow:yyyyMMdd-HHmmss}.json";
            }
            
            return File(fileContent, contentType, fileName);
        }
    }
} 