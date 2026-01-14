using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using Backend.Services;

namespace Backend.Controllers;

/// <summary>
/// Controller for correlation engine operations and results
/// </summary>
[ApiController]
[Route("api/[controller]")]
[Authorize]
public class CorrelationController : ControllerBase
{
    private readonly ICorrelationService _correlationService;
    private readonly ILogger<CorrelationController> _logger;

    public CorrelationController(
        ICorrelationService correlationService,
        ILogger<CorrelationController> logger)
    {
        _correlationService = correlationService;
        _logger = logger;
    }

    /// <summary>
    /// Get correlation statistics
    /// </summary>
    [HttpGet("statistics")]
    [ProducesResponseType(StatusCodes.Status200OK)]
    public async Task<ActionResult<object>> GetStatistics(
        [FromQuery] DateTime? startDate = null,
        [FromQuery] DateTime? endDate = null)
    {
        try
        {
            var statistics = await _correlationService.GetStatisticsAsync(startDate, endDate);
            return Ok(statistics);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting correlation statistics");
            return StatusCode(500, new { Error = "Failed to get correlation statistics" });
        }
    }

    /// <summary>
    /// Get active correlation rules
    /// </summary>
    [HttpGet("rules")]
    [ProducesResponseType(StatusCodes.Status200OK)]
    public async Task<ActionResult<object>> GetRules()
    {
        try
        {
            var rules = await _correlationService.GetRulesAsync();
            return Ok(rules);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting correlation rules");
            return StatusCode(500, new { Error = "Failed to get correlation rules" });
        }
    }

    /// <summary>
    /// Manually trigger correlation for a specific log entry
    /// </summary>
    [HttpPost("trigger/{logEntryId}")]
    [Authorize(Roles = "Admin")]
    [ProducesResponseType(StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status404NotFound)]
    public async Task<ActionResult<object>> TriggerCorrelation(string logEntryId)
    {
        try
        {
            var result = await _correlationService.TriggerCorrelationAsync(logEntryId);
            
            if (result == null)
                return NotFound(new { Error = "Log entry not found or not normalized" });

            return Ok(new
            {
                LogEntryId = logEntryId,
                CorrelationResult = new
                {
                    result.RuleName,
                    result.RuleDescription,
                    result.CorrelationId,
                    result.Type,
                    result.Confidence,
                    CorrelatedLogCount = result.CorrelatedLogs.Count,
                    result.Metadata
                }
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error triggering correlation for log {LogEntryId}", logEntryId);
            return StatusCode(500, new { Error = "Failed to trigger correlation" });
        }
    }
}
