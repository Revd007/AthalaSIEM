using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using Backend.Services;
using Backend.Infrastructure.Data.Repositories;
using Microsoft.EntityFrameworkCore;

namespace Backend.Controllers;

/// <summary>
/// Controller for log normalization operations and statistics
/// </summary>
[ApiController]
[Route("api/[controller]")]
[Authorize]
public class NormalizationController : ControllerBase
{
    private readonly INormalizationService _normalizationService;
    private readonly INormalizedLogRepository _normalizedLogRepository;
    private readonly ILogger<NormalizationController> _logger;

    public NormalizationController(
        INormalizationService normalizationService,
        INormalizedLogRepository normalizedLogRepository,
        ILogger<NormalizationController> logger)
    {
        _normalizationService = normalizationService;
        _normalizedLogRepository = normalizedLogRepository;
        _logger = logger;
    }

    /// <summary>
    /// Get normalization statistics
    /// </summary>
    [HttpGet("statistics")]
    [ProducesResponseType(StatusCodes.Status200OK)]
    public async Task<ActionResult<object>> GetStatistics(
        [FromQuery] DateTime? startDate = null,
        [FromQuery] DateTime? endDate = null)
    {
        try
        {
            var statistics = await _normalizationService.GetStatisticsAsync(startDate, endDate);
            return Ok(statistics);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting normalization statistics");
            return StatusCode(500, new { Error = "Failed to get normalization statistics" });
        }
    }

    /// <summary>
    /// Get normalized logs with ECS fields
    /// </summary>
    [HttpGet("normalized")]
    [ProducesResponseType(StatusCodes.Status200OK)]
    public async Task<ActionResult<object>> GetNormalizedLogs(
        [FromQuery] int page = 1,
        [FromQuery] int pageSize = 50,
        [FromQuery] string? eventType = null,
        [FromQuery] string? sourceIp = null,
        [FromQuery] int? minSeverity = null,
        [FromQuery] DateTime? startDate = null,
        [FromQuery] DateTime? endDate = null)
    {
        try
        {
            var start = startDate ?? DateTime.UtcNow.AddDays(-1);
            var end = endDate ?? DateTime.UtcNow;

            // Use repository search method
            var normalizedLogs = await _normalizedLogRepository.SearchAsync(
                sourceIp: sourceIp,
                destinationIp: null,
                processName: null,
                userName: null,
                startTime: start,
                endTime: end,
                limit: pageSize * 10, // Get more to filter
                cancellationToken: default);

            // Apply additional filters
            var filtered = normalizedLogs.AsQueryable();

            if (!string.IsNullOrEmpty(eventType))
                filtered = filtered.Where(nl => nl.EventType == eventType);

            if (minSeverity.HasValue)
                filtered = filtered.Where(nl => nl.SiemSeverity >= minSeverity.Value);

            var totalCount = filtered.Count();
            var logs = filtered
                .OrderByDescending(nl => nl.Timestamp)
                .Skip((page - 1) * pageSize)
                .Take(pageSize)
                .Select(nl => new
                {
                    nl.Id,
                    nl.LogEntryId,
                    nl.Timestamp,
                    nl.SourceIp,
                    nl.DestinationIp,
                    nl.EventType,
                    nl.EventAction,
                    nl.EventCategory,
                    Severity = nl.SiemSeverity,
                    nl.UserName,
                    nl.ProcessName,
                    nl.ProcessId,
                    nl.Protocol,
                    nl.AgentId,
                    nl.HostName
                })
                .ToList();

            return Ok(new
            {
                Items = logs,
                TotalCount = totalCount,
                Page = page,
                PageSize = pageSize,
                TotalPages = (int)Math.Ceiling(totalCount / (double)pageSize)
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting normalized logs");
            return StatusCode(500, new { Error = "Failed to get normalized logs" });
        }
    }
}
