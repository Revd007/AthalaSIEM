using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging;
using Backend.Data;
using Backend.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace Backend.Controllers
{
    /// <summary>
    /// Controller for File Integrity Monitoring operations
    /// </summary>
    [ApiController]
    [Route("api/[controller]")]
    [Authorize]
    public class FileIntegrityController : ControllerBase
    {
        private readonly ApplicationDbContext _context;
        private readonly ILogger<FileIntegrityController> _logger;

        public FileIntegrityController(
            ApplicationDbContext context,
            ILogger<FileIntegrityController> logger)
        {
            _context = context ?? throw new ArgumentNullException(nameof(context));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }

        /// <summary>
        /// Gets File Integrity events with filtering and pagination
        /// </summary>
        /// <param name="agentId">Filter by agent ID</param>
        /// <param name="severity">Filter by severity</param>
        /// <param name="changeType">Filter by change type</param>
        /// <param name="acknowledged">Filter by acknowledgment status</param>
        /// <param name="startDate">Filter events from this date</param>
        /// <param name="endDate">Filter events to this date</param>
        /// <param name="page">Page number (1-based)</param>
        /// <param name="pageSize">Number of items per page</param>
        /// <returns>Paginated list of FIM events</returns>
        [HttpGet("events")]
        public async Task<ActionResult<PagedResult<FileIntegrityEventDto>>> GetFileIntegrityEvents(
            string? agentId = null,
            string? severity = null,
            string? changeType = null,
            bool? acknowledged = null,
            DateTime? startDate = null,
            DateTime? endDate = null,
            int page = 1,
            int pageSize = 50)
        {
            try
            {
                var query = _context.FileIntegrityEvents
                    .Include(e => e.Agent)
                    .AsQueryable();

                // Apply filters
                if (!string.IsNullOrEmpty(agentId))
                    query = query.Where(e => e.AgentId == agentId);

                if (!string.IsNullOrEmpty(severity))
                    query = query.Where(e => e.Severity == severity);

                if (!string.IsNullOrEmpty(changeType))
                    query = query.Where(e => e.ChangeType == changeType);

                if (acknowledged.HasValue)
                    query = query.Where(e => e.IsAcknowledged == acknowledged.Value);

                if (startDate.HasValue)
                    query = query.Where(e => e.DetectedAt >= startDate.Value);

                if (endDate.HasValue)
                    query = query.Where(e => e.DetectedAt <= endDate.Value);

                // Get total count
                var totalCount = await query.CountAsync();

                // Apply pagination and ordering
                var events = await query
                    .OrderByDescending(e => e.DetectedAt)
                    .Skip((page - 1) * pageSize)
                    .Take(pageSize)
                    .Select(e => new FileIntegrityEventDto
                    {
                        Id = e.Id,
                        AgentId = e.AgentId,
                        AgentName = e.Agent != null ? e.Agent.Name : "Unknown",
                        FilePath = e.FilePath,
                        ChangeType = e.ChangeType,
                        BaselineHash = e.BaselineHash,
                        CurrentHash = e.CurrentHash,
                        BaselineSize = e.BaselineSize,
                        CurrentSize = e.CurrentSize,
                        BaselineModified = e.BaselineModified,
                        CurrentModified = e.CurrentModified,
                        FileAttributes = e.FileAttributes,
                        Severity = e.Severity,
                        DetectedAt = e.DetectedAt,
                        ProcessedAt = e.ProcessedAt,
                        IsAcknowledged = e.IsAcknowledged,
                        AcknowledgedBy = e.AcknowledgedBy,
                        AcknowledgedAt = e.AcknowledgedAt,
                        Details = e.Details
                    })
                    .ToListAsync();

                var result = new PagedResult<FileIntegrityEventDto>
                {
                    Items = events,
                    TotalCount = totalCount,
                    Page = page,
                    PageSize = pageSize,
                    TotalPages = (int)Math.Ceiling((double)totalCount / pageSize)
                };

                return Ok(result);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error retrieving file integrity events");
                return StatusCode(500, "An error occurred while retrieving file integrity events");
            }
        }

        /// <summary>
        /// Gets a specific File Integrity event by ID
        /// </summary>
        /// <param name="id">Event ID</param>
        /// <returns>File Integrity event details</returns>
        [HttpGet("events/{id}")]
        public async Task<ActionResult<FileIntegrityEventDto>> GetFileIntegrityEvent(string id)
        {
            try
            {
                var fimEvent = await _context.FileIntegrityEvents
                    .Include(e => e.Agent)
                    .FirstOrDefaultAsync(e => e.Id == id);

                if (fimEvent == null)
                {
                    return NotFound($"File integrity event with ID {id} not found");
                }

                var eventDto = new FileIntegrityEventDto
                {
                    Id = fimEvent.Id,
                    AgentId = fimEvent.AgentId,
                    AgentName = fimEvent.Agent?.Name ?? "Unknown",
                    FilePath = fimEvent.FilePath,
                    ChangeType = fimEvent.ChangeType,
                    BaselineHash = fimEvent.BaselineHash,
                    CurrentHash = fimEvent.CurrentHash,
                    BaselineSize = fimEvent.BaselineSize,
                    CurrentSize = fimEvent.CurrentSize,
                    BaselineModified = fimEvent.BaselineModified,
                    CurrentModified = fimEvent.CurrentModified,
                    FileAttributes = fimEvent.FileAttributes,
                    Severity = fimEvent.Severity,
                    DetectedAt = fimEvent.DetectedAt,
                    ProcessedAt = fimEvent.ProcessedAt,
                    IsAcknowledged = fimEvent.IsAcknowledged,
                    AcknowledgedBy = fimEvent.AcknowledgedBy,
                    AcknowledgedAt = fimEvent.AcknowledgedAt,
                    Details = fimEvent.Details
                };

                return Ok(eventDto);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error retrieving file integrity event {EventId}", id);
                return StatusCode(500, "An error occurred while retrieving the file integrity event");
            }
        }

        /// <summary>
        /// Acknowledges one or more File Integrity events
        /// </summary>
        /// <param name="request">Acknowledgment request</param>
        /// <returns>Success response</returns>
        [HttpPost("events/acknowledge")]
        public async Task<ActionResult> AcknowledgeEvents([FromBody] AcknowledgeFimEventRequest request)
        {
            try
            {
                if (request.EventIds == null || !request.EventIds.Any())
                {
                    return BadRequest("Event IDs are required");
                }

                var currentUser = User.Identity?.Name ?? "System";
                var acknowledgedAt = DateTime.UtcNow;

                var events = await _context.FileIntegrityEvents
                    .Where(e => request.EventIds.Contains(e.Id))
                    .ToListAsync();

                if (!events.Any())
                {
                    return NotFound("No events found with the provided IDs");
                }

                foreach (var fimEvent in events)
                {
                    fimEvent.IsAcknowledged = true;
                    fimEvent.AcknowledgedBy = currentUser;
                    fimEvent.AcknowledgedAt = acknowledgedAt;
                }

                await _context.SaveChangesAsync();

                _logger.LogInformation("User {User} acknowledged {Count} file integrity events", 
                    currentUser, events.Count);

                return Ok(new { Message = $"Successfully acknowledged {events.Count} events" });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error acknowledging file integrity events");
                return StatusCode(500, "An error occurred while acknowledging events");
            }
        }

        /// <summary>
        /// Gets File Integrity rules
        /// </summary>
        /// <returns>List of FIM rules</returns>
        [HttpGet("rules")]
        [Authorize(Roles = "Admin,Analyst")]
        public async Task<ActionResult<List<FileIntegrityRuleDto>>> GetFileIntegrityRules()
        {
            try
            {
                var rules = await _context.FileIntegrityRules
                    .OrderBy(r => r.Name)
                    .Select(r => new FileIntegrityRuleDto
                    {
                        Id = r.Id,
                        Name = r.Name,
                        Description = r.Description,
                        IsEnabled = r.IsEnabled,
                        MonitoredPaths = r.MonitoredPaths,
                        ExcludePatterns = r.ExcludePatterns,
                        RealTimeMonitoring = r.RealTimeMonitoring,
                        ScanIntervalMinutes = r.ScanIntervalMinutes,
                        Severity = r.Severity,
                        AlertOnCreation = r.AlertOnCreation,
                        AlertOnModification = r.AlertOnModification,
                        AlertOnDeletion = r.AlertOnDeletion,
                        AlertOnRename = r.AlertOnRename,
                        CreatedAt = r.CreatedAt,
                        UpdatedAt = r.UpdatedAt,
                        CreatedBy = r.CreatedBy,
                        TargetAgents = r.TargetAgents
                    })
                    .ToListAsync();

                return Ok(rules);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error retrieving file integrity rules");
                return StatusCode(500, "An error occurred while retrieving file integrity rules");
            }
        }

        /// <summary>
        /// Creates a new File Integrity rule
        /// </summary>
        /// <param name="request">Rule creation request</param>
        /// <returns>Created rule</returns>
        [HttpPost("rules")]
        [Authorize(Roles = "Admin,Analyst")]
        public async Task<ActionResult<FileIntegrityRuleDto>> CreateFileIntegrityRule([FromBody] CreateFileIntegrityRuleDto request)
        {
            try
            {
                var currentUser = User.Identity?.Name ?? "System";

                var rule = new FileIntegrityRule
                {
                    Name = request.Name,
                    Description = request.Description,
                    IsEnabled = request.IsEnabled,
                    MonitoredPaths = request.MonitoredPaths,
                    ExcludePatterns = request.ExcludePatterns,
                    RealTimeMonitoring = request.RealTimeMonitoring,
                    ScanIntervalMinutes = request.ScanIntervalMinutes,
                    Severity = request.Severity,
                    AlertOnCreation = request.AlertOnCreation,
                    AlertOnModification = request.AlertOnModification,
                    AlertOnDeletion = request.AlertOnDeletion,
                    AlertOnRename = request.AlertOnRename,
                    CreatedBy = currentUser,
                    TargetAgents = request.TargetAgents
                };

                _context.FileIntegrityRules.Add(rule);
                await _context.SaveChangesAsync();

                var ruleDto = new FileIntegrityRuleDto
                {
                    Id = rule.Id,
                    Name = rule.Name,
                    Description = rule.Description,
                    IsEnabled = rule.IsEnabled,
                    MonitoredPaths = rule.MonitoredPaths,
                    ExcludePatterns = rule.ExcludePatterns,
                    RealTimeMonitoring = rule.RealTimeMonitoring,
                    ScanIntervalMinutes = rule.ScanIntervalMinutes,
                    Severity = rule.Severity,
                    AlertOnCreation = rule.AlertOnCreation,
                    AlertOnModification = rule.AlertOnModification,
                    AlertOnDeletion = rule.AlertOnDeletion,
                    AlertOnRename = rule.AlertOnRename,
                    CreatedAt = rule.CreatedAt,
                    UpdatedAt = rule.UpdatedAt,
                    CreatedBy = rule.CreatedBy,
                    TargetAgents = rule.TargetAgents
                };

                _logger.LogInformation("User {User} created file integrity rule {RuleName}", currentUser, rule.Name);

                return CreatedAtAction(nameof(GetFileIntegrityRule), new { id = rule.Id }, ruleDto);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error creating file integrity rule");
                return StatusCode(500, "An error occurred while creating the file integrity rule");
            }
        }

        /// <summary>
        /// Gets a specific File Integrity rule by ID
        /// </summary>
        /// <param name="id">Rule ID</param>
        /// <returns>File Integrity rule details</returns>
        [HttpGet("rules/{id}")]
        [Authorize(Roles = "Admin,Analyst")]
        public async Task<ActionResult<FileIntegrityRuleDto>> GetFileIntegrityRule(string id)
        {
            try
            {
                var rule = await _context.FileIntegrityRules.FindAsync(id);

                if (rule == null)
                {
                    return NotFound($"File integrity rule with ID {id} not found");
                }

                var ruleDto = new FileIntegrityRuleDto
                {
                    Id = rule.Id,
                    Name = rule.Name,
                    Description = rule.Description,
                    IsEnabled = rule.IsEnabled,
                    MonitoredPaths = rule.MonitoredPaths,
                    ExcludePatterns = rule.ExcludePatterns,
                    RealTimeMonitoring = rule.RealTimeMonitoring,
                    ScanIntervalMinutes = rule.ScanIntervalMinutes,
                    Severity = rule.Severity,
                    AlertOnCreation = rule.AlertOnCreation,
                    AlertOnModification = rule.AlertOnModification,
                    AlertOnDeletion = rule.AlertOnDeletion,
                    AlertOnRename = rule.AlertOnRename,
                    CreatedAt = rule.CreatedAt,
                    UpdatedAt = rule.UpdatedAt,
                    CreatedBy = rule.CreatedBy,
                    TargetAgents = rule.TargetAgents
                };

                return Ok(ruleDto);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error retrieving file integrity rule {RuleId}", id);
                return StatusCode(500, "An error occurred while retrieving the file integrity rule");
            }
        }

        /// <summary>
        /// Updates a File Integrity rule
        /// </summary>
        /// <param name="id">Rule ID</param>
        /// <param name="request">Rule update request</param>
        /// <returns>Updated rule</returns>
        [HttpPut("rules/{id}")]
        [Authorize(Roles = "Admin,Analyst")]
        public async Task<ActionResult<FileIntegrityRuleDto>> UpdateFileIntegrityRule(string id, [FromBody] CreateFileIntegrityRuleDto request)
        {
            try
            {
                var rule = await _context.FileIntegrityRules.FindAsync(id);

                if (rule == null)
                {
                    return NotFound($"File integrity rule with ID {id} not found");
                }

                rule.Name = request.Name;
                rule.Description = request.Description;
                rule.IsEnabled = request.IsEnabled;
                rule.MonitoredPaths = request.MonitoredPaths;
                rule.ExcludePatterns = request.ExcludePatterns;
                rule.RealTimeMonitoring = request.RealTimeMonitoring;
                rule.ScanIntervalMinutes = request.ScanIntervalMinutes;
                rule.Severity = request.Severity;
                rule.AlertOnCreation = request.AlertOnCreation;
                rule.AlertOnModification = request.AlertOnModification;
                rule.AlertOnDeletion = request.AlertOnDeletion;
                rule.AlertOnRename = request.AlertOnRename;
                rule.TargetAgents = request.TargetAgents;
                rule.UpdatedAt = DateTime.UtcNow;

                await _context.SaveChangesAsync();

                var ruleDto = new FileIntegrityRuleDto
                {
                    Id = rule.Id,
                    Name = rule.Name,
                    Description = rule.Description,
                    IsEnabled = rule.IsEnabled,
                    MonitoredPaths = rule.MonitoredPaths,
                    ExcludePatterns = rule.ExcludePatterns,
                    RealTimeMonitoring = rule.RealTimeMonitoring,
                    ScanIntervalMinutes = rule.ScanIntervalMinutes,
                    Severity = rule.Severity,
                    AlertOnCreation = rule.AlertOnCreation,
                    AlertOnModification = rule.AlertOnModification,
                    AlertOnDeletion = rule.AlertOnDeletion,
                    AlertOnRename = rule.AlertOnRename,
                    CreatedAt = rule.CreatedAt,
                    UpdatedAt = rule.UpdatedAt,
                    CreatedBy = rule.CreatedBy,
                    TargetAgents = rule.TargetAgents
                };

                _logger.LogInformation("User {User} updated file integrity rule {RuleName}", User.Identity?.Name, rule.Name);

                return Ok(ruleDto);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error updating file integrity rule {RuleId}", id);
                return StatusCode(500, "An error occurred while updating the file integrity rule");
            }
        }

        /// <summary>
        /// Deletes a File Integrity rule
        /// </summary>
        /// <param name="id">Rule ID</param>
        /// <returns>Success response</returns>
        [HttpDelete("rules/{id}")]
        [Authorize(Roles = "Admin")]
        public async Task<ActionResult> DeleteFileIntegrityRule(string id)
        {
            try
            {
                var rule = await _context.FileIntegrityRules.FindAsync(id);

                if (rule == null)
                {
                    return NotFound($"File integrity rule with ID {id} not found");
                }

                _context.FileIntegrityRules.Remove(rule);
                await _context.SaveChangesAsync();

                _logger.LogInformation("User {User} deleted file integrity rule {RuleName}", User.Identity?.Name, rule.Name);

                return Ok(new { Message = "File integrity rule deleted successfully" });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error deleting file integrity rule {RuleId}", id);
                return StatusCode(500, "An error occurred while deleting the file integrity rule");
            }
        }

        /// <summary>
        /// Gets File Integrity statistics
        /// </summary>
        /// <param name="agentId">Optional agent ID filter</param>
        /// <param name="days">Number of days to include in statistics (default: 7)</param>
        /// <returns>FIM statistics</returns>
        [HttpGet("statistics")]
        public async Task<ActionResult<object>> GetFileIntegrityStatistics(string? agentId = null, int days = 7)
        {
            try
            {
                var startDate = DateTime.UtcNow.AddDays(-days);
                var query = _context.FileIntegrityEvents.Where(e => e.DetectedAt >= startDate);

                if (!string.IsNullOrEmpty(agentId))
                    query = query.Where(e => e.AgentId == agentId);

                var statistics = new
                {
                    TotalEvents = await query.CountAsync(),
                    EventsBySeverity = await query
                        .GroupBy(e => e.Severity)
                        .Select(g => new { Severity = g.Key, Count = g.Count() })
                        .ToListAsync(),
                    EventsByChangeType = await query
                        .GroupBy(e => e.ChangeType)
                        .Select(g => new { ChangeType = g.Key, Count = g.Count() })
                        .ToListAsync(),
                    EventsByAgent = await query
                        .Include(e => e.Agent)
                        .GroupBy(e => new { e.AgentId, e.Agent!.Name })
                        .Select(g => new { AgentId = g.Key.AgentId, AgentName = g.Key.Name, Count = g.Count() })
                        .ToListAsync(),
                    AcknowledgedEvents = await query.CountAsync(e => e.IsAcknowledged),
                    UnacknowledgedEvents = await query.CountAsync(e => !e.IsAcknowledged),
                    EventsOverTime = await query
                        .GroupBy(e => e.DetectedAt.Date)
                        .Select(g => new { Date = g.Key, Count = g.Count() })
                        .OrderBy(x => x.Date)
                        .ToListAsync()
                };

                return Ok(statistics);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error retrieving file integrity statistics");
                return StatusCode(500, "An error occurred while retrieving file integrity statistics");
            }
        }
    }

    /// <summary>
    /// Paged result wrapper
    /// </summary>
    /// <typeparam name="T">Type of items</typeparam>
    public class PagedResult<T>
    {
        public List<T> Items { get; set; } = new();
        public int TotalCount { get; set; }
        public int Page { get; set; }
        public int PageSize { get; set; }
        public int TotalPages { get; set; }
    }
} 