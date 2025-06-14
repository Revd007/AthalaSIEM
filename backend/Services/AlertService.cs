using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging;
using Backend.Data;
using Backend.DTOs;
using Backend.Models;
using Backend.Data.Repositories;
using System.IO;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.Configuration;

namespace Backend.Services
{
    /// <summary>
    /// Enhanced Alert Service with multi-collector support and advanced notification capabilities
    /// </summary>
    public class AlertService : IAlertService
    {
        private readonly ILogger<AlertService> _logger;
        private readonly IAlertRepository _alertRepository;
        private readonly ApplicationDbContext _context;
        private readonly IThreatIntelligenceService _threatIntelligenceService;
        private readonly IConfiguration _configuration;
        private readonly Dictionary<string, CollectorAlertProfile> _collectorProfiles;
        private readonly Dictionary<string, AlertRule> _activeRules = new();

        /// <summary>
        /// Initializes a new instance of the <see cref="AlertService"/> class
        /// </summary>
        /// <param name="alertRepository">The alert repository</param>
        /// <param name="dbContext">The database context</param>
        /// <param name="logger">The logger</param>
        /// <param name="threatIntelligenceService">The threat intelligence service</param>
        /// <param name="configuration">The configuration</param>
        public AlertService(
            IAlertRepository alertRepository,
            ApplicationDbContext dbContext,
            ILogger<AlertService> logger,
            IThreatIntelligenceService threatIntelligenceService,
            IConfiguration configuration)
        {
            _alertRepository = alertRepository ?? throw new ArgumentNullException(nameof(alertRepository));
            _context = dbContext ?? throw new ArgumentNullException(nameof(dbContext));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _threatIntelligenceService = threatIntelligenceService ?? throw new ArgumentNullException(nameof(threatIntelligenceService));
            _configuration = configuration ?? throw new ArgumentNullException(nameof(configuration));

            _collectorProfiles = InitializeCollectorProfiles();
            LoadActiveRules();
        }

        /// <inheritdoc/>
        public async Task<IEnumerable<AlertDto>> GetAllAlertsAsync()
        {
            try
            {
                var alerts = await _alertRepository.GetAllAsync();
                return MapToAlertDtos(alerts);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting all alerts");
                throw;
            }
        }

        /// <inheritdoc/>
        public async Task<IEnumerable<AlertDto>> GetAllAlertsAsync(int limit = 100, int offset = 0)
        {
            try
            {
                var alerts = await _alertRepository.GetAllAsync();
                return alerts.Skip(offset).Take(limit).Select(alert => MapToAlertDto(alert));
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting all alerts with pagination");
                throw;
            }
        }

        /// <inheritdoc/>
        public async Task<AlertDto?> GetAlertByIdAsync(string id)
        {
            try
            {
                var alert = await _alertRepository.GetByIdAsync(id);
                return alert != null ? MapToAlertDto(alert) : null;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting alert by ID: {AlertId}", id);
                throw;
            }
        }

        /// <inheritdoc/>
        public async Task<IEnumerable<AlertDto>> GetAlertsByStatusAsync(AlertStatusModels status)
        {
            try
            {
                var alerts = await _alertRepository.GetByStatusAsync(status);
                return MapToAlertDtos(alerts);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting alerts by status: {Status}", status);
                throw;
            }
        }

        /// <inheritdoc/>
        public async Task<IEnumerable<AlertDto>> GetAlertsByStatusAsync(AlertStatusModels status, int limit = 100, int offset = 0)
        {
            try
            {
                var alerts = await _alertRepository.GetByStatusAsync(status);
                return alerts.Skip(offset).Take(limit).Select(alert => MapToAlertDto(alert));
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting alerts by status with pagination: {Status}", status);
                throw;
            }
        }

        /// <inheritdoc/>
        public async Task<IEnumerable<AlertDto>> GetAlertsBySeverityAsync(Models.SeverityModels severity)
        {
            try
            {
                var alerts = await _alertRepository.GetBySeverityAsync(severity);
                return MapToAlertDtos(alerts);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting alerts by severity: {Severity}", severity);
                throw;
            }
        }

        /// <inheritdoc/>
        public async Task<IEnumerable<AlertDto>> GetAlertsBySeverityAsync(Models.SeverityModels severity, int limit = 100, int offset = 0)
        {
            try
            {
                var alerts = await _alertRepository.GetBySeverityAsync(severity);
                return alerts.Skip(offset).Take(limit).Select(alert => MapToAlertDto(alert));
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting alerts by severity with pagination: {Severity}", severity);
                throw;
            }
        }

        /// <inheritdoc/>
        public async Task<IEnumerable<AlertDto>> GetAlertsByAgentIdAsync(string agentId)
        {
            try
            {
                var alerts = await _alertRepository.GetByAgentIdAsync(agentId);
                return MapToAlertDtos(alerts);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting alerts by agent ID: {AgentId}", agentId);
                throw;
            }
        }

        /// <inheritdoc/>
        public async Task<IEnumerable<AlertDto>> GetAlertsByUserIdAsync(string userId)
        {
            try
            {
                var alerts = await _alertRepository.GetByUserIdAsync(userId);
                return MapToAlertDtos(alerts);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting alerts by user ID: {UserId}", userId);
                throw;
            }
        }

        /// <inheritdoc/>
        public async Task<AlertDto> CreateAlertAsync(AlertDto alertDto)
        {
            try
            {
                var alert = MapToAlert(alertDto);
                alert.Id = Guid.NewGuid().ToString();
                alert.CreatedAt = DateTime.UtcNow;
                alert.UpdatedAt = DateTime.UtcNow;

                await _alertRepository.AddAsync(alert);
                _logger.LogInformation("Alert created: {AlertId}", alert.Id);

                return MapToAlertDto(alert);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error creating alert: {Title}", alertDto.Title);
                throw;
            }
        }

        /// <inheritdoc/>
        public async Task<AlertDto?> UpdateAlertAsync(string id, AlertDto alertDto)
        {
            try
            {
                var alert = await _alertRepository.GetByIdAsync(id);
                if (alert == null)
                {
                    _logger.LogWarning("Alert not found: {AlertId}", id);
                    return null!;
                }

                // Parse enum values
                AlertSeverityModels severity;
                AlertStatusModels status;

                if (!Enum.TryParse<AlertSeverityModels>(alertDto.Severity, out severity))
                {
                    severity = AlertSeverityModels.Low;
                }

                if (!Enum.TryParse<AlertStatusModels>(alertDto.Status, out status))
                {
                    status = AlertStatusModels.New;
                }

                // Update alert properties
                alert.Title = alertDto.Title;
                alert.Description = alertDto.Description ?? string.Empty;
                alert.Severity = severity;
                alert.Status = status;
                alert.Source = alertDto.Source ?? string.Empty;
                // Alert model doesn't have Category, Tags, Data properties
                // alert.Category = alertDto.Category;
                // alert.Tags = alertDto.Tags != null ? string.Join(",", alertDto.Tags) : null;
                // alert.Data = alertDto.Data;
                // alert.AssignedToUserId = alertDto.AssignedToUserId;
                alert.UpdatedAt = DateTime.UtcNow;

                await _alertRepository.UpdateAsync(alert);
                _logger.LogInformation("Alert updated: {AlertId}", id);

                return MapToAlertDto(alert);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error updating alert: {AlertId}", id);
                throw;
            }
        }

        /// <inheritdoc/>
        public async Task<AlertDto?> UpdateAlertStatusAsync(string id, AlertStatusModels status)
        {
            try
            {
                var alert = await _alertRepository.GetByIdAsync(id);
                if (alert == null)
                {
                    _logger.LogWarning("Alert not found: {AlertId}", id);
                    return null!;
                }

                alert.Status = status;
                alert.UpdatedAt = DateTime.UtcNow;

                await _alertRepository.UpdateAsync(alert);
                _logger.LogInformation("Alert status updated: {AlertId}, {Status}", id, status);

                return MapToAlertDto(alert);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error updating alert status: {AlertId}, {Status}", id, status);
                throw;
            }
        }

        /// <inheritdoc/>
        public async Task<AlertDto?> UpdateAlertStatusAsync(string id, UpdateAlertStatusDto updateStatusDto)
        {
            try
            {
                var alert = await _context.Alert.FindAsync(id);
                if (alert == null)
                {
                    _logger.LogWarning("Alert not found: {AlertId}", id);
                    return null!;
                }

                // Parse the status
                if (Enum.TryParse<AlertStatusModels>(updateStatusDto.Status, out var status))
                {
                    alert.Status = status;
                }
                else
                {
                    _logger.LogWarning("Invalid status: {Status}", updateStatusDto.Status);
                    return null!;
                }

                alert.UpdatedAt = updateStatusDto.UpdatedAt;

                // Set the user who updated the status
                if (status == AlertStatusModels.Acknowledged || status == AlertStatusModels.InProgress)
                {
                    alert.AcknowledgedBy = updateStatusDto.AssignedTo;
                    alert.AcknowledgedAt = DateTime.UtcNow;
                }
                else if (status == AlertStatusModels.Resolved || status == AlertStatusModels.Closed || status == AlertStatusModels.FalsePositive)
                {
                    alert.ResolvedBy = updateStatusDto.UpdatedBy;
                    alert.ResolvedAt = DateTime.UtcNow;
                    alert.ResolutionNotes = updateStatusDto.Comment;
                }

                await _context.SaveChangesAsync();
                _logger.LogInformation("Alert status updated: {AlertId}, {Status}, by {UserId}", id, status, updateStatusDto.UpdatedBy);

                return MapToAlertDto(alert);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error updating alert status: {AlertId}, {Status}", id, updateStatusDto.Status);
                throw;
            }
        }

        /// <inheritdoc/>
        public async Task<AlertDto?> AssignAlertAsync(string id, string userId)
        {
            try
            {
                var alert = await _alertRepository.GetByIdAsync(id);
                if (alert == null)
                {
                    _logger.LogWarning("Alert not found: {AlertId}", id);
                    return null!;
                }

                // Alert model doesn't have AssignedToUserId property
                // alert.AssignedToUserId = userId;
                alert.AcknowledgedBy = userId;
                alert.AcknowledgedAt = DateTime.UtcNow;
                alert.UpdatedAt = DateTime.UtcNow;

                await _alertRepository.UpdateAsync(alert);
                _logger.LogInformation("Alert assigned: {AlertId}, {UserId}", id, userId);

                return MapToAlertDto(alert);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error assigning alert: {AlertId}, {UserId}", id, userId);
                throw;
            }
        }

        /// <inheritdoc/>
        public async Task<bool> DeleteAlertAsync(string id)
        {
            try
            {
                var alert = await _alertRepository.GetByIdAsync(id);
                if (alert == null)
                {
                    _logger.LogWarning("Alert not found: {AlertId}", id);
                    return false;
                }

                await _alertRepository.DeleteByIdAsync(id);
                _logger.LogInformation("Alert deleted: {AlertId}", id);

                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error deleting alert: {AlertId}", id);
                throw;
            }
        }

        /// <inheritdoc/>
        public async Task<IEnumerable<AlertDto>> GetAlertsByAgentAsync(string agentId, int limit = 100, int offset = 0)
        {
            var alerts = await _alertRepository.GetByAgentIdAsync(agentId);
            return alerts.Skip(offset).Take(limit).Select(alert => MapToAlertDto(alert));
        }

        /// <inheritdoc/>
        public async Task<IEnumerable<AlertDto>> GetAlertsByTimeRangeAsync(DateTime startTime, DateTime endTime, int limit = 100, int offset = 0)
        {
            var alerts = await _alertRepository.GetByTimeRangeAsync(startTime, endTime);
            return alerts.Skip(offset).Take(limit).Select(alert => MapToAlertDto(alert));
        }

        /// <inheritdoc/>
        public async Task<IEnumerable<AlertDto>> GetUnresolvedAlertsAsync(int limit = 100, int offset = 0)
        {
            var alerts = await _alertRepository.GetUnresolvedAsync();
            return alerts.Skip(offset).Take(limit).Select(alert => MapToAlertDto(alert));
        }

        /// <inheritdoc/>
        public async Task<Dictionary<string, int>> GetAlertStatsByAgentAsync(DateTime startTime, DateTime endTime)
        {
            var stats = await _context.Alert
                .Where(a => a.Timestamp >= startTime && a.Timestamp <= endTime)
                .GroupBy(a => a.AgentId ?? string.Empty)
                .Select(g => new { AgentId = g.Key, Count = g.Count() })
                .ToDictionaryAsync(x => x.AgentId, x => x.Count);

            return stats;
        }

        /// <inheritdoc/>
        public async Task<Dictionary<string, int>> GetAlertStatsBySeverityAsync(DateTime startTime, DateTime endTime)
        {
            var stats = await _context.Alert
                .Where(a => a.Timestamp >= startTime && a.Timestamp <= endTime)
                .GroupBy(a => a.Severity)
                .Select(g => new { Severity = g.Key, Count = g.Count() })
                .ToDictionaryAsync(x => x.Severity.ToString(), x => x.Count);

            return stats;
        }

        /// <inheritdoc/>
        public async Task<Dictionary<string, int>> GetAlertStatsByStatusAsync(DateTime startTime, DateTime endTime)
        {
            var stats = await _context.Alert
                .Where(a => a.Timestamp >= startTime && a.Timestamp <= endTime)
                .GroupBy(a => a.Status)
                .Select(g => new { Status = g.Key, Count = g.Count() })
                .ToDictionaryAsync(x => x.Status.ToString(), x => x.Count);

            return stats;
        }

        /// <inheritdoc/>
        public async Task<Dictionary<DateTime, int>> GetAlertStatsByTimeAsync(DateTime startTime, DateTime endTime, string interval)
        {
            var result = new Dictionary<DateTime, int>();

            // Define the grouping function based on the interval
            Func<DateTime, DateTime> truncateTime;

            switch (interval.ToLower())
            {
                case "hour":
                    truncateTime = dt => new DateTime(dt.Year, dt.Month, dt.Day, dt.Hour, 0, 0);
                    break;
                case "day":
                    truncateTime = dt => dt.Date;
                    break;
                case "week":
                    truncateTime = dt => dt.Date.AddDays(-(int)dt.DayOfWeek);
                    break;
                case "month":
                    truncateTime = dt => new DateTime(dt.Year, dt.Month, 1);
                    break;
                default:
                    throw new ArgumentException($"Invalid interval: {interval}", nameof(interval));
            }

            // Get all alerts within the time range
            var alerts = await _context.Alert
                .Where(a => a.Timestamp >= startTime && a.Timestamp <= endTime)
                .ToListAsync();

            // Group by the truncated timestamp
            var groupedAlerts = alerts
                .GroupBy(a => truncateTime(a.Timestamp))
                .Select(g => new { Timestamp = g.Key, Count = g.Count() })
                .OrderBy(x => x.Timestamp)
                .ToDictionary(x => x.Timestamp, x => x.Count);

            // Fill in missing intervals with zero counts
            var current = truncateTime(startTime);
            var end = truncateTime(endTime);

            while (current <= end)
            {
                if (!groupedAlerts.TryGetValue(current, out var count))
                {
                    result[current] = 0;
                }
                else
                {
                    result[current] = count;
                }

                // Increment current based on the interval
                switch (interval.ToLower())
                {
                    case "hour":
                        current = current.AddHours(1);
                        break;
                    case "day":
                        current = current.AddDays(1);
                        break;
                    case "week":
                        current = current.AddDays(7);
                        break;
                    case "month":
                        current = current.AddMonths(1);
                        break;
                }
            }

            return result;
        }

        /// <summary>
        /// Searches alerts based on a query
        /// </summary>
        /// <param name="query">The alert query</param>
        /// <returns>Paginated result of alerts</returns>
        public async Task<PaginatedResult<AlertDto>> SearchAlertsAsync(AlertQueryDto query)
        {
            try
            {
                var alertsQuery = _context.Alert.AsQueryable();

                // Apply filters
                if (!string.IsNullOrEmpty(query.SearchTerm))
                {
                    alertsQuery = alertsQuery.Where(a => 
                        a.Title.Contains(query.SearchTerm) || 
                        a.Description.Contains(query.SearchTerm) ||
                        a.Message.Contains(query.SearchTerm));
                }

                if (!string.IsNullOrEmpty(query.Severity))
                {
                    if (Enum.TryParse<AlertSeverityModels>(query.Severity, out var severity))
                    {
                        alertsQuery = alertsQuery.Where(a => a.Severity == severity);
                    }
                }

                if (!string.IsNullOrEmpty(query.Status))
                {
                    if (Enum.TryParse<AlertStatusModels>(query.Status, out var status))
                    {
                        alertsQuery = alertsQuery.Where(a => a.Status == status);
                    }
                }

                if (!string.IsNullOrEmpty(query.AgentId))
                {
                    alertsQuery = alertsQuery.Where(a => a.AgentId == query.AgentId);
                }

                if (!string.IsNullOrEmpty(query.Source))
                {
                    alertsQuery = alertsQuery.Where(a => a.Source == query.Source);
                }

                if (!string.IsNullOrEmpty(query.AssignedTo))
                {
                    alertsQuery = alertsQuery.Where(a => a.AcknowledgedBy == query.AssignedTo);
                }

                if (query.StartTime.HasValue)
                {
                    alertsQuery = alertsQuery.Where(a => a.Timestamp >= query.StartTime.Value);
                }

                if (query.EndTime.HasValue)
                {
                    alertsQuery = alertsQuery.Where(a => a.Timestamp <= query.EndTime.Value);
                }

                // Apply sorting
                alertsQuery = ApplySorting(alertsQuery, query.SortField, query.SortDirection);

                // Get total count
                var totalCount = await alertsQuery.CountAsync();

                // Apply pagination
                var alerts = await alertsQuery
                    .Skip(query.Offset)
                    .Take(query.Limit)
                    .ToListAsync();

                // Map to DTOs
                var alertDtos = alerts.Select(a => MapToAlertDto(a)).ToList();

                return new PaginatedResult<AlertDto>
                {
                    Items = alertDtos,
                    TotalCount = totalCount,
                    Page = (query.Offset / query.Limit) + 1,
                    PageSize = query.Limit,
                    TotalPages = (int)Math.Ceiling((double)totalCount / query.Limit),
                    HasPreviousPage = query.Offset > 0,
                    HasNextPage = (query.Offset + query.Limit) < totalCount
                };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error searching alerts");
                throw;
            }
        }

        /// <summary>
        /// Gets an alert summary
        /// </summary>
        /// <param name="startTime">Start time</param>
        /// <param name="endTime">End time</param>
        /// <returns>Alert summary</returns>
        public async Task<AlertSummaryDto> GetAlertSummaryAsync(DateTime? startTime, DateTime? endTime)
        {
            try
            {
                var start = startTime ?? DateTime.UtcNow.AddDays(-30);
                var end = endTime ?? DateTime.UtcNow;

                var query = _context.Alert.AsQueryable();
                query = query.Where(a => a.Timestamp >= start && a.Timestamp <= end);

                var totalAlerts = await query.CountAsync();
                var openAlerts = await query.CountAsync(a => a.Status != AlertStatusModels.Resolved && a.Status != AlertStatusModels.Closed && a.Status != AlertStatusModels.FalsePositive);
                var closedAlerts = await query.CountAsync(a => a.Status == AlertStatusModels.Resolved || a.Status == AlertStatusModels.Closed || a.Status == AlertStatusModels.FalsePositive);

                // Get severity counts
                var severityCounts = await query
                    .GroupBy(a => a.Severity)
                    .Select(g => new { Severity = g.Key.ToString(), Count = g.Count() })
                    .ToDictionaryAsync(x => x.Severity, x => x.Count);

                // Get status counts
                var statusCounts = await query
                    .GroupBy(a => a.Status)
                    .Select(g => new { Status = g.Key.ToString(), Count = g.Count() })
                    .ToDictionaryAsync(x => x.Status, x => x.Count);

                // Get source counts
                var sourceCounts = await query
                    .GroupBy(a => a.Source)
                    .Select(g => new { Source = g.Key, Count = g.Count() })
                    .ToDictionaryAsync(x => x.Source, x => x.Count);

                // Get agent counts
                var agentCounts = await query
                    .Where(a => a.AgentId != null)
                    .GroupBy(a => a.AgentId)
                    .Select(g => new { AgentId = g.Key, Count = g.Count() })
                    .ToDictionaryAsync(x => x.AgentId ?? "unknown", x => x.Count);

                // Get hourly distribution
                var hourlyDistribution = await query
                    .GroupBy(a => a.Timestamp.Hour)
                    .Select(g => new { Hour = g.Key.ToString(), Count = g.Count() })
                    .ToDictionaryAsync(x => x.Hour, x => x.Count);

                return new AlertSummaryDto
                {
                    TotalAlerts = totalAlerts,
                    OpenAlerts = openAlerts,
                    ClosedAlerts = closedAlerts,
                    StartTime = start,
                    EndTime = end,
                    SeverityCounts = severityCounts,
                    StatusCounts = statusCounts,
                    SourceCounts = sourceCounts,
                    AgentCounts = agentCounts!,
                    HourlyDistribution = hourlyDistribution
                };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting alert summary");
                throw;
            }
        }

        /// <summary>
        /// Gets alert trends
        /// </summary>
        /// <param name="startTime">Start time</param>
        /// <param name="endTime">End time</param>
        /// <param name="interval">Time interval (hour, day, week, month)</param>
        /// <returns>Alert trends</returns>
        public async Task<AlertTrendsDto> GetAlertTrendsAsync(DateTime? startTime, DateTime? endTime, string interval)
        {
            try
            {
                var start = startTime ?? DateTime.UtcNow.AddDays(-30);
                var end = endTime ?? DateTime.UtcNow;
                var timeInterval = interval ?? "day";

                var query = _context.Alert.AsQueryable();
                query = query.Where(a => a.Timestamp >= start && a.Timestamp <= end);

                // Define the grouping function based on the interval
                Func<DateTime, DateTime> truncateTime;

                switch (timeInterval.ToLower())
                {
                    case "hour":
                        truncateTime = dt => new DateTime(dt.Year, dt.Month, dt.Day, dt.Hour, 0, 0);
                        break;
                    case "day":
                        truncateTime = dt => dt.Date;
                        break;
                    case "week":
                        truncateTime = dt => dt.Date.AddDays(-(int)dt.DayOfWeek);
                        break;
                    case "month":
                        truncateTime = dt => new DateTime(dt.Year, dt.Month, 1);
                        break;
                    default:
                        truncateTime = dt => dt.Date;
                        break;
                }

                // Get all alerts within the time range
                var alerts = await query.ToListAsync();

                // Group by the truncated timestamp
                var timePoints = new List<DateTime>();
                var totalCounts = new List<int>();
                var severityCounts = new Dictionary<string, List<int>>();
                var statusCounts = new Dictionary<string, List<int>>();
                var sourceCounts = new Dictionary<string, List<int>>();

                // Initialize dictionaries for each severity, status, and source
                foreach (AlertSeverityModels severity in Enum.GetValues(typeof(AlertSeverityModels)))
                {
                    severityCounts[severity.ToString()] = new List<int>();
                }

                foreach (AlertStatusModels status in Enum.GetValues(typeof(AlertStatusModels)))
                {
                    statusCounts[status.ToString()] = new List<int>();
                }

                var sources = alerts.Select(a => a.Source).Distinct().ToList();
                foreach (var source in sources)
                {
                    sourceCounts[source] = new List<int>();
                }

                // Generate time points based on the interval
                var current = truncateTime(start);
                var end2 = truncateTime(end);

                while (current <= end2)
                {
                    timePoints.Add(current);

                    var periodAlerts = alerts.Where(a => truncateTime(a.Timestamp) == current).ToList();
                    totalCounts.Add(periodAlerts.Count);

                    // Count by severity
                    foreach (var severity in severityCounts.Keys)
                    {
                        var count = periodAlerts.Count(a => a.Severity.ToString() == severity);
                        severityCounts[severity].Add(count);
                    }

                    // Count by status
                    foreach (var status in statusCounts.Keys)
                    {
                        var count = periodAlerts.Count(a => a.Status.ToString() == status);
                        statusCounts[status].Add(count);
                    }

                    // Count by source
                    foreach (var source in sourceCounts.Keys)
                    {
                        var count = periodAlerts.Count(a => a.Source == source);
                        sourceCounts[source].Add(count);
                    }

                    // Increment current based on the interval
                    switch (timeInterval.ToLower())
                    {
                        case "hour":
                            current = current.AddHours(1);
                            break;
                        case "day":
                            current = current.AddDays(1);
                            break;
                        case "week":
                            current = current.AddDays(7);
                            break;
                        case "month":
                            current = current.AddMonths(1);
                            break;
                        default:
                            current = current.AddDays(1);
                            break;
                    }
                }

                return new AlertTrendsDto
                {
                    TimePoints = timePoints,
                    TotalCounts = totalCounts,
                    SeverityCounts = severityCounts,
                    StatusCounts = statusCounts,
                    SourceCounts = sourceCounts,
                    TimeInterval = timeInterval,
                    StartTime = start,
                    EndTime = end
                };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting alert trends");
                throw;
            }
        }

        /// <summary>
        /// Gets related alerts
        /// </summary>
        /// <param name="alertId">The alert ID</param>
        /// <param name="maxResults">Maximum number of results</param>
        /// <returns>Related alerts</returns>
        public async Task<IEnumerable<AlertDto>> GetRelatedAlertsAsync(string alertId, int maxResults)
        {
            try
            {
                var alert = await _context.Alert.FindAsync(alertId);
                if (alert == null)
                {
                    return Enumerable.Empty<AlertDto>();
                }

                // Find alerts with the same source, agent, or similar title
                var relatedAlerts = await _context.Alert
                    .Where(a => a.Id != alertId && (
                        a.Source == alert.Source ||
                        a.AgentId == alert.AgentId ||
                        a.Title.Contains(alert.Title) ||
                        alert.Title.Contains(a.Title)))
                    .OrderByDescending(a => a.Timestamp)
                    .Take(maxResults)
                    .ToListAsync();

                return relatedAlerts.Select(a => MapToAlertDto(a));
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting related alerts for alert {AlertId}", alertId);
                throw;
            }
        }

        /// <summary>
        /// Exports alerts to CSV
        /// </summary>
        /// <param name="query">The alert query</param>
        /// <returns>CSV data</returns>
        public async Task<byte[]> ExportAlertsToCsvAsync(AlertQueryDto query)
        {
            try
            {
                var result = await SearchAlertsAsync(query);
                var alerts = result.Items;

                using (var memoryStream = new MemoryStream())
                using (var writer = new StreamWriter(memoryStream))
                {
                    // Write header
                    writer.WriteLine("Id,Title,Description,Timestamp,Severity,Status,AgentId,Source,AssignedTo,CreatedAt");

                    // Write data
                    foreach (var alert in alerts)
                    {
                        writer.WriteLine($"{alert.Id},{EscapeCsvField(alert.Title)},{EscapeCsvField(alert.Description)},{alert.Timestamp},{alert.Severity},{alert.Status},{alert.AgentId},{alert.Source},{alert.AssignedTo},{alert.CreatedAt}");
                    }

                    writer.Flush();
                    return memoryStream.ToArray();
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error exporting alerts to CSV");
                throw;
            }
        }

        /// <summary>
        /// Exports alerts to JSON
        /// </summary>
        /// <param name="query">The alert query</param>
        /// <returns>JSON data</returns>
        public async Task<byte[]> ExportAlertsToJsonAsync(AlertQueryDto query)
        {
            try
            {
                var result = await SearchAlertsAsync(query);
                var alerts = result.Items;

                var json = System.Text.Json.JsonSerializer.Serialize(alerts, new System.Text.Json.JsonSerializerOptions
                {
                    WriteIndented = true
                });

                return System.Text.Encoding.UTF8.GetBytes(json);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error exporting alerts to JSON");
                throw;
            }
        }

        /// <summary>
        /// Adds an alert comment
        /// </summary>
        /// <param name="alertId">The alert ID</param>
        /// <param name="commentDto">The comment data</param>
        /// <returns>The added comment</returns>
        public async Task<AlertDto> AddAlertCommentAsync(string alertId, AddAlertCommentDto commentDto)
        {
            try
            {
                var alert = await _context.Alert.FindAsync(alertId);
                if (alert == null)
                {
                    throw new KeyNotFoundException($"Alert with ID {alertId} not found");
                }

                // In a real implementation, you would have a comments table
                // For simplicity, we're just updating the alert
                alert.UpdatedAt = DateTime.UtcNow;

                await _context.SaveChangesAsync();

                return MapToAlertDto(alert);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error adding comment to alert {AlertId}", alertId);
                throw;
            }
        }

        /// <summary>
        /// Bulk updates alert status
        /// </summary>
        /// <param name="bulkUpdateDto">The bulk update data</param>
        /// <returns>Bulk update result</returns>
        public async Task<BulkUpdateResultDto> BulkUpdateAlertStatusAsync(BulkUpdateAlertsDto bulkUpdateDto)
        {
            try
            {
                var result = new BulkUpdateResultDto
                {
                    UpdatedCount = 0,
                    FailedCount = 0,
                    FailedAlertIds = new List<string>(),
                    ErrorMessages = new Dictionary<string, string>()
                };

                foreach (var alertId in bulkUpdateDto.AlertIds)
                {
                    try
                    {
                        var alert = await _context.Alert.FindAsync(alertId);
                        if (alert == null)
                        {
                            result.FailedCount++;
                            result.FailedAlertIds.Add(alertId);
                            result.ErrorMessages[alertId] = "Alert not found";
                            continue;
                        }

                        // Parse the status
                        if (Enum.TryParse<AlertStatusModels>(bulkUpdateDto.Status, out var status))
                        {
                            alert.Status = status;
                        }
                        else
                        {
                            result.FailedCount++;
                            result.FailedAlertIds.Add(alertId);
                            result.ErrorMessages[alertId] = $"Invalid status: {bulkUpdateDto.Status}";
                            continue;
                        }

                        alert.UpdatedAt = DateTime.UtcNow;

                        // Set the user who updated the status
                        if (status == AlertStatusModels.Acknowledged || status == AlertStatusModels.InProgress)
                        {
                            alert.AcknowledgedBy = bulkUpdateDto.AssignedTo;
                            alert.AcknowledgedAt = DateTime.UtcNow;
                        }
                        else if (status == AlertStatusModels.Resolved || status == AlertStatusModels.Closed || status == AlertStatusModels.FalsePositive)
                        {
                            alert.ResolvedBy = bulkUpdateDto.UpdatedBy;
                            alert.ResolvedAt = DateTime.UtcNow;
                            alert.ResolutionNotes = bulkUpdateDto.Comment;
                        }

                        result.UpdatedCount++;
                    }
                    catch (Exception ex)
                    {
                        _logger.LogError(ex, "Error updating alert status in bulk: {AlertId}", alertId);
                        result.FailedCount++;
                        result.FailedAlertIds.Add(alertId);
                        result.ErrorMessages[alertId] = ex.Message;
                    }
                }

                await _context.SaveChangesAsync();
                _logger.LogInformation("Bulk updated alert statuses: {SuccessCount} succeeded, {FailureCount} failed", 
                    result.UpdatedCount, result.FailedCount);

                return result;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error performing bulk update of alert statuses");
                throw;
            }
        }

        /// <summary>
        /// Escapes a field for CSV output
        /// </summary>
        /// <param name="field">The field to escape</param>
        /// <returns>The escaped field</returns>
        private string EscapeCsvField(string field)
        {
            if (string.IsNullOrEmpty(field))
            {
                return string.Empty;
            }

            if (field.Contains(",") || field.Contains("\"") || field.Contains("\n"))
            {
                return $"\"{field.Replace("\"", "\"\"")}\"";
            }

            return field;
        }

        /// <summary>
        /// Applies sorting to a query
        /// </summary>
        /// <param name="query">The query</param>
        /// <param name="sortField">The sort field</param>
        /// <param name="sortDirection">The sort direction</param>
        /// <returns>The sorted query</returns>
        private IQueryable<AlertModels> ApplySorting(IQueryable<AlertModels> query, string sortField, string sortDirection)
        {
            var isAscending = string.Equals(sortDirection, "asc", StringComparison.OrdinalIgnoreCase);

            switch (sortField.ToLower())
            {
                case "title":
                    return isAscending ? query.OrderBy(a => a.Title) : query.OrderByDescending(a => a.Title);
                case "severity":
                    return isAscending ? query.OrderBy(a => a.Severity) : query.OrderByDescending(a => a.Severity);
                case "status":
                    return isAscending ? query.OrderBy(a => a.Status) : query.OrderByDescending(a => a.Status);
                case "source":
                    return isAscending ? query.OrderBy(a => a.Source) : query.OrderByDescending(a => a.Source);
                case "agentid":
                    return isAscending ? query.OrderBy(a => a.AgentId) : query.OrderByDescending(a => a.AgentId);
                case "createdat":
                    return isAscending ? query.OrderBy(a => a.CreatedAt) : query.OrderByDescending(a => a.CreatedAt);
                case "updatedat":
                    return isAscending ? query.OrderBy(a => a.UpdatedAt) : query.OrderByDescending(a => a.UpdatedAt);
                case "timestamp":
                default:
                    return isAscending ? query.OrderBy(a => a.Timestamp) : query.OrderByDescending(a => a.Timestamp);
            }
        }

        /// <summary>
        /// Maps an Alert entity to an AlertDto
        /// </summary>
        /// <param name="alert">The alert entity</param>
        /// <returns>The alert DTO</returns>
        private AlertDto MapToAlertDto(AlertModels alert)
        {
            return new AlertDto
            {
                Id = alert.Id,
                AgentId = alert.AgentId ?? string.Empty,
                Title = alert.Title,
                Description = alert.Description,
                Message = alert.Message,
                Severity = alert.Severity.ToString(),
                Status = alert.Status.ToString(),
                Timestamp = alert.Timestamp,
                Source = alert.Source,
                ResolutionNotes = alert.ResolutionNotes ?? string.Empty,
                AssignedTo = alert.AcknowledgedBy ?? string.Empty,
                AssignedToUserId = alert.AcknowledgedBy ?? string.Empty,
                ResolvedAt = alert.ResolvedAt,
                ResolvedBy = alert.ResolvedBy ?? string.Empty,
                CreatedAt = alert.CreatedAt,
                UpdatedAt = alert.UpdatedAt
            };
        }

        /// <summary>
        /// Maps a collection of Alert entities to AlertDtos
        /// </summary>
        /// <param name="alerts">The alert entities</param>
        /// <returns>The alert DTOs</returns>
        private IEnumerable<AlertDto> MapToAlertDtos(IEnumerable<AlertModels> alerts)
        {
            return alerts.Select(alert => MapToAlertDto(alert));
        }

        /// <summary>
        /// Maps an AlertDto to an Alert entity
        /// </summary>
        /// <param name="alertDto">The alert DTO</param>
        /// <returns>The alert entity</returns>
        private AlertModels MapToAlert(AlertDto alertDto)
        {
            AlertSeverityModels severity;
            AlertStatusModels status;

            // Parse enum values
            if (!Enum.TryParse<AlertSeverityModels>(alertDto.Severity, out severity))
            {
                severity = AlertSeverityModels.Low;
            }

            if (!Enum.TryParse<AlertStatusModels>(alertDto.Status, out status))
            {
                status = AlertStatusModels.New;
            }

            return new AlertModels
            {
                Id = alertDto.Id,
                AgentId = alertDto.AgentId,
                Title = alertDto.Title,
                Description = alertDto.Description,
                Message = alertDto.Message,
                Severity = severity,
                Status = status,
                Timestamp = alertDto.Timestamp,
                Source = alertDto.Source,
                ResolutionNotes = alertDto.ResolutionNotes,
                AcknowledgedBy = alertDto.AssignedToUserId,
                AcknowledgedAt = alertDto.AssignedAt,
                ResolvedBy = alertDto.ResolvedBy,
                ResolvedAt = alertDto.ResolvedAt,
                CreatedAt = alertDto.CreatedAt,
                UpdatedAt = alertDto.UpdatedAt
            };
        }

        /// <summary>
        /// Creates an alert from a CreateAlertDto
        /// </summary>
        /// <param name="createAlertDto">The alert data</param>
        /// <returns>The created alert</returns>
        public async Task<AlertDto> CreateAlertAsync(CreateAlertDto createAlertDto)
        {
            try
            {
                // Convert CreateAlertDto to AlertDto
                var alertDto = new AlertDto
                {
                    Id = Guid.NewGuid().ToString(),
                    Title = createAlertDto.Title,
                    Description = createAlertDto.Description,
                    Severity = createAlertDto.Severity,
                    Status = AlertStatusModels.New.ToString(),
                    AgentId = createAlertDto.AgentId,
                    Source = createAlertDto.Source,
                    RuleId = createAlertDto.RuleId,
                    GeneratedBy = createAlertDto.GeneratedBy,
                    RelatedLogIds = createAlertDto.RelatedLogIds,
                    Details = createAlertDto.Details,
                    Timestamp = DateTime.UtcNow,
                    CreatedAt = DateTime.UtcNow,
                    UpdatedAt = DateTime.UtcNow
                };

                return await CreateAlertAsync(alertDto);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error creating alert from CreateAlertDto: {Title}", createAlertDto.Title);
                throw;
            }
        }

        public async Task<AlertModels> CreateAlertAsync(CreateAlertRequest request)
        {
            try
            {
                _logger.LogInformation("Creating alert for collector {CollectorType}", request.CollectorType);

                var profile = _collectorProfiles.GetValueOrDefault(request.CollectorType, _collectorProfiles["General"]);
                var severity = DetermineAlertSeverity(request, profile);

                var alert = new AlertModels
                {
                    Id = Guid.NewGuid().ToString(),
                    AgentId = request.AgentId,
                    Title = GenerateAlertTitle(request, profile),
                    Description = GenerateAlertDescription(request, profile),
                    Message = request.Message,
                    Severity = severity,
                    Status = AlertStatusModels.New,
                    Source = $"{request.CollectorType}Alert",
                    Timestamp = DateTime.UtcNow,
                    CreatedAt = DateTime.UtcNow,
                    UpdatedAt = DateTime.UtcNow
                };

                // Add collector-specific metadata
                var metadata = new AlertMetadata
                {
                    CollectorType = request.CollectorType,
                    ThreatLevel = request.ThreatLevel,
                    OriginalLogId = request.LogEntryId,
                    ThreatIndicators = request.ThreatIndicators ?? new List<string>(),
                    CollectorSpecificData = request.CollectorSpecificData ?? new Dictionary<string, object>(),
                    AutoEscalationEnabled = profile.EnableAutoEscalation,
                    EscalationThresholds = profile.EscalationThresholds,
                    NotificationChannels = profile.NotificationChannels.ToList()
                };

                // Store metadata as JSON in a custom field or separate table
                var metadataJson = JsonSerializer.Serialize(metadata);
                // For now, we'll store it in the alert's details or create a separate metadata table

                await _alertRepository.AddAsync(alert);

                // Process escalation if needed
                await ProcessAlertEscalationAsync(alert, profile);

                // Send notifications
                await SendAlertNotificationsAsync(alert, metadata, profile);

                _logger.LogInformation("Alert created with ID {AlertId} for collector {CollectorType}", 
                    alert.Id, request.CollectorType);

                return alert;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error creating alert for collector {CollectorType}", request.CollectorType);
                throw;
            }
        }

        public async Task<AlertModels> ProcessLogEntryAlertAsync(LogEntryModels logEntry)
        {
            try
            {
                // Determine collector type from log source
                var collectorType = DetermineCollectorType(logEntry.Source);
                var profile = _collectorProfiles.GetValueOrDefault(collectorType, _collectorProfiles["General"]);

                // Check if this log entry should generate an alert
                if (!ShouldGenerateAlert(logEntry, profile))
                {
                    return null!;
                }

                // Perform threat analysis
                var threatAnalysis = await _threatIntelligenceService.AnalyzeLogEntryAsync(logEntry);

                // Create alert request
                var alertRequest = new CreateAlertRequest
                {
                    AgentId = logEntry.AgentId,
                    CollectorType = collectorType,
                    Message = logEntry.Message,
                    LogEntryId = logEntry.Id,
                    ThreatLevel = threatAnalysis.ThreatLevel,
                    ThreatIndicators = threatAnalysis.Indicators.Select(i => i.Type).ToList(),
                    CollectorSpecificData = ExtractCollectorSpecificData(logEntry, collectorType)
                };

                return await CreateAlertAsync(alertRequest);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error processing log entry alert for log {LogId}", logEntry.Id);
                return null!;
            }
        }

        public async Task<List<AlertSummary>> GetCollectorAlertSummaryAsync(DateTime? since = null)
        {
            try
            {
                var sinceDate = since ?? DateTime.UtcNow.AddDays(-7);
                
                var alerts = await _context.Alerts
                    .Where(a => a.CreatedAt >= sinceDate)
                    .ToListAsync();

                var summaries = new List<AlertSummary>();

                var collectorGroups = alerts
                    .GroupBy(a => DetermineCollectorType(a.Source))
                    .ToList();

                foreach (var group in collectorGroups)
                {
                    var collectorType = group.Key;
                    var collectorAlerts = group.ToList();

                    var summary = new AlertSummary
                    {
                        CollectorType = collectorType,
                        Period = sinceDate,
                        TotalAlerts = collectorAlerts.Count,
                        AlertsBySeverity = collectorAlerts
                            .GroupBy(a => a.Severity)
                            .ToDictionary(g => g.Key, g => g.Count()),
                        AlertsByStatus = collectorAlerts
                            .GroupBy(a => a.Status)
                            .ToDictionary(g => g.Key, g => g.Count()),
                        CriticalAlerts = collectorAlerts.Count(a => a.Severity == AlertSeverityModels.Critical),
                        UnresolvedAlerts = collectorAlerts.Count(a => a.Status != AlertStatusModels.Resolved),
                        AverageResolutionTime = CalculateAverageResolutionTime(collectorAlerts),
                        TopAlertSources = GetTopAlertSources(collectorAlerts)
                    };

                    summaries.Add(summary);
                }

                return summaries.OrderByDescending(s => s.TotalAlerts).ToList();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting collector alert summary");
                throw;
            }
        }

        public async Task<List<AlertCorrelation>> FindAlertCorrelationsAsync(TimeSpan timeWindow, int minimumOccurrences = 2)
        {
            try
            {
                var correlations = new List<AlertCorrelation>();
                var cutoffTime = DateTime.UtcNow - timeWindow;

                var recentAlerts = await _context.Alerts
                    .Where(a => a.CreatedAt >= cutoffTime)
                    .OrderBy(a => a.CreatedAt)
                    .ToListAsync();

                // Group by collector type and similar patterns
                var collectorGroups = recentAlerts
                    .GroupBy(a => new { 
                        CollectorType = DetermineCollectorType(a.Source),
                        Severity = a.Severity,
                        Pattern = ExtractAlertPattern(a.Message)
                    })
                    .Where(g => g.Count() >= minimumOccurrences);

                foreach (var group in collectorGroups)
                {
                    var alerts = group.ToList();
                    var correlation = new AlertCorrelation
                    {
                        CollectorType = group.Key.CollectorType,
                        Pattern = group.Key.Pattern,
                        Severity = group.Key.Severity,
                        Occurrences = alerts.Count,
                        TimeWindow = timeWindow,
                        FirstAlert = alerts.Min(a => a.CreatedAt),
                        LastAlert = alerts.Max(a => a.CreatedAt),
                        AffectedAgents = alerts.Select(a => a.AgentId).Distinct().Count(),
                        RecommendedActions = GenerateCorrelationRecommendations(group.Key.CollectorType, group.Key.Pattern, alerts)
                    };

                    correlations.Add(correlation);
                }

                return correlations.OrderByDescending(c => c.Occurrences).ToList();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error finding alert correlations");
                throw;
            }
        }

        public Task<AlertRule> CreateAlertRuleAsync(CreateAlertRuleRequest request)
        {
            try
            {
                var rule = new AlertRule
                {
                    Id = Guid.NewGuid().ToString(),
                    Name = request.Name,
                    CollectorType = request.CollectorType,
                    Condition = request.Condition,
                    Severity = request.Severity,
                    Enabled = true,
                    CreatedAt = DateTime.UtcNow,
                    UpdatedAt = DateTime.UtcNow,
                    NotificationChannels = request.NotificationChannels ?? new List<string>(),
                    Actions = request.Actions ?? new List<AlertAction>()
                };

                // Store in database (assuming we have an AlertRules table)
                // For now, we'll add to active rules dictionary
                _activeRules[rule.Id] = rule;

                _logger.LogInformation("Created alert rule {RuleId} for collector {CollectorType}", 
                    rule.Id, request.CollectorType);

                return Task.FromResult(rule);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error creating alert rule for collector {CollectorType}", request.CollectorType);
                throw;
            }
        }

        private Dictionary<string, CollectorAlertProfile> InitializeCollectorProfiles()
        {
            var profiles = new Dictionary<string, CollectorAlertProfile>();

            // Container alerts
            profiles["Container"] = new CollectorAlertProfile
            {
                AlertThresholds = new AlertThresholds
                {
                    ErrorLogThreshold = 10,
                    WarningLogThreshold = 50,
                    CriticalEventThreshold = 5,
                    TimeWindowMinutes = 15
                },
                EnableAutoEscalation = true,
                EscalationThresholds = new Dictionary<AlertSeverityModels, TimeSpan>
                {
                    { AlertSeverityModels.Critical, TimeSpan.FromMinutes(5) },
                    { AlertSeverityModels.High, TimeSpan.FromMinutes(15) },
                    { AlertSeverityModels.Medium, TimeSpan.FromMinutes(60) }
                },
                NotificationChannels = new[] { "email", "slack", "teams" },
                AlertKeywords = new[] { "container", "docker", "kubernetes", "pod", "image", "privilege" },
                SeverityBoostKeywords = new[] { "privileged", "escape", "breakout", "unauthorized" }
            };

            // Cloud Services alerts
            profiles["CloudServices"] = new CollectorAlertProfile
            {
                AlertThresholds = new AlertThresholds
                {
                    ErrorLogThreshold = 5,
                    WarningLogThreshold = 20,
                    CriticalEventThreshold = 3,
                    TimeWindowMinutes = 10
                },
                EnableAutoEscalation = true,
                EscalationThresholds = new Dictionary<AlertSeverityModels, TimeSpan>
                {
                    { AlertSeverityModels.Critical, TimeSpan.FromMinutes(2) },
                    { AlertSeverityModels.High, TimeSpan.FromMinutes(10) },
                    { AlertSeverityModels.Medium, TimeSpan.FromMinutes(30) }
                },
                NotificationChannels = new[] { "email", "slack", "pagerduty" },
                AlertKeywords = new[] { "aws", "azure", "gcp", "api", "authentication", "access" },
                SeverityBoostKeywords = new[] { "credential", "compromise", "breach", "unauthorized", "escalation" }
            };

            // Database alerts
            profiles["Database"] = new CollectorAlertProfile
            {
                AlertThresholds = new AlertThresholds
                {
                    ErrorLogThreshold = 8,
                    WarningLogThreshold = 30,
                    CriticalEventThreshold = 3,
                    TimeWindowMinutes = 20
                },
                EnableAutoEscalation = true,
                EscalationThresholds = new Dictionary<AlertSeverityModels, TimeSpan>
                {
                    { AlertSeverityModels.Critical, TimeSpan.FromMinutes(3) },
                    { AlertSeverityModels.High, TimeSpan.FromMinutes(15) },
                    { AlertSeverityModels.Medium, TimeSpan.FromMinutes(45) }
                },
                NotificationChannels = new[] { "email", "slack" },
                AlertKeywords = new[] { "sql", "injection", "database", "query", "table", "schema" },
                SeverityBoostKeywords = new[] { "injection", "drop", "delete", "truncate", "exfiltration" }
            };

            // IoT alerts
            profiles["IoT"] = new CollectorAlertProfile
            {
                AlertThresholds = new AlertThresholds
                {
                    ErrorLogThreshold = 15,
                    WarningLogThreshold = 100,
                    CriticalEventThreshold = 5,
                    TimeWindowMinutes = 30
                },
                EnableAutoEscalation = false, // IoT devices can be noisy
                EscalationThresholds = new Dictionary<AlertSeverityModels, TimeSpan>
                {
                    { AlertSeverityModels.Critical, TimeSpan.FromMinutes(10) },
                    { AlertSeverityModels.High, TimeSpan.FromMinutes(30) }
                },
                NotificationChannels = new[] { "email" },
                AlertKeywords = new[] { "sensor", "device", "iot", "modbus", "mqtt", "scada" },
                SeverityBoostKeywords = new[] { "compromise", "hijack", "anomaly", "protocol violation" }
            };

            // File Integrity alerts
            profiles["FileIntegrity"] = new CollectorAlertProfile
            {
                AlertThresholds = new AlertThresholds
                {
                    ErrorLogThreshold = 5,
                    WarningLogThreshold = 20,
                    CriticalEventThreshold = 2,
                    TimeWindowMinutes = 5
                },
                EnableAutoEscalation = true,
                EscalationThresholds = new Dictionary<AlertSeverityModels, TimeSpan>
                {
                    { AlertSeverityModels.Critical, TimeSpan.FromMinutes(1) },
                    { AlertSeverityModels.High, TimeSpan.FromMinutes(5) },
                    { AlertSeverityModels.Medium, TimeSpan.FromMinutes(15) }
                },
                NotificationChannels = new[] { "email", "slack", "sms" },
                AlertKeywords = new[] { "file", "integrity", "modification", "change", "hash" },
                SeverityBoostKeywords = new[] { "system", "critical", "malware", "ransomware", "rootkit" }
            };

            // General/Default alerts
            profiles["General"] = new CollectorAlertProfile
            {
                AlertThresholds = new AlertThresholds
                {
                    ErrorLogThreshold = 20,
                    WarningLogThreshold = 100,
                    CriticalEventThreshold = 10,
                    TimeWindowMinutes = 60
                },
                EnableAutoEscalation = false,
                EscalationThresholds = new Dictionary<AlertSeverityModels, TimeSpan>
                {
                    { AlertSeverityModels.Critical, TimeSpan.FromMinutes(30) },
                    { AlertSeverityModels.High, TimeSpan.FromMinutes(60) }
                },
                NotificationChannels = new[] { "email" },
                AlertKeywords = new[] { "error", "warning", "failed", "exception" },
                SeverityBoostKeywords = new[] { "critical", "severe", "emergency" }
            };

            return profiles;
        }

        private void LoadActiveRules()
        {
            // Load alert rules from database
            // This is a placeholder - in real implementation, load from AlertRules table
            _logger.LogInformation("Loaded {Count} active alert rules", _activeRules.Count);
        }

        private string DetermineCollectorType(string source)
        {
            if (source.Contains("Container") || source.Contains("Docker") || source.Contains("Kubernetes"))
                return "Container";
            if (source.Contains("AWS") || source.Contains("Azure") || source.Contains("GCP") || source.Contains("CloudServices"))
                return "CloudServices";
            if (source.Contains("Database") || source.Contains("SQL") || source.Contains("MySQL") || source.Contains("PostgreSQL") || source.Contains("MongoDB"))
                return "Database";
            if (source.Contains("IoT") || source.Contains("Sensor") || source.Contains("SCADA") || source.Contains("Modbus") || source.Contains("MQTT"))
                return "IoT";
            if (source.Contains("FIM") || source.Contains("FileIntegrity"))
                return "FileIntegrity";
            
            return "General";
        }

        private bool ShouldGenerateAlert(LogEntryModels logEntry, CollectorAlertProfile profile)
        {
            // Check log level
            if (logEntry.Level == "Error" || logEntry.Level == "Critical" || logEntry.Level == "Warning")
            {
                return true;
            }

            // Check for alert keywords
            var message = logEntry.Message?.ToLowerInvariant() ?? "";
            if (profile.AlertKeywords.Any(keyword => message.Contains(keyword.ToLowerInvariant())))
            {
                return true;
            }

            return false;
        }

        private AlertSeverityModels DetermineAlertSeverity(CreateAlertRequest request, CollectorAlertProfile profile)
        {
            var baseSeverity = request.ThreatLevel switch
            {
                ThreatLevel.Critical => AlertSeverityModels.Critical,
                ThreatLevel.High => AlertSeverityModels.High,
                ThreatLevel.Medium => AlertSeverityModels.Medium,
                ThreatLevel.Low => AlertSeverityModels.Low,
                _ => AlertSeverityModels.Info
            };

            // Boost severity based on keywords
            var message = request.Message?.ToLowerInvariant() ?? "";
            if (profile.SeverityBoostKeywords.Any(keyword => message.Contains(keyword.ToLowerInvariant())))
            {
                baseSeverity = baseSeverity switch
                {
                    AlertSeverityModels.Info => AlertSeverityModels.Low,
                    AlertSeverityModels.Low => AlertSeverityModels.Medium,
                    AlertSeverityModels.Medium => AlertSeverityModels.High,
                    AlertSeverityModels.High => AlertSeverityModels.Critical,
                    _ => baseSeverity
                };
            }

            return baseSeverity;
        }

        private string GenerateAlertTitle(CreateAlertRequest request, CollectorAlertProfile profile)
        {
            var collectorName = request.CollectorType;
            var severity = DetermineAlertSeverity(request, profile);
            
            return $"{severity} Alert - {collectorName} Security Event";
        }

        private string GenerateAlertDescription(CreateAlertRequest request, CollectorAlertProfile profile)
        {
            var description = $"Security alert generated by {request.CollectorType} collector. ";
            
            if (request.ThreatIndicators?.Any() == true)
            {
                description += $"Threat indicators detected: {string.Join(", ", request.ThreatIndicators)}. ";
            }

            description += $"Original message: {request.Message}";
            
            return description;
        }

        private Dictionary<string, object> ExtractCollectorSpecificData(LogEntryModels logEntry, string collectorType)
        {
            var data = new Dictionary<string, object>();
            
            if (!string.IsNullOrEmpty(logEntry.Details))
            {
                try
                {
                    var details = JsonSerializer.Deserialize<Dictionary<string, object>>(logEntry.Details);
                    if (details != null)
                    {
                        data = details;
                    }
                }
                catch
                {
                    // Ignore JSON parsing errors
                }
            }

            // Add collector-specific fields
            data["collector_type"] = collectorType;
            data["log_level"] = logEntry.Level;
            data["agent_id"] = logEntry.AgentId ?? "";
            data["timestamp"] = logEntry.Timestamp;

            return data;
        }

        private Task ProcessAlertEscalationAsync(AlertModels alert, CollectorAlertProfile profile)
        {
            if (!profile.EscalationThresholds.TryGetValue(alert.Severity, out var escalationTime))
            {
                return Task.CompletedTask;
            }

            // Schedule escalation (in a real implementation, this would use a background job scheduler)
            _logger.LogInformation("Scheduling escalation for alert {AlertId} in {EscalationTime}", 
                alert.Id, escalationTime);
                
            return Task.CompletedTask;
        }

        private async Task SendAlertNotificationsAsync(AlertModels alert, AlertMetadata metadata, CollectorAlertProfile profile)
        {
            foreach (var channel in profile.NotificationChannels)
            {
                try
                {
                    await SendNotificationToChannelAsync(alert, metadata, channel);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error sending notification to channel {Channel} for alert {AlertId}", 
                        channel, alert.Id);
                }
            }
        }

        private Task SendNotificationToChannelAsync(AlertModels alert, AlertMetadata metadata, string channel)
        {
            // Placeholder for notification implementation
            _logger.LogInformation("Sending alert {AlertId} notification to {Channel}", alert.Id, channel);
            
            // In real implementation, this would integrate with:
            // - Email service
            // - Slack API
            // - Microsoft Teams API
            // - PagerDuty API
            // - SMS service
            // etc.
            
            return Task.CompletedTask;
        }

        private Task SendResolutionNotificationAsync(AlertModels alert)
        {
            _logger.LogInformation("Sending resolution notification for alert {AlertId}", alert.Id);
            return Task.CompletedTask;
        }

        private TimeSpan? CalculateAverageResolutionTime(List<AlertModels> alerts)
        {
            var resolvedAlerts = alerts.Where(a => a.ResolvedAt.HasValue && a.CreatedAt < a.ResolvedAt).ToList();
            
            if (!resolvedAlerts.Any())
            {
                return null!;
            }

            var totalTicks = resolvedAlerts.Sum(a => (a.ResolvedAt!.Value - a.CreatedAt).Ticks);
            return new TimeSpan(totalTicks / resolvedAlerts.Count);
        }

        private List<string> GetTopAlertSources(List<AlertModels> alerts)
        {
            return alerts
                .GroupBy(a => a.Source)
                .OrderByDescending(g => g.Count())
                .Take(5)
                .Select(g => g.Key)
                .ToList();
        }

        private string ExtractAlertPattern(string message)
        {
            // Simple pattern extraction - could be more sophisticated
            var words = message.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            return words.Length > 2 ? string.Join(" ", words.Take(3)) : message;
        }

        private List<string> GenerateCorrelationRecommendations(string collectorType, string pattern, List<AlertModels> alerts)
        {
            // Generate specific recommendations based on correlation patterns
            var recommendations = new List<string>();
            
            // Default recommendations based on collector type
            recommendations.Add($"Review {collectorType} configuration for pattern: {pattern}");
            recommendations.Add($"Investigate {alerts.Count} related alerts");
            
            return recommendations;
        }

        public async Task<bool> UpdateAlertStatusAsync(string alertId, AlertStatusModels newStatus, string? notes = null, string? userId = null)
        {
            try
            {
                var alert = await _alertRepository.GetByIdAsync(alertId);
                if (alert == null)
                {
                    return false;
                }

                var oldStatus = alert.Status;
                alert.Status = newStatus;
                alert.UpdatedAt = DateTime.UtcNow;

                switch (newStatus)
                {
                    case AlertStatusModels.Acknowledged:
                        alert.AcknowledgedBy = userId;
                        alert.AcknowledgedAt = DateTime.UtcNow;
                        break;
                    case AlertStatusModels.Resolved:
                        alert.ResolvedBy = userId;
                        alert.ResolvedAt = DateTime.UtcNow;
                        break;
                }

                if (!string.IsNullOrEmpty(notes))
                {
                    alert.ResolutionNotes = notes;
                }

                await _alertRepository.UpdateAsync(alert);

                // Log status change
                _logger.LogInformation("Alert {AlertId} status changed from {OldStatus} to {NewStatus} by {UserId}", 
                    alertId, oldStatus, newStatus, userId ?? "System");

                // Send status change notifications if needed
                if (newStatus == AlertStatusModels.Resolved)
                {
                    await SendResolutionNotificationAsync(alert);
                }

                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error updating alert status for alert {AlertId}", alertId);
                return false;
            }
        }
    }

    // Data Transfer Objects and Models
    public class CreateAlertRequest
    {
        public string? AgentId { get; set; }
        public string CollectorType { get; set; } = string.Empty;
        public string Message { get; set; } = string.Empty;
        public string? LogEntryId { get; set; }
        public ThreatLevel ThreatLevel { get; set; }
        public List<string>? ThreatIndicators { get; set; }
        public Dictionary<string, object>? CollectorSpecificData { get; set; }
    }

    public class AlertMetadata
    {
        public string CollectorType { get; set; } = string.Empty;
        public ThreatLevel ThreatLevel { get; set; }
        public string? OriginalLogId { get; set; }
        public List<string> ThreatIndicators { get; set; } = new();
        public Dictionary<string, object> CollectorSpecificData { get; set; } = new();
        public bool AutoEscalationEnabled { get; set; }
        public Dictionary<AlertSeverityModels, TimeSpan> EscalationThresholds { get; set; } = new();
        public List<string> NotificationChannels { get; set; } = new();
    }

    public class CollectorAlertProfile
    {
        public AlertThresholds AlertThresholds { get; set; } = new();
        public bool EnableAutoEscalation { get; set; }
        public Dictionary<AlertSeverityModels, TimeSpan> EscalationThresholds { get; set; } = new();
        public string[] NotificationChannels { get; set; } = Array.Empty<string>();
        public string[] AlertKeywords { get; set; } = Array.Empty<string>();
        public string[] SeverityBoostKeywords { get; set; } = Array.Empty<string>();
    }

    public class AlertThresholds
    {
        public int ErrorLogThreshold { get; set; }
        public int WarningLogThreshold { get; set; }
        public int CriticalEventThreshold { get; set; }
        public int TimeWindowMinutes { get; set; }
    }

    public class AlertSummary
    {
        public string CollectorType { get; set; } = string.Empty;
        public DateTime Period { get; set; }
        public int TotalAlerts { get; set; }
        public Dictionary<AlertSeverityModels, int> AlertsBySeverity { get; set; } = new();
        public Dictionary<AlertStatusModels, int> AlertsByStatus { get; set; } = new();
        public int CriticalAlerts { get; set; }
        public int UnresolvedAlerts { get; set; }
        public TimeSpan? AverageResolutionTime { get; set; }
        public List<string> TopAlertSources { get; set; } = new();
    }

    public class AlertCorrelation
    {
        public string CollectorType { get; set; } = string.Empty;
        public string Pattern { get; set; } = string.Empty;
        public AlertSeverityModels Severity { get; set; }
        public int Occurrences { get; set; }
        public TimeSpan TimeWindow { get; set; }
        public DateTime FirstAlert { get; set; }
        public DateTime LastAlert { get; set; }
        public int AffectedAgents { get; set; }
        public List<string> RecommendedActions { get; set; } = new();
    }

    public class AlertRule
    {
        public string Id { get; set; } = string.Empty;
        public string Name { get; set; } = string.Empty;
        public string CollectorType { get; set; } = string.Empty;
        public string Condition { get; set; } = string.Empty;
        public AlertSeverityModels Severity { get; set; }
        public bool Enabled { get; set; }
        public DateTime CreatedAt { get; set; }
        public DateTime UpdatedAt { get; set; }
        public List<string> NotificationChannels { get; set; } = new();
        public List<AlertAction> Actions { get; set; } = new();
    }

    public class CreateAlertRuleRequest
    {
        public string Name { get; set; } = string.Empty;
        public string CollectorType { get; set; } = string.Empty;
        public string Condition { get; set; } = string.Empty;
        public AlertSeverityModels Severity { get; set; }
        public List<string>? NotificationChannels { get; set; }
        public List<AlertAction>? Actions { get; set; }
    }

    public class AlertAction
    {
        public string Type { get; set; } = string.Empty;
        public Dictionary<string, object> Parameters { get; set; } = new();
    }
}


