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

namespace Backend.Services
{
    /// <summary>
    /// Service for alert operations
    /// </summary>
    public class AlertService : IAlertService
    {
        private readonly IAlertRepository _alertRepository;
        private readonly ApplicationDbContext _dbContext;
        private readonly ILogger<AlertService> _logger;

        /// <summary>
        /// Initializes a new instance of the <see cref="AlertService"/> class
        /// </summary>
        /// <param name="alertRepository">The alert repository</param>
        /// <param name="dbContext">The database context</param>
        /// <param name="logger">The logger</param>
        public AlertService(
            IAlertRepository alertRepository,
            ApplicationDbContext dbContext,
            ILogger<AlertService> logger)
        {
            _alertRepository = alertRepository ?? throw new ArgumentNullException(nameof(alertRepository));
            _dbContext = dbContext ?? throw new ArgumentNullException(nameof(dbContext));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
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
                    return null;
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
                    return null;
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
                var alert = await _dbContext.Alert.FindAsync(id);
                if (alert == null)
                {
                    _logger.LogWarning("Alert not found: {AlertId}", id);
                    return null;
                }

                // Parse the status
                if (Enum.TryParse<AlertStatusModels>(updateStatusDto.Status, out var status))
                {
                    alert.Status = status;
                }
                else
                {
                    _logger.LogWarning("Invalid status: {Status}", updateStatusDto.Status);
                    return null;
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

                await _dbContext.SaveChangesAsync();
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
                    return null;
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
            var stats = await _dbContext.Alert
                .Where(a => a.Timestamp >= startTime && a.Timestamp <= endTime)
                .GroupBy(a => a.AgentId ?? string.Empty)
                .Select(g => new { AgentId = g.Key, Count = g.Count() })
                .ToDictionaryAsync(x => x.AgentId, x => x.Count);

            return stats;
        }

        /// <inheritdoc/>
        public async Task<Dictionary<string, int>> GetAlertStatsBySeverityAsync(DateTime startTime, DateTime endTime)
        {
            var stats = await _dbContext.Alert
                .Where(a => a.Timestamp >= startTime && a.Timestamp <= endTime)
                .GroupBy(a => a.Severity)
                .Select(g => new { Severity = g.Key, Count = g.Count() })
                .ToDictionaryAsync(x => x.Severity.ToString(), x => x.Count);

            return stats;
        }

        /// <inheritdoc/>
        public async Task<Dictionary<string, int>> GetAlertStatsByStatusAsync(DateTime startTime, DateTime endTime)
        {
            var stats = await _dbContext.Alert
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
            var alerts = await _dbContext.Alert
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
                var alertsQuery = _dbContext.Alert.AsQueryable();

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

                var query = _dbContext.Alert.AsQueryable();
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

                var query = _dbContext.Alert.AsQueryable();
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
                var alert = await _dbContext.Alert.FindAsync(alertId);
                if (alert == null)
                {
                    return Enumerable.Empty<AlertDto>();
                }

                // Find alerts with the same source, agent, or similar title
                var relatedAlerts = await _dbContext.Alert
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
                var alert = await _dbContext.Alert.FindAsync(alertId);
                if (alert == null)
                {
                    throw new KeyNotFoundException($"Alert with ID {alertId} not found");
                }

                // In a real implementation, you would have a comments table
                // For simplicity, we're just updating the alert
                alert.UpdatedAt = DateTime.UtcNow;

                await _dbContext.SaveChangesAsync();

                return MapToAlertDto(alert);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error adding comment to alert {AlertId}", alertId);
                throw;
            }
        }

        /// <summary>
        /// Updates an alert status with additional information
        /// </summary>
        /// <param name="id">The alert ID</param>
        /// <param name="status">The new status</param>
        /// <param name="userId">The user ID making the change</param>
        /// <param name="notes">Optional notes about the status change</param>
        /// <returns>The updated alert</returns>
        public async Task<AlertDto?> UpdateAlertStatusAsync(string id, AlertStatusModels status, string userId, string? notes = null)
        {
            try
            {
                var alert = await _alertRepository.GetByIdAsync(id);
                if (alert == null)
                {
                    _logger.LogWarning("Alert not found: {AlertId}", id);
                    return null;
                }

                alert.Status = status;
                alert.UpdatedAt = DateTime.UtcNow;

                // Set the user who updated the status
                if (status == AlertStatusModels.Acknowledged || status == AlertStatusModels.InProgress)
                {
                    alert.AcknowledgedBy = userId;
                    alert.AcknowledgedAt = DateTime.UtcNow;
                }
                else if (status == AlertStatusModels.Resolved || status == AlertStatusModels.Closed || status == AlertStatusModels.FalsePositive)
                {
                    alert.ResolvedBy = userId;
                    alert.ResolvedAt = DateTime.UtcNow;
                    alert.ResolutionNotes = notes;
                }

                await _alertRepository.UpdateAsync(alert);
                _logger.LogInformation("Alert status updated: {AlertId}, {Status}, by {UserId}", id, status, userId);

                return MapToAlertDto(alert);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error updating alert status: {AlertId}, {Status}", id, status);
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
                        var alert = await _dbContext.Alert.FindAsync(alertId);
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

                await _dbContext.SaveChangesAsync();
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
    }
}