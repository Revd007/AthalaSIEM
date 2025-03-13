using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Backend.DTOs;
using Backend.Models;

namespace Backend.Services
{
    /// <summary>
    /// Service interface for alert operations
    /// </summary>
    public interface IAlertService
    {
        /// <summary>
        /// Gets all alerts
        /// </summary>
        /// <returns>All alerts</returns>
        Task<IEnumerable<AlertDto>> GetAllAlertsAsync();
        
        /// <summary>
        /// Gets all alerts with pagination
        /// </summary>
        /// <param name="limit">Maximum number of alerts to return</param>
        /// <param name="offset">Number of alerts to skip</param>
        /// <returns>Paginated alerts</returns>
        Task<IEnumerable<AlertDto>> GetAllAlertsAsync(int limit = 100, int offset = 0);
        
        /// <summary>
        /// Gets an alert by ID
        /// </summary>
        /// <param name="id">The alert ID</param>
        /// <returns>The alert, or null if not found</returns>
        Task<AlertDto?> GetAlertByIdAsync(string id);
        
        /// <summary>
        /// Gets alerts by status
        /// </summary>
        /// <param name="status">The alert status</param>
        /// <returns>The alerts with the specified status</returns>
        Task<IEnumerable<AlertDto>> GetAlertsByStatusAsync(AlertStatusModels status);
        
        /// <summary>
        /// Gets alerts by status with pagination
        /// </summary>
        /// <param name="status">The alert status</param>
        /// <param name="limit">Maximum number of alerts to return</param>
        /// <param name="offset">Number of alerts to skip</param>
        /// <returns>Paginated alerts with the specified status</returns>
        Task<IEnumerable<AlertDto>> GetAlertsByStatusAsync(AlertStatusModels status, int limit = 100, int offset = 0);
        
        /// <summary>
        /// Gets alerts by severity
        /// </summary>
        /// <param name="severity">The alert severity</param>
        /// <returns>The alerts with the specified severity</returns>
        Task<IEnumerable<AlertDto>> GetAlertsBySeverityAsync(SeverityModels severity);
        
        /// <summary>
        /// Gets alerts by severity with pagination
        /// </summary>
        /// <param name="severity">The alert severity</param>
        /// <param name="limit">Maximum number of alerts to return</param>
        /// <param name="offset">Number of alerts to skip</param>
        /// <returns>Paginated alerts with the specified severity</returns>
        Task<IEnumerable<AlertDto>> GetAlertsBySeverityAsync(SeverityModels severity, int limit = 100, int offset = 0);
        
        /// <summary>
        /// Gets alerts by agent ID
        /// </summary>
        /// <param name="agentId">The agent ID</param>
        /// <returns>The alerts for the specified agent</returns>
        Task<IEnumerable<AlertDto>> GetAlertsByAgentIdAsync(string agentId);
        
        /// <summary>
        /// Gets alerts by user ID
        /// </summary>
        /// <param name="userId">The user ID</param>
        /// <returns>The alerts assigned to the specified user</returns>
        Task<IEnumerable<AlertDto>> GetAlertsByUserIdAsync(string userId);
        
        /// <summary>
        /// Gets alerts by agent with pagination
        /// </summary>
        /// <param name="agentId">The agent ID</param>
        /// <param name="limit">Maximum number of alerts to return</param>
        /// <param name="offset">Number of alerts to skip</param>
        /// <returns>Paginated alerts for the specified agent</returns>
        Task<IEnumerable<AlertDto>> GetAlertsByAgentAsync(string agentId, int limit = 100, int offset = 0);
        
        /// <summary>
        /// Gets alerts by time range with pagination
        /// </summary>
        /// <param name="startTime">Start time</param>
        /// <param name="endTime">End time</param>
        /// <param name="limit">Maximum number of alerts to return</param>
        /// <param name="offset">Number of alerts to skip</param>
        /// <returns>Paginated alerts within the specified time range</returns>
        Task<IEnumerable<AlertDto>> GetAlertsByTimeRangeAsync(DateTime startTime, DateTime endTime, int limit = 100, int offset = 0);
        
        /// <summary>
        /// Gets unresolved alerts with pagination
        /// </summary>
        /// <param name="limit">Maximum number of alerts to return</param>
        /// <param name="offset">Number of alerts to skip</param>
        /// <returns>Paginated unresolved alerts</returns>
        Task<IEnumerable<AlertDto>> GetUnresolvedAlertsAsync(int limit = 100, int offset = 0);
        
        /// <summary>
        /// Creates an alert
        /// </summary>
        /// <param name="alertDto">The alert data</param>
        /// <returns>The created alert</returns>
        Task<AlertDto> CreateAlertAsync(AlertDto alertDto);
        
        /// <summary>
        /// Creates an alert from a CreateAlertDto
        /// </summary>
        /// <param name="createAlertDto">The alert data</param>
        /// <returns>The created alert</returns>
        Task<AlertDto> CreateAlertAsync(CreateAlertDto createAlertDto);
        
        /// <summary>
        /// Updates an alert
        /// </summary>
        /// <param name="id">The alert ID</param>
        /// <param name="alertDto">The alert data</param>
        /// <returns>The updated alert</returns>
        Task<AlertDto?> UpdateAlertAsync(string id, AlertDto alertDto);
        
        /// <summary>
        /// Updates an alert status
        /// </summary>
        /// <param name="id">The alert ID</param>
        /// <param name="status">The new status</param>
        /// <returns>The updated alert</returns>
        Task<AlertDto?> UpdateAlertStatusAsync(string id, AlertStatusModels status);
        
        /// <summary>
        /// Updates an alert status with additional information
        /// </summary>
        /// <param name="id">The alert ID</param>
        /// <param name="updateStatusDto">The status update DTO</param>
        /// <returns>The updated alert</returns>
        Task<AlertDto?> UpdateAlertStatusAsync(string id, UpdateAlertStatusDto updateStatusDto);
        
        /// <summary>
        /// Assigns an alert to a user
        /// </summary>
        /// <param name="id">The alert ID</param>
        /// <param name="userId">The user ID</param>
        /// <returns>The updated alert</returns>
        Task<AlertDto?> AssignAlertAsync(string id, string userId);
        
        /// <summary>
        /// Deletes an alert
        /// </summary>
        /// <param name="id">The alert ID</param>
        /// <returns>True if successful, false otherwise</returns>
        Task<bool> DeleteAlertAsync(string id);
        
        /// <summary>
        /// Gets alert statistics by agent
        /// </summary>
        /// <param name="startTime">Start time</param>
        /// <param name="endTime">End time</param>
        /// <returns>Alert counts by agent</returns>
        Task<Dictionary<string, int>> GetAlertStatsByAgentAsync(DateTime startTime, DateTime endTime);
        
        /// <summary>
        /// Gets alert statistics by severity
        /// </summary>
        /// <param name="startTime">Start time</param>
        /// <param name="endTime">End time</param>
        /// <returns>Alert counts by severity</returns>
        Task<Dictionary<string, int>> GetAlertStatsBySeverityAsync(DateTime startTime, DateTime endTime);
        
        /// <summary>
        /// Gets alert statistics by status
        /// </summary>
        /// <param name="startTime">Start time</param>
        /// <param name="endTime">End time</param>
        /// <returns>Alert counts by status</returns>
        Task<Dictionary<string, int>> GetAlertStatsByStatusAsync(DateTime startTime, DateTime endTime);
        
        /// <summary>
        /// Gets alert statistics by time
        /// </summary>
        /// <param name="startTime">Start time</param>
        /// <param name="endTime">End time</param>
        /// <param name="interval">Time interval (hour, day, week, month)</param>
        /// <returns>Alert counts by time interval</returns>
        Task<Dictionary<DateTime, int>> GetAlertStatsByTimeAsync(DateTime startTime, DateTime endTime, string interval);
        
        /// <summary>
        /// Searches alerts based on a query
        /// </summary>
        /// <param name="query">The alert query</param>
        /// <returns>Paginated result of alerts</returns>
        Task<PaginatedResult<AlertDto>> SearchAlertsAsync(AlertQueryDto query);
        
        /// <summary>
        /// Gets an alert summary
        /// </summary>
        /// <param name="startTime">Start time</param>
        /// <param name="endTime">End time</param>
        /// <returns>Alert summary</returns>
        Task<AlertSummaryDto> GetAlertSummaryAsync(DateTime? startTime, DateTime? endTime);
        
        /// <summary>
        /// Gets alert trends
        /// </summary>
        /// <param name="startTime">Start time</param>
        /// <param name="endTime">End time</param>
        /// <param name="interval">Time interval (hour, day, week, month)</param>
        /// <returns>Alert trends</returns>
        Task<AlertTrendsDto> GetAlertTrendsAsync(DateTime? startTime, DateTime? endTime, string interval);
        
        /// <summary>
        /// Gets related alerts
        /// </summary>
        /// <param name="alertId">The alert ID</param>
        /// <param name="maxResults">Maximum number of results</param>
        /// <returns>Related alerts</returns>
        Task<IEnumerable<AlertDto>> GetRelatedAlertsAsync(string alertId, int maxResults);
        
        /// <summary>
        /// Exports alerts to CSV
        /// </summary>
        /// <param name="query">The alert query</param>
        /// <returns>CSV data</returns>
        Task<byte[]> ExportAlertsToCsvAsync(AlertQueryDto query);
        
        /// <summary>
        /// Exports alerts to JSON
        /// </summary>
        /// <param name="query">The alert query</param>
        /// <returns>JSON data</returns>
        Task<byte[]> ExportAlertsToJsonAsync(AlertQueryDto query);
        
        /// <summary>
        /// Adds an alert comment
        /// </summary>
        /// <param name="alertId">The alert ID</param>
        /// <param name="commentDto">The comment data</param>
        /// <returns>The added comment</returns>
        Task<AlertDto> AddAlertCommentAsync(string alertId, AddAlertCommentDto commentDto);
        
        /// <summary>
        /// Bulk updates alert status
        /// </summary>
        /// <param name="bulkUpdateDto">The bulk update data</param>
        /// <returns>Bulk update result</returns>
        Task<BulkUpdateResultDto> BulkUpdateAlertStatusAsync(BulkUpdateAlertsDto bulkUpdateDto);
    }
} 