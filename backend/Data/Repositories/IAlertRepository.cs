using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Backend.Models;

namespace Backend.Data.Repositories
{
    /// <summary>
    /// Repository interface for alert operations
    /// </summary>
    public interface IAlertRepository : IRepository<AlertModels, string>
    {
        /// <summary>
        /// Gets alerts by status
        /// </summary>
        /// <param name="status">The alert status</param>
        /// <returns>The alerts with the specified status</returns>
        Task<IEnumerable<AlertModels>> GetByStatusAsync(AlertStatusModels status);
        
        /// <summary>
        /// Gets alerts by severity
        /// </summary>
        /// <param name="severity">The alert severity</param>
        /// <returns>The alerts with the specified severity</returns>
        Task<IEnumerable<AlertModels>> GetBySeverityAsync(SeverityModels severity);
        
        /// <summary>
        /// Gets alerts by agent ID
        /// </summary>
        /// <param name="agentId">The agent ID</param>
        /// <returns>The alerts for the specified agent</returns>
        Task<IEnumerable<AlertModels>> GetByAgentIdAsync(string agentId);
        
        /// <summary>
        /// Gets alerts by user ID
        /// </summary>
        /// <param name="userId">The user ID</param>
        /// <returns>The alerts assigned to the specified user</returns>
        Task<IEnumerable<AlertModels>> GetByUserIdAsync(string userId);
        
        /// <summary>
        /// Gets alerts within a date range
        /// </summary>
        /// <param name="startDate">The start date</param>
        /// <param name="endDate">The end date</param>
        /// <returns>Matching alerts</returns>
        Task<IEnumerable<AlertModels>> GetByDateRangeAsync(DateTime startDate, DateTime endDate);
        
        /// <summary>
        /// Updates an alert's status
        /// </summary>
        /// <param name="id">The alert ID</param>
        /// <param name="status">The new status</param>
        /// <returns>The updated alert</returns>
        Task<AlertModels> UpdateStatusAsync(string id, AlertStatusModels status);
        
        /// <summary>
        /// Gets all unresolved alerts
        /// </summary>
        /// <returns>Unresolved alerts</returns>
        Task<IEnumerable<AlertModels>> GetUnresolvedAlertsAsync();

        Task<IEnumerable<AlertModels>> GetByTimeRangeAsync(DateTime startTime, DateTime endTime);
        Task<IEnumerable<AlertModels>> GetUnresolvedAsync();
        Task<IEnumerable<AlertModels>> GetBySeverityAsync(AlertSeverityModels severity);
    }
} 