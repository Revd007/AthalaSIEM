using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Backend.Models;
using Microsoft.Graph.Models;

namespace Backend.Data.Repositories
{
    /// <summary>
    /// Repository interface for report operations
    /// </summary>
    public interface IReportRepository : IRepository<ReportModels, string>
    {
        /// <summary>
        /// Gets reports by user ID
        /// </summary>
        /// <param name="userId">The user ID</param>
        /// <returns>Matching reports</returns>
        Task<IEnumerable<ReportModels>> GetByUserIdAsync(string userId);
        
        /// <summary>
        /// Gets reports by name
        /// </summary>
        /// <param name="name">The report name</param>
        /// <returns>The reports with the specified name</returns>
        Task<IEnumerable<ReportModels>> GetByNameAsync(string name);
        
        /// <summary>
        /// Gets reports by type
        /// </summary>
        /// <param name="type">The report type</param>
        /// <returns>The reports with the specified type</returns>
        Task<IEnumerable<ReportModels>> GetByTypeAsync(string type);
        
        /// <summary>
        /// Gets reports by time range
        /// </summary>
        /// <param name="startTime">The start time</param>
        /// <param name="endTime">The end time</param>
        /// <returns>The reports within the specified time range</returns>
        Task<IEnumerable<ReportModels>> GetByTimeRangeAsync(DateTime startTime, DateTime endTime);
        
        /// <summary>
        /// Gets scheduled reports
        /// </summary>
        /// <returns>Scheduled reports</returns>
        Task<IEnumerable<ReportModels>> GetScheduledReportsAsync();
        
        /// <summary>
        /// Updates a report's schedule
        /// </summary>
        /// <param name="id">The report ID</param>
        /// <param name="schedule">The new schedule</param>
        /// <returns>The updated report</returns>
        Task<ReportModels> UpdateScheduleAsync(string id, string schedule);
        
        /// <summary>
        /// Checks if a user has access to a report
        /// </summary>
        /// <param name="reportId">The report ID</param>
        /// <param name="userId">The user ID</param>
        /// <returns>True if the user has access, false otherwise</returns>
        Task<bool> UserHasAccessAsync(string reportId, string userId);
    }
} 