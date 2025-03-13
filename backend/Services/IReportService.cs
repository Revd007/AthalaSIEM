 using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Backend.Models;

namespace Backend.Services
{
    /// <summary>
    /// Service interface for report operations
    /// </summary>
    public interface IReportService
    {
        /// <summary>
        /// Gets all reports
        /// </summary>
        /// <returns>All reports</returns>
        Task<IEnumerable<ReportModels>> GetAllReportsAsync();
        
        /// <summary>
        /// Gets a report by ID
        /// </summary>
        /// <param name="id">The report ID</param>
        /// <returns>The report, or null if not found</returns>
        Task<ReportModels?> GetReportByIdAsync(string id);
        
        /// <summary>
        /// Gets reports by user ID
        /// </summary>
        /// <param name="userId">The user ID</param>
        /// <returns>The reports for the specified user</returns>
        Task<IEnumerable<ReportModels>> GetReportsByUserAsync(string userId);
        
        /// <summary>
        /// Gets reports by name
        /// </summary>
        /// <param name="name">The report name</param>
        /// <returns>The reports with the specified name</returns>
        Task<IEnumerable<ReportModels>> GetReportsByNameAsync(string name);
        
        /// <summary>
        /// Gets reports by type
        /// </summary>
        /// <param name="type">The report type</param>
        /// <returns>The reports with the specified type</returns>
        Task<IEnumerable<ReportModels>> GetReportsByTypeAsync(string type);
        
        /// <summary>
        /// Gets reports by time range
        /// </summary>
        /// <param name="startTime">The start time</param>
        /// <param name="endTime">The end time</param>
        /// <returns>The reports within the specified time range</returns>
        Task<IEnumerable<ReportModels>> GetReportsByTimeRangeAsync(DateTime startTime, DateTime endTime);
        
        /// <summary>
        /// Gets scheduled reports
        /// </summary>
        /// <returns>The scheduled reports</returns>
        Task<IEnumerable<ReportModels>> GetScheduledReportsAsync();
        
        /// <summary>
        /// Creates a new report
        /// </summary>
        /// <param name="report">The report to create</param>
        /// <returns>The created report</returns>
        Task<ReportModels> CreateReportAsync(ReportModels report);
        
        /// <summary>
        /// Updates a report
        /// </summary>
        /// <param name="report">The report to update</param>
        /// <returns>The updated report</returns>
        Task<ReportModels> UpdateReportAsync(ReportModels report);
        
        /// <summary>
        /// Updates a report's schedule
        /// </summary>
        /// <param name="id">The report ID</param>
        /// <param name="schedule">The new schedule</param>
        /// <returns>The updated report</returns>
        Task<ReportModels> UpdateReportScheduleAsync(string id, string schedule);
        
        /// <summary>
        /// Generates a report
        /// </summary>
        /// <param name="id">The report ID</param>
        /// <returns>The generated report content</returns>
        Task<string> GenerateReportAsync(string id);
        
        /// <summary>
        /// Deletes a report
        /// </summary>
        /// <param name="id">The report ID</param>
        /// <returns>True if successful, false otherwise</returns>
        Task<bool> DeleteReportAsync(string id);
    }
}