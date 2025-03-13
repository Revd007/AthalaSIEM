using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Backend.Models;
using Backend.DTOs;

namespace Backend.Services
{
    /// <summary>
    /// Interface for log analysis operations
    /// </summary>
    public interface ILogAnalysisService
    {
        /// <summary>
        /// Gets logs by agent ID
        /// </summary>
        /// <param name="agentId">The agent ID</param>
        /// <param name="limit">The maximum number of logs to return</param>
        /// <param name="offset">The number of logs to skip</param>
        /// <returns>Matching logs</returns>
        Task<IEnumerable<LogEntryModels>> GetLogsByAgentAsync(string agentId, int limit = 100, int offset = 0);

        /// <summary>
        /// Gets logs by level
        /// </summary>
        /// <param name="level">The log level</param>
        /// <param name="limit">The maximum number of logs to return</param>
        /// <param name="offset">The number of logs to skip</param>
        /// <returns>Matching logs</returns>
        Task<IEnumerable<LogEntryModels>> GetLogsByLevelAsync(string level, int limit = 100, int offset = 0);

        /// <summary>
        /// Gets logs by time range
        /// </summary>
        /// <param name="startTime">The start time</param>
        /// <param name="endTime">The end time</param>
        /// <param name="limit">The maximum number of logs to return</param>
        /// <param name="offset">The number of logs to skip</param>
        /// <returns>Matching logs</returns>
        Task<IEnumerable<LogEntryModels>> GetLogsByTimeRangeAsync(DateTime startTime, DateTime endTime, int limit = 100, int offset = 0);

        /// <summary>
        /// Gets logs by source
        /// </summary>
        /// <param name="source">The log source</param>
        /// <param name="limit">The maximum number of logs to return</param>
        /// <param name="offset">The number of logs to skip</param>
        /// <returns>Matching logs</returns>
        Task<IEnumerable<LogEntryModels>> GetLogsBySourceAsync(string source, int limit = 100, int offset = 0);

        /// <summary>
        /// Gets logs by category
        /// </summary>
        /// <param name="category">The log category</param>
        /// <param name="limit">The maximum number of logs to return</param>
        /// <param name="offset">The number of logs to skip</param>
        /// <returns>Matching logs</returns>
        Task<IEnumerable<LogEntryModels>> GetLogsByCategoryAsync(string category, int limit = 100, int offset = 0);

        /// <summary>
        /// Searches logs by query
        /// </summary>
        /// <param name="query">The search query</param>
        /// <param name="limit">The maximum number of results to return</param>
        /// <returns>Matching logs</returns>
        Task<IEnumerable<LogEntryModels>> SearchLogsAsync(string query, int limit = 100);
        
        /// <summary>
        /// Gets filtered logs
        /// </summary>
        /// <param name="agentId">Optional agent ID filter</param>
        /// <param name="level">Optional level filter</param>
        /// <param name="startDate">Optional start date filter</param>
        /// <param name="endDate">Optional end date filter</param>
        /// <param name="searchQuery">Optional search query filter</param>
        /// <param name="limit">Optional limit on results</param>
        /// <returns>Filtered logs</returns>
        Task<IEnumerable<LogEntryModels>> GetFilteredLogsAsync(
            string? agentId = null,
            string? level = null,
            DateTime? startDate = null,
            DateTime? endDate = null,
            string? searchQuery = null,
            int? limit = 100);
        
        /// <summary>
        /// Gets log count by level
        /// </summary>
        /// <param name="startDate">The start date</param>
        /// <param name="endDate">The end date</param>
        /// <returns>Log count by level</returns>
        Task<Dictionary<string, int>> GetLogCountByLevelAsync(DateTime startDate, DateTime endDate);
        
        /// <summary>
        /// Gets log count by source
        /// </summary>
        /// <param name="startDate">The start date</param>
        /// <param name="endDate">The end date</param>
        /// <returns>Log count by source</returns>
        Task<Dictionary<string, int>> GetLogCountBySourceAsync(DateTime startDate, DateTime endDate);
        
        /// <summary>
        /// Gets log count by time
        /// </summary>
        /// <param name="startDate">The start date</param>
        /// <param name="endDate">The end date</param>
        /// <param name="interval">The time interval (hour, day, week, month)</param>
        /// <returns>Log count by time</returns>
        Task<Dictionary<DateTime, int>> GetLogCountByTimeAsync(DateTime startDate, DateTime endDate, string interval = "hour");
        
        /// <summary>
        /// Gets common patterns in logs
        /// </summary>
        /// <param name="startDate">The start date</param>
        /// <param name="endDate">The end date</param>
        /// <param name="limit">The maximum number of patterns to return</param>
        /// <returns>Common patterns</returns>
        Task<IEnumerable<string>> GetCommonPatternsAsync(DateTime startDate, DateTime endDate, int limit = 10);
        
        /// <summary>
        /// Gets errors by agent
        /// </summary>
        /// <param name="startDate">The start date</param>
        /// <param name="endDate">The end date</param>
        /// <returns>Errors by agent</returns>
        Task<Dictionary<string, int>> GetErrorsByAgentAsync(DateTime startDate, DateTime endDate);

        /// <summary>
        /// Gets log statistics by agent
        /// </summary>
        /// <param name="agentId">The agent ID</param>
        /// <param name="startTime">The start time</param>
        /// <param name="endTime">The end time</param>
        /// <returns>Log statistics by agent</returns>
        Task<Dictionary<string, int>> GetLogStatsByAgentAsync(string agentId, DateTime startTime, DateTime endTime);

        /// <summary>
        /// Gets log statistics by level
        /// </summary>
        /// <param name="startTime">The start time</param>
        /// <param name="endTime">The end time</param>
        /// <returns>Log statistics by level</returns>
        Task<Dictionary<string, int>> GetLogStatsByLevelAsync(DateTime startTime, DateTime endTime);

        /// <summary>
        /// Gets log statistics by time
        /// </summary>
        /// <param name="startTime">The start time</param>
        /// <param name="endTime">The end time</param>
        /// <param name="interval">The time interval</param>
        /// <returns>Log statistics by time</returns>
        Task<Dictionary<DateTime, int>> GetLogStatsByTimeAsync(DateTime startTime, DateTime endTime, string interval);

        /// <summary>
        /// Gets log trends over time
        /// </summary>
        /// <param name="startTime">The start time</param>
        /// <param name="endTime">The end time</param>
        /// <param name="interval">The time interval</param>
        /// <returns>Log trends data</returns>
        Task<LogTrendsDto> GetLogTrendsAsync(DateTime startTime, DateTime endTime, TimeInterval interval);
        
        /// <summary>
        /// Gets log anomalies
        /// </summary>
        /// <param name="startTime">The start time</param>
        /// <param name="endTime">The end time</param>
        /// <param name="limit">Maximum number of anomalies to return</param>
        /// <returns>Log anomalies</returns>
        Task<IEnumerable<LogAnomalyDto>> GetLogAnomaliesAsync(DateTime startTime, DateTime endTime, int limit);
        
        /// <summary>
        /// Gets log patterns
        /// </summary>
        /// <param name="startTime">The start time</param>
        /// <param name="endTime">The end time</param>
        /// <param name="limit">Maximum number of patterns to return</param>
        /// <returns>Log patterns</returns>
        Task<IEnumerable<LogPatternDto>> GetLogPatternsAsync(DateTime startTime, DateTime endTime, int limit);
        
        /// <summary>
        /// Gets log correlation
        /// </summary>
        /// <param name="logId">The log ID</param>
        /// <param name="timeWindow">The time window</param>
        /// <returns>Log correlation data</returns>
        Task<LogCorrelationDto> GetLogCorrelationAsync(string logId, TimeSpan timeWindow);
    }
}