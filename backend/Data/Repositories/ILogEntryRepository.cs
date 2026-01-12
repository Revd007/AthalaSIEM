using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Backend.Models;
using Microsoft.Extensions.Logging;

namespace Backend.Data.Repositories
{
    /// <summary>
    /// Legacy repository interface for log entry operations (uses LogEntryModels)
    /// </summary>
    public interface ILegacyLogEntryRepository : IRepository<LogEntryModels, string>
    {
        /// <summary>
        /// Gets log entries by agent ID
        /// </summary>
        /// <param name="agentId">The agent ID</param>
        /// <param name="limit">Optional limit on results</param>
        /// <param name="offset">Optional offset for pagination</param>
        /// <returns>Matching log entries</returns>
        Task<IEnumerable<LogEntryModels>> GetByAgentIdAsync(string agentId, int limit = 100, int offset = 0);
        
        /// <summary>
        /// Gets log entries by level
        /// </summary>
        /// <param name="level">The log level</param>
        /// <param name="limit">Optional limit on results</param>
        /// <param name="offset">Optional offset for pagination</param>
        /// <returns>Matching log entries</returns>
        Task<IEnumerable<LogEntryModels>> GetByLevelAsync(string level, int limit = 100, int offset = 0);
        
        /// <summary>
        /// Gets log entries by level
        /// </summary>
        /// <param name="level">The log level</param>
        /// <returns>Matching log entries</returns>
        Task<IEnumerable<LogEntryModels>> GetByLevelAsync(LogLevel level);
        
        /// <summary>
        /// Gets log entries within a date range
        /// </summary>
        /// <param name="startDate">The start date</param>
        /// <param name="endDate">The end date</param>
        /// <returns>Matching log entries</returns>
        Task<IEnumerable<LogEntryModels>> GetByDateRangeAsync(DateTime startDate, DateTime endDate);
        
        /// <summary>
        /// Gets log entries within a time range
        /// </summary>
        /// <param name="startTime">The start time</param>
        /// <param name="endTime">The end time</param>
        /// <param name="limit">Optional limit on results</param>
        /// <param name="offset">Optional offset for pagination</param>
        /// <returns>Matching log entries</returns>
        Task<IEnumerable<LogEntryModels>> GetByTimeRangeAsync(DateTime startTime, DateTime endTime, int limit = 100, int offset = 0);
        
        /// <summary>
        /// Gets log entries by source
        /// </summary>
        /// <param name="source">The log source</param>
        /// <param name="limit">Optional limit on results</param>
        /// <param name="offset">Optional offset for pagination</param>
        /// <returns>Matching log entries</returns>
        Task<IEnumerable<LogEntryModels>> GetBySourceAsync(string source, int limit = 100, int offset = 0);
        
        /// <summary>
        /// Gets log entries by category
        /// </summary>
        /// <param name="category">The log category</param>
        /// <param name="limit">Optional limit on results</param>
        /// <param name="offset">Optional offset for pagination</param>
        /// <returns>Matching log entries</returns>
        Task<IEnumerable<LogEntryModels>> GetByCategoryAsync(string category, int limit = 100, int offset = 0);
        
        /// <summary>
        /// Searches log entries by query
        /// </summary>
        /// <param name="query">The search query</param>
        /// <returns>Matching log entries</returns>
        Task<IEnumerable<LogEntryModels>> SearchAsync(string query);
        
        /// <summary>
        /// Gets recent log entries
        /// </summary>
        /// <param name="count">The number of entries to retrieve</param>
        /// <returns>Recent log entries</returns>
        Task<IEnumerable<LogEntryModels>> GetRecentLogsAsync(int count);
        
        /// <summary>
        /// Gets filtered log entries
        /// </summary>
        /// <param name="agentId">Optional agent ID filter</param>
        /// <param name="level">Optional level filter</param>
        /// <param name="startDate">Optional start date filter</param>
        /// <param name="endDate">Optional end date filter</param>
        /// <param name="searchQuery">Optional search query filter</param>
        /// <param name="limit">Optional limit on results</param>
        /// <returns>Filtered log entries</returns>
        Task<IEnumerable<LogEntryModels>> GetFilteredLogsAsync(
            string? agentId = null,
            string? level = null,
            DateTime? startDate = null,
            DateTime? endDate = null,
            string? searchQuery = null,
            int? limit = null);
    }
} 