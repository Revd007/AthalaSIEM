using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Backend.DTOs;
using Backend.Models;

namespace Backend.Services
{
    /// <summary>
    /// Interface for log service operations
    /// </summary>
    public interface ILogService
    {
        /// <summary>
        /// Processes a batch of logs from an agent
        /// </summary>
        /// <param name="agentId">The agent ID</param>
        /// <param name="logBatch">The log batch to process</param>
        /// <returns>Processing result</returns>
        Task<LogBatchProcessingResult> ProcessLogBatchAsync(string agentId, LogBatchDto logBatch);

        /// <summary>
        /// Searches logs based on query parameters
        /// </summary>
        /// <param name="query">Query parameters</param>
        /// <returns>Paginated log results</returns>
        Task<PaginatedResult<LogEntryDto>> SearchLogsAsync(LogQueryDto query);

        /// <summary>
        /// Gets a log entry by ID
        /// </summary>
        /// <param name="id">The log entry ID</param>
        /// <returns>The log entry or null if not found</returns>
        Task<LogEntryDto?> GetLogByIdAsync(string id);

        /// <summary>
        /// Gets logs by agent ID with pagination
        /// </summary>
        /// <param name="agentId">The agent ID</param>
        /// <param name="offset">Offset for pagination</param>
        /// <param name="limit">Maximum number of logs to return</param>
        /// <returns>Paginated log results</returns>
        Task<PaginatedResult<LogEntryDto>> GetLogsByAgentIdAsync(string agentId, int offset, int limit);

        /// <summary>
        /// Gets log summary statistics
        /// </summary>
        /// <param name="startTime">Start time for statistics</param>
        /// <param name="endTime">End time for statistics</param>
        /// <returns>Log summary statistics</returns>
        Task<LogSummaryDto> GetLogSummaryAsync(DateTime startTime, DateTime endTime);

        /// <summary>
        /// Exports logs to CSV format
        /// </summary>
        /// <param name="query">Query parameters</param>
        /// <returns>CSV data as byte array</returns>
        Task<byte[]> ExportLogsToCsvAsync(LogQueryDto query);

        /// <summary>
        /// Exports logs to JSON format
        /// </summary>
        /// <param name="query">Query parameters</param>
        /// <returns>JSON data as byte array</returns>
        Task<byte[]> ExportLogsToJsonAsync(LogQueryDto query);

        /// <summary>
        /// Creates a log entry
        /// </summary>
        /// <param name="logEntry">The log entry to create</param>
        /// <returns>The created log entry</returns>
        Task<LogEntryDto> CreateLogEntryAsync(LogEntryDto logEntry);

        /// <summary>
        /// Bulk creates log entries
        /// </summary>
        /// <param name="logEntries">The log entries to create</param>
        /// <returns>Number of created entries</returns>
        Task<int> BulkCreateLogEntriesAsync(List<LogEntryDto> logEntries);

        /// <summary>
        /// Deletes old log entries based on retention policy
        /// </summary>
        /// <param name="retentionDays">Number of days to retain logs</param>
        /// <returns>Number of deleted entries</returns>
        Task<int> DeleteOldLogsAsync(int retentionDays);

        /// <summary>
        /// Gets log count by severity
        /// </summary>
        /// <param name="startTime">Start time</param>
        /// <param name="endTime">End time</param>
        /// <returns>Dictionary of severity counts</returns>
        Task<Dictionary<string, int>> GetLogCountBySeverityAsync(DateTime startTime, DateTime endTime);

        /// <summary>
        /// Gets log count by source
        /// </summary>
        /// <param name="startTime">Start time</param>
        /// <param name="endTime">End time</param>
        /// <returns>Dictionary of source counts</returns>
        Task<Dictionary<string, int>> GetLogCountBySourceAsync(DateTime startTime, DateTime endTime);

        /// <summary>
        /// Gets log count by agent
        /// </summary>
        /// <param name="startTime">Start time</param>
        /// <param name="endTime">End time</param>
        /// <returns>Dictionary of agent counts</returns>
        Task<Dictionary<string, int>> GetLogCountByAgentAsync(DateTime startTime, DateTime endTime);

        /// <summary>
        /// Gets recent logs for dashboard
        /// </summary>
        /// <param name="limit">Maximum number of logs to return</param>
        /// <returns>Recent log entries</returns>
        Task<List<LogEntryDto>> GetRecentLogsAsync(int limit = 100);

        /// <summary>
        /// Gets critical logs for dashboard
        /// </summary>
        /// <param name="limit">Maximum number of logs to return</param>
        /// <returns>Critical log entries</returns>
        Task<List<LogEntryDto>> GetCriticalLogsAsync(int limit = 50);

        /// <summary>
        /// Searches logs with full-text search
        /// </summary>
        /// <param name="searchTerm">Search term</param>
        /// <param name="limit">Maximum number of results</param>
        /// <param name="offset">Offset for pagination</param>
        /// <returns>Search results</returns>
        Task<PaginatedResult<LogEntryDto>> FullTextSearchAsync(string searchTerm, int limit, int offset);
    }
} 