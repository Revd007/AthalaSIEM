using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging;
using Backend.Data;
using Backend.DTOs;
using Backend.Models;

namespace Backend.Services
{
    /// <summary>
    /// Service for log operations
    /// </summary>
    public class LogService : ILogService
    {
        private readonly ApplicationDbContext _context;
        private readonly ILogger<LogService> _logger;

        /// <summary>
        /// Initializes a new instance of the <see cref="LogService"/> class
        /// </summary>
        /// <param name="context">The database context</param>
        /// <param name="logger">The logger</param>
        public LogService(ApplicationDbContext context, ILogger<LogService> logger)
        {
            _context = context ?? throw new ArgumentNullException(nameof(context));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }

        /// <summary>
        /// Processes a batch of logs from an agent
        /// </summary>
        /// <param name="agentId">The agent ID</param>
        /// <param name="logBatch">The log batch to process</param>
        /// <returns>Processing result</returns>
        public async Task<LogBatchProcessingResult> ProcessLogBatchAsync(string agentId, LogBatchDto logBatch)
        {
            var stopwatch = Stopwatch.StartNew();
            var result = new LogBatchProcessingResult
            {
                BatchId = logBatch.BatchId ?? Guid.NewGuid().ToString(),
                Errors = new List<string>()
            };

            try
            {
                _logger.LogInformation("Processing log batch from agent {AgentId} with {LogCount} logs", 
                    agentId, logBatch.Logs.Count);

                var logEntries = new List<LogEntryModels>();
                var processedCount = 0;
                var failedCount = 0;

                foreach (var logDto in logBatch.Logs)
                {
                    try
                    {
                        var logEntry = MapToLogEntryModel(logDto, agentId);
                        logEntries.Add(logEntry);
                        processedCount++;
                    }
                    catch (Exception ex)
                    {
                        failedCount++;
                        result.Errors.Add($"Failed to process log {logDto.Id}: {ex.Message}");
                        _logger.LogWarning(ex, "Failed to process log {LogId} from agent {AgentId}", 
                            logDto.Id, agentId);
                    }
                }

                // Bulk insert logs for better performance
                if (logEntries.Count > 0)
                {
                    await _context.LogEntries.AddRangeAsync(logEntries);
                await _context.SaveChangesAsync();

                    result.ProcessedLogs = logEntries;
                }

                result.ProcessedCount = processedCount;
                result.FailedCount = failedCount;
                
                stopwatch.Stop();
                result.ProcessingTimeMs = stopwatch.ElapsedMilliseconds;

                _logger.LogInformation("Processed {ProcessedCount} logs, failed {FailedCount} logs from agent {AgentId} in {ElapsedMs}ms", 
                    processedCount, failedCount, agentId, stopwatch.ElapsedMilliseconds);

                return result;
            }
            catch (Exception ex)
            {
                stopwatch.Stop();
                result.ProcessingTimeMs = stopwatch.ElapsedMilliseconds;
                result.FailedCount = logBatch.Logs.Count;
                result.Errors.Add($"Batch processing failed: {ex.Message}");
                
                _logger.LogError(ex, "Failed to process log batch from agent {AgentId}", agentId);
                return result;
            }
        }

        /// <summary>
        /// Searches logs based on query parameters
        /// </summary>
        /// <param name="query">Query parameters</param>
        /// <returns>Paginated log results</returns>
        public async Task<PaginatedResult<LogEntryDto>> SearchLogsAsync(LogQueryDto query)
        {
            var queryable = _context.LogEntries.AsQueryable();

            // Apply filters
            if (!string.IsNullOrEmpty(query.SearchTerm))
            {
                queryable = queryable.Where(l => 
                    l.Message.Contains(query.SearchTerm) ||
                    l.Source.Contains(query.SearchTerm) ||
                    l.Category != null && l.Category.Contains(query.SearchTerm));
            }

            if (!string.IsNullOrEmpty(query.Level))
            {
                queryable = queryable.Where(l => l.Level == query.Level);
            }

            if (!string.IsNullOrEmpty(query.Severity))
            {
                queryable = queryable.Where(l => l.Level == query.Severity);
            }

            if (!string.IsNullOrEmpty(query.Source))
            {
                queryable = queryable.Where(l => l.Source == query.Source);
            }

            if (!string.IsNullOrEmpty(query.AgentId))
            {
                queryable = queryable.Where(l => l.AgentId == query.AgentId);
            }

            if (query.StartTime.HasValue)
            {
                queryable = queryable.Where(l => l.Timestamp >= query.StartTime.Value);
            }

            if (query.EndTime.HasValue)
            {
                queryable = queryable.Where(l => l.Timestamp <= query.EndTime.Value);
            }

            if (query.Categories != null && query.Categories.Count > 0)
            {
                queryable = queryable.Where(l => l.Category != null && query.Categories.Contains(l.Category));
            }

            if (query.EventIds != null && query.EventIds.Count > 0)
            {
                var eventIds = query.EventIds.Select(long.Parse).ToList();
                queryable = queryable.Where(l => eventIds.Contains(l.EventId));
            }

            // Get total count
            var totalCount = await queryable.CountAsync();

            // Apply sorting
            if (!string.IsNullOrEmpty(query.SortBy))
            {
                switch (query.SortBy.ToLower())
                {
                    case "timestamp":
                        queryable = query.SortOrder?.ToLower() == "asc" 
                            ? queryable.OrderBy(l => l.Timestamp)
                            : queryable.OrderByDescending(l => l.Timestamp);
                        break;
                    case "level":
                        queryable = query.SortOrder?.ToLower() == "asc" 
                            ? queryable.OrderBy(l => l.Level)
                            : queryable.OrderByDescending(l => l.Level);
                        break;
                    case "source":
                        queryable = query.SortOrder?.ToLower() == "asc" 
                            ? queryable.OrderBy(l => l.Source)
                            : queryable.OrderByDescending(l => l.Source);
                        break;
                    default:
                        queryable = queryable.OrderByDescending(l => l.Timestamp);
                        break;
                }
            }
            else
            {
                queryable = queryable.OrderByDescending(l => l.Timestamp);
            }

            // Apply pagination
            var logs = await queryable
                .Skip(query.Offset)
                .Take(query.Limit)
                .Select(l => MapToLogEntryDto(l))
                .ToListAsync();

            return new PaginatedResult<LogEntryDto>
            {
                Items = logs,
                TotalCount = totalCount,
                Page = (query.Offset / query.Limit) + 1,
                PageSize = query.Limit
            };
        }

        /// <summary>
        /// Gets a log entry by ID
        /// </summary>
        /// <param name="id">The log entry ID</param>
        /// <returns>The log entry or null if not found</returns>
        public async Task<LogEntryDto?> GetLogByIdAsync(string id)
        {
            var log = await _context.LogEntries.FirstOrDefaultAsync(l => l.Id == id);
            return log != null ? MapToLogEntryDto(log) : null;
        }

        /// <summary>
        /// Gets logs by agent ID with pagination
        /// </summary>
        /// <param name="agentId">The agent ID</param>
        /// <param name="offset">Offset for pagination</param>
        /// <param name="limit">Maximum number of logs to return</param>
        /// <returns>Paginated log results</returns>
        public async Task<PaginatedResult<LogEntryDto>> GetLogsByAgentIdAsync(string agentId, int offset, int limit)
        {
            var query = new LogQueryDto
            {
                AgentId = agentId,
                Offset = offset,
                Limit = limit
            };
            
            return await SearchLogsAsync(query);
        }

        /// <summary>
        /// Gets log summary statistics
        /// </summary>
        /// <param name="startTime">Start time for statistics</param>
        /// <param name="endTime">End time for statistics</param>
        /// <returns>Log summary statistics</returns>
        public async Task<LogSummaryDto> GetLogSummaryAsync(DateTime startTime, DateTime endTime)
        {
            var logs = _context.LogEntries.Where(l => l.Timestamp >= startTime && l.Timestamp <= endTime);

            var totalLogs = await logs.CountAsync();
            var logsByLevel = await logs.GroupBy(l => l.Level)
                .Select(g => new { Level = g.Key, Count = g.Count() })
                .ToDictionaryAsync(x => x.Level, x => x.Count);

            var logsBySource = await logs.GroupBy(l => l.Source)
                .Select(g => new { Source = g.Key, Count = g.Count() })
                .ToDictionaryAsync(x => x.Source, x => x.Count);

            var logsByAgent = await logs.GroupBy(l => l.AgentId)
                .Select(g => new { AgentId = g.Key, Count = g.Count() })
                .ToDictionaryAsync(x => x.AgentId, x => x.Count);

            var criticalCount = logsByLevel.GetValueOrDefault("Critical", 0) + logsByLevel.GetValueOrDefault("Error", 0);
            var errorCount = logsByLevel.GetValueOrDefault("Error", 0);
            var warningCount = logsByLevel.GetValueOrDefault("Warning", 0);
            var infoCount = logsByLevel.GetValueOrDefault("Information", 0);

            var timeSpan = endTime - startTime;
            var logsPerHour = timeSpan.TotalHours > 0 ? totalLogs / timeSpan.TotalHours : 0;

            // Get hourly breakdown
            var hourlyBreakdown = await logs
                .GroupBy(l => new { 
                    Year = l.Timestamp.Year, 
                    Month = l.Timestamp.Month, 
                    Day = l.Timestamp.Day, 
                    Hour = l.Timestamp.Hour 
                })
                .Select(g => new HourlyLogCount
                {
                    Hour = new DateTime(g.Key.Year, g.Key.Month, g.Key.Day, g.Key.Hour, 0, 0),
                    Count = g.Count(),
                    ByLevel = g.GroupBy(l => l.Level).ToDictionary(lg => lg.Key, lg => lg.Count())
                })
                .OrderBy(h => h.Hour)
                .ToListAsync();

            return new LogSummaryDto
            {
                TotalLogs = totalLogs,
                LogsByLevel = logsByLevel,
                LogsBySource = logsBySource,
                LogsByAgent = logsByAgent,
                StartTime = startTime,
                EndTime = endTime,
                CriticalCount = criticalCount,
                ErrorCount = errorCount,
                WarningCount = warningCount,
                InfoCount = infoCount,
                LogsPerHour = logsPerHour,
                HourlyBreakdown = hourlyBreakdown
            };
        }

        /// <summary>
        /// Exports logs to CSV format
        /// </summary>
        /// <param name="query">Query parameters</param>
        /// <returns>CSV data as byte array</returns>
        public async Task<byte[]> ExportLogsToCsvAsync(LogQueryDto query)
        {
            var logs = await SearchLogsAsync(query);
            var csv = new StringBuilder();
            
            // CSV Header
            csv.AppendLine("Id,AgentId,Timestamp,Level,Message,Source,Category,EventId,IPAddress,MachineName,ProcessId,ThreadId");
            
            // CSV Data
            foreach (var log in logs.Items)
            {
                csv.AppendLine($"\"{log.Id}\",\"{log.AgentId}\",\"{log.Timestamp:yyyy-MM-dd HH:mm:ss}\",\"{log.Level}\",\"{EscapeCsv(log.Message)}\",\"{log.Source}\",\"{log.Category}\",\"{log.EventId}\",\"{log.IPAddress}\",\"{log.MachineName}\",\"{log.ProcessId}\",\"{log.ThreadId}\"");
            }
            
            return Encoding.UTF8.GetBytes(csv.ToString());
        }

        /// <summary>
        /// Exports logs to JSON format
        /// </summary>
        /// <param name="query">Query parameters</param>
        /// <returns>JSON data as byte array</returns>
        public async Task<byte[]> ExportLogsToJsonAsync(LogQueryDto query)
        {
            var logs = await SearchLogsAsync(query);
            var json = JsonSerializer.Serialize(logs, new JsonSerializerOptions { WriteIndented = true });
            return Encoding.UTF8.GetBytes(json);
        }

        /// <summary>
        /// Creates a log entry
        /// </summary>
        /// <param name="logEntry">The log entry to create</param>
        /// <returns>The created log entry</returns>
        public async Task<LogEntryDto> CreateLogEntryAsync(LogEntryDto logEntry)
        {
            var model = MapToLogEntryModel(logEntry, logEntry.AgentId);
            _context.LogEntries.Add(model);
            await _context.SaveChangesAsync();
            
            return MapToLogEntryDto(model);
        }

        /// <summary>
        /// Bulk creates log entries
        /// </summary>
        /// <param name="logEntries">The log entries to create</param>
        /// <returns>Number of created entries</returns>
        public async Task<int> BulkCreateLogEntriesAsync(List<LogEntryDto> logEntries)
        {
            var models = logEntries.Select(dto => MapToLogEntryModel(dto, dto.AgentId)).ToList();
            await _context.LogEntries.AddRangeAsync(models);
            await _context.SaveChangesAsync();
            
            return models.Count;
        }

        /// <summary>
        /// Deletes old log entries based on retention policy
        /// </summary>
        /// <param name="retentionDays">Number of days to retain logs</param>
        /// <returns>Number of deleted entries</returns>
        public async Task<int> DeleteOldLogsAsync(int retentionDays)
        {
            var cutoffDate = DateTime.UtcNow.AddDays(-retentionDays);
            var oldLogs = _context.LogEntries.Where(l => l.Timestamp < cutoffDate);
            var count = await oldLogs.CountAsync();
            
            _context.LogEntries.RemoveRange(oldLogs);
            await _context.SaveChangesAsync();
            
            _logger.LogInformation("Deleted {Count} old log entries older than {CutoffDate}", count, cutoffDate);
            return count;
        }

        /// <summary>
        /// Gets log count by severity
        /// </summary>
        /// <param name="startTime">Start time</param>
        /// <param name="endTime">End time</param>
        /// <returns>Dictionary of severity counts</returns>
        public async Task<Dictionary<string, int>> GetLogCountBySeverityAsync(DateTime startTime, DateTime endTime)
        {
            return await _context.LogEntries
                .Where(l => l.Timestamp >= startTime && l.Timestamp <= endTime)
                .GroupBy(l => l.Level)
                .Select(g => new { Level = g.Key, Count = g.Count() })
                .ToDictionaryAsync(x => x.Level, x => x.Count);
        }

        /// <summary>
        /// Gets log count by source
        /// </summary>
        /// <param name="startTime">Start time</param>
        /// <param name="endTime">End time</param>
        /// <returns>Dictionary of source counts</returns>
        public async Task<Dictionary<string, int>> GetLogCountBySourceAsync(DateTime startTime, DateTime endTime)
        {
            return await _context.LogEntries
                .Where(l => l.Timestamp >= startTime && l.Timestamp <= endTime)
                .GroupBy(l => l.Source)
                .Select(g => new { Source = g.Key, Count = g.Count() })
                .ToDictionaryAsync(x => x.Source, x => x.Count);
        }

        /// <summary>
        /// Gets log count by agent
        /// </summary>
        /// <param name="startTime">Start time</param>
        /// <param name="endTime">End time</param>
        /// <returns>Dictionary of agent counts</returns>
        public async Task<Dictionary<string, int>> GetLogCountByAgentAsync(DateTime startTime, DateTime endTime)
        {
            return await _context.LogEntries
                .Where(l => l.Timestamp >= startTime && l.Timestamp <= endTime)
                .GroupBy(l => l.AgentId)
                .Select(g => new { AgentId = g.Key, Count = g.Count() })
                .ToDictionaryAsync(x => x.AgentId, x => x.Count);
        }

        /// <summary>
        /// Gets recent logs for dashboard
        /// </summary>
        /// <param name="limit">Maximum number of logs to return</param>
        /// <returns>Recent log entries</returns>
        public async Task<List<LogEntryDto>> GetRecentLogsAsync(int limit = 100)
        {
            var logs = await _context.LogEntries
                .OrderByDescending(l => l.Timestamp)
                .Take(limit)
                .Select(l => MapToLogEntryDto(l))
                .ToListAsync();
                
            return logs;
        }

        /// <summary>
        /// Gets critical logs for dashboard
        /// </summary>
        /// <param name="limit">Maximum number of logs to return</param>
        /// <returns>Critical log entries</returns>
        public async Task<List<LogEntryDto>> GetCriticalLogsAsync(int limit = 50)
        {
            var logs = await _context.LogEntries
                .Where(l => l.Level == "Critical" || l.Level == "Error")
                .OrderByDescending(l => l.Timestamp)
                .Take(limit)
                .Select(l => MapToLogEntryDto(l))
                .ToListAsync();
                
            return logs;
        }

        /// <summary>
        /// Searches logs with full-text search
        /// </summary>
        /// <param name="searchTerm">Search term</param>
        /// <param name="limit">Maximum number of results</param>
        /// <param name="offset">Offset for pagination</param>
        /// <returns>Search results</returns>
        public async Task<PaginatedResult<LogEntryDto>> FullTextSearchAsync(string searchTerm, int limit, int offset)
        {
            var query = new LogQueryDto
            {
                SearchTerm = searchTerm,
                Limit = limit,
                Offset = offset
            };
            
            return await SearchLogsAsync(query);
        }

        // Private helper methods

        /// <summary>
        /// Maps a LogEntryDto to LogEntryModels
        /// </summary>
        /// <param name="dto">The DTO to map</param>
        /// <param name="agentId">The agent ID</param>
        /// <returns>The mapped model</returns>
        private LogEntryModels MapToLogEntryModel(LogEntryDto dto, string agentId)
        {
            return new LogEntryModels
            {
                Id = dto.Id ?? Guid.NewGuid().ToString(),
                AgentId = agentId,
                Timestamp = dto.Timestamp,
                Level = dto.Level,
                Message = dto.Message,
                Source = dto.Source,
                Category = dto.Category,
                EventId = dto.EventId,
                IPAddress = dto.IPAddress ?? string.Empty,
                Exception = dto.Exception,
                MachineName = dto.MachineName,
                ProcessId = dto.ProcessId,
                ThreadId = dto.ThreadId,
                UserId = dto.UserId,
                RequestPath = dto.RequestPath,
                RequestId = dto.RequestId,
                ClientIp = dto.ClientIp,
                Properties = dto.Properties != null ? JsonSerializer.Serialize(dto.Properties) : null,
                ReceivedAt = DateTime.UtcNow,
                Processed = false,
                CreatedAt = DateTime.UtcNow,
                StackTrace = dto.StackTrace,
                Details = dto.Details
            };
        }

        /// <summary>
        /// Maps a LogEntryModels to LogEntryDto
        /// </summary>
        /// <param name="model">The model to map</param>
        /// <returns>The mapped DTO</returns>
        private LogEntryDto MapToLogEntryDto(LogEntryModels model)
        {
            Dictionary<string, object>? properties = null;
            if (!string.IsNullOrEmpty(model.Properties))
            {
                try
                {
                    properties = JsonSerializer.Deserialize<Dictionary<string, object>>(model.Properties);
                }
                catch (JsonException ex)
                {
                    _logger.LogWarning(ex, "Failed to deserialize properties for log {LogId}", model.Id);
                    properties = new Dictionary<string, object> { ["original"] = model.Properties };
                }
            }

            return new LogEntryDto
            {
                Id = model.Id,
                AgentId = model.AgentId,
                Timestamp = model.Timestamp,
                Level = model.Level,
                Message = model.Message,
                Source = model.Source,
                Category = model.Category,
                EventId = model.EventId,
                IPAddress = model.IPAddress,
                Exception = model.Exception,
                MachineName = model.MachineName,
                ProcessId = model.ProcessId,
                ThreadId = model.ThreadId,
                UserId = model.UserId,
                RequestPath = model.RequestPath,
                RequestId = model.RequestId,
                ClientIp = model.ClientIp,
                Properties = properties,
                ReceivedAt = model.ReceivedAt,
                Processed = model.Processed,
                ProcessedAt = model.ProcessedAt,
                CreatedAt = model.CreatedAt,
                StackTrace = model.StackTrace,
                Details = model.Details
            };
    }

    /// <summary>
        /// Escapes CSV values
    /// </summary>
        /// <param name="value">The value to escape</param>
        /// <returns>The escaped value</returns>
        private string EscapeCsv(string value)
        {
            if (string.IsNullOrEmpty(value)) return string.Empty;
            
            return value.Replace("\"", "\"\"");
        }
    }
} 