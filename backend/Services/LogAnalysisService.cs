using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using Backend.Data;
using Backend.Data.Repositories;
using Backend.Models;
using Backend.DTOs;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging;

namespace Backend.Services
{
    /// <summary>
    /// Service for log analysis operations
    /// </summary>
    public class LogAnalysisService : ILogAnalysisService
    {
        private readonly Backend.Data.Repositories.ILegacyLogEntryRepository _logEntryRepository;
        private readonly ApplicationDbContext _context;
        private readonly ILogger<LogAnalysisService> _logger;
        
        /// <summary>
        /// Initializes a new instance of the <see cref="LogAnalysisService"/> class
        /// </summary>
        /// <param name="logEntryRepository">The log entry repository</param>
        /// <param name="context">The database context</param>
        /// <param name="logger">The logger</param>
        public LogAnalysisService(
            Backend.Data.Repositories.ILegacyLogEntryRepository logEntryRepository,
            ApplicationDbContext context,
            ILogger<LogAnalysisService> logger)
        {
            _logEntryRepository = logEntryRepository ?? throw new ArgumentNullException(nameof(logEntryRepository));
            _context = context ?? throw new ArgumentNullException(nameof(context));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }
        
        /// <inheritdoc/>
        public async Task<IEnumerable<LogEntryModels>> GetLogsByAgentAsync(string agentId, int limit = 100, int offset = 0)
        {
            return await _logEntryRepository.GetByAgentIdAsync(agentId, limit, offset);
        }
        
        /// <inheritdoc/>
        public async Task<IEnumerable<LogEntryModels>> GetLogsByLevelAsync(string level, int limit = 100, int offset = 0)
        {
            return await _logEntryRepository.GetByLevelAsync(level, limit, offset);
        }
        
        /// <inheritdoc/>
        public async Task<IEnumerable<LogEntryModels>> GetLogsByTimeRangeAsync(DateTime startTime, DateTime endTime, int limit = 100, int offset = 0)
        {
            return await _logEntryRepository.GetByTimeRangeAsync(startTime, endTime, limit, offset);
        }
        
        /// <inheritdoc/>
        public async Task<IEnumerable<LogEntryModels>> GetLogsBySourceAsync(string source, int limit = 100, int offset = 0)
        {
            return await _logEntryRepository.GetBySourceAsync(source, limit, offset);
        }
        
        /// <inheritdoc/>
        public async Task<IEnumerable<LogEntryModels>> GetLogsByCategoryAsync(string category, int limit = 100, int offset = 0)
        {
            return await _logEntryRepository.GetByCategoryAsync(category, limit, offset);
        }
        
        /// <inheritdoc/>
        public async Task<IEnumerable<LogEntryModels>> SearchLogsAsync(string query, int limit = 100)
        {
            try
            {
                return await _logEntryRepository.SearchAsync(query);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error searching logs with query: {Query}", query);
                throw;
            }
        }
        
        /// <inheritdoc/>
        public async Task<Dictionary<string, int>> GetLogStatsByAgentAsync(string agentId, DateTime startTime, DateTime endTime)
        {
            var stats = await _context.LogEntry
                .Where(l => l.AgentId == agentId && l.Timestamp >= startTime && l.Timestamp <= endTime)
                .GroupBy(l => l.Level)
                .Select(g => new { Level = g.Key, Count = g.Count() })
                .ToDictionaryAsync(x => x.Level, x => x.Count);
            
            return stats;
        }
        
        /// <inheritdoc/>
        public async Task<Dictionary<string, int>> GetLogStatsByLevelAsync(DateTime startTime, DateTime endTime)
        {
            var stats = await _context.LogEntry
                .Where(l => l.Timestamp >= startTime && l.Timestamp <= endTime)
                .GroupBy(l => l.Level)
                .Select(g => new { Level = g.Key, Count = g.Count() })
                .ToDictionaryAsync(x => x.Level, x => x.Count);
            
            return stats;
        }
        
        /// <inheritdoc/>
        public async Task<Dictionary<string, int>> GetLogStatsBySourceAsync(DateTime startTime, DateTime endTime)
        {
            var stats = await _context.LogEntry
                .Where(l => l.Timestamp >= startTime && l.Timestamp <= endTime)
                .GroupBy(l => l.Source)
                .Select(g => new { Source = g.Key, Count = g.Count() })
                .ToDictionaryAsync(x => x.Source, x => x.Count);
            
            return stats;
        }
        
        /// <inheritdoc/>
        public async Task<Dictionary<string, int>> GetLogStatsByCategoryAsync(DateTime startTime, DateTime endTime)
        {
            var stats = await _context.LogEntry
                .Where(l => l.Timestamp >= startTime && l.Timestamp <= endTime)
                .GroupBy(l => l.Category)
                .Select(g => new { Category = g.Key, Count = g.Count() })
                .ToDictionaryAsync(x => x.Category ?? "Unknown", x => x.Count);
            
            return stats;
        }
        
        /// <inheritdoc/>
        public async Task<Dictionary<DateTime, int>> GetLogStatsByTimeAsync(DateTime startTime, DateTime endTime, string interval)
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
            
            // Get all log entries within the time range
            var logs = await _context.LogEntry
                .Where(l => l.Timestamp >= startTime && l.Timestamp <= endTime)
                .ToListAsync();
            
            // Group by the truncated timestamp
            var groupedLogs = logs
                .GroupBy(l => truncateTime(l.Timestamp))
                .Select(g => new { Timestamp = g.Key, Count = g.Count() })
                .OrderBy(x => x.Timestamp)
                .ToDictionary(x => x.Timestamp, x => x.Count);
            
            // Fill in missing intervals with zero counts
            var current = truncateTime(startTime);
            var end = truncateTime(endTime);
            
            while (current <= end)
            {
                if (!groupedLogs.TryGetValue(current, out var count))
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

        /// <inheritdoc/>
        public async Task<IEnumerable<LogEntryModels>> GetFilteredLogsAsync(
            string? agentId = null,
            string? level = null,
            DateTime? startDate = null,
            DateTime? endDate = null,
            string? searchQuery = null,
            int? limit = 100)
        {
            try
            {
                return await _logEntryRepository.GetFilteredLogsAsync(
                    agentId,
                    level,
                    startDate,
                    endDate,
                    searchQuery,
                    limit);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting filtered logs");
                throw;
            }
        }

        /// <inheritdoc/>
        public async Task<Dictionary<string, int>> GetLogCountByLevelAsync(DateTime startDate, DateTime endDate)
        {
            try
            {
                var logs = await _context.LogEntries
                    .Where(l => l.Timestamp >= startDate && l.Timestamp <= endDate)
                    .GroupBy(l => l.Level)
                    .Select(g => new { Level = g.Key, Count = g.Count() })
                    .ToListAsync();

                return logs.ToDictionary(l => l.Level, l => l.Count);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting log count by level");
                throw;
            }
        }

        /// <inheritdoc/>
        public async Task<Dictionary<string, int>> GetLogCountBySourceAsync(DateTime startDate, DateTime endDate)
        {
            try
            {
                var logs = await _context.LogEntries
                    .Where(l => l.Timestamp >= startDate && l.Timestamp <= endDate)
                    .GroupBy(l => l.Source)
                    .Select(g => new { Source = g.Key, Count = g.Count() })
                    .ToListAsync();

                return logs.ToDictionary(l => l.Source, l => l.Count);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting log count by source");
                throw;
            }
        }

        /// <inheritdoc/>
        public async Task<Dictionary<DateTime, int>> GetLogCountByTimeAsync(DateTime startDate, DateTime endDate, string interval = "hour")
        {
            try
            {
                var result = new Dictionary<DateTime, int>();
                var logs = await _context.LogEntries
                    .Where(l => l.Timestamp >= startDate && l.Timestamp <= endDate)
                    .ToListAsync();

                // Group logs by the specified interval
                switch (interval.ToLower())
                {
                    case "hour":
                        result = logs
                            .GroupBy(l => new DateTime(l.Timestamp.Year, l.Timestamp.Month, l.Timestamp.Day, l.Timestamp.Hour, 0, 0))
                            .ToDictionary(g => g.Key, g => g.Count());
                        break;
                    case "day":
                        result = logs
                            .GroupBy(l => new DateTime(l.Timestamp.Year, l.Timestamp.Month, l.Timestamp.Day))
                            .ToDictionary(g => g.Key, g => g.Count());
                        break;
                    case "week":
                        result = logs
                            .GroupBy(l => {
                                var firstDayOfWeek = l.Timestamp.AddDays(-(int)l.Timestamp.DayOfWeek);
                                return new DateTime(firstDayOfWeek.Year, firstDayOfWeek.Month, firstDayOfWeek.Day);
                            })
                            .ToDictionary(g => g.Key, g => g.Count());
                        break;
                    case "month":
                        result = logs
                            .GroupBy(l => new DateTime(l.Timestamp.Year, l.Timestamp.Month, 1))
                            .ToDictionary(g => g.Key, g => g.Count());
                        break;
                    default:
                        throw new ArgumentException($"Invalid interval: {interval}. Supported intervals are: hour, day, week, month");
                }

                return result;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting log count by time with interval: {Interval}", interval);
                throw;
            }
        }

        /// <inheritdoc/>
        public async Task<IEnumerable<string>> GetCommonPatternsAsync(DateTime startDate, DateTime endDate, int limit = 10)
        {
            try
            {
                // This is a simplified implementation
                // In a real system, you would use more sophisticated pattern recognition algorithms

                var logs = await _context.LogEntries
                    .Where(l => l.Timestamp >= startDate && l.Timestamp <= endDate)
                    .Select(l => l.Message)
                    .ToListAsync();

                // Extract common words or phrases
                var words = logs
                    .SelectMany(l => l.Split(new[] { ' ', '.', ',', ':', ';', '(', ')', '[', ']', '{', '}', '\n', '\r', '\t' }, StringSplitOptions.RemoveEmptyEntries))
                    .Where(w => w.Length > 3) // Filter out short words
                    .GroupBy(w => w.ToLower())
                    .OrderByDescending(g => g.Count())
                    .Take(limit)
                    .Select(g => $"{g.Key} ({g.Count()})");

                return words;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting common patterns");
                throw;
            }
        }

        /// <inheritdoc/>
        public async Task<Dictionary<string, int>> GetErrorsByAgentAsync(DateTime startDate, DateTime endDate)
        {
            try
            {
                var errors = await _context.LogEntries
                    .Where(l => l.Timestamp >= startDate && l.Timestamp <= endDate && (l.Level == "Error" || l.Level == "Critical"))
                    .GroupBy(l => l.AgentId)
                    .Select(g => new { AgentId = g.Key, Count = g.Count() })
                    .ToListAsync();

                // Get agent hostnames
                var agentIds = errors.Select(e => e.AgentId).ToList();
                var agents = await _context.Agents
                    .Where(a => agentIds.Contains(a.Id))
                    .Select(a => new { a.Id, a.Hostname })
                    .ToListAsync();

                // Map agent IDs to hostnames
                var result = new Dictionary<string, int>();
                foreach (var error in errors)
                {
                    var agent = agents.FirstOrDefault(a => a.Id == error.AgentId);
                    var key = agent != null ? agent.Hostname : error.AgentId;
                    result[key] = error.Count;
                }

                return result;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting errors by agent");
                throw;
            }
        }

        /// <inheritdoc/>
        public async Task<LogTrendsDto> GetLogTrendsAsync(DateTime startTime, DateTime endTime, Backend.Models.TimeInterval interval)
        {
            _logger.LogInformation("Getting log trends from {StartTime} to {EndTime} with interval {Interval}", startTime, endTime, interval);
            
            // Get log counts by time
            var logCountsByTime = await GetLogCountByTimeAsync(startTime, endTime, interval.ToString().ToLower());
            
            // Get severity counts
            var severityCounts = await GetLogCountByLevelAsync(startTime, endTime);
            
            // Get source counts
            var sourceCounts = await GetLogCountBySourceAsync(startTime, endTime);
            
            return new LogTrendsDto
            {
                TimePoints = logCountsByTime.Keys.ToList(),
                TotalCounts = logCountsByTime.Values.ToList(),
                SeverityCounts = ConvertToListOfInts(severityCounts),
                SourceCounts = ConvertToListOfInts(sourceCounts),
                TimeInterval = interval.ToString()
            };
        }
        
        /// <inheritdoc/>
        public async Task<IEnumerable<LogAnomalyDto>> GetLogAnomaliesAsync(DateTime startTime, DateTime endTime, int limit)
        {
            _logger.LogInformation("Getting log anomalies from {StartTime} to {EndTime} with limit {Limit}", startTime, endTime, limit);
            
            // This is a placeholder implementation
            // In a real implementation, this would use machine learning or statistical analysis
            var anomalies = new List<LogAnomalyDto>();
            
            // Get logs with errors or warnings that might indicate anomalies
            var logs = await _context.LogEntries
                .Where(l => l.Timestamp >= startTime && l.Timestamp <= endTime && 
                           (l.Level == "Error" || l.Level == "Warning"))
                .OrderByDescending(l => l.Timestamp)
                .Take(limit)
                .ToListAsync();
                
            foreach (var log in logs)
            {
                anomalies.Add(new LogAnomalyDto
                {
                    Id = Guid.NewGuid().ToString(),
                    Timestamp = log.Timestamp,
                    AnomalyType = log.Level == "Error" ? "Error Spike" : "Warning Pattern",
                    Description = $"Unusual {log.Level.ToLower()} activity detected: {log.Message}",
                    Severity = log.Level,
                    ConfidenceScore = log.Level == "Error" ? 0.9 : 0.7,
                    RelatedLogIds = new List<string> { log.Id },
                    AffectedAgents = new List<string> { log.AgentId },
                    Details = new Dictionary<string, string>
                    {
                        { "Source", log.Source },
                        { "Category", log.Category ?? "Unknown" },
                        { "Message", log.Message }
                    }
                });
            }
            
            return anomalies;
        }
        
        /// <inheritdoc/>
        public async Task<IEnumerable<LogPatternDto>> GetLogPatternsAsync(DateTime startTime, DateTime endTime, int limit)
        {
            _logger.LogInformation("Getting log patterns from {StartTime} to {EndTime} with limit {Limit}", startTime, endTime, limit);
            
            // This is a placeholder implementation
            // In a real implementation, this would use pattern recognition algorithms
            var patterns = new List<LogPatternDto>();
            
            // Get most common sources and categories
            var sources = await _context.LogEntries
                .Where(l => l.Timestamp >= startTime && l.Timestamp <= endTime)
                .GroupBy(l => l.Source)
                .Select(g => new { Source = g.Key, Count = g.Count() })
                .OrderByDescending(x => x.Count)
                .Take(limit)
                .ToListAsync();
                
            foreach (var source in sources)
            {
                // Get sample logs for this source
                var sampleLogs = await _context.LogEntries
                    .Where(l => l.Timestamp >= startTime && l.Timestamp <= endTime && l.Source == source.Source)
                    .OrderByDescending(l => l.Timestamp)
                    .Take(5)
                    .ToListAsync();
                    
                patterns.Add(new LogPatternDto
                {
                    Id = Guid.NewGuid().ToString(),
                    Signature = $"Common logs from {source.Source}",
                    Description = $"Pattern of logs from {source.Source} occurring {source.Count} times",
                    OccurrenceCount = source.Count,
                    FirstSeen = sampleLogs.Min(l => l.Timestamp),
                    LastSeen = sampleLogs.Max(l => l.Timestamp),
                    Severity = sampleLogs.Any(l => l.Level == "Error") ? "Error" : 
                               sampleLogs.Any(l => l.Level == "Warning") ? "Warning" : "Information",
                    Sources = new List<string> { source.Source },
                    SampleLogs = sampleLogs.Select(l => new LogEntryDto 
                    { 
                        Id = l.Id,
                        AgentId = l.AgentId,
                        Timestamp = l.Timestamp,
                        Severity = l.Level,
                        Source = l.Source,
                        Message = l.Message
                    }).ToList()
                });
            }
            
            return patterns;
        }
        
        /// <inheritdoc/>
        public async Task<LogCorrelationDto> GetLogCorrelationAsync(string logId, TimeSpan timeWindow)
        {
            _logger.LogInformation("Getting log correlation for log {LogId} with time window {TimeWindow}", logId, timeWindow);
            
            // Get the base log
            var baseLog = await _context.LogEntries.FindAsync(logId);
            if (baseLog == null)
            {
                throw new KeyNotFoundException($"Log with ID {logId} not found");
            }
            
            // Define the time window
            var startTime = baseLog.Timestamp.Subtract(timeWindow);
            var endTime = baseLog.Timestamp.Add(timeWindow);
            
            // Get correlated logs in the time window
            var correlatedLogs = await _context.LogEntries
                .Where(l => l.Id != logId && 
                           l.Timestamp >= startTime && 
                           l.Timestamp <= endTime &&
                           (l.AgentId == baseLog.AgentId || l.Source == baseLog.Source))
                .OrderBy(l => l.Timestamp)
                .Take(50)
                .ToListAsync();
                
            // Get related alerts
            var relatedAlerts = await _context.Alerts
                .Where(a => a.Timestamp >= startTime && 
                           a.Timestamp <= endTime &&
                           a.AgentId == baseLog.AgentId)
                .OrderBy(a => a.Timestamp)
                .Take(10)
                .ToListAsync();
                
            return new LogCorrelationDto
            {
                BaseLog = new LogEntryDto 
                { 
                    Id = baseLog.Id,
                    AgentId = baseLog.AgentId,
                    Timestamp = baseLog.Timestamp,
                    Severity = baseLog.Level,
                    Source = baseLog.Source,
                    Message = baseLog.Message
                },
                CorrelatedLogs = correlatedLogs.Select(l => new LogEntryDto
                {
                    Id = l.Id,
                    AgentId = l.AgentId,
                    Timestamp = l.Timestamp,
                    Severity = l.Level,
                    Source = l.Source,
                    Message = l.Message
                }).ToList(),
                CorrelationType = "Temporal and Source",
                CorrelationScore = 0.8,
                TimeWindowMinutes = (int)timeWindow.TotalMinutes,
                RelatedAlerts = relatedAlerts.Select(a => new AlertDto
                {
                    Id = a.Id,
                    Title = a.Id, // Use appropriate properties from your AlertModels
                    Severity = "High", // Use appropriate properties
                    Source = "Correlation" // Use appropriate properties
                }).ToList()
            };
        }

        // Helper method to convert Dictionary<string, int> to Dictionary<string, List<int>>
        private Dictionary<string, List<int>> ConvertToListOfInts(Dictionary<string, int> source)
        {
            return source.ToDictionary(
                kvp => kvp.Key,
                kvp => new List<int> { kvp.Value }
            );
        }
    }
}