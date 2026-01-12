using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Backend.Models;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging;

namespace Backend.Data.Repositories
{
    /// <summary>
    /// Repository for log entry operations
    /// </summary>
    public class LogEntryRepository : Repository<LogEntryModels, string>, ILegacyLogEntryRepository
    {
        private readonly ILogger<LogEntryRepository> _logger;
        
        /// <summary>
        /// Initializes a new instance of the <see cref="LogEntryRepository"/> class
        /// </summary>
        /// <param name="context">The database context</param>
        /// <param name="logger">The logger</param>
        public LogEntryRepository(ApplicationDbContext context, ILogger<LogEntryRepository> logger)
            : base(context)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }
        
        /// <inheritdoc/>
        public async Task<IEnumerable<LogEntryModels>> GetByAgentIdAsync(string agentId, int limit = 100, int offset = 0)
        {
            return await DbSet
                .Where(l => l.AgentId == agentId)
                .OrderByDescending(l => l.Timestamp)
                .Skip(offset)
                .Take(limit)
                .ToListAsync();
        }
        
        /// <inheritdoc/>
        public async Task<IEnumerable<LogEntryModels>> GetByLevelAsync(string level, int limit = 100, int offset = 0)
        {
            return await DbSet
                .Where(l => l.Level == level)
                .OrderByDescending(l => l.Timestamp)
                .Skip(offset)
                .Take(limit)
                .ToListAsync();
        }
        
        /// <inheritdoc/>
        public async Task<IEnumerable<LogEntryModels>> GetByLevelAsync(LogLevel level)
        {
            return await DbSet
                .Where(l => l.Level == level.ToString())
                .OrderByDescending(l => l.Timestamp)
                .ToListAsync();
        }
        
        /// <inheritdoc/>
        public async Task<IEnumerable<LogEntryModels>> GetByDateRangeAsync(DateTime startDate, DateTime endDate)
        {
            return await DbSet
                .Where(l => l.Timestamp >= startDate && l.Timestamp <= endDate)
                .OrderByDescending(l => l.Timestamp)
                .ToListAsync();
        }
        
        /// <inheritdoc/>
        public async Task<IEnumerable<LogEntryModels>> GetByTimeRangeAsync(DateTime startTime, DateTime endTime, int limit = 100, int offset = 0)
        {
            return await DbSet
                .Where(l => l.Timestamp >= startTime && l.Timestamp <= endTime)
                .OrderByDescending(l => l.Timestamp)
                .Skip(offset)
                .Take(limit)
                .ToListAsync();
        }
        
        /// <inheritdoc/>
        public async Task<IEnumerable<LogEntryModels>> GetBySourceAsync(string source, int limit = 100, int offset = 0)
        {
            return await DbSet
                .Where(l => l.Source == source)
                .OrderByDescending(l => l.Timestamp)
                .Skip(offset)
                .Take(limit)
                .ToListAsync();
        }
        
        /// <inheritdoc/>
        public async Task<IEnumerable<LogEntryModels>> GetByCategoryAsync(string category, int limit = 100, int offset = 0)
        {
            return await DbSet
                .Where(l => l.Category == category)
                .OrderByDescending(l => l.Timestamp)
                .Skip(offset)
                .Take(limit)
                .ToListAsync();
        }
        
        /// <inheritdoc/>
        public async Task<IEnumerable<LogEntryModels>> SearchAsync(string query)
        {
            return await DbSet
                .Where(l => l.Message.Contains(query) || l.Source.Contains(query))
                .OrderByDescending(l => l.Timestamp)
                .ToListAsync();
        }
        
        /// <inheritdoc/>
        public async Task<IEnumerable<LogEntryModels>> GetRecentLogsAsync(int count)
        {
            return await DbSet
                .OrderByDescending(l => l.Timestamp)
                .Take(count)
                .ToListAsync();
        }
        
        /// <inheritdoc/>
        public async Task<IEnumerable<LogEntryModels>> GetFilteredLogsAsync(
            string? agentId = null,
            string? level = null,
            DateTime? startDate = null,
            DateTime? endDate = null,
            string? searchQuery = null,
            int? limit = null)
        {
            var query = DbSet.AsQueryable();
            
            if (!string.IsNullOrEmpty(agentId))
            {
                query = query.Where(l => l.AgentId == agentId);
            }
            
            if (!string.IsNullOrEmpty(level))
            {
                query = query.Where(l => l.Level == level);
            }
            
            if (startDate.HasValue)
            {
                query = query.Where(l => l.Timestamp >= startDate.Value);
            }
            
            if (endDate.HasValue)
            {
                query = query.Where(l => l.Timestamp <= endDate.Value);
            }
            
            if (!string.IsNullOrEmpty(searchQuery))
            {
                query = query.Where(l => l.Message.Contains(searchQuery) || l.Source.Contains(searchQuery));
            }
            
            query = query.OrderByDescending(l => l.Timestamp);
            
            if (limit.HasValue)
            {
                query = query.Take(limit.Value);
            }
            
            return await query.ToListAsync();
        }
    }
} 