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
    /// Repository for alert operations
    /// </summary>
    public class AlertRepository : Repository<AlertModels, string>, IAlertRepository
    {
        private readonly ILogger<AlertRepository> _logger;
        
        /// <summary>
        /// Initializes a new instance of the <see cref="AlertRepository"/> class
        /// </summary>
        /// <param name="context">The database context</param>
        /// <param name="logger">The logger</param>
        public AlertRepository(ApplicationDbContext context, ILogger<AlertRepository> logger)
            : base(context)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }
        
        /// <inheritdoc/>
        public async Task<IEnumerable<AlertModels>> GetByStatusAsync(AlertStatusModels status)
        {
            return await DbSet.Where(a => a.Status == status).ToListAsync();
        }
        
        /// <inheritdoc/>
        public async Task<IEnumerable<AlertModels>> GetBySeverityAsync(SeverityModels severity)
        {
            return await DbSet
                .Where(a => (int)a.Severity == (int)severity)
                .OrderByDescending(a => a.CreatedAt)
                .ToListAsync();
        }
        
        /// <inheritdoc/>
        public async Task<IEnumerable<AlertModels>> GetByAgentIdAsync(string agentId)
        {
            return await DbSet.Where(a => a.AgentId == agentId).ToListAsync();
        }
        
        /// <inheritdoc/>
        public async Task<IEnumerable<AlertModels>> GetByDateRangeAsync(DateTime startDate, DateTime endDate)
        {
            return await DbSet
                .Where(a => a.CreatedAt >= startDate && a.CreatedAt <= endDate)
                .ToListAsync();
        }
        
        /// <inheritdoc/>
        public async Task<AlertModels> UpdateStatusAsync(string id, AlertStatusModels status)
        {
            var alert = await DbSet.FindAsync(id);
            
            if (alert == null)
            {
                throw new KeyNotFoundException($"Alert with ID {id} not found");
            }
            
            alert.Status = status;
            alert.UpdatedAt = DateTime.UtcNow;
            
            await Context.SaveChangesAsync();
            
            return alert;
        }
        
        /// <inheritdoc/>
        public async Task<IEnumerable<AlertModels>> GetUnresolvedAlertsAsync()
        {
            return await DbSet
                .Where(a => a.Status != AlertStatusModels.Resolved)
                .OrderByDescending(a => a.Severity)
                .ThenByDescending(a => a.CreatedAt)
                .ToListAsync();
        }

        /// <inheritdoc/>
        public async Task<IEnumerable<AlertModels>> GetByUserIdAsync(string userId)
        {
            return await DbSet
                .Where(a => a.AcknowledgedBy == userId || a.ResolvedBy == userId)
                .OrderByDescending(a => a.CreatedAt)
                .ToListAsync();
        }

        public async Task<IEnumerable<AlertModels>> GetByTimeRangeAsync(DateTime startTime, DateTime endTime)
        {
            return await Context.Alerts
                .Where(a => a.Timestamp >= startTime && a.Timestamp <= endTime)
                .ToListAsync();
        }

        public async Task<IEnumerable<AlertModels>> GetUnresolvedAsync()
        {
            return await Context.Alerts
                .Where(a => a.Status != AlertStatusModels.Resolved)
                .ToListAsync();
        }

        /// <inheritdoc/>
        public async Task<IEnumerable<AlertModels>> GetBySeverityAsync(AlertSeverityModels severity)
        {
            return await DbSet
                .Where(a => (int)a.Severity == (int)severity)
                .OrderByDescending(a => a.CreatedAt)
                .ToListAsync();
        }
    }
}