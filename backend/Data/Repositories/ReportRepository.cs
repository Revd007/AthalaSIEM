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
    /// Repository for report operations
    /// </summary>
    public class ReportRepository : Repository<ReportModels, string>, IReportRepository
    {
        private readonly ILogger<ReportRepository> _logger;
        
        /// <summary>
        /// Initializes a new instance of the <see cref="ReportRepository"/> class
        /// </summary>
        /// <param name="context">The database context</param>
        /// <param name="logger">The logger</param>
        public ReportRepository(ApplicationDbContext context, ILogger<ReportRepository> logger)
            : base(context)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }
        
        /// <inheritdoc/>
        public async Task<IEnumerable<ReportModels>> GetByUserIdAsync(string userId)
        {
            return await DbSet
                .Where(r => r.UserId == userId)
                .OrderByDescending(r => r.CreatedAt)
                .ToListAsync();
        }
        
        /// <inheritdoc/>
        public async Task<IEnumerable<ReportModels>> GetByNameAsync(string name)
        {
            return await DbSet
                .Where(r => r.Name.Contains(name))
                .OrderByDescending(r => r.CreatedAt)
                .ToListAsync();
        }
        
        /// <inheritdoc/>
        public async Task<IEnumerable<ReportModels>> GetByTypeAsync(string type)
        {
            return await DbSet
                .Where(r => r.Type == type)
                .OrderByDescending(r => r.CreatedAt)
                .ToListAsync();
        }
        
        /// <inheritdoc/>
        public async Task<IEnumerable<ReportModels>> GetByTimeRangeAsync(DateTime startTime, DateTime endTime)
        {
            return await DbSet
                .Where(r => r.CreatedAt >= startTime && r.CreatedAt <= endTime)
                .OrderByDescending(r => r.CreatedAt)
                .ToListAsync();
        }
        
        /// <inheritdoc/>
        public async Task<IEnumerable<ReportModels>> GetScheduledReportsAsync()
        {
            return await DbSet
                .Where(r => !string.IsNullOrEmpty(r.Schedule))
                .OrderBy(r => r.Name)
                .ToListAsync();
        }
        
        /// <inheritdoc/>
        public async Task<ReportModels> UpdateScheduleAsync(string id, string schedule)
        {
            var report = await DbSet.FindAsync(id);
            
            if (report == null)
            {
                throw new KeyNotFoundException($"Report with ID {id} not found");
            }
            
            report.Schedule = schedule;
            report.UpdatedAt = DateTime.UtcNow;
            
            await Context.SaveChangesAsync();
            
            return report;
        }
        
        /// <inheritdoc/>
        public async Task<bool> UserHasAccessAsync(string reportId, string userId)
        {
            var report = await DbSet.FindAsync(reportId);
            
            if (report == null)
            {
                return false;
            }
            
            // User has access if they created the report
            return report.UserId == userId;
        }
    }
}