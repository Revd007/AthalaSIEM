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
    /// Repository for dashboard operations
    /// </summary>
    public class DashboardRepository : Repository<DashboardModels, string>, IDashboardRepository
    {
        private readonly ILogger<DashboardRepository> _logger;
        
        /// <summary>
        /// Initializes a new instance of the <see cref="DashboardRepository"/> class
        /// </summary>
        /// <param name="context">The database context</param>
        /// <param name="logger">The logger</param>
        public DashboardRepository(ApplicationDbContext context, ILogger<DashboardRepository> logger)
            : base(context)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }
        
        /// <inheritdoc/>
        public async Task<IEnumerable<DashboardModels>> GetByUserIdAsync(string userId)
        {
            return await DbSet
                .Where(d => d.UserId == userId)
                .OrderBy(d => d.Name)
                .ToListAsync();
        }
        
        /// <inheritdoc/>
        public async Task<IEnumerable<DashboardModels>> GetByNameAsync(string name)
        {
            return await DbSet
                .Where(d => d.Name.Contains(name))
                .OrderBy(d => d.Name)
                .ToListAsync();
        }
        
        /// <inheritdoc/>
        public async Task<IEnumerable<DashboardModels>> GetByTypeAsync(string type)
        {
            return await DbSet
                .Where(d => d.Type == type)
                .OrderBy(d => d.Name)
                .ToListAsync();
        }
        
        /// <inheritdoc/>
        public async Task<IEnumerable<DashboardModels>> GetSharedDashboardsAsync()
        {
            return await DbSet
                .Where(d => d.IsShared)
                .OrderBy(d => d.Name)
                .ToListAsync();
        }
        
        /// <inheritdoc/>
        public async Task<DashboardModels> UpdateLayoutAsync(string id, string layoutJson)
        {
            var dashboard = await DbSet.FindAsync(id);
            
            if (dashboard == null)
            {
                throw new KeyNotFoundException($"Dashboard with ID {id} not found");
            }
            
            dashboard.LayoutJson = layoutJson;
            dashboard.UpdatedAt = DateTime.UtcNow;
            
            await Context.SaveChangesAsync();
            
            return dashboard;
        }
        
        /// <inheritdoc/>
        public async Task<bool> UserHasAccessAsync(string dashboardId, string userId)
        {
            var dashboard = await DbSet.FindAsync(dashboardId);
            
            if (dashboard == null)
            {
                return false;
            }
            
            // User has access if they created the dashboard or if it's shared
            return dashboard.UserId == userId || dashboard.IsShared;
        }

        public new async Task<DashboardModels?> GetByIdAsync(string id)
        {
            return await DbSet
                .Include(d => d.Widgets)
                .FirstOrDefaultAsync(d => d.Id == id);
        }

        public new async Task<IEnumerable<DashboardModels>> GetAllAsync()
        {
            return await DbSet
                .Include(d => d.Widgets)
                .ToListAsync();
        }

        public async Task<IEnumerable<DashboardModels>> GetSharedAsync()
        {
            return await DbSet
                .Include(d => d.Widgets)
                .Where(d => d.IsShared)
                .ToListAsync();
        }

        public new async Task<DashboardModels> AddAsync(DashboardModels dashboard)
        {
            await DbSet.AddAsync(dashboard);
            await Context.SaveChangesAsync();
            return dashboard;
        }

        public new async Task<DashboardModels> UpdateAsync(DashboardModels dashboard)
        {
            Context.Entry(dashboard).State = EntityState.Modified;
            await Context.SaveChangesAsync();
            return dashboard;
        }

        public async Task DeleteAsync(string id)
        {
            var dashboard = await DbSet.FindAsync(id);
            if (dashboard != null)
            {
                DbSet.Remove(dashboard);
                await Context.SaveChangesAsync();
            }
        }
    }
} 