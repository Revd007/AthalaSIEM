using System.Collections.Generic;
using System.Threading.Tasks;
using Backend.Models;

namespace Backend.Data.Repositories
{
    /// <summary>
    /// Repository interface for dashboard operations
    /// </summary>
    public interface IDashboardRepository : IRepository<DashboardModels, string>
    {
        /// <summary>
        /// Gets dashboards by user ID
        /// </summary>
        /// <param name="userId">The user ID</param>
        /// <returns>Matching dashboards</returns>
        Task<IEnumerable<DashboardModels>> GetByUserIdAsync(string userId);
        
        /// <summary>
        /// Gets dashboards by name
        /// </summary>
        /// <param name="name">The dashboard name</param>
        /// <returns>The dashboards with the specified name</returns>
        Task<IEnumerable<DashboardModels>> GetByNameAsync(string name);
        
        /// <summary>
        /// Gets dashboards by type
        /// </summary>
        /// <param name="type">The dashboard type</param>
        /// <returns>The dashboards with the specified type</returns>
        Task<IEnumerable<DashboardModels>> GetByTypeAsync(string type);
        
        /// <summary>
        /// Gets shared dashboards
        /// </summary>
        /// <returns>Shared dashboards</returns>
        Task<IEnumerable<DashboardModels>> GetSharedDashboardsAsync();
        
        /// <summary>
        /// Updates a dashboard's layout
        /// </summary>
        /// <param name="id">The dashboard ID</param>
        /// <param name="layoutJson">The new layout JSON</param>
        /// <returns>The updated dashboard</returns>
        Task<DashboardModels> UpdateLayoutAsync(string id, string layoutJson);
        
        /// <summary>
        /// Checks if a user has access to a dashboard
        /// </summary>
        /// <param name="dashboardId">The dashboard ID</param>
        /// <param name="userId">The user ID</param>
        /// <returns>True if the user has access, false otherwise</returns>
        Task<bool> UserHasAccessAsync(string dashboardId, string userId);
        Task<IEnumerable<DashboardModels>> GetSharedAsync();
    }
} 