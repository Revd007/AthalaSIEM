using System.Collections.Generic;
using System.Threading.Tasks;
using Backend.Models;

namespace Backend.Services
{
    /// <summary>
    /// Service interface for dashboard operations
    /// </summary>
    public interface IDashboardService
    {
        /// <summary>
        /// Gets all dashboards
        /// </summary>
        /// <returns>All dashboards</returns>
        Task<IEnumerable<DashboardModels>> GetAllDashboardsAsync();
        
        /// <summary>
        /// Gets a dashboard by ID
        /// </summary>
        /// <param name="id">The dashboard ID</param>
        /// <returns>The dashboard, or null if not found</returns>
        Task<DashboardModels?> GetDashboardByIdAsync(string id);
        
        /// <summary>
        /// Gets dashboards by user ID
        /// </summary>
        /// <param name="userId">The user ID</param>
        /// <returns>The dashboards for the specified user</returns>
        Task<IEnumerable<DashboardModels>> GetDashboardsByUserAsync(string userId);
        
        /// <summary>
        /// Gets dashboards by name
        /// </summary>
        /// <param name="name">The dashboard name</param>
        /// <returns>The dashboards with the specified name</returns>
        Task<IEnumerable<DashboardModels>> GetDashboardsByNameAsync(string name);
        
        /// <summary>
        /// Gets dashboards by type
        /// </summary>
        /// <param name="type">The dashboard type</param>
        /// <returns>The dashboards with the specified type</returns>
        Task<IEnumerable<DashboardModels>> GetDashboardsByTypeAsync(string type);
        
        /// <summary>
        /// Gets shared dashboards
        /// </summary>
        /// <returns>The shared dashboards</returns>
        Task<IEnumerable<DashboardModels>> GetSharedDashboardsAsync();
        
        /// <summary>
        /// Creates a new dashboard
        /// </summary>
        /// <param name="dashboard">The dashboard to create</param>
        /// <returns>The created dashboard</returns>
        Task<DashboardModels> CreateDashboardAsync(DashboardModels dashboard);
        
        /// <summary>
        /// Updates a dashboard
        /// </summary>
        /// <param name="dashboard">The dashboard to update</param>
        /// <returns>The updated dashboard</returns>
        Task<DashboardModels> UpdateDashboardAsync(DashboardModels dashboard);
        
        /// <summary>
        /// Updates a dashboard's layout
        /// </summary>
        /// <param name="id">The dashboard ID</param>
        /// <param name="layout">The new layout</param>
        /// <returns>The updated dashboard</returns>
        Task<DashboardModels> UpdateDashboardLayoutAsync(string id, string layout);
        
        /// <summary>
        /// Deletes a dashboard
        /// </summary>
        /// <param name="id">The dashboard ID</param>
        /// <returns>True if successful, false otherwise</returns>
        Task<bool> DeleteDashboardAsync(string id);
    }
} 