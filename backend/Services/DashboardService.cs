using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Backend.Data.Repositories;
using Backend.Models;
using Microsoft.Extensions.Logging;

namespace Backend.Services
{
    /// <summary>
    /// Service for dashboard operations
    /// </summary>
    public class DashboardService : IDashboardService
    {
        private readonly IDashboardRepository _dashboardRepository;
        private readonly ILogger<DashboardService> _logger;
        
        /// <summary>
        /// Initializes a new instance of the <see cref="DashboardService"/> class
        /// </summary>
        /// <param name="dashboardRepository">The dashboard repository</param>
        /// <param name="logger">The logger</param>
        public DashboardService(IDashboardRepository dashboardRepository, ILogger<DashboardService> logger)
        {
            _dashboardRepository = dashboardRepository ?? throw new ArgumentNullException(nameof(dashboardRepository));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }
        
        /// <inheritdoc/>
        public async Task<IEnumerable<DashboardModels>> GetAllDashboardsAsync()
        {
            return await _dashboardRepository.GetAllAsync();
        }
        
        /// <inheritdoc/>
        public async Task<DashboardModels?> GetDashboardByIdAsync(string id)
        {
            return await _dashboardRepository.GetByIdAsync(id);
        }
        
        /// <inheritdoc/>
        public async Task<IEnumerable<DashboardModels>> GetDashboardsByUserAsync(string userId)
        {
            return await _dashboardRepository.GetByUserIdAsync(userId);
        }
        
        /// <inheritdoc/>
        public async Task<IEnumerable<DashboardModels>> GetDashboardsByNameAsync(string name)
        {
            return await _dashboardRepository.GetByNameAsync(name);
        }
        
        /// <inheritdoc/>
        public async Task<IEnumerable<DashboardModels>> GetDashboardsByTypeAsync(string type)
        {
            return await _dashboardRepository.GetByTypeAsync(type);
        }
        
        /// <inheritdoc/>
        public async Task<IEnumerable<DashboardModels>> GetSharedDashboardsAsync()
        {
            return await _dashboardRepository.GetSharedAsync();
        }
        
        /// <inheritdoc/>
        public async Task<DashboardModels> CreateDashboardAsync(DashboardModels dashboard)
        {
            if (dashboard == null)
            {
                throw new ArgumentNullException(nameof(dashboard));
            }
            
            // Set default values if not provided
            dashboard.Id = string.IsNullOrEmpty(dashboard.Id) ? Guid.NewGuid().ToString() : dashboard.Id;
            dashboard.CreatedAt = DateTime.UtcNow;
            dashboard.UpdatedAt = DateTime.UtcNow;
            
            // Add dashboard to database
            await _dashboardRepository.AddAsync(dashboard);
            
            _logger.LogInformation("Dashboard created: {DashboardId} ({Name})", dashboard.Id, dashboard.Name);
            
            return dashboard;
        }
        
        /// <inheritdoc/>
        public async Task<DashboardModels> UpdateDashboardAsync(DashboardModels dashboard)
        {
            if (dashboard == null)
            {
                throw new ArgumentNullException(nameof(dashboard));
            }
            
            var existingDashboard = await _dashboardRepository.GetByIdAsync(dashboard.Id);
            if (existingDashboard == null)
            {
                throw new KeyNotFoundException($"Dashboard with ID {dashboard.Id} not found");
            }
            
            // Update dashboard properties
            existingDashboard.Name = dashboard.Name;
            existingDashboard.Description = dashboard.Description;
            existingDashboard.IsShared = dashboard.IsShared;
            existingDashboard.UpdatedAt = DateTime.UtcNow;
            
            // Update dashboard in database
            await _dashboardRepository.UpdateAsync(existingDashboard);
            
            _logger.LogInformation("Dashboard updated: {DashboardId} ({Name})", existingDashboard.Id, existingDashboard.Name);
            
            return existingDashboard;
        }
        
        /// <inheritdoc/>
        public async Task<DashboardModels> UpdateDashboardLayoutAsync(string id, string layout)
        {
            return await _dashboardRepository.UpdateLayoutAsync(id, layout);
        }
        
        /// <inheritdoc/>
        public async Task<bool> DeleteDashboardAsync(string id)
        {
            try
            {
                await _dashboardRepository.DeleteByIdAsync(id);
                _logger.LogInformation("Dashboard deleted: {DashboardId}", id);
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error deleting dashboard: {DashboardId}", id);
                return false;
            }
        }
    }
} 