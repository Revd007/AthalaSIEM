using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Backend.Models;
using Backend.DTOs;
using Backend.Services;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;

namespace Backend.Controllers
{
    /// <summary>
    /// Controller for dashboard operations
    /// </summary>
    [ApiController]
    [Route("api/[controller]")]
    [Authorize]
    public class DashboardsController : ControllerBase
    {
        private readonly IDashboardService _dashboardService;
        private readonly IAuthService _authService;
        private readonly ILogger<DashboardsController> _logger;
        
        /// <summary>
        /// Initializes a new instance of the <see cref="DashboardsController"/> class
        /// </summary>
        /// <param name="dashboardService">The dashboard service</param>
        /// <param name="authService">The authentication service</param>
        /// <param name="logger">The logger</param>
        public DashboardsController(
            IDashboardService dashboardService,
            IAuthService authService,
            ILogger<DashboardsController> logger)
        {
            _dashboardService = dashboardService ?? throw new ArgumentNullException(nameof(dashboardService));
            _authService = authService ?? throw new ArgumentNullException(nameof(authService));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }
        
        /// <summary>
        /// Gets all dashboards
        /// </summary>
        /// <returns>All dashboards</returns>
        [HttpGet]
        [Authorize(Roles = "Admin")]
        public async Task<ActionResult<IEnumerable<DashboardModels>>> GetAllDashboards()
        {
            try
            {
                var dashboards = await _dashboardService.GetAllDashboardsAsync();
                return Ok(dashboards);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting all dashboards");
                return StatusCode(500, "An error occurred while retrieving dashboards");
            }
        }
        
        /// <summary>
        /// Gets a dashboard by ID
        /// </summary>
        /// <param name="id">The dashboard ID</param>
        /// <returns>The dashboard</returns>
        [HttpGet("{id}")]
        public async Task<ActionResult<DashboardModels>> GetDashboardById(string id)
        {
            try
            {
                var dashboard = await _dashboardService.GetDashboardByIdAsync(id);
                
                if (dashboard == null)
                {
                    return NotFound();
                }
                
                // Check if the user has access to the dashboard
                var token = HttpContext.Request.Headers["Authorization"].ToString().Replace("Bearer ", "");
                var user = await _authService.GetUserFromTokenAsync(token);
                
                if (user == null)
                {
                    return Unauthorized();
                }
                
                if (dashboard.UserId != user.Id && !dashboard.IsShared && !await _authService.UserHasRoleAsync(user.Id, "Admin"))
                {
                    return Forbid();
                }
                
                return Ok(dashboard);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting dashboard {DashboardId}", id);
                return StatusCode(500, "An error occurred while retrieving the dashboard");
            }
        }
        
        /// <summary>
        /// Gets dashboards by user ID
        /// </summary>
        /// <returns>The dashboards for the current user</returns>
        [HttpGet("my")]
        public async Task<ActionResult<IEnumerable<DashboardModels>>> GetMyDashboards()
        {
            try
            {
                var token = HttpContext.Request.Headers["Authorization"].ToString().Replace("Bearer ", "");
                var user = await _authService.GetUserFromTokenAsync(token);
                
                if (user == null)
                {
                    return Unauthorized();
                }
                
                var dashboards = await _dashboardService.GetDashboardsByUserAsync(user.Id);
                return Ok(dashboards);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting dashboards for current user");
                return StatusCode(500, "An error occurred while retrieving dashboards");
            }
        }
        
        /// <summary>
        /// Gets shared dashboards
        /// </summary>
        /// <returns>The shared dashboards</returns>
        [HttpGet("shared")]
        public async Task<ActionResult<IEnumerable<DashboardModels>>> GetSharedDashboards()
        {
            try
            {
                var dashboards = await _dashboardService.GetSharedDashboardsAsync();
                return Ok(dashboards);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting shared dashboards");
                return StatusCode(500, "An error occurred while retrieving dashboards");
            }
        }
        
        /// <summary>
        /// Creates a new dashboard
        /// </summary>
        /// <param name="dashboard">The dashboard to create</param>
        /// <returns>The created dashboard</returns>
        [HttpPost]
        public async Task<ActionResult<DashboardModels>> CreateDashboard(DashboardModels dashboard)
        {
            try
            {
                var token = HttpContext.Request.Headers["Authorization"].ToString().Replace("Bearer ", "");
                var user = await _authService.GetUserFromTokenAsync(token);
                
                if (user == null)
                {
                    return Unauthorized();
                }
                
                // Set the user ID
                dashboard.UserId = user.Id;
                
                var createdDashboard = await _dashboardService.CreateDashboardAsync(dashboard);
                return CreatedAtAction(nameof(GetDashboardById), new { id = createdDashboard.Id }, createdDashboard);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error creating dashboard");
                return StatusCode(500, "An error occurred while creating the dashboard");
            }
        }
        
        /// <summary>
        /// Updates a dashboard
        /// </summary>
        /// <param name="id">The dashboard ID</param>
        /// <param name="dashboard">The updated dashboard</param>
        /// <returns>The updated dashboard</returns>
        [HttpPut("{id}")]
        public async Task<ActionResult<DashboardModels>> UpdateDashboard(string id, DashboardModels dashboard)
        {
            try
            {
                if (id != dashboard.Id)
                {
                    return BadRequest("Dashboard ID mismatch");
                }
                
                var existingDashboard = await _dashboardService.GetDashboardByIdAsync(id);
                
                if (existingDashboard == null)
                {
                    return NotFound();
                }
                
                // Check if the user has access to update the dashboard
                var token = HttpContext.Request.Headers["Authorization"].ToString().Replace("Bearer ", "");
                var user = await _authService.GetUserFromTokenAsync(token);
                
                if (user == null)
                {
                    return Unauthorized();
                }
                
                if (existingDashboard.UserId != user.Id && !await _authService.UserHasRoleAsync(user.Id, "Admin"))
                {
                    return Forbid();
                }
                
                // Preserve the user ID
                dashboard.UserId = existingDashboard.UserId;
                
                var updatedDashboard = await _dashboardService.UpdateDashboardAsync(dashboard);
                return Ok(updatedDashboard);
            }
            catch (KeyNotFoundException)
            {
                return NotFound();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error updating dashboard {DashboardId}", id);
                return StatusCode(500, "An error occurred while updating the dashboard");
            }
        }
        
        /// <summary>
        /// Updates a dashboard's layout
        /// </summary>
        /// <param name="id">The dashboard ID</param>
        /// <param name="request">The update request</param>
        /// <returns>The updated dashboard</returns>
        [HttpPut("{id}/layout")]
        public async Task<ActionResult<DashboardModels>> UpdateDashboardLayout(string id, UpdateDashboardLayoutRequest request)
        {
            try
            {
                var existingDashboard = await _dashboardService.GetDashboardByIdAsync(id);
                
                if (existingDashboard == null)
                {
                    return NotFound();
                }
                
                // Check if the user has access to update the dashboard
                var token = HttpContext.Request.Headers["Authorization"].ToString().Replace("Bearer ", "");
                var user = await _authService.GetUserFromTokenAsync(token);
                
                if (user == null)
                {
                    return Unauthorized();
                }
                
                if (existingDashboard.UserId != user.Id && !await _authService.UserHasRoleAsync(user.Id, "Admin"))
                {
                    return Forbid();
                }
                
                var updatedDashboard = await _dashboardService.UpdateDashboardLayoutAsync(id, request.Layout);
                return Ok(updatedDashboard);
            }
            catch (KeyNotFoundException)
            {
                return NotFound();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error updating dashboard layout {DashboardId}", id);
                return StatusCode(500, "An error occurred while updating the dashboard layout");
            }
        }
        
        /// <summary>
        /// Deletes a dashboard
        /// </summary>
        /// <param name="id">The dashboard ID</param>
        /// <returns>No content</returns>
        [HttpDelete("{id}")]
        public async Task<ActionResult> DeleteDashboard(string id)
        {
            try
            {
                var existingDashboard = await _dashboardService.GetDashboardByIdAsync(id);
                
                if (existingDashboard == null)
                {
                    return NotFound();
                }
                
                // Check if the user has access to delete the dashboard
                var token = HttpContext.Request.Headers["Authorization"].ToString().Replace("Bearer ", "");
                var user = await _authService.GetUserFromTokenAsync(token);
                
                if (user == null)
                {
                    return Unauthorized();
                }
                
                if (existingDashboard.UserId != user.Id && !await _authService.UserHasRoleAsync(user.Id, "Admin"))
                {
                    return Forbid();
                }
                
                var result = await _dashboardService.DeleteDashboardAsync(id);
                
                if (!result)
                {
                    return NotFound();
                }
                
                return NoContent();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error deleting dashboard {DashboardId}", id);
                return StatusCode(500, "An error occurred while deleting the dashboard");
            }
        }
    }
} 