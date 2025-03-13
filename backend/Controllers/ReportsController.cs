using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Backend.Models;
using Backend.Services;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;

namespace Backend.Controllers
{
    /// <summary>
    /// Controller for report operations
    /// </summary>
    [ApiController]
    [Route("api/[controller]")]
    [Authorize]
    public class ReportsController : ControllerBase
    {
        private readonly IReportService _reportService;
        private readonly IAuthService _authService;
        private readonly ILogger<ReportsController> _logger;
        
        /// <summary>
        /// Initializes a new instance of the <see cref="ReportsController"/> class
        /// </summary>
        /// <param name="reportService">The report service</param>
        /// <param name="authService">The authentication service</param>
        /// <param name="logger">The logger</param>
        public ReportsController(
            IReportService reportService,
            IAuthService authService,
            ILogger<ReportsController> logger)
        {
            _reportService = reportService ?? throw new ArgumentNullException(nameof(reportService));
            _authService = authService ?? throw new ArgumentNullException(nameof(authService));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }
        
        /// <summary>
        /// Gets all reports
        /// </summary>
        /// <returns>All reports</returns>
        [HttpGet]
        [Authorize(Roles = "Admin")]
        public async Task<ActionResult<IEnumerable<ReportModels>>> GetAllReports()
        {
            try
            {
                var reports = await _reportService.GetAllReportsAsync();
                return Ok(reports);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting all reports");
                return StatusCode(500, "An error occurred while retrieving reports");
            }
        }
        
        /// <summary>
        /// Gets a report by ID
        /// </summary>
        /// <param name="id">The report ID</param>
        /// <returns>The report</returns>
        [HttpGet("{id}")]
        public async Task<ActionResult<ReportModels>> GetReportById(string id)
        {
            try
            {
                var report = await _reportService.GetReportByIdAsync(id);
                
                if (report == null)
                {
                    return NotFound();
                }
                
                // Check if the user has access to the report
                var token = HttpContext.Request.Headers["Authorization"].ToString().Replace("Bearer ", "");
                var user = await _authService.GetUserFromTokenAsync(token);
                
                if (user == null)
                {
                    return Unauthorized();
                }
                
                if (report.UserId != user.Id && !await _authService.UserHasRoleAsync(user.Id, "Admin"))
                {
                    return Forbid();
                }
                
                return Ok(report);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting report {ReportId}", id);
                return StatusCode(500, "An error occurred while retrieving the report");
            }
        }
        
        /// <summary>
        /// Gets reports by user ID
        /// </summary>
        /// <returns>The reports for the current user</returns>
        [HttpGet("my")]
        public async Task<ActionResult<IEnumerable<ReportModels>>> GetMyReports()
        {
            try
            {
                var token = HttpContext.Request.Headers["Authorization"].ToString().Replace("Bearer ", "");
                var user = await _authService.GetUserFromTokenAsync(token);
                
                if (user == null)
                {
                    return Unauthorized();
                }
                
                var reports = await _reportService.GetReportsByUserAsync(user.Id);
                return Ok(reports);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting reports for current user");
                return StatusCode(500, "An error occurred while retrieving reports");
            }
        }
        
        /// <summary>
        /// Gets scheduled reports
        /// </summary>
        /// <returns>The scheduled reports</returns>
        [HttpGet("scheduled")]
        [Authorize(Roles = "Admin,Operator")]
        public async Task<ActionResult<IEnumerable<ReportModels>>> GetScheduledReports()
        {
            try
            {
                var reports = await _reportService.GetScheduledReportsAsync();
                return Ok(reports);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting scheduled reports");
                return StatusCode(500, "An error occurred while retrieving reports");
            }
        }
        
        /// <summary>
        /// Creates a new report
        /// </summary>
        /// <param name="report">The report to create</param>
        /// <returns>The created report</returns>
        [HttpPost]
        public async Task<ActionResult<ReportModels>> CreateReport(ReportModels report)
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
                report.UserId = user.Id;
                
                // Add null check
                if (report?.Parameters == null)
                {
                    return BadRequest("Report parameters cannot be null");
                }

                var parameters = report.Parameters;
                
                var createdReport = await _reportService.CreateReportAsync(report);
                return CreatedAtAction(nameof(GetReportById), new { id = createdReport.Id }, createdReport);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error creating report");
                return StatusCode(500, "An error occurred while creating the report");
            }
        }
        
        /// <summary>
        /// Updates a report
        /// </summary>
        /// <param name="id">The report ID</param>
        /// <param name="report">The updated report</param>
        /// <returns>The updated report</returns>
        [HttpPut("{id}")]
        public async Task<ActionResult<ReportModels>> UpdateReport(string id, ReportModels report)
        {
            try
            {
                if (id != report.Id)
                {
                    return BadRequest("Report ID mismatch");
                }
                
                var existingReport = await _reportService.GetReportByIdAsync(id);
                
                if (existingReport == null)
                {
                    return NotFound();
                }
                
                // Check if the user has access to update the report
                var token = HttpContext.Request.Headers["Authorization"].ToString().Replace("Bearer ", "");
                var user = await _authService.GetUserFromTokenAsync(token);
                
                if (user == null)
                {
                    return Unauthorized();
                }
                
                if (existingReport.UserId != user.Id && !await _authService.UserHasRoleAsync(user.Id, "Admin"))
                {
                    return Forbid();
                }
                
                // Preserve the user ID
                report.UserId = existingReport.UserId;
                
                var updatedReport = await _reportService.UpdateReportAsync(report);
                return Ok(updatedReport);
            }
            catch (KeyNotFoundException)
            {
                return NotFound();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error updating report {ReportId}", id);
                return StatusCode(500, "An error occurred while updating the report");
            }
        }
        
        /// <summary>
        /// Updates a report's schedule
        /// </summary>
        /// <param name="id">The report ID</param>
        /// <param name="request">The update request</param>
        /// <returns>The updated report</returns>
        [HttpPut("{id}/schedule")]
        public async Task<ActionResult<ReportModels>> UpdateReportSchedule(string id, UpdateReportScheduleRequest request)
        {
            try
            {
                var existingReport = await _reportService.GetReportByIdAsync(id);
                
                if (existingReport == null)
                {
                    return NotFound();
                }
                
                // Check if the user has access to update the report
                var token = HttpContext.Request.Headers["Authorization"].ToString().Replace("Bearer ", "");
                var user = await _authService.GetUserFromTokenAsync(token);
                
                if (user == null)
                {
                    return Unauthorized();
                }
                
                if (existingReport.UserId != user.Id && !await _authService.UserHasRoleAsync(user.Id, "Admin"))
                {
                    return Forbid();
                }
                
                var updatedReport = await _reportService.UpdateReportScheduleAsync(id, request.Schedule);
                return Ok(updatedReport);
            }
            catch (KeyNotFoundException)
            {
                return NotFound();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error updating report schedule {ReportId}", id);
                return StatusCode(500, "An error occurred while updating the report schedule");
            }
        }
        
        /// <summary>
        /// Generates a report
        /// </summary>
        /// <param name="id">The report ID</param>
        /// <returns>The generated report content</returns>
        [HttpPost("{id}/generate")]
        public async Task<ActionResult<string>> GenerateReport(string id)
        {
            try
            {
                var existingReport = await _reportService.GetReportByIdAsync(id);
                
                if (existingReport == null)
                {
                    return NotFound();
                }
                
                // Check if the user has access to generate the report
                var token = HttpContext.Request.Headers["Authorization"].ToString().Replace("Bearer ", "");
                var user = await _authService.GetUserFromTokenAsync(token);
                
                if (user == null)
                {
                    return Unauthorized();
                }
                
                if (existingReport.UserId != user.Id && !await _authService.UserHasRoleAsync(user.Id, "Admin") && !await _authService.UserHasRoleAsync(user.Id, "Operator"))
                {
                    return Forbid();
                }
                
                var content = await _reportService.GenerateReportAsync(id);
                return Ok(new { Content = content });
            }
            catch (KeyNotFoundException)
            {
                return NotFound();
            }
            catch (NotSupportedException ex)
            {
                return BadRequest(new { message = ex.Message });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error generating report {ReportId}", id);
                return StatusCode(500, "An error occurred while generating the report");
            }
        }
        
        /// <summary>
        /// Deletes a report
        /// </summary>
        /// <param name="id">The report ID</param>
        /// <returns>No content</returns>
        [HttpDelete("{id}")]
        public async Task<ActionResult> DeleteReport(string id)
        {
            try
            {
                var existingReport = await _reportService.GetReportByIdAsync(id);
                
                if (existingReport == null)
                {
                    return NotFound();
                }
                
                // Check if the user has access to delete the report
                var token = HttpContext.Request.Headers["Authorization"].ToString().Replace("Bearer ", "");
                var user = await _authService.GetUserFromTokenAsync(token);
                
                if (user == null)
                {
                    return Unauthorized();
                }
                
                if (existingReport.UserId != user.Id && !await _authService.UserHasRoleAsync(user.Id, "Admin"))
                {
                    return Forbid();
                }
                
                var result = await _reportService.DeleteReportAsync(id);
                
                if (!result)
                {
                    return NotFound();
                }
                
                return NoContent();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error deleting report {ReportId}", id);
                return StatusCode(500, "An error occurred while deleting the report");
            }
        }
    }
    
    /// <summary>
    /// Update report schedule request
    /// </summary>
    public class UpdateReportScheduleRequest
    {
        /// <summary>
        /// Gets or sets the schedule
        /// </summary>
        public string Schedule { get; set; } = string.Empty;
    }
} 