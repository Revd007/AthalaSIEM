using Microsoft.AspNetCore.Mvc;
using Backend.Models;
using Backend.Data;
using Microsoft.AspNetCore.Authorization;
using System.Security.Claims;
using Backend.DTOs;
using Backend.Services;
using Microsoft.IdentityModel.Tokens;
using System.Text;
using Microsoft.EntityFrameworkCore;
using System.IdentityModel.Tokens.Jwt;
using System.IO.Compression;
using System.IO;
using System.Text.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using System.Security.Cryptography;
using Microsoft.AspNetCore.Http;
using System.Net;
using Microsoft.AspNetCore.Cors;

namespace Backend.Controllers
{
    /// <summary>
    /// Controller for managing agents
    /// </summary>
    [ApiController]
    [Route("api/[controller]")]
    [Authorize] // Allow any authenticated user to view agents
    [EnableCors("AllowFrontend")]
    public class AgentsController : ControllerBase
    {
        private readonly ApplicationDbContext _context;
        private readonly IConfiguration _configuration;
        private readonly IAgentService _agentService;
        private readonly IInstallerService _installerService;
        private readonly ILogger<AgentsController> _logger;
        private readonly IAlertService _alertService;

        public AgentsController(
            ApplicationDbContext context,
            IConfiguration configuration,
            IAgentService agentService,
            IInstallerService installerService,
            IAlertService alertService,
            ILogger<AgentsController> logger)
        {
            _context = context ?? throw new ArgumentNullException(nameof(context));
            _configuration = configuration ?? throw new ArgumentNullException(nameof(configuration));
            _agentService = agentService ?? throw new ArgumentNullException(nameof(agentService));
            _installerService = installerService ?? throw new ArgumentNullException(nameof(installerService));
            _alertService = alertService ?? throw new ArgumentNullException(nameof(alertService));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }

        [HttpOptions]
        [Route("{*url}")]
        [AllowAnonymous]
        public IActionResult HandleOptions()
        {
            return Ok();
        }

        private bool ValidateApiKey(string apiKey)
        {
            if (string.IsNullOrEmpty(apiKey))
            {
                return false;
            }

            var configApiKey = _configuration["ApiKey"];
            return apiKey == configApiKey;
        }

        /// <summary>
        /// Gets all agents
        /// </summary>
        /// <returns>All agents</returns>
        [HttpGet]
        [ProducesResponseType(StatusCodes.Status200OK)]
        public async Task<ActionResult<IEnumerable<AgentDto>>> GetAllAgents()
        {
            try
            {
                var origin = Request.Headers["Origin"].ToString();
                if (!string.IsNullOrEmpty(origin))
                {
                    Response.Headers["Access-Control-Allow-Origin"] = origin;
                    Response.Headers["Access-Control-Allow-Credentials"] = "true";
                }

                // Debug: Log how many agents are in the database
                var totalAgentsInDb = await _context.Agents.CountAsync();
                _logger.LogInformation("Total agents in database: {Count}", totalAgentsInDb);
                
                var agents = await _agentService.GetAllAgentsAsync();
                var agentDtos = agents.Select(a => MapToDto(a)).ToList();
                
                _logger.LogInformation("Returning {Count} agents to frontend", agentDtos.Count);
                
                return Ok(agentDtos);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting all agents");
                return StatusCode(500, new { Error = "An internal server error occurred while getting agents" });
            }
        }

        /// <summary>
        /// Gets an agent by ID
        /// </summary>
        /// <param name="id">The agent ID</param>
        /// <returns>The agent</returns>
        [HttpGet("{id}")]
        [ProducesResponseType(StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status404NotFound)]
        [ProducesResponseType(StatusCodes.Status500InternalServerError)]
        public async Task<ActionResult<AgentDto>> GetAgentById(string id)
        {
            try
            {
                var agent = await _agentService.GetAgentByIdAsync(id);
                if (agent == null)
                {
                    return NotFound(new { Error = $"Agent with ID {id} not found" });
                }
                
                return Ok(MapToDto(agent));
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error retrieving agent {AgentId}", id);
                return StatusCode(500, new { Error = "An error occurred while retrieving the agent", Details = ex.Message });
            }
        }

        /// <summary>
        /// Gets agent status/health by ID (alias for GetAgentById for frontend compatibility)
        /// </summary>
        /// <param name="id">The agent ID</param>
        /// <returns>The agent with status information</returns>
        [HttpGet("{id}/status")]
        [ProducesResponseType(StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status404NotFound)]
        [ProducesResponseType(StatusCodes.Status500InternalServerError)]
        public async Task<ActionResult<AgentDto>> GetAgentStatus(string id)
        {
            try
            {
                var agent = await _agentService.GetAgentByIdAsync(id);
                if (agent == null)
                {
                    return NotFound(new { Error = $"Agent with ID {id} not found" });
                }
                
                return Ok(MapToDto(agent));
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error retrieving agent status {AgentId}", id);
                return StatusCode(500, new { Error = "An error occurred while retrieving the agent status", Details = ex.Message });
            }
        }

        /// <summary>
        /// Gets agents by status
        /// </summary>
        /// <param name="status">The agent status</param>
        /// <returns>Agents with the specified status</returns>
        [HttpGet("status/{status}")]
        [ProducesResponseType(StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status400BadRequest)]
        public async Task<ActionResult<IEnumerable<AgentDto>>> GetAgentsByStatus(string status)
        {
            if (!Enum.TryParse<AgentStatus>(status, true, out var agentStatus))
            {
                return BadRequest(new { Error = $"Invalid agent status: {status}" });
            }
            
            var agents = await _agentService.GetAgentsByStatusAsync(agentStatus);
            var agentDtos = agents.Select(a => MapToDto(a)).ToList();
            return Ok(agentDtos);
        }

        /// <summary>
        /// Gets agents by type
        /// </summary>
        /// <param name="type">The agent type</param>
        /// <returns>Agents with the specified type</returns>
        [HttpGet("type/{type}")]
        [ProducesResponseType(StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status400BadRequest)]
        public async Task<ActionResult<IEnumerable<AgentDto>>> GetAgentsByType(string type)
        {
            if (!Enum.TryParse<AgentType>(type, true, out var agentType))
            {
                return BadRequest(new { Error = $"Invalid agent type: {type}" });
            }
            
            var agents = await _agentService.GetAgentsByTypeAsync(agentType);
            var agentDtos = agents.Select(a => MapToDto(a)).ToList();
            return Ok(agentDtos);
        }

        /// <summary>
        /// Registers a new agent
        /// </summary>
        /// <param name="agentDto">The agent registration data</param>
        /// <returns>The registered agent</returns>
        [HttpPost("register")]
        [AllowAnonymous]
        [ProducesResponseType(StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status400BadRequest)]
        [ProducesResponseType(StatusCodes.Status401Unauthorized)]
        [ProducesResponseType(StatusCodes.Status500InternalServerError)]
        public async Task<ActionResult<AgentRegistrationResultDto>> RegisterAgent([FromBody] AgentRegistrationDto agentDto)
        {
            if (agentDto == null)
            {
                return BadRequest(new { Error = "Agent data is required" });
            }
            
            string? keyValue = null;
            if (Request.Headers.TryGetValue("X-Registration-Key", out var headerKey))
            {
                keyValue = headerKey;
            }
            else if (!string.IsNullOrEmpty(agentDto.RegistrationKey))
            {
                keyValue = agentDto.RegistrationKey;
            }
                
            if (string.IsNullOrEmpty(keyValue))
            {
                _logger.LogWarning("Registration attempt without registration key from {IP}", HttpContext.Connection.RemoteIpAddress);
                return Unauthorized(new { Error = "Registration key is required in the headers or body" });
            }

            var configRegistrationKey = _configuration["AgentSettings:ServerRegistrationKey"];
            if (string.IsNullOrEmpty(configRegistrationKey))
            {
                _logger.LogError("Registration key not configured in server settings");
                return StatusCode(500, new { Error = "Server registration key not configured" });
            }

            bool keysMatch = SecureCompare(keyValue, configRegistrationKey);
            
            if (!keysMatch)
            {
                _logger.LogWarning("Invalid registration key attempt from {IP}", HttpContext.Connection.RemoteIpAddress);
                return Unauthorized(new { Error = "Invalid registration key" });
            }

            _logger.LogInformation("Valid registration key from {IP} for host {Hostname}", 
                HttpContext.Connection.RemoteIpAddress, agentDto.Hostname);

            var result = await _agentService.RegisterAgentAsync(agentDto);
            if (!result.Success)
            {
                return BadRequest(new { Error = result.ErrorMessage });
            }

            _logger.LogInformation("Successfully registered agent {AgentId} for host {Hostname}", 
                result.AgentId, agentDto.Hostname);

            return Ok(new 
            { 
                AgentId = result.AgentId,
                ApiKey = result.ApiKey,
                Success = result.Success
            });
        }

        /// <summary>
        /// Updates an agent's basic information
        /// </summary>
        /// <param name="id">The agent ID</param>
        /// <param name="updateDto">The update data</param>
        /// <returns>The updated agent</returns>
        [HttpPut("{id}")]
        [ProducesResponseType(StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status404NotFound)]
        [ProducesResponseType(StatusCodes.Status400BadRequest)]
        public async Task<ActionResult<AgentDto>> UpdateAgent(string id, [FromBody] UpdateAgentDto updateDto)
        {
            if (updateDto == null)
            {
                return BadRequest(new { Error = "Update data is required" });
            }

            try
            {
                var agent = await _agentService.UpdateAgentAsync(id, updateDto);
                return Ok(MapToDto(agent));
            }
            catch (KeyNotFoundException)
            {
                return NotFound(new { Error = $"Agent with ID {id} not found" });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error updating agent {AgentId}", id);
                return StatusCode(500, new { Error = "An error occurred while updating the agent" });
            }
        }

        /// <summary>
        /// Updates an agent's status
        /// </summary>
        /// <param name="id">The agent ID</param>
        /// <param name="status">The new status</param>
        /// <returns>The updated agent</returns>
        [HttpPut("{id}/status")]
        [ProducesResponseType(StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status404NotFound)]
        public async Task<ActionResult<AgentDto>> UpdateAgentStatus(string id, [FromBody] string status)
        {
            if (!Enum.TryParse<AgentStatus>(status, true, out var agentStatus))
            {
                return BadRequest(new { Error = $"Invalid agent status: {status}" });
            }
            
            var agent = await _agentService.UpdateAgentStatusAsync(id, agentStatus);
            return Ok(MapToDto(agent));
        }

        /// <summary>
        /// Updates an agent's configuration
        /// </summary>
        /// <param name="id">The agent ID</param>
        /// <param name="configDto">The configuration data</param>
        /// <returns>The updated agent</returns>
        [HttpPut("{id}/config")]
        [ProducesResponseType(StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status404NotFound)]
        public async Task<ActionResult<AgentDto>> UpdateAgentConfig(string id, [FromBody] AgentConfigDto configDto)
        {
            try
            {
                var agent = await _agentService.UpdateAgentConfigAsync(id, configDto);
                return Ok(MapToDto(agent));
            }
            catch (KeyNotFoundException)
            {
                return NotFound(new { Error = $"Agent with ID {id} not found" });
            }
        }

        /// <summary>
        /// Deletes an agent
        /// </summary>
        /// <param name="id">The agent ID</param>
        /// <returns>No content</returns>
        [HttpDelete("{id}")]
        [Authorize(Roles = "Admin")]
        [ProducesResponseType(StatusCodes.Status204NoContent)]
        [ProducesResponseType(StatusCodes.Status404NotFound)]
        [ProducesResponseType(StatusCodes.Status500InternalServerError)]
        public async Task<IActionResult> DeleteAgent(string id)
        {
            try
            {
                var agent = await _agentService.GetAgentByIdAsync(id);
                if (agent == null)
                {
                    return NotFound(new { Error = $"Agent with ID {id} not found" });
                }

                // Delete related records first to avoid foreign key constraint violations
                var agentId = id;
                
                // Delete agent configuration
                var config = await _context.AgentConfigs.FirstOrDefaultAsync(c => c.AgentId == agentId);
                if (config != null)
                {
                    _context.AgentConfigs.Remove(config);
                }

                // Delete log entries
                var logEntries = await _context.LogEntries.Where(l => l.AgentId == agentId).ToListAsync();
                if (logEntries.Any())
                {
                    _context.LogEntries.RemoveRange(logEntries);
                }

                // Delete alerts
                var alerts = await _context.Alerts.Where(a => a.AgentId == agentId).ToListAsync();
                if (alerts.Any())
                {
                    _context.Alerts.RemoveRange(alerts);
                }

                // Delete health reports
                var healthReports = await _context.AgentHealthReports.Where(hr => hr.AgentId == agentId).ToListAsync();
                if (healthReports.Any())
                {
                    _context.AgentHealthReports.RemoveRange(healthReports);
                }

                // Delete heartbeats
                var heartbeats = await _context.AgentHeartbeats.Where(h => h.AgentId == agentId).ToListAsync();
                if (heartbeats.Any())
                {
                    _context.AgentHeartbeats.RemoveRange(heartbeats);
                }

                // Delete health metrics
                var healthMetrics = await _context.HealthMetrics.Where(h => h.AgentId == agentId).ToListAsync();
                if (healthMetrics.Any())
                {
                    _context.HealthMetrics.RemoveRange(healthMetrics);
                }

                // Delete security events
                var securityEvents = await _context.SecurityEvents.Where(s => s.AgentId == agentId).ToListAsync();
                if (securityEvents.Any())
                {
                    _context.SecurityEvents.RemoveRange(securityEvents);
                }

                // Now delete the agent itself
                _context.Agents.Remove(agent);
                await _context.SaveChangesAsync();

                _logger.LogInformation("Successfully deleted agent {AgentId} and all related records", id);
                return NoContent();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error deleting agent {AgentId}: {Message}", id, ex.Message);
                return StatusCode(500, new { Error = "An error occurred while deleting the agent", Details = ex.Message });
            }
        }

        /// <summary>
        /// Gets agent health metrics
        /// </summary>
        /// <param name="id">The agent ID</param>
        /// <returns>The agent health metrics</returns>
        [HttpGet("{id}/health")]
        [ProducesResponseType(StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status404NotFound)]
        public async Task<ActionResult<object>> GetAgentHealth(string id)
        {
            var agent = await _agentService.GetAgentByIdAsync(id);
            if (agent == null)
            {
                return NotFound(new { Error = $"Agent with ID {id} not found" });
            }
            
            var health = new
            {
                AgentId = id,
                Status = agent.Status.ToString(),
                LastHeartbeat = agent.LastHeartbeat,
                CpuUsage = agent.CpuUsage ?? 0,
                MemoryUsage = agent.MemoryUsage ?? 0,
                DiskUsage = agent.DiskUsage ?? 0
            };
            
            return Ok(health);
        }

        /// <summary>
        /// Records a heartbeat from an agent
        /// </summary>
        /// <param name="id">The agent ID</param>
        /// <param name="heartbeat">The heartbeat data</param>
        /// <returns>The updated agent</returns>
        [HttpPost("{id}/heartbeat")]
        [AllowAnonymous]
        [ProducesResponseType(StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status404NotFound)]
        public async Task<ActionResult<AgentDto>> RecordHeartbeat(string id, [FromBody] AgentHeartbeatDto heartbeat)
        {
            try
            {
                var agent = await _agentService.RecordHeartbeatAsync(id, heartbeat);
                return Ok(MapToDto(agent));
            }
            catch (KeyNotFoundException)
            {
                return NotFound(new { Error = $"Agent with ID {id} not found" });
            }
        }

        /// <summary>
        /// Downloads the agent installer for a specific OS
        /// </summary>
        /// <param name="os">The operating system</param>
        /// <returns>The installer file</returns>
        [HttpGet("download/{os}")]
        [ProducesResponseType(StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status400BadRequest)]
        public async Task<IActionResult> DownloadAgent(string os)
        {
            try
            {
                var installer = await _installerService.GenerateInstallerPackage(os);
                if (installer == null)
                {
                    return BadRequest(new { Error = $"Installer for {os} is not available" });
                }
                
                return File(installer.Content ?? Array.Empty<byte>(), installer.ContentType ?? "application/octet-stream", installer.FileName ?? "installer.msi");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error generating installer for {OS}", os);
                return StatusCode(500, new { Error = "An error occurred while generating the installer" });
            }
        }

        /// <summary>
        /// Registers a new agent using deployment token (alternative endpoint)
        /// </summary>
        [HttpPost("register-with-token")]
        [AllowAnonymous]
        [ProducesResponseType(StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status400BadRequest)]
        public async Task<ActionResult<AgentRegistrationResultDto>> RegisterAgentWithToken([FromBody] AgentRegistrationDto agentDto)
        {
            try
            {
                if (agentDto == null)
                {
                    return BadRequest(new { Error = "Agent data is required" });
                }
                
                if (string.IsNullOrEmpty(agentDto.DeploymentToken))
                {
                    return BadRequest(new { Error = "Deployment token is required" });
                }
                
                var preConfig = await _agentService.GetAgentPreConfigurationAsync(agentDto.DeploymentToken);
                if (preConfig == null)
                {
                    return BadRequest(new { Error = "Invalid or expired deployment token" });
                }
                
                if (!string.IsNullOrEmpty(preConfig.Name))
                {
                    agentDto.Hostname = preConfig.Name;
                }
                
                var result = await _agentService.RegisterAgentAsync(agentDto);
                if (!result.Success)
                {
                    return BadRequest(new { Error = result.ErrorMessage });
                }
                
                // Note: Collector configuration is handled separately via agent configuration
                // The preConfig.Collectors list is used during agent installation, not via AgentConfigDto
                
                return Ok(result);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error registering agent with token");
                return StatusCode(500, new { Error = "An error occurred while registering the agent" });
            }
        }

        /// <summary>
        /// Maps an AgentModels to an AgentDto
        /// </summary>
        private AgentDto MapToDto(AgentModels agent)
        {
            var eventLogsList = new List<string>();
            if (!string.IsNullOrEmpty(agent.Configuration?.EventLogsToMonitor))
            {
                eventLogsList.AddRange(agent.Configuration.EventLogsToMonitor.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries));
            }
            
            return new AgentDto
            {
                Id = agent.Id,
                Name = agent.Name ?? agent.Hostname ?? "Unknown",
                Hostname = agent.Hostname ?? "Unknown",
                IpAddress = agent.IPAddress ?? "Unknown",
                OperatingSystem = agent.OperatingSystem ?? "Unknown",
                Version = agent.Version ?? agent.AgentVersion ?? "Unknown",
                Status = agent.Status.ToString(),
                Type = agent.Type.ToString(),
                LastConnected = agent.LastConnected,
                InstallDate = agent.InstallDate,
                IsEnabled = agent.IsEnabled,
                CpuUsage = agent.CpuUsage ?? 0,
                MemoryUsage = agent.MemoryUsage ?? 0,
                DiskUsage = agent.DiskUsage ?? 0,
                HealthStatus = agent.Status == AgentStatus.Online ? "Healthy" : 
                              agent.Status == AgentStatus.Offline ? "Offline" : "Unknown",
                EventLogsToMonitor = eventLogsList,
                CollectEventLogs = agent.CollectEventLogs,
                CollectSystemMetrics = agent.Configuration?.CollectSystemMetrics ?? true,
                Tags = new List<string>() // Tags not available in AgentModels
            };
        }

        /// <summary>
        /// Securely compares two strings to prevent timing attacks
        /// </summary>
        private bool SecureCompare(string a, string b)
        {
            if (a == null || b == null || a.Length != b.Length)
            {
                return false;
            }
            
            int result = 0;
            for (int i = 0; i < a.Length; i++)
            {
                result |= a[i] ^ b[i];
            }
            
            return result == 0;
        }
    }
}
