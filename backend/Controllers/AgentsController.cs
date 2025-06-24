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
    [Authorize(Roles = "Admin,Operator")]
    [EnableCors("AllowFrontend")]  // Enable CORS for this controller
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

        // Special endpoint just for CORS preflight
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
                // Ensure CORS headers are added for this response
                var origin = Request.Headers["Origin"].ToString();
                if (!string.IsNullOrEmpty(origin))
                {
                    Response.Headers["Access-Control-Allow-Origin"] = origin;
                    Response.Headers["Access-Control-Allow-Credentials"] = "true";
                }

                var agents = await _agentService.GetAllAgentsAsync();
                var agentDtos = agents.Select(a => MapToDto(a)).ToList();
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
        public async Task<ActionResult<AgentDto>> GetAgentById(string id)
        {
            var agent = await _agentService.GetAgentByIdAsync(id);
            if (agent == null)
            {
                return NotFound(new { Error = $"Agent with ID {id} not found" });
            }
            
            return Ok(MapToDto(agent));
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
            
            // Get registration key from request headers or body
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

                // Get configured registration key
                var configRegistrationKey = _configuration["AgentSettings:ServerRegistrationKey"];
                if (string.IsNullOrEmpty(configRegistrationKey))
                {
                    _logger.LogError("Registration key not configured in server settings");
                    return StatusCode(500, new { Error = "Server registration key not configured" });
                }

                // Validate registration key (secure comparison to prevent timing attacks)
                bool keysMatch = SecureCompare(keyValue, configRegistrationKey);
                
                if (!keysMatch)
                {
                    _logger.LogWarning("Invalid registration key attempt from {IP}", HttpContext.Connection.RemoteIpAddress);
                    return Unauthorized(new { Error = "Invalid registration key" });
                }

                _logger.LogInformation("Valid registration key from {IP} for host {Hostname}", 
                    HttpContext.Connection.RemoteIpAddress, agentDto.Hostname);

                // Register the agent
                var result = await _agentService.RegisterAgentAsync(agentDto);
                if (!result.Success)
                {
                    return BadRequest(new { Error = result.ErrorMessage });
                }

                _logger.LogInformation("Successfully registered agent {AgentId} for host {Hostname}", 
                    result.AgentId, agentDto.Hostname);

                // Return agent ID and API key
                return Ok(new 
                { 
                    AgentId = result.AgentId,
                    ApiKey = result.ApiKey,
                Success = result.Success
            });
        }

        /// <summary>
        /// Updates an agent's status
        /// </summary>
        /// <param name="id">The agent ID</param>
        /// <param name="statusDto">The status update data</param>
        /// <returns>The updated agent</returns>
        [HttpPut("{id}/status")]
        [ProducesResponseType(StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status400BadRequest)]
        [ProducesResponseType(StatusCodes.Status404NotFound)]
        public async Task<ActionResult<AgentDto>> UpdateAgentStatus(string id, [FromBody] UpdateStatusDto statusDto)
        {
            if (statusDto == null || string.IsNullOrEmpty(statusDto.Status))
            {
                return BadRequest(new { Error = "Status is required" });
            }
            
            if (!Enum.TryParse<AgentStatus>(statusDto.Status, true, out var status))
            {
                return BadRequest(new { Error = $"Invalid agent status: {statusDto.Status}" });
            }
            
            var agent = await _agentService.GetAgentByIdAsync(id);
            if (agent == null)
            {
                return NotFound(new { Error = $"Agent with ID {id} not found" });
            }
            
            var updatedAgent = await _agentService.UpdateAgentStatusAsync(id, status);
            return Ok(MapToDto(updatedAgent));
        }

        /// <summary>
        /// Updates an agent's configuration
        /// </summary>
        /// <param name="id">The agent ID</param>
        /// <param name="configDto">The configuration update data</param>
        /// <returns>The updated agent</returns>
        [HttpPut("{id}/config")]
        [ProducesResponseType(StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status400BadRequest)]
        [ProducesResponseType(StatusCodes.Status404NotFound)]
        public async Task<ActionResult<AgentDto>> UpdateAgentConfig(string id, [FromBody] AgentConfigDto configDto)
        {
            if (configDto == null)
            {
                return BadRequest(new { Error = "Configuration data is required" });
            }
            
            var agent = await _agentService.GetAgentByIdAsync(id);
            if (agent == null)
            {
                return NotFound(new { Error = $"Agent with ID {id} not found" });
            }
            
            var updatedAgent = await _agentService.UpdateAgentConfigAsync(id, configDto);
            return Ok(MapToDto(updatedAgent));
        }

        /// <summary>
        /// Gets logs from an agent
        /// </summary>
        /// <param name="id">The agent ID</param>
        /// <param name="count">The number of logs to return</param>
        /// <param name="offset">The offset to start from</param>
        /// <returns>The agent logs</returns>
        [HttpGet("{id}/logs")]
        [ProducesResponseType(StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status404NotFound)]
        public async Task<ActionResult<PaginatedResult<LogEntryDto>>> GetAgentLogs(string id, [FromQuery] int count = 100, [FromQuery] int offset = 0)
        {
            var agent = await _agentService.GetAgentByIdAsync(id);
            if (agent == null)
            {
                return NotFound(new { Error = $"Agent with ID {id} not found" });
            }
            
            // This would be implemented in a LogService
            var logService = HttpContext.RequestServices.GetService(typeof(ILogService)) as ILogService;
            
            if (logService == null)
            {
                return NotFound("Log service is not available");
            }
            
            var logs = await logService.GetLogsByAgentIdAsync(id, offset, count);
            
            return Ok(logs);
        }

        /// <summary>
        /// Gets health reports from an agent
        /// </summary>
        /// <param name="id">The agent ID</param>
        /// <param name="count">The number of reports to return</param>
        /// <param name="offset">The offset to start from</param>
        /// <returns>The agent health reports</returns>
        [HttpGet("{id}/health")]
        [ProducesResponseType(StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status404NotFound)]
        public async Task<ActionResult<PaginatedResult<AgentHealthReportDto>>> GetAgentHealthReports(string id, [FromQuery] int count = 10, [FromQuery] int offset = 0)
        {
            var agent = await _agentService.GetAgentByIdAsync(id);
            if (agent == null)
            {
                return NotFound(new { Error = $"Agent with ID {id} not found" });
            }
            
            // This would be implemented in a AgentService
            var healthReports = await _agentService.GetAgentHealthReportsAsync(id, offset, count);
            
            return Ok(healthReports);
        }

        /// <summary>
        /// Deletes an agent
        /// </summary>
        /// <param name="id">The agent ID</param>
        /// <returns>Success message</returns>
        [HttpDelete("{id}")]
        [ProducesResponseType(StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status404NotFound)]
        public async Task<ActionResult> DeleteAgent(string id)
        {
            var agent = await _agentService.GetAgentByIdAsync(id);
            if (agent == null)
            {
                return NotFound(new { Error = $"Agent with ID {id} not found" });
            }
            
            var result = await _agentService.DeleteAgentAsync(id);
            if (result)
            {
                return Ok(new { Message = $"Agent {id} deleted successfully" });
            }
            else
            {
                return StatusCode(500, new { Error = $"Failed to delete agent {id}" });
            }
        }

        /// <summary>
        /// Generates a new API key for an agent
        /// </summary>
        /// <param name="id">The agent ID</param>
        /// <returns>The new API key</returns>
        [HttpPost("{id}/rotate-key")]
        [ProducesResponseType(StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status404NotFound)]
        public async Task<ActionResult<string>> RotateApiKey(string id)
        {
            var agent = await _agentService.GetAgentByIdAsync(id);
            if (agent == null)
            {
                return NotFound(new { Error = $"Agent with ID {id} not found" });
            }
            
            var newKey = await _agentService.RotateApiKeyAsync(id);
            return Ok(new { ApiKey = newKey });
        }

        /// <summary>
        /// Creates a download link for the agent installer
        /// </summary>
        /// <param name="os">Operating system (Windows, Linux)</param>
        /// <param name="version">Agent version</param>
        /// <returns>Download link</returns>
        [HttpGet("download/{os}")]
        [AllowAnonymous]
        [ProducesResponseType(StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status400BadRequest)]
        public async Task<IActionResult> GetAgentDownloadLink(string os, [FromQuery] string version = "latest")
        {
            try
            {
                var installer = await _installerService.GenerateInstallerPackage(os.ToLowerInvariant());
                if (installer == null)
                {
                    return NotFound(new { Error = $"Installer for {os} not found" });
                }

                return File(installer.Content, installer.ContentType, installer.FileName);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error generating installer for {os}", os);
                return StatusCode(500, new { Error = $"Error generating installer: {ex.Message}" });
            }
        }

        /// <summary>
        /// Validates an API key
        /// </summary>
        /// <param name="apiKey">The API key to validate</param>
        /// <returns>Validation result</returns>
        [HttpPost("validate-api-key")]
        [AllowAnonymous]
        [ProducesResponseType(StatusCodes.Status200OK)]
        public async Task<ActionResult<bool>> ValidateApiKey([FromBody] ValidateApiKeyDto request)
        {
            if (request == null || string.IsNullOrEmpty(request.ApiKey))
            {
                return BadRequest(new { Error = "API key is required" });
            }
            
            var result = await _agentService.ValidateApiKeyAsync(request.ApiKey);
            return Ok(new { Valid = result });
        }

        /// <summary>
        /// Maps an agent model to a DTO
        /// </summary>
        /// <param name="agent">The agent model</param>
        /// <returns>The agent DTO</returns>
        private AgentDto? MapToDto(AgentModels? agent)
        {
            if (agent == null) return null;
            
            return new AgentDto
            {
                Id = agent.Id,
                Name = agent.Name,
                Status = agent.Status.ToString(),
                Type = agent.Type.ToString(),
                Hostname = agent.Hostname,
                IpAddress = agent.IPAddress,
                Version = agent.Version,
                LastConnected = agent.LastConnected,
                InstallDate = agent.InstallDate,
                IsEnabled = agent.IsEnabled,
                OperatingSystem = agent.OperatingSystem,
                CpuUsage = agent.CpuUsage,
                MemoryUsage = agent.MemoryUsage,
                DiskUsage = agent.DiskUsage,
                CollectEventLogs = agent.CollectEventLogs,
                CollectSystemMetrics = agent.CollectSystemMetrics,
                EventLogsToMonitor = string.IsNullOrEmpty(agent.EventLogsToMonitor) 
                    ? new List<string>() 
                    : agent.EventLogsToMonitor.Split(',').ToList(),
                HealthStatus = agent.HealthReports.OrderByDescending(r => r.Timestamp).FirstOrDefault()?.OverallStatus ?? "Unknown",
                Tags = new List<string>() // Default empty list for tags
            };
        }

        /// <summary>
        /// Securely compares two strings to prevent timing attacks
        /// </summary>
        /// <param name="a">First string</param>
        /// <param name="b">Second string</param>
        /// <returns>True if strings are equal, false otherwise</returns>
        private bool SecureCompare(string a, string b)
        {
            if (string.IsNullOrEmpty(a) || string.IsNullOrEmpty(b) || a.Length != b.Length)
            {
                return false;
            }

            return a.Equals(b, StringComparison.Ordinal);
        }

        /// <summary>
        /// Configures an agent
        /// </summary>
        /// <param name="agentId">The agent ID</param>
        /// <param name="configDto">The configuration data</param>
        /// <returns>Success or failure</returns>
        [HttpPut("{agentId}/configure")]
        [Authorize(Policy = "RequireAdminRole")]
        public async Task<IActionResult> ConfigureAgent(
            [FromRoute] Guid agentId,
            [FromBody] AgentConfigDto configDto)
        {
            if (!ModelState.IsValid)
            {
                return BadRequest(new { Errors = ModelState.Values.SelectMany(v => v.Errors).Select(e => e.ErrorMessage) });
            }

            try
            {
                var success = await _agentService.UpdateAgentConfigurationAsync(agentId, configDto);
                if (!success)
                {
                    return NotFound(new { Error = "Agent not found" });
                }

                return Ok(new { Message = "Agent configuration updated successfully" });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error configuring agent {AgentId}", agentId);
                return StatusCode(500, new { Error = "An internal server error occurred while configuring the agent" });
            }
        }

        /// <summary>
        /// Receives a heartbeat from an agent
        /// </summary>
        /// <param name="agentId">The agent ID</param>
        /// <param name="heartbeatDto">The heartbeat data</param>
        /// <returns>Success or failure</returns>
        [HttpPost("{agentId}/heartbeat")]
        public async Task<IActionResult> ReceiveHeartbeat(
            [FromRoute] string agentId,
            [FromBody] AgentHeartbeatDto heartbeatDto)
        {
            if (!ModelState.IsValid)
            {
                return BadRequest(new { Errors = ModelState.Values.SelectMany(v => v.Errors).Select(e => e.ErrorMessage) });
            }

            try
            {
                // Validate agent ID
                if (string.IsNullOrEmpty(agentId))
                {
                    return BadRequest(new { Error = "Agent ID cannot be empty" });
                }
                
                // Validate API key
                if (!Request.Headers.TryGetValue("X-API-Key", out var apiKey))
                {
                    return Unauthorized(new { Error = "API key is required in the headers" });
                }

                // Convert StringValues to string to avoid null reference warnings
                string apiKeyStr = apiKey.ToString();

                var isValidApiKey = await _agentService.ValidateApiKeyAsync(Guid.Parse(agentId), apiKeyStr);
                if (!isValidApiKey)
                {
                    return Unauthorized(new { Error = "Invalid API key" });
                }

                var agent = await _agentService.RecordHeartbeatAsync(agentId, heartbeatDto);
                if (agent == null)
                {
                    return NotFound(new { Error = "Agent not found" });
                }

                return Ok(new { Message = "Heartbeat received successfully" });
            }
            catch (FormatException ex)
            {
                _logger.LogError(ex, "Invalid agent ID format: {AgentId}", agentId);
                return BadRequest(new { Error = "Invalid agent ID format" });
            }
            catch (KeyNotFoundException ex)
            {
                _logger.LogWarning(ex, "Agent not found: {AgentId}", agentId);
                return NotFound(new { Error = ex.Message });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error processing heartbeat from agent {AgentId}", agentId);
                return StatusCode(500, new { Error = "An internal server error occurred while processing the heartbeat" });
            }
        }

        /// <summary>
        /// Receives system metrics from an agent
        /// </summary>
        /// <param name="agentId">The agent ID</param>
        /// <param name="metrics">The system metrics data</param>
        /// <returns>Success or failure</returns>
        [HttpPost("{agentId}/metrics")]
        public async Task<IActionResult> ReceiveSystemMetrics(
            [FromRoute] Guid agentId,
            [FromBody] SystemMetricsDto metrics)
        {
            if (!ModelState.IsValid)
            {
                return BadRequest(new { Errors = ModelState.Values.SelectMany(v => v.Errors).Select(e => e.ErrorMessage) });
            }

            try
            {
                // Validate agent ID
                if (agentId == Guid.Empty)
                {
                    return BadRequest(new { Error = "Agent ID cannot be empty" });
                }
                
                // Validate API key
                if (!Request.Headers.TryGetValue("X-API-Key", out var apiKey))
                {
                    return Unauthorized(new { Error = "API key is required in the headers" });
                }

                // Convert StringValues to string to avoid null reference warnings
                string apiKeyStr = apiKey.ToString();

                var isValidApiKey = await _agentService.ValidateApiKeyAsync(agentId, apiKeyStr);
                if (!isValidApiKey)
                {
                    return Unauthorized(new { Error = "Invalid API key" });
                }

                // TODO: Implement system metrics processing
                // For now, just return success
                return Ok(new { Message = "System metrics received successfully" });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error processing system metrics from agent {AgentId}", agentId);
                return StatusCode(500, new { Error = "An internal server error occurred while processing the system metrics" });
            }
        }

        /// <summary>
        /// Gets agent configuration
        /// </summary>
        /// <param name="agentId">The agent ID</param>
        /// <returns>Agent configuration</returns>
        [HttpGet("{agentId}/config")]
        public async Task<IActionResult> GetAgentConfiguration(Guid agentId)
        {
            try
            {
                // Validate agent ID
                if (agentId == Guid.Empty)
                {
                    return BadRequest(new { Error = "Agent ID cannot be empty" });
                }
                
                // Validate API key
                if (!Request.Headers.TryGetValue("X-API-Key", out var apiKey))
                {
                    return Unauthorized(new { Error = "API key is required in the headers" });
                }
                
                string apiKeyStr = apiKey.ToString();
                var isValidApiKey = await _agentService.ValidateApiKeyAsync(agentId, apiKeyStr);
                if (!isValidApiKey)
                {
                    return Unauthorized(new { Error = "Invalid API key" });
                }
                
                // Get agent configuration
                var config = await _agentService.GetAgentConfigurationAsync(agentId);
                if (config == null)
                {
                    return NotFound(new { Error = "Agent configuration not found" });
                }
                
                return Ok(config);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting configuration for agent {AgentId}", agentId);
                return StatusCode(500, new { Error = "An internal server error occurred while getting the agent configuration" });
            }
        }

        /// <summary>
        /// Generates a deployment token with pre-configuration
        /// </summary>
        /// <param name="request">The token generation request</param>
        /// <returns>The generated token</returns>
        [HttpPost("generate-token")]
        [Authorize]
        [ProducesResponseType(StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status400BadRequest)]
        [ProducesResponseType(StatusCodes.Status500InternalServerError)]
        public async Task<ActionResult<AgentTokenDto>> GenerateDeploymentToken([FromBody] GenerateTokenRequestDto request)
        {
            try
            {
                if (request == null)
                {
                    return BadRequest(new { Error = "Request data is required" });
                }
                
                // Get the current user
                var userId = User.FindFirstValue(ClaimTypes.NameIdentifier);
                if (string.IsNullOrEmpty(userId))
                {
                    return Unauthorized(new { Error = "User not authenticated" });
                }
                
                // Generate a random token
                var token = Guid.NewGuid().ToString("N");
                var expiresAt = DateTime.UtcNow.AddHours(24); // Token valid for 24 hours
                
                // Save the pre-configuration with the token
                if (request.Configuration != null)
                {
                    await _agentService.SaveAgentPreConfigurationAsync(token, request.Configuration, userId, expiresAt);
                }
                
                // Construct the download URL
                var baseUrl = $"{Request.Scheme}://{Request.Host}";
                var downloadUrl = $"{baseUrl}/api/agents/token-download";
                
                return Ok(new AgentTokenDto
                {
                    Token = token,
                    ExpiresAt = expiresAt,
                    DownloadUrl = downloadUrl
                });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error generating deployment token");
                return StatusCode(500, new { Error = $"Error generating deployment token: {ex.Message}" });
            }
        }
        
        /// <summary>
        /// Downloads an agent installer using a deployment token
        /// </summary>
        /// <param name="token">The deployment token</param>
        /// <param name="type">The installer type</param>
        /// <returns>The installer file</returns>
        [HttpGet("token-download")]
        [AllowAnonymous]
        [ProducesResponseType(StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status400BadRequest)]
        [ProducesResponseType(StatusCodes.Status404NotFound)]
        public async Task<ActionResult> DownloadInstallerWithToken([FromQuery] string token, [FromQuery] string type = "windows")
        {
            try
            {
                if (string.IsNullOrEmpty(token))
                {
                    return BadRequest(new { Error = "Token is required" });
                }
                
                // Validate the token
                var preConfig = await _agentService.GetAgentPreConfigurationAsync(token);
                if (preConfig == null)
                {
                    return BadRequest(new { Error = "Invalid or expired token" });
                }
                
                var installer = await _installerService.GenerateInstallerPackage(type);
                
                if (installer == null)
                {
                    return NotFound(new { Error = $"Installer for type {type} not found" });
                }
                
                // Stream the installer to the client
                return File(installer.Content, installer.ContentType, installer.FileName);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error downloading installer with token");
                return StatusCode(500, new { Error = $"Error downloading installer: {ex.Message}" });
            }
        }
        
        /// <summary>
        /// Registers an agent using a deployment token
        /// </summary>
        /// <param name="agentDto">The agent registration data</param>
        /// <returns>The registration result</returns>
        [HttpPost("token-register")]
        [AllowAnonymous]
        [ProducesResponseType(StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status400BadRequest)]
        [ProducesResponseType(StatusCodes.Status401Unauthorized)]
        [ProducesResponseType(StatusCodes.Status500InternalServerError)]
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
                
                // Validate the token and get the pre-configuration
                var preConfig = await _agentService.GetAgentPreConfigurationAsync(agentDto.DeploymentToken);
                if (preConfig == null)
                {
                    return BadRequest(new { Error = "Invalid or expired deployment token" });
                }
                
                // Apply pre-configuration values to the agent registration
                if (!string.IsNullOrEmpty(preConfig.Name))
                {
                    agentDto.Hostname = preConfig.Name;
                }
                
                // Register the agent
                var result = await _agentService.RegisterAgentAsync(agentDto);
                if (!result.Success)
                {
                    return BadRequest(new { Error = result.ErrorMessage });
                }
                
                // Apply the collectors configuration
                if (preConfig.Collectors != null && preConfig.Collectors.Count > 0)
                {
                    var configDto = new AgentConfigDto
                    {
                        Enabled = true,
                        CollectEventLogs = preConfig.Collectors.Contains("windows"),
                        CollectSystemMetrics = preConfig.Collectors.Contains("metrics"),
                        EnableRealTimeMonitoring = preConfig.Collectors.Contains("network"),
                        IpAddress = preConfig.IpAddress,
                        UseSSL = preConfig.UseSSL
                        // Add other properties as needed
                    };
                    
                    await _agentService.UpdateAgentConfigAsync(result.AgentId, configDto);
                }
                
                // Delete the token to prevent reuse
                await _agentService.DeleteAgentPreConfigurationAsync(agentDto.DeploymentToken);
                
                _logger.LogInformation("Successfully registered agent {AgentId} using deployment token", 
                    result.AgentId);
                
                return Ok(result);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error registering agent with token");
                return StatusCode(500, new { Error = $"Error registering agent: {ex.Message}" });
            }
        }

        // Note: Log ingestion is now handled by the LogsController
        
        /// <summary>
        /// Securely downloads an agent installer using a secure identifier
        /// </summary>
        /// <param name="secureId">The secure download identifier</param>
        /// <returns>The installer file</returns>
        [HttpGet("secure-download/{secureId}")]
        [AllowAnonymous]
        [ProducesResponseType(StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status400BadRequest)]
        [ProducesResponseType(StatusCodes.Status404NotFound)]
        public async Task<ActionResult> SecureDownloadInstaller(string secureId)
        {
            try
            {
                if (string.IsNullOrEmpty(secureId))
                {
                    return BadRequest(new { Error = "Secure download ID is required" });
                }
                
                _logger.LogInformation("Secure download requested with ID: {SecureId}", secureId);
                
                // Check if the secureId matches the configured download code
                var configuredCode = _configuration["InstallerDownloadCode"];
                if (string.IsNullOrEmpty(configuredCode) || secureId != configuredCode)
                {
                    // If there's no matching ID or it's incorrect, return 404
                    _logger.LogWarning("Invalid secure download ID: {SecureId}", secureId);
                    return NotFound(new { Error = "Invalid secure download ID" });
                }
                
                // Default to Windows installer if no platform is specified
                string installerType = "windows";
                
                // Generate the installer package
                var installer = await _installerService.GenerateInstallerPackage(installerType);
                
                if (installer == null)
                {
                    return NotFound(new { Error = $"Installer for type {installerType} not found" });
                }
                
                // Stream the installer to the client
                return File(installer.Content, installer.ContentType, installer.FileName);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error processing secure download for ID {SecureId}", secureId);
                return StatusCode(500, new { Error = $"Error downloading installer: {ex.Message}" });
            }
        }
    }
} 