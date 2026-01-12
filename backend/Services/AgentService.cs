using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Backend.Data.Repositories;
using Backend.DTOs;
using Backend.Models;
using Microsoft.Extensions.Logging;
using System.Text.Json;
using Microsoft.EntityFrameworkCore;
using System.Linq;
using Backend.Data;
using AthalaSIEM.Backend.Models;
using AthalaSIEM.Backend.Repositories;

namespace Backend.Services
{
    /// <summary>
    /// Service for agent operations
    /// </summary>
    public class AgentService : IAgentService
    {
        private readonly Backend.Data.Repositories.ILegacyAgentRepository _agentRepository;
        private readonly ILogger<AgentService> _logger;
        private readonly ApplicationDbContext _dbContext;
        private readonly IAgentDeploymentTokenRepository _tokenRepository;

        /// <summary>
        /// Initializes a new instance of the <see cref="AgentService"/> class
        /// </summary>
        /// <param name="agentRepository">The agent repository</param>
        /// <param name="logger">The logger</param>
        /// <param name="dbContext">The database context</param>
        /// <param name="tokenRepository">The deployment token repository</param>
        public AgentService(
            Backend.Data.Repositories.ILegacyAgentRepository agentRepository, 
            ILogger<AgentService> logger, 
            ApplicationDbContext dbContext,
            IAgentDeploymentTokenRepository tokenRepository)
        {
            _agentRepository = agentRepository ?? throw new ArgumentNullException(nameof(agentRepository));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _dbContext = dbContext ?? throw new ArgumentNullException(nameof(dbContext));
            _tokenRepository = tokenRepository ?? throw new ArgumentNullException(nameof(tokenRepository));
        }

        /// <inheritdoc/>
        public async Task<IEnumerable<AgentModels>> GetAllAgentsAsync()
        {
            return await _agentRepository.GetAllAsync();
        }
        
        /// <inheritdoc/>
        public async Task<AgentModels?> GetAgentByIdAsync(string id)
        {
            return await _agentRepository.GetByIdAsync(id);
        }
        
        /// <inheritdoc/>
        public async Task<IEnumerable<AgentModels>> GetAgentsByStatusAsync(AgentStatus status)
        {
            return await _agentRepository.GetByStatusAsync(status);
        }
        
        /// <inheritdoc/>
        public async Task<IEnumerable<AgentModels>> GetAgentsByTypeAsync(AgentType type)
        {
            return await _agentRepository.GetByTypeAsync(type);
        }
        
        /// <inheritdoc/>
        public async Task<IEnumerable<AgentModels>> GetOfflineAgentsAsync(TimeSpan offlineThreshold)
        {
            return await _agentRepository.GetOfflineAgentsAsync(offlineThreshold);
        }
        
        /// <inheritdoc/>
        public async Task<AgentRegistrationResultDto> RegisterAgentAsync(AgentRegistrationDto registrationDto)
        {
            try
            {
                // Create a new agent
                var agent = new AgentModels
                {
                    Id = Guid.NewGuid().ToString(),
                    Hostname = registrationDto.Hostname,
                    IPAddress = registrationDto.IPAddress,
                    OperatingSystem = registrationDto.OperatingSystem,
                    Version = registrationDto.Version,
                    Status = AgentStatus.Pending,
                    Type = DetermineAgentType(registrationDto.OperatingSystem),
                    InstallDate = DateTime.UtcNow,
                    LastConnected = DateTime.UtcNow,
                    ApiKey = GenerateApiKey(),
                    IsEnabled = true,
                    CreatedAt = DateTime.UtcNow,
                    UpdatedAt = DateTime.UtcNow
                };

                // Save the agent
                await _agentRepository.AddAgentAsync(agent);

                _logger.LogInformation("Registered new agent: {Id} ({Hostname})", agent.Id, agent.Hostname);

                // Return success result
                return new AgentRegistrationResultDto
                {
                    Success = true,
                    AgentId = agent.Id,
                    ApiKey = agent.ApiKey,
                    ErrorMessage = string.Empty
                };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error registering agent");
                return new AgentRegistrationResultDto
                {
                    Success = false,
                    ErrorMessage = $"Error registering agent: {ex.Message}"
                };
            }
        }
        
        /// <inheritdoc/>
        public async Task<AgentModels> UpdateAgentStatusAsync(string id, AgentStatus status)
        {
            return await _agentRepository.UpdateStatusAsync(id, status);
        }
        
        /// <inheritdoc/>
        public async Task<AgentModels> UpdateAgentConfigAsync(string id, AgentConfigDto configDto)
        {
            var agent = await _agentRepository.GetByIdAsync(id);
            
            if (agent == null)
            {
                throw new KeyNotFoundException($"Agent with ID {id} not found");
            }
            
            if (agent.Configuration == null)
            {
                agent.Configuration = new AgentConfigModels
                {
                    Id = Guid.NewGuid().ToString(),
                    AgentId = agent.Id,
                    Enabled = true,
                    CollectEventLogs = true,
                    CollectSystemMetrics = true,
                    EventLogsToMonitor = "Application,System,Security",
                    LogCollectionIntervalSeconds = 60,
                    MaxLogBufferCount = 1000,
                    MaxLogBufferTimeSeconds = 300,
                    EnableRealTimeMonitoring = false,
                    EnableAlerting = true,
                    CpuAlertThresholdPercent = 90,
                    MemoryAlertThresholdPercent = 90,
                    DiskAlertThresholdPercent = 90,
                    CreatedAt = DateTime.UtcNow,
                    UpdatedAt = DateTime.UtcNow
                };
            }
            
            // Update configuration properties
            agent.Configuration.Enabled = configDto.Enabled;
            agent.Configuration.CollectEventLogs = configDto.CollectEventLogs;
            agent.Configuration.CollectSystemMetrics = configDto.CollectSystemMetrics;
            agent.Configuration.EventLogsToMonitor = configDto.EventLogsToMonitor;
            agent.Configuration.LogCollectionIntervalSeconds = configDto.LogCollectionIntervalSeconds;
            agent.Configuration.MaxLogBufferCount = configDto.MaxLogBufferCount;
            agent.Configuration.MaxLogBufferTimeSeconds = configDto.MaxLogBufferTimeSeconds;
            agent.Configuration.EnableRealTimeMonitoring = configDto.EnableRealTimeMonitoring;
            agent.Configuration.EnableAlerting = configDto.EnableAlerting;
            agent.Configuration.CpuAlertThresholdPercent = configDto.CpuAlertThresholdPercent;
            agent.Configuration.MemoryAlertThresholdPercent = configDto.MemoryAlertThresholdPercent;
            agent.Configuration.DiskAlertThresholdPercent = configDto.DiskAlertThresholdPercent;
            agent.Configuration.UpdatedAt = DateTime.UtcNow;
            
            await _agentRepository.UpdateAsync(agent);
            
            _logger.LogInformation("Agent configuration updated: {AgentId}", id);
            
            return agent;
        }
        
        /// <inheritdoc/>
        public async Task<AgentModels> ProcessHeartbeatAsync(string agentId, AgentHeartbeatDto heartbeatDto)
        {
            var agent = await _agentRepository.GetByIdAsync(agentId);
            
            if (agent == null)
            {
                throw new KeyNotFoundException($"Agent with ID {agentId} not found");
            }
            
            // Create heartbeat entity
            var heartbeat = new AgentHeartbeatModels
            {
                Id = Guid.NewGuid().ToString(),
                AgentId = agentId,
                Timestamp = heartbeatDto.Timestamp,
                Status = heartbeatDto.Status,
                CpuUsage = heartbeatDto.CpuUsage,
                MemoryUsage = heartbeatDto.MemoryUsage,
                DiskUsage = heartbeatDto.DiskUsage,
                IpAddress = heartbeatDto.IpAddress,
                AdditionalInfo = heartbeatDto.AdditionalInfo,
                CreatedAt = DateTime.UtcNow
            };
            
            // Update agent with latest health metrics
            agent.LastHeartbeat = DateTime.UtcNow;
            agent.Status = heartbeatDto.Status;
            agent.CpuUsage = heartbeatDto.CpuUsage;
            agent.MemoryUsage = heartbeatDto.MemoryUsage;
            agent.DiskUsage = heartbeatDto.DiskUsage;
            agent.UpdatedAt = DateTime.UtcNow;
            
            // Add health report to database
            // Note: In a real implementation, you would have a health report repository
            // For simplicity, we're just updating the agent here
            await _agentRepository.UpdateAsync(agent);
            
            _logger.LogDebug("Heartbeat processed for agent: {AgentId}", agentId);
            
            return agent;
        }
        
        /// <inheritdoc/>
        public async Task<AgentModels> RecordHeartbeatAsync(string agentId, AgentHeartbeatDto heartbeatDto)
        {
            try
            {
                var agent = await _agentRepository.GetByIdAsync(agentId);
                
                if (agent == null)
                {
                    throw new KeyNotFoundException($"Agent with ID {agentId} not found");
                }
                
                // Create heartbeat entity
                var heartbeat = new AgentHeartbeatModels
                {
                    Id = Guid.NewGuid().ToString(),
                    AgentId = agentId,
                    Timestamp = heartbeatDto.Timestamp,
                    Status = heartbeatDto.Status,
                    CpuUsage = heartbeatDto.CpuUsage,
                    MemoryUsage = heartbeatDto.MemoryUsage,
                    DiskUsage = heartbeatDto.DiskUsage,
                    IpAddress = heartbeatDto.IpAddress,
                    AdditionalInfo = heartbeatDto.AdditionalInfo,
                    CreatedAt = DateTime.UtcNow
                };
                
                // Update agent with latest heartbeat info
                agent.LastHeartbeat = DateTime.UtcNow;
                agent.Status = heartbeatDto.Status;
                agent.CpuUsage = heartbeatDto.CpuUsage;
                agent.MemoryUsage = heartbeatDto.MemoryUsage;
                agent.DiskUsage = heartbeatDto.DiskUsage;
                agent.UpdatedAt = DateTime.UtcNow;
                
                // Record heartbeat and update agent
                await _agentRepository.RecordHeartbeatAsync(agentId, heartbeat);
                
                _logger.LogDebug("Heartbeat recorded for agent: {AgentId}", agentId);
                
                return agent;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error recording heartbeat for agent: {AgentId}", agentId);
                throw;
            }
        }
        
        /// <inheritdoc/>
        public async Task<bool> UpdateAgentConfigurationAsync(Guid agentId, AgentConfigDto configDto)
        {
            try
            {
                var agent = await _agentRepository.GetByIdAsync(agentId.ToString());
                
                if (agent == null)
                {
                    _logger.LogWarning("Agent not found: {AgentId}", agentId);
                    return false;
                }
                
                if (agent.Configuration == null)
                {
                    agent.Configuration = new AgentConfigModels
                    {
                        Id = Guid.NewGuid().ToString(),
                        AgentId = agent.Id,
                        CreatedAt = DateTime.UtcNow
                    };
                }
                
                // Update configuration properties
                agent.Configuration.Enabled = configDto.Enabled;
                agent.Configuration.CollectEventLogs = configDto.CollectEventLogs;
                agent.Configuration.CollectSystemMetrics = configDto.CollectSystemMetrics;
                agent.Configuration.EventLogsToMonitor = configDto.EventLogsToMonitor;
                agent.Configuration.LogCollectionIntervalSeconds = configDto.LogCollectionIntervalSeconds;
                agent.Configuration.MaxLogBufferCount = configDto.MaxLogBufferCount;
                agent.Configuration.MaxLogBufferTimeSeconds = configDto.MaxLogBufferTimeSeconds;
                agent.Configuration.EnableRealTimeMonitoring = configDto.EnableRealTimeMonitoring;
                agent.Configuration.EnableAlerting = configDto.EnableAlerting;
                agent.Configuration.CpuAlertThresholdPercent = configDto.CpuAlertThresholdPercent;
                agent.Configuration.MemoryAlertThresholdPercent = configDto.MemoryAlertThresholdPercent;
                agent.Configuration.DiskAlertThresholdPercent = configDto.DiskAlertThresholdPercent;
                agent.Configuration.UpdatedAt = DateTime.UtcNow;
                
                await _agentRepository.UpdateAsync(agent);
                
                _logger.LogInformation("Agent configuration updated: {AgentId}", agentId);
                
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error updating agent configuration: {AgentId}", agentId);
                return false;
            }
        }
        
        /// <inheritdoc/>
        public async Task<bool> ValidateApiKeyAsync(Guid agentId, string apiKey)
        {
            try
            {
                _logger.LogInformation("🔍 API Key validation - AgentId: {AgentId}, ApiKey provided: {ApiKeyProvided}", 
                    agentId, string.IsNullOrEmpty(apiKey) ? "EMPTY" : "PROVIDED");
                
                var agent = await _agentRepository.GetByIdAsync(agentId.ToString());
                
                if (agent == null)
                {
                    _logger.LogWarning("❌ Agent not found during API key validation: {AgentId}", agentId);
                    return false;
                }
                
                _logger.LogInformation("✅ Agent found - Name: {AgentName}, ApiKey match: {Match}", 
                    agent.Name, agent.ApiKey == apiKey);
                
                return agent.ApiKey == apiKey;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error validating API key for agent: {AgentId}", agentId);
                return false;
            }
        }
        
        /// <inheritdoc/>
        public async Task<AgentConfigDto?> GetAgentConfigurationAsync(Guid agentId)
        {
            try
            {
                var agent = await _agentRepository.GetByIdAsync(agentId.ToString());
                
                if (agent == null || agent.Configuration == null)
                {
                    _logger.LogWarning("Agent or agent configuration not found: {AgentId}", agentId);
                    return null;
                }
                
                return new AgentConfigDto
                {
                    Enabled = agent.Configuration.Enabled,
                    CollectEventLogs = agent.Configuration.CollectEventLogs,
                    CollectSystemMetrics = agent.Configuration.CollectSystemMetrics,
                    EventLogsToMonitor = agent.Configuration.EventLogsToMonitor,
                    LogCollectionIntervalSeconds = agent.Configuration.LogCollectionIntervalSeconds,
                    MaxLogBufferCount = agent.Configuration.MaxLogBufferCount,
                    MaxLogBufferTimeSeconds = agent.Configuration.MaxLogBufferTimeSeconds,
                    EnableRealTimeMonitoring = agent.Configuration.EnableRealTimeMonitoring,
                    EnableAlerting = agent.Configuration.EnableAlerting,
                    CpuAlertThresholdPercent = agent.Configuration.CpuAlertThresholdPercent,
                    MemoryAlertThresholdPercent = agent.Configuration.MemoryAlertThresholdPercent,
                    DiskAlertThresholdPercent = agent.Configuration.DiskAlertThresholdPercent
                };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting agent configuration: {AgentId}", agentId);
                return null;
            }
        }
        
        /// <summary>
        /// Processes a health report from an agent
        /// </summary>
        /// <param name="agentId">The agent ID</param>
        /// <param name="healthReportDto">The health report data</param>
        /// <returns>The updated agent</returns>
        public async Task<AgentModels> ProcessHealthReportAsync(string agentId, AgentHealthReportDto healthReportDto)
        {
            try
            {
                var agent = await _agentRepository.GetByIdAsync(agentId);
                if (agent == null)
                {
                    throw new KeyNotFoundException($"Agent not found: {agentId}");
                }

                // Create a new health report
                var healthReport = new AgentHealthReport
                {
                    AgentId = agentId,
                    Timestamp = DateTime.UtcNow,
                    OverallStatus = healthReportDto.OverallStatus,
                    Metrics = System.Text.Json.JsonSerializer.Serialize(healthReportDto.Metrics)
                };

                // Update agent metrics
                agent.CpuUsage = healthReportDto.CpuUsage;
                agent.MemoryUsage = healthReportDto.MemoryUsage;
                agent.DiskUsage = healthReportDto.DiskUsage;
                agent.LastHeartbeat = DateTime.UtcNow;
                agent.Status = healthReportDto.OverallStatus == "Critical" ? AgentStatus.Warning :
                              healthReportDto.OverallStatus == "Healthy" ? AgentStatus.Online :
                              AgentStatus.Warning;

                // Add health report to agent
                agent.HealthReports.Add(healthReport);

                // Update agent in database
                await _agentRepository.UpdateAsync(agent);

                _logger.LogInformation("Health report processed for agent: {AgentId}", agentId);

                return agent;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error processing health report for agent: {AgentId}", agentId);
                throw;
            }
        }

        public async Task<bool> ProcessHealthReportAsync(Guid agentId, AgentHealthReportDto healthReportDto)
        {
            try
            {
                await ProcessHealthReportAsync(agentId.ToString(), healthReportDto);
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error processing health report for agent: {AgentId}", agentId);
                return false;
            }
        }
        
        /// <inheritdoc/>
        public async Task<bool> DeleteAgentAsync(string id)
        {
            try
            {
                await _agentRepository.DeleteByIdAsync(id);
                _logger.LogInformation("Agent deleted: {AgentId}", id);
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error deleting agent: {AgentId}", id);
                return false;
            }
        }

        /// <summary>
        /// Generates a random API key
        /// </summary>
        /// <returns>The generated API key</returns>
        public string GenerateApiKey()
        {
            return Guid.NewGuid().ToString("N");
        }
        
        /// <summary>
        /// Determines the agent type based on the operating system
        /// </summary>
        /// <param name="operatingSystem">The operating system</param>
        /// <returns>The determined agent type</returns>
        private AgentType DetermineAgentType(string operatingSystem)
        {
            if (operatingSystem.Contains("Windows", StringComparison.OrdinalIgnoreCase))
            {
                return AgentType.Windows;
            }
            else if (operatingSystem.Contains("Linux", StringComparison.OrdinalIgnoreCase))
            {
                return AgentType.Linux;
            }
            else
            {
                return AgentType.Custom;
            }
        }

        /// <summary>
        /// Validates an API key
        /// </summary>
        /// <param name="apiKey">The API key to validate</param>
        /// <returns>True if the API key is valid, otherwise false</returns>
        public async Task<bool> ValidateApiKeyAsync(string apiKey)
        {
            if (string.IsNullOrEmpty(apiKey))
                return false;
                
            var agent = await _dbContext.Agents.FirstOrDefaultAsync(a => a.ApiKey == apiKey);
            return agent != null;
        }
        
        /// <summary>
        /// Validates an API key for a specific agent (string agentId overload)
        /// </summary>
        /// <param name="agentId">The agent ID as string</param>
        /// <param name="apiKey">The API key to validate</param>
        /// <returns>True if the API key is valid for the agent, otherwise false</returns>
        public async Task<bool> ValidateApiKeyAsync(string agentId, string apiKey)
        {
            try
            {
                if (string.IsNullOrEmpty(agentId) || string.IsNullOrEmpty(apiKey))
                {
                    return false;
                }

                var agent = await _dbContext.Agents.FirstOrDefaultAsync(a => a.Id == agentId && a.ApiKey == apiKey);
                
                if (agent == null)
                {
                    _logger.LogWarning("Agent not found or API key mismatch during validation: {AgentId}", agentId);
                    return false;
                }

                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error validating API key for agent: {AgentId}", agentId);
                return false;
            }
        }
        
        /// <summary>
        /// Gets the agent health history
        /// </summary>
        /// <param name="agentId">The agent ID</param>
        /// <param name="count">Maximum number of records to return</param>
        /// <param name="offset">Number of records to skip</param>
        /// <returns>Paginated health history</returns>
        public async Task<PaginatedResult<AgentHealthReportDto>> GetAgentHealthHistoryAsync(string agentId, int count = 10, int offset = 0)
        {
            var query = _dbContext.AgentHealthReports
                .Where(h => h.AgentId == agentId)
                .OrderByDescending(h => h.Timestamp);
                
            var totalCount = await query.CountAsync();
            var items = await query
                .Skip(offset)
                .Take(count)
                .Select(h => new AgentHealthReportDto
                {
                    Id = h.Id,
                    AgentId = h.AgentId,
                    Timestamp = h.Timestamp,
                    OverallStatus = h.OverallStatus,
                    CpuUsage = 0,
                    MemoryUsage = 0,
                    DiskUsage = 0,
                    SystemUptime = 0
                })
                .ToListAsync();
                
            return new PaginatedResult<AgentHealthReportDto>
            {
                Items = items,
                TotalCount = totalCount,
                Page = (offset / count) + 1,
                PageSize = count,
                TotalPages = (int)Math.Ceiling((double)totalCount / count),
                HasPreviousPage = offset > 0,
                HasNextPage = (offset + count) < totalCount
            };
        }
        
        /// <summary>
        /// Updates an agent's API key
        /// </summary>
        /// <param name="agentId">The agent ID</param>
        /// <returns>The new API key</returns>
        public async Task<string> UpdateAgentApiKeyAsync(string agentId)
        {
            var agent = await _dbContext.Agents.FindAsync(agentId);
            if (agent == null)
                throw new KeyNotFoundException($"Agent with ID {agentId} not found");
                
            var newApiKey = Guid.NewGuid().ToString("N");
            agent.ApiKey = newApiKey;
            agent.UpdatedAt = DateTime.UtcNow;
            
            await _dbContext.SaveChangesAsync();
            
            return newApiKey;
        }

        /// <summary>
        /// Saves agent pre-configuration with a deployment token
        /// </summary>
        /// <param name="token">The deployment token</param>
        /// <param name="preConfig">The agent pre-configuration</param>
        /// <param name="userId">The user ID of the creator</param>
        /// <param name="expiresAt">When the token expires</param>
        /// <returns>A task representing the asynchronous operation</returns>
        public async Task SaveAgentPreConfigurationAsync(string token, AgentPreConfigDto preConfig, string userId, DateTime expiresAt)
        {
            try
            {
                var deploymentToken = new AgentDeploymentToken
                {
                    Token = token,
                    CreatedById = userId,
                    ExpiresAt = expiresAt,
                    IpAddress = preConfig.IpAddress,
                    Port = preConfig.Port,
                    AgentName = preConfig.Name,
                    UseSSL = preConfig.UseSSL
                };

                deploymentToken.SetCollectors(preConfig.Collectors.ToArray());
                
                await _tokenRepository.CreateTokenAsync(deploymentToken);
                
                _logger.LogInformation("Saved agent pre-configuration for token {Token}", token);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error saving agent pre-configuration for token {Token}", token);
                throw;
            }
        }
        
        /// <summary>
        /// Gets agent pre-configuration by token
        /// </summary>
        /// <param name="token">The deployment token</param>
        /// <returns>The agent pre-configuration or null if the token is invalid or expired</returns>
        public async Task<AgentPreConfigDto?> GetAgentPreConfigurationAsync(string token)
        {
            try
            {
                var deploymentToken = await _tokenRepository.GetTokenAsync(token);
                
                if (deploymentToken == null)
                {
                    _logger.LogWarning("No pre-configuration found for token {Token}", token);
                    return null;
                }

                return new AgentPreConfigDto
                {
                    IpAddress = deploymentToken.IpAddress,
                    Port = deploymentToken.Port,
                    Name = deploymentToken.AgentName ?? string.Empty,
                    UseSSL = deploymentToken.UseSSL,
                    Collectors = deploymentToken.GetCollectors().ToList()
                };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error retrieving agent pre-configuration for token {Token}", token);
                return null;
            }
        }
        
        /// <summary>
        /// Deletes agent pre-configuration by token
        /// </summary>
        /// <param name="token">The deployment token</param>
        /// <returns>A task representing the asynchronous operation</returns>
        public async Task DeleteAgentPreConfigurationAsync(string token)
        {
            try
            {
                await _tokenRepository.DeleteTokenAsync(token);
                _logger.LogInformation("Deleted agent pre-configuration for token {Token}", token);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error deleting agent pre-configuration for token {Token}", token);
                throw;
            }
        }
    }
} 