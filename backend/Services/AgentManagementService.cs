using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging;
using Backend.Data;
using Backend.Models;
using Backend.DTOs;

namespace Backend.Services
{
    /// <summary>
    /// Service for managing agents
    /// </summary>
    public class AgentManagementService : IAgentManagementService
    {
        private readonly ApplicationDbContext _dbContext;
        private readonly ILogger<AgentManagementService> _logger;
        
        /// <summary>
        /// Initializes a new instance of the <see cref="AgentManagementService"/> class
        /// </summary>
        /// <param name="dbContext">The database context</param>
        /// <param name="logger">The logger</param>
        public AgentManagementService(ApplicationDbContext dbContext, ILogger<AgentManagementService> logger)
        {
            _dbContext = dbContext ?? throw new ArgumentNullException(nameof(dbContext));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }
        
        /// <inheritdoc/>
        public async Task<IEnumerable<AgentModels>> GetAllAgentsAsync()
        {
            return await _dbContext.Agents
                .Include(a => a.Configuration)
                .ToListAsync();
        }
        
        /// <inheritdoc/>
        public async Task<AgentModels?> GetAgentByIdAsync(string id)
        {
            return await _dbContext.Agents
                .Include(a => a.Configuration)
                .FirstOrDefaultAsync(a => a.Id == id);
        }
        
        /// <inheritdoc/>
        public async Task<AgentModels?> GetAgentByApiKeyAsync(string apiKey)
        {
            return await _dbContext.Agents
                .Include(a => a.Configuration)
                .FirstOrDefaultAsync(a => a.ApiKey == apiKey);
        }
        
        /// <inheritdoc/>
        public async Task<AgentModels> CreateAgentAsync(AgentModels agent)
        {
            if (string.IsNullOrEmpty(agent.ApiKey))
            {
                agent.ApiKey = GenerateApiKey();
            }
            
            agent.CreatedAt = DateTime.UtcNow;
            agent.UpdatedAt = DateTime.UtcNow;
            
            _dbContext.Agents.Add(agent);
            await _dbContext.SaveChangesAsync();
            
            _logger.LogInformation("Created agent {AgentId} with name {AgentName}", agent.Id, agent.Name);
            
            return agent;
        }
        
        /// <inheritdoc/>
        public async Task<AgentModels> UpdateAgentAsync(AgentModels agent)
        {
            var existingAgent = await _dbContext.Agents
                .Include(a => a.Configuration)
                .FirstOrDefaultAsync(a => a.Id == agent.Id);
                
            if (existingAgent == null)
            {
                throw new KeyNotFoundException($"Agent with ID {agent.Id} not found");
            }
            
            // Update properties
            existingAgent.Name = agent.Name;
            existingAgent.Version = agent.Version;
            existingAgent.IPAddress = agent.IPAddress;
            existingAgent.Hostname = agent.Hostname;
            existingAgent.OS = agent.OS;
            existingAgent.Status = agent.Status;
            existingAgent.Type = agent.Type;
            existingAgent.IsEnabled = agent.IsEnabled;
            existingAgent.UpdatedAt = DateTime.UtcNow;
            existingAgent.CollectEventLogs = agent.CollectEventLogs;
            existingAgent.CollectSystemMetrics = agent.CollectSystemMetrics;
            existingAgent.EventLogsToMonitor = agent.EventLogsToMonitor;
            
            // Update configuration if provided
            if (agent.Configuration != null)
            {
                if (existingAgent.Configuration == null)
                {
                    existingAgent.Configuration = new AgentConfigModels
                    {
                        AgentId = agent.Id,
                        CreatedAt = DateTime.UtcNow
                    };
                }
                
                existingAgent.Configuration.ServerUrl = agent.Configuration.ServerUrl;
                existingAgent.Configuration.ConfigRefreshIntervalMinutes = agent.Configuration.ConfigRefreshIntervalMinutes;
                existingAgent.Configuration.LogCollectionIntervalSeconds = agent.Configuration.LogCollectionIntervalSeconds;
                existingAgent.Configuration.MaxLogBufferCount = agent.Configuration.MaxLogBufferCount;
                existingAgent.Configuration.MaxLogBufferTimeSeconds = agent.Configuration.MaxLogBufferTimeSeconds;
                existingAgent.Configuration.EnableRealTimeMonitoring = agent.Configuration.EnableRealTimeMonitoring;
                existingAgent.Configuration.EnableAlerting = agent.Configuration.EnableAlerting;
                existingAgent.Configuration.CpuAlertThresholdPercent = agent.Configuration.CpuAlertThresholdPercent;
                existingAgent.Configuration.MemoryAlertThresholdPercent = agent.Configuration.MemoryAlertThresholdPercent;
                existingAgent.Configuration.DiskAlertThresholdPercent = agent.Configuration.DiskAlertThresholdPercent;
                existingAgent.Configuration.IncludeProcessDetails = agent.Configuration.IncludeProcessDetails;
                existingAgent.Configuration.LogLevelFilters = agent.Configuration.LogLevelFilters;
                existingAgent.Configuration.UseSSL = agent.Configuration.UseSSL;
                existingAgent.Configuration.ValidateServerCertificate = agent.Configuration.ValidateServerCertificate;
                existingAgent.Configuration.LogSources = agent.Configuration.LogSources;
                existingAgent.Configuration.LogFilePaths = agent.Configuration.LogFilePaths;
                existingAgent.Configuration.UpdatedAt = DateTime.UtcNow;
                existingAgent.Configuration.Version++;
            }
            
            await _dbContext.SaveChangesAsync();
            
            _logger.LogInformation("Updated agent {AgentId} with name {AgentName}", existingAgent.Id, existingAgent.Name);
            
            return existingAgent;
        }
        
        /// <inheritdoc/>
        public async Task<bool> DeleteAgentAsync(string id)
        {
            var agent = await _dbContext.Agents.FindAsync(id);
            
            if (agent == null)
            {
                return false;
            }
            
            _dbContext.Agents.Remove(agent);
            await _dbContext.SaveChangesAsync();
            
            _logger.LogInformation("Deleted agent {AgentId} with name {AgentName}", agent.Id, agent.Name);
            
            return true;
        }
        
        /// <inheritdoc/>
        public async Task<IEnumerable<AgentModels>> GetOfflineAgentsAsync(TimeSpan offlineThreshold)
        {
            var thresholdTime = DateTime.UtcNow.Subtract(offlineThreshold);
            
            return await _dbContext.Agents
                .Include(a => a.Configuration)
                .Where(a => a.LastHeartbeat == null || a.LastHeartbeat < thresholdTime)
                .ToListAsync();
        }
        
        /// <inheritdoc/>
        public async Task<AgentModels> UpdateAgentStatusAsync(string id, AgentStatus status)
        {
            var agent = await _dbContext.Agents.FindAsync(id);
            
            if (agent == null)
            {
                throw new KeyNotFoundException($"Agent with ID {id} not found");
            }
            
            agent.Status = status;
            agent.UpdatedAt = DateTime.UtcNow;
            
            await _dbContext.SaveChangesAsync();
            
            _logger.LogInformation("Updated status of agent {AgentId} to {Status}", agent.Id, status);
            
            return agent;
        }
        
        /// <inheritdoc/>
        public async Task<string> GenerateNewApiKeyAsync(string id)
        {
            var agent = await _dbContext.Agents.FindAsync(id);
            
            if (agent == null)
            {
                throw new KeyNotFoundException($"Agent with ID {id} not found");
            }
            
            agent.ApiKey = GenerateApiKey();
            agent.UpdatedAt = DateTime.UtcNow;
            
            await _dbContext.SaveChangesAsync();
            
            _logger.LogInformation("Generated new API key for agent {AgentId}", agent.Id);
            
            return agent.ApiKey;
        }
        
        /// <inheritdoc/>
        public async Task<AgentModels> RecordHeartbeatAsync(string agentId, AgentHeartbeatModels heartbeat)
        {
            var agent = await _dbContext.Agents.FindAsync(agentId);
            
            if (agent == null)
            {
                throw new KeyNotFoundException($"Agent with ID {agentId} not found");
            }
            
            // Update agent status based on heartbeat
            agent.LastHeartbeat = heartbeat.Timestamp;
            agent.Status = AgentStatus.Online;
            agent.CpuUsage = heartbeat.CpuUsage;
            agent.MemoryUsage = heartbeat.MemoryUsage;
            agent.DiskUsage = heartbeat.DiskUsage;
            agent.UpdatedAt = DateTime.UtcNow;
            
            // Add heartbeat to database
            heartbeat.AgentId = agentId;
            heartbeat.CreatedAt = DateTime.UtcNow;
            
            _dbContext.AgentHeartbeats.Add(heartbeat);
            await _dbContext.SaveChangesAsync();
            
            _logger.LogDebug("Recorded heartbeat for agent {AgentId}", agent.Id);
            
            return agent;
        }
        
        private string GenerateApiKey()
        {
            return Guid.NewGuid().ToString("N");
        }
    }
}
