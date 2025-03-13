using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Backend.DTOs;
using Backend.Models;

namespace Backend.Services
{
    /// <summary>
    /// Service interface for agent operations
    /// </summary>
    public interface IAgentService
    {

        /// <summary>
        /// Gets all agents
        /// </summary>
        /// <returns>All agents</returns>
        Task<IEnumerable<AgentModels>> GetAllAgentsAsync();
        
        /// <summary>
        /// Gets an agent by ID
        /// </summary>
        /// <param name="id">The agent ID</param>
        /// <returns>The agent, or null if not found</returns>
        Task<AgentModels?> GetAgentByIdAsync(string id);
        
        /// <summary>
        /// Gets agents by status
        /// </summary>
        /// <param name="status">The agent status</param>
        /// <returns>The agents with the specified status</returns>
        Task<IEnumerable<AgentModels>> GetAgentsByStatusAsync(AgentStatus status);
        
        /// <summary>
        /// Gets agents by type
        /// </summary>
        /// <param name="type">The agent type</param>
        /// <returns>The agents with the specified type</returns>
        Task<IEnumerable<AgentModels>> GetAgentsByTypeAsync(AgentType type);
        
        /// <summary>
        /// Gets offline agents
        /// </summary>
        /// <param name="offlineThreshold">The offline threshold</param>
        /// <returns>The offline agents</returns>
        Task<IEnumerable<AgentModels>> GetOfflineAgentsAsync(TimeSpan offlineThreshold);
        
        /// <summary>
        /// Registers a new agent
        /// </summary>
        /// <param name="registrationDto">The agent registration data</param>
        /// <returns>The registration result</returns>
        Task<AgentRegistrationResultDto> RegisterAgentAsync(AgentRegistrationDto registrationDto);
        
        /// <summary>
        /// Updates an agent's status
        /// </summary>
        /// <param name="id">The agent ID</param>
        /// <param name="status">The new status</param>
        /// <returns>The updated agent</returns>
        Task<AgentModels> UpdateAgentStatusAsync(string id, AgentStatus status);
        
        /// <summary>
        /// Updates an agent's configuration
        /// </summary>
        /// <param name="id">The agent ID</param>
        /// <param name="configDto">The new configuration</param>
        /// <returns>The updated agent</returns>
        Task<AgentModels> UpdateAgentConfigAsync(string id, AgentConfigDto configDto);
        
        /// <summary>
        /// Processes a heartbeat from an agent
        /// </summary>
        /// <param name="agentId">The agent ID</param>
        /// <param name="heartbeatDto">The heartbeat data</param>
        /// <returns>The updated agent</returns>
        Task<AgentModels> ProcessHeartbeatAsync(string agentId, AgentHeartbeatDto heartbeatDto);
        
        /// <summary>
        /// Processes a health report from an agent
        /// </summary>
        /// <param name="agentId">The agent ID</param>
        /// <param name="healthReportDto">The health report data</param>
        /// <returns>The updated agent</returns>
        Task<AgentModels> ProcessHealthReportAsync(string agentId, AgentHealthReportDto healthReportDto);
        
        /// <summary>
        /// Deletes an agent
        /// </summary>
        /// <param name="id">The agent ID</param>
        /// <returns>True if successful, false otherwise</returns>
        Task<bool> DeleteAgentAsync(string id);

        /// <summary>
        /// Records a heartbeat from an agent
        /// </summary>
        /// <param name="agentId">The agent ID</param>
        /// <param name="heartbeatDto">The heartbeat data</param>
        /// <returns>The updated agent</returns>
        Task<AgentModels> RecordHeartbeatAsync(string agentId, AgentHeartbeatDto heartbeatDto);
        
        /// <summary>
        /// Updates an agent's configuration
        /// </summary>
        /// <param name="agentId">The agent ID</param>
        /// <param name="configDto">The new configuration</param>
        /// <returns>True if successful, false otherwise</returns>
        Task<bool> UpdateAgentConfigurationAsync(Guid agentId, AgentConfigDto configDto);
        
        /// <summary>
        /// Validates an API key
        /// </summary>
        /// <param name="apiKey">The API key to validate</param>
        /// <returns>True if the API key is valid, otherwise false</returns>
        Task<bool> ValidateApiKeyAsync(string apiKey);
        
        /// <summary>
        /// Validates an API key for a specific agent
        /// </summary>
        /// <param name="agentId">The agent ID</param>
        /// <param name="apiKey">The API key to validate</param>
        /// <returns>True if the API key is valid for the agent, otherwise false</returns>
        Task<bool> ValidateApiKeyAsync(Guid agentId, string apiKey);
        
        /// <summary>
        /// Gets an agent's configuration
        /// </summary>
        /// <param name="agentId">The agent ID</param>
        /// <returns>The agent configuration, or null if not found</returns>
        Task<AgentConfigDto?> GetAgentConfigurationAsync(Guid agentId);
        
        /// <summary>
        /// Processes a health report from an agent
        /// </summary>
        /// <param name="agentId">The agent ID</param>
        /// <param name="healthReportDto">The health report data</param>
        /// <returns>True if successful, false otherwise</returns>
        Task<bool> ProcessHealthReportAsync(Guid agentId, AgentHealthReportDto healthReportDto);
        
        /// <summary>
        /// Gets the agent health history
        /// </summary>
        /// <param name="agentId">The agent ID</param>
        /// <param name="count">Maximum number of records to return</param>
        /// <param name="offset">Number of records to skip</param>
        /// <returns>Paginated health history</returns>
        Task<PaginatedResult<AgentHealthReportDto>> GetAgentHealthHistoryAsync(string agentId, int count = 10, int offset = 0);
        
        /// <summary>
        /// Updates an agent's API key
        /// </summary>
        /// <param name="agentId">The agent ID</param>
        /// <returns>The new API key</returns>
        Task<string> UpdateAgentApiKeyAsync(string agentId);
    }
} 