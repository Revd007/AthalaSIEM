using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Backend.Models;

namespace Backend.Services
{
    /// <summary>
    /// Interface for agent management service
    /// </summary>
    public interface IAgentManagementService
    {
        /// <summary>
        /// Gets all agents
        /// </summary>
        /// <returns>A collection of agents</returns>
        Task<IEnumerable<AgentModels>> GetAllAgentsAsync();
        
        /// <summary>
        /// Gets an agent by ID
        /// </summary>
        /// <param name="id">The agent ID</param>
        /// <returns>The agent, or null if not found</returns>
        Task<AgentModels?> GetAgentByIdAsync(string id);
        
        /// <summary>
        /// Gets an agent by API key
        /// </summary>
        /// <param name="apiKey">The API key</param>
        /// <returns>The agent, or null if not found</returns>
        Task<AgentModels?> GetAgentByApiKeyAsync(string apiKey);
        
        /// <summary>
        /// Creates a new agent
        /// </summary>
        /// <param name="agent">The agent to create</param>
        /// <returns>The created agent</returns>
        Task<AgentModels> CreateAgentAsync(AgentModels agent);
        
        /// <summary>
        /// Updates an existing agent
        /// </summary>
        /// <param name="agent">The agent to update</param>
        /// <returns>The updated agent</returns>
        Task<AgentModels> UpdateAgentAsync(AgentModels agent);
        
        /// <summary>
        /// Deletes an agent
        /// </summary>
        /// <param name="id">The agent ID</param>
        /// <returns>True if the agent was deleted, false otherwise</returns>
        Task<bool> DeleteAgentAsync(string id);
        
        /// <summary>
        /// Gets agents that have not sent a heartbeat within the specified time
        /// </summary>
        /// <param name="offlineThreshold">The time threshold for considering an agent offline</param>
        /// <returns>A collection of offline agents</returns>
        Task<IEnumerable<AgentModels>> GetOfflineAgentsAsync(TimeSpan offlineThreshold);
        
        /// <summary>
        /// Updates the status of an agent
        /// </summary>
        /// <param name="id">The agent ID</param>
        /// <param name="status">The new status</param>
        /// <returns>The updated agent</returns>
        Task<AgentModels> UpdateAgentStatusAsync(string id, AgentStatus status);
        
        /// <summary>
        /// Generates a new API key for an agent
        /// </summary>
        /// <param name="id">The agent ID</param>
        /// <returns>The new API key</returns>
        Task<string> GenerateNewApiKeyAsync(string id);
        
        /// <summary>
        /// Records a heartbeat from an agent
        /// </summary>
        /// <param name="agentId">The agent ID</param>
        /// <param name="heartbeat">The heartbeat data</param>
        /// <returns>The updated agent</returns>
        Task<AgentModels> RecordHeartbeatAsync(string agentId, AgentHeartbeatModels heartbeat);
    }
} 