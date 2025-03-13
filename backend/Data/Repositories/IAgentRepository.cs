using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Backend.Models;

namespace Backend.Data.Repositories
{
    /// <summary>
    /// Interface for agent repository
    /// </summary>
    public interface IAgentRepository : IRepository<AgentModels, string>
    {
        /// <summary>
        /// Gets an agent by API key
        /// </summary>
        /// <param name="apiKey">The API key</param>
        /// <returns>The agent, or null if not found</returns>
        Task<AgentModels?> GetByApiKeyAsync(string apiKey);
        
        /// <summary>
        /// Gets agents by status
        /// </summary>
        /// <param name="status">The status</param>
        /// <returns>A collection of agents</returns>
        Task<IEnumerable<AgentModels>> GetByStatusAsync(AgentStatus status);
        
        /// <summary>
        /// Gets agents by type
        /// </summary>
        /// <param name="type">The type</param>
        /// <returns>A collection of agents</returns>
        Task<IEnumerable<AgentModels>> GetByTypeAsync(AgentType type);
        
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
        Task<AgentModels> UpdateStatusAsync(string id, AgentStatus status);
        
        /// <summary>
        /// Records a heartbeat from an agent
        /// </summary>
        /// <param name="agentId">The agent ID</param>
        /// <param name="heartbeat">The heartbeat data</param>
        /// <returns>The updated agent</returns>
        Task<AgentModels> RecordHeartbeatAsync(string agentId, AgentHeartbeatModels heartbeat);
        
        /// <summary>
        /// Adds a new agent
        /// </summary>
        /// <param name="agent">The agent to add</param>
        /// <returns>The added agent</returns>
        Task<AgentModels> AddAgentAsync(AgentModels agent);
    }
} 