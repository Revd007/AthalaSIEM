using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Backend.Models;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging;

namespace Backend.Data.Repositories
{
    /// <summary>
    /// Repository for agent operations
    /// </summary>
    public class AgentRepository : Repository<AgentModels, string>, ILegacyAgentRepository
    {
        private readonly ILogger<AgentRepository> _logger;
        
        /// <summary>
        /// Initializes a new instance of the <see cref="AgentRepository"/> class
        /// </summary>
        /// <param name="context">The database context</param>
        /// <param name="logger">The logger</param>
        public AgentRepository(ApplicationDbContext context, ILogger<AgentRepository> logger)
            : base(context)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }

        /// <summary>
        /// Gets an agent by ID with Configuration included
        /// </summary>
        /// <param name="id">The agent ID</param>
        /// <returns>The agent with configuration, or null if not found</returns>
        public new async Task<AgentModels?> GetByIdAsync(string id)
        {
            try
            {
                return await DbSet
                    .Include(a => a.Configuration)
                    .FirstOrDefaultAsync(a => a.Id == id);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting agent by ID: {AgentId}", id);
                // Fallback to base implementation without includes
                return await base.GetByIdAsync(id);
            }
        }

        /// <summary>
        /// Gets all agents with Configuration included
        /// </summary>
        /// <returns>All agents with their configurations</returns>
        public new async Task<IEnumerable<AgentModels>> GetAllAsync()
        {
            try
            {
                return await DbSet
                    .Include(a => a.Configuration)
                    .ToListAsync();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting all agents");
                // Fallback to base implementation without includes
                return await base.GetAllAsync();
            }
        }
        
        /// <inheritdoc/>
        public async Task<AgentModels?> GetByApiKeyAsync(string apiKey)
        {
            return await DbSet
                .Include(a => a.Configuration)
                .FirstOrDefaultAsync(a => a.ApiKey == apiKey);
        }
        
        /// <inheritdoc/>
        public async Task<IEnumerable<AgentModels>> GetByStatusAsync(AgentStatus status)
        {
            return await DbSet.Where(a => a.Status == status).ToListAsync();
        }
        
        /// <inheritdoc/>
        public async Task<IEnumerable<AgentModels>> GetByTypeAsync(AgentType type)
        {
            return await DbSet.Where(a => a.Type == type).ToListAsync();
        }
        
        /// <inheritdoc/>
        public async Task<IEnumerable<AgentModels>> GetOfflineAgentsAsync(TimeSpan offlineThreshold)
        {
            var thresholdTime = DateTime.UtcNow.Subtract(offlineThreshold);
            
            return await DbSet
                .Where(a => a.LastHeartbeat == null || a.LastHeartbeat < thresholdTime)
                .ToListAsync();
        }
        
        /// <inheritdoc/>
        public async Task<AgentModels> UpdateStatusAsync(string id, AgentStatus status)
        {
            var agent = await DbSet.FindAsync(id);
            
            if (agent == null)
            {
                throw new KeyNotFoundException($"Agent with ID {id} not found");
            }
            
            agent.Status = status;
            agent.UpdatedAt = DateTime.UtcNow;
            
            await Context.SaveChangesAsync();
            
            return agent;
        }
        
        /// <inheritdoc/>
        public async Task<AgentModels> RecordHeartbeatAsync(string agentId, AgentHeartbeatModels heartbeat)
        {
            var agent = await DbSet.FindAsync(agentId);
            
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
            
            Context.Set<AgentHeartbeatModels>().Add(heartbeat);
            await Context.SaveChangesAsync();
            
            return agent;
        }
        
        /// <inheritdoc/>
        public async Task<AgentModels> AddAgentAsync(AgentModels agent)
        {
            if (agent == null)
            {
                throw new ArgumentNullException(nameof(agent));
            }
            
            await DbSet.AddAsync(agent);
            await Context.SaveChangesAsync();
            
            _logger.LogInformation("Added new agent with ID {AgentId}", agent.Id);
            
            return agent;
        }
    }
} 