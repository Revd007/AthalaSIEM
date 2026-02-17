using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging;
using Backend.Data;
using Backend.Domain.Entities;
using Backend.Domain.Interfaces;
using System.Text.Json;

namespace Backend.Infrastructure.Data.Repositories;

public class AgentRepository : IAgentRepository
{
    private readonly ApplicationDbContext _context;
    private readonly ILogger<AgentRepository> _logger;

    public AgentRepository(ApplicationDbContext context, ILogger<AgentRepository> logger)
    {
        _context = context;
        _logger = logger;
    }

    public async Task<Agent?> GetByIdAsync(string id, CancellationToken cancellationToken = default)
    {
        var model = await _context.Agents.FindAsync(new object[] { id }, cancellationToken);
        return model != null ? MapToDomain(model) : null;
    }

    public async Task<Agent?> GetByApiKeyAsync(string apiKey, CancellationToken cancellationToken = default)
    {
        var model = await _context.Agents
            .FirstOrDefaultAsync(a => a.ApiKey == apiKey, cancellationToken);
        return model != null ? MapToDomain(model) : null;
    }

    public async Task<IEnumerable<Agent>> GetAllAsync(CancellationToken cancellationToken = default)
    {
        var models = await _context.Agents.ToListAsync(cancellationToken);
        return models.Select(MapToDomain);
    }

    public async Task<IEnumerable<Agent>> GetOnlineAgentsAsync(CancellationToken cancellationToken = default)
    {
        var models = await _context.Agents
            .Where(a => a.Status == Models.AgentStatus.Online)
            .ToListAsync(cancellationToken);
        return models.Select(MapToDomain);
    }

    public async Task<IEnumerable<Agent>> GetOfflineAgentsAsync(TimeSpan threshold, CancellationToken cancellationToken = default)
    {
        var cutoffTime = DateTime.UtcNow.Subtract(threshold);
        var models = await _context.Agents
            .Where(a => a.LastHeartbeat == null || a.LastHeartbeat < cutoffTime)
            .ToListAsync(cancellationToken);
        return models.Select(MapToDomain);
    }

    public async Task AddAsync(Agent agent, CancellationToken cancellationToken = default)
    {
        var model = MapToModel(agent);
        await _context.Agents.AddAsync(model, cancellationToken);
        await _context.SaveChangesAsync(cancellationToken);
    }

    public async Task UpdateAsync(Agent agent, CancellationToken cancellationToken = default)
    {
        var model = await _context.Agents.FindAsync(new object[] { agent.Id }, cancellationToken);
        if (model != null)
        {
            UpdateModel(model, agent);
            await _context.SaveChangesAsync(cancellationToken);
        }
    }

    public async Task DeleteAsync(string id, CancellationToken cancellationToken = default)
    {
        var model = await _context.Agents.FindAsync(new object[] { id }, cancellationToken);
        if (model != null)
        {
            _context.Agents.Remove(model);
            await _context.SaveChangesAsync(cancellationToken);
        }
    }

    public async Task<bool> ValidateApiKeyAsync(string agentId, string apiKey, CancellationToken cancellationToken = default)
    {
        var agent = await GetByIdAsync(agentId, cancellationToken);
        return agent != null && agent.ApiKey == apiKey;
    }

    private Agent MapToDomain(Models.AgentModels model)
    {
        return new Agent
        {
            Id = model.Id,
            Name = model.Name,
            Hostname = model.Hostname ?? string.Empty,
            IpAddress = model.IpAddress ?? string.Empty,
            OperatingSystem = model.OperatingSystem,
            AgentVersion = model.Version,
            ApiKey = model.ApiKey,
            Status = MapStatus(model.Status),
            LastHeartbeat = model.LastHeartbeat,
            CreatedAt = model.CreatedAt,
            UpdatedAt = model.UpdatedAt
        };
    }

    private Models.AgentModels MapToModel(Agent agent)
    {
        // Ensure all DateTime values are UTC (PostgreSQL requirement)
        static DateTime? EnsureUtcNullable(DateTime? dt) => dt.HasValue 
            ? (dt.Value.Kind == DateTimeKind.Utc ? dt.Value : dt.Value.ToUniversalTime())
            : null;
        static DateTime EnsureUtc(DateTime dt) => dt.Kind == DateTimeKind.Utc ? dt : dt.ToUniversalTime();

        return new Models.AgentModels
        {
            Id = agent.Id,
            Name = agent.Name,
            Hostname = agent.Hostname,
            IpAddress = agent.IpAddress,
            OperatingSystem = agent.OperatingSystem ?? string.Empty,
            Version = agent.AgentVersion ?? string.Empty,
            ApiKey = agent.ApiKey,
            Status = MapStatus(agent.Status),
            LastHeartbeat = EnsureUtcNullable(agent.LastHeartbeat),
            CreatedAt = EnsureUtc(agent.CreatedAt),
            UpdatedAt = EnsureUtc(agent.UpdatedAt)
        };
    }

    private void UpdateModel(Models.AgentModels model, Agent agent)
    {
        // Ensure all DateTime values are UTC (PostgreSQL requirement)
        static DateTime? EnsureUtcNullable(DateTime? dt) => dt.HasValue 
            ? (dt.Value.Kind == DateTimeKind.Utc ? dt.Value : dt.Value.ToUniversalTime())
            : null;
        static DateTime EnsureUtc(DateTime dt) => dt.Kind == DateTimeKind.Utc ? dt : dt.ToUniversalTime();

        model.Name = agent.Name;
        model.Hostname = agent.Hostname;
        model.IpAddress = agent.IpAddress;
        model.OperatingSystem = agent.OperatingSystem ?? string.Empty;
        model.Version = agent.AgentVersion ?? string.Empty;
        model.Status = MapStatus(agent.Status);
        model.LastHeartbeat = EnsureUtcNullable(agent.LastHeartbeat);
        model.UpdatedAt = EnsureUtc(agent.UpdatedAt);
    }

    private AgentStatus MapStatus(Models.AgentStatus status)
    {
        return status switch
        {
            Models.AgentStatus.Online => AgentStatus.Online,
            Models.AgentStatus.Offline => AgentStatus.Offline,
            _ => AgentStatus.Offline
        };
    }

    private Models.AgentStatus MapStatus(AgentStatus status)
    {
        return status switch
        {
            AgentStatus.Online => Models.AgentStatus.Online,
            AgentStatus.Offline => Models.AgentStatus.Offline,
            AgentStatus.Degraded => Models.AgentStatus.Offline, // Map to closest
            AgentStatus.Error => Models.AgentStatus.Offline,
            _ => Models.AgentStatus.Offline
        };
    }
}
