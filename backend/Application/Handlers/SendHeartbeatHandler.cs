using MediatR;
using Microsoft.Extensions.Logging;
using Backend.Application.Commands;
using Backend.Domain.Interfaces;

namespace Backend.Application.Handlers;

public class SendHeartbeatHandler : IRequestHandler<SendHeartbeatCommand, SendHeartbeatResult>
{
    private readonly IAgentRepository _agentRepository;
    private readonly ILogger<SendHeartbeatHandler> _logger;

    public SendHeartbeatHandler(
        IAgentRepository agentRepository,
        ILogger<SendHeartbeatHandler> logger)
    {
        _agentRepository = agentRepository;
        _logger = logger;
    }

    public async Task<SendHeartbeatResult> Handle(SendHeartbeatCommand request, CancellationToken cancellationToken)
    {
        try
        {
            // Validate API key
            var isValid = await _agentRepository.ValidateApiKeyAsync(request.AgentId, request.ApiKey, cancellationToken);
            if (!isValid)
            {
                return new SendHeartbeatResult
                {
                    Success = false,
                    ErrorMessage = "Invalid API key"
                };
            }

            // Update agent heartbeat
            var agent = await _agentRepository.GetByIdAsync(request.AgentId, cancellationToken);
            if (agent == null)
            {
                return new SendHeartbeatResult
                {
                    Success = false,
                    ErrorMessage = "Agent not found"
                };
            }

            agent.LastHeartbeat = DateTime.UtcNow;
            agent.Status = Domain.Entities.AgentStatus.Online;

            // Update health metrics if provided
            if (request.HealthMetrics != null)
            {
                if (request.HealthMetrics.TryGetValue("cpu_usage", out var cpuUsage))
                    agent.CpuUsage = Convert.ToDouble(cpuUsage);
                if (request.HealthMetrics.TryGetValue("memory_usage", out var memoryUsage))
                    agent.MemoryUsage = Convert.ToDouble(memoryUsage);
                if (request.HealthMetrics.TryGetValue("logs_sent_count", out var logsSent))
                    agent.LogsSentCount = Convert.ToInt64(logsSent);
            }

            agent.UpdatedAt = DateTime.UtcNow;
            await _agentRepository.UpdateAsync(agent, cancellationToken);

            _logger.LogDebug("Heartbeat received from agent {AgentId}", request.AgentId);

            return new SendHeartbeatResult
            {
                Success = true,
                Configuration = agent.Configuration
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error processing heartbeat from agent {AgentId}", request.AgentId);
            return new SendHeartbeatResult
            {
                Success = false,
                ErrorMessage = ex.Message
            };
        }
    }
}
