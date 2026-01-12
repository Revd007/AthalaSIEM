using MediatR;
using Microsoft.Extensions.Logging;
using Backend.Application.Commands;
using Backend.Domain.Entities;
using Backend.Domain.Interfaces;

namespace Backend.Application.Handlers;

public class RegisterAgentHandler : IRequestHandler<RegisterAgentCommand, RegisterAgentResult>
{
    private readonly IAgentRepository _agentRepository;
    private readonly ILogger<RegisterAgentHandler> _logger;

    public RegisterAgentHandler(
        IAgentRepository agentRepository,
        ILogger<RegisterAgentHandler> logger)
    {
        _agentRepository = agentRepository;
        _logger = logger;
    }

    public async Task<RegisterAgentResult> Handle(RegisterAgentCommand request, CancellationToken cancellationToken)
    {
        try
        {
            // Generate API key
            var apiKey = GenerateApiKey();

            var agent = new Agent
            {
                Name = request.Name,
                Hostname = request.Hostname,
                IpAddress = request.IpAddress,
                OperatingSystem = request.OperatingSystem,
                AgentVersion = request.AgentVersion,
                ApiKey = apiKey,
                Status = AgentStatus.Online,
                LastHeartbeat = DateTime.UtcNow,
                CreatedAt = DateTime.UtcNow,
                UpdatedAt = DateTime.UtcNow
            };

            await _agentRepository.AddAsync(agent, cancellationToken);

            _logger.LogInformation("Agent registered: {AgentId} ({Hostname})", agent.Id, request.Hostname);

            return new RegisterAgentResult
            {
                AgentId = agent.Id,
                ApiKey = apiKey,
                Success = true
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error registering agent {Name}", request.Name);
            return new RegisterAgentResult
            {
                Success = false,
                ErrorMessage = ex.Message
            };
        }
    }

    private string GenerateApiKey()
    {
        using var rng = System.Security.Cryptography.RandomNumberGenerator.Create();
        var bytes = new byte[32];
        rng.GetBytes(bytes);
        return Convert.ToBase64String(bytes).Replace("+", "-").Replace("/", "_").TrimEnd('=');
    }
}
