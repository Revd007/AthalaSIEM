using MediatR;

namespace Backend.Application.Commands;

public class RegisterAgentCommand : IRequest<RegisterAgentResult>
{
    public string Name { get; set; } = string.Empty;
    public string Hostname { get; set; } = string.Empty;
    public string IpAddress { get; set; } = string.Empty;
    public string? OperatingSystem { get; set; }
    public string? AgentVersion { get; set; }
    public Dictionary<string, object>? Metadata { get; set; }
}

public class RegisterAgentResult
{
    public string AgentId { get; set; } = string.Empty;
    public string ApiKey { get; set; } = string.Empty;
    public bool Success { get; set; }
    public string? ErrorMessage { get; set; }
}
