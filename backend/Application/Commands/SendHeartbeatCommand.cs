using MediatR;

namespace Backend.Application.Commands;

public class SendHeartbeatCommand : IRequest<SendHeartbeatResult>
{
    public string AgentId { get; set; } = string.Empty;
    public string ApiKey { get; set; } = string.Empty;
    public Dictionary<string, object>? HealthMetrics { get; set; }
}

public class SendHeartbeatResult
{
    public bool Success { get; set; }
    public string? ErrorMessage { get; set; }
    public Dictionary<string, object>? Configuration { get; set; }
}
