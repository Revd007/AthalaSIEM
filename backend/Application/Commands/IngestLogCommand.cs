using MediatR;
using Backend.Domain.Entities;

namespace Backend.Application.Commands;

public class IngestLogCommand : IRequest<IngestLogResult>
{
    public string AgentId { get; set; } = string.Empty;
    public string Message { get; set; } = string.Empty;
    public string Source { get; set; } = string.Empty;
    public string? Category { get; set; }
    public long? EventId { get; set; }
    public DateTime Timestamp { get; set; }
    public Dictionary<string, object>? Properties { get; set; }
}

public class IngestLogResult
{
    public string LogId { get; set; } = string.Empty;
    public bool Success { get; set; }
    public string? ErrorMessage { get; set; }
}
