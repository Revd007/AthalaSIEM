using MediatR;
using Backend.Domain.Entities;

namespace Backend.Domain.Events;

public class LogNormalizedEvent : INotification
{
    public LogEntry LogEntry { get; set; } = null!;
    public DateTime NormalizedAt { get; set; } = DateTime.UtcNow;
}
