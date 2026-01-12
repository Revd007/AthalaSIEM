using MediatR;
using Backend.Domain.Entities;

namespace Backend.Domain.Events;

public class LogIngestedEvent : INotification
{
    public LogEntry LogEntry { get; set; } = null!;
    public DateTime IngestedAt { get; set; } = DateTime.UtcNow;
}
