using MediatR;
using Backend.Domain.Entities;

namespace Backend.Domain.Events;

public class DetectionFiredEvent : INotification
{
    public DetectionRule Rule { get; set; } = null!;
    public LogEntry LogEntry { get; set; } = null!;
    public Dictionary<string, object> MatchContext { get; set; } = new();
    public DateTime DetectedAt { get; set; } = DateTime.UtcNow;
}
