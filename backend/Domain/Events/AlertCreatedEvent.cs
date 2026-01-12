using MediatR;
using Backend.Domain.Entities;

namespace Backend.Domain.Events;

public class AlertCreatedEvent : INotification
{
    public Alert Alert { get; set; } = null!;
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
}
