using MediatR;
using Microsoft.Extensions.Logging;
using Backend.Domain.Events;

namespace Backend.Application.Handlers;

public class LogNormalizedEventHandler : INotificationHandler<LogNormalizedEvent>
{
    private readonly ILogger<LogNormalizedEventHandler> _logger;

    public LogNormalizedEventHandler(ILogger<LogNormalizedEventHandler> logger)
    {
        _logger = logger;
    }

    public Task Handle(LogNormalizedEvent notification, CancellationToken cancellationToken)
    {
        // Log normalized - detection worker will pick it up
        _logger.LogDebug("Log {LogId} normalized, ready for detection", notification.LogEntry.Id);
        return Task.CompletedTask;
    }
}
