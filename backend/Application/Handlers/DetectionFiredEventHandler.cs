using MediatR;
using Microsoft.Extensions.Logging;
using Backend.Domain.Entities;
using Backend.Domain.Events;
using Backend.Domain.Interfaces;
using Backend.Infrastructure.AlertProcessing;

namespace Backend.Application.Handlers;

public class DetectionFiredEventHandler : INotificationHandler<DetectionFiredEvent>
{
    private readonly IAlertRepository _alertRepository;
    private readonly IAlertDeduplicator _deduplicator;
    private readonly IAlertSeverityScorer _severityScorer;
    private readonly IMediator _mediator;
    private readonly ILogger<DetectionFiredEventHandler> _logger;

    public DetectionFiredEventHandler(
        IAlertRepository alertRepository,
        IAlertDeduplicator deduplicator,
        IAlertSeverityScorer severityScorer,
        IMediator mediator,
        ILogger<DetectionFiredEventHandler> logger)
    {
        _alertRepository = alertRepository;
        _deduplicator = deduplicator;
        _severityScorer = severityScorer;
        _mediator = mediator;
        _logger = logger;
    }

    public async Task Handle(DetectionFiredEvent notification, CancellationToken cancellationToken)
    {
        try
        {
            var rule = notification.Rule;
            var logEntry = notification.LogEntry;

            // Create alert
            var alert = new Alert
            {
                AgentId = logEntry.AgentId,
                Title = $"{rule.DefaultSeverity} Alert - {rule.Name}",
                Description = rule.Description,
                Message = logEntry.RawMessage,
                Severity = rule.DefaultSeverity,
                Source = logEntry.Source,
                Timestamp = logEntry.Timestamp,
                RuleId = rule.Id,
                TechniqueIds = rule.TechniqueIds.ToList(),
                Confidence = 0.8, // Base confidence, can be enhanced
                DetectionReason = BuildDetectionReason(notification),
                DetectionMetadata = notification.MatchContext,
                RelatedLogIds = new List<string> { logEntry.Id }
            };

            // Generate deduplication key
            alert.DeduplicationKey = _deduplicator.GenerateDeduplicationKey(alert);

            // Check for duplicates
            var duplicate = await _deduplicator.FindDuplicateAsync(alert, cancellationToken);
            if (duplicate != null)
            {
                await _deduplicator.MergeDuplicateAsync(duplicate, alert, cancellationToken);
                await _alertRepository.UpdateAsync(duplicate, cancellationToken);
                _logger.LogDebug("Merged duplicate alert {AlertId}", duplicate.Id);
                return;
            }

            // Calculate severity score
            alert.SeverityScore = _severityScorer.CalculateSeverity(alert, logEntry);
            alert.Severity = alert.SeverityScore.CalculateSeverity();

            // Set occurrence tracking
            alert.FirstOccurrence = DateTime.UtcNow;
            alert.LastOccurrence = DateTime.UtcNow;

            // Save alert
            await _alertRepository.AddAsync(alert, cancellationToken);

            // Publish alert created event
            await _mediator.Publish(new AlertCreatedEvent { Alert = alert }, cancellationToken);

            _logger.LogInformation("Alert created: {AlertId} from rule {RuleId}", alert.Id, rule.Id);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error handling detection fired event");
        }
    }

    private string BuildDetectionReason(DetectionFiredEvent notification)
    {
        var reason = $"Rule '{notification.Rule.Name}' matched";
        
        if (notification.MatchContext.Any())
        {
            var matchedFields = string.Join(", ", notification.MatchContext.Keys);
            reason += $" on fields: {matchedFields}";
        }

        if (notification.Rule.TechniqueIds.Any())
        {
            reason += $" (MITRE ATT&CK: {string.Join(", ", notification.Rule.TechniqueIds)})";
        }

        return reason;
    }
}
