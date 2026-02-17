using MediatR;
using Microsoft.Extensions.Logging;
using Backend.Domain.Events;
using Backend.Domain.Entities;
using Backend.Infrastructure.Normalizers;
using Backend.Infrastructure.Data.Repositories;
using Backend.Domain.Interfaces;

namespace Backend.Application.Handlers;

public class NormalizeAndStoreHandler : INotificationHandler<LogIngestedEvent>
{
    private readonly ILogNormalizer _normalizer;
    private readonly INormalizedLogRepository _normalizedLogRepository;
    private readonly ILogRepository _logRepository;
    private readonly IMediator _mediator;
    private readonly ILogger<NormalizeAndStoreHandler> _logger;

    public NormalizeAndStoreHandler(
        ILogNormalizer normalizer,
        INormalizedLogRepository normalizedLogRepository,
        ILogRepository logRepository,
        IMediator mediator,
        ILogger<NormalizeAndStoreHandler> logger)
    {
        _normalizer = normalizer;
        _normalizedLogRepository = normalizedLogRepository;
        _logRepository = logRepository;
        _mediator = mediator;
        _logger = logger;
    }

    public async Task Handle(LogIngestedEvent notification, CancellationToken cancellationToken)
    {
        try
        {
            var logEntry = notification.LogEntry;

            // Normalize log
            var ecsFields = await _normalizer.NormalizeAsync(logEntry, cancellationToken);
            if (ecsFields == null)
            {
                _logger.LogWarning("Failed to normalize log {LogId}", logEntry.Id);
                return;
            }

            // Ensure DateTime values are UTC (PostgreSQL requirement)
            DateTime EnsureUtc(DateTime dt) => dt.Kind == DateTimeKind.Utc ? dt : dt.ToUniversalTime();
            
            // Store normalized log
            var normalizedLog = new NormalizedLog
            {
                Id = Guid.NewGuid().ToString(),
                LogEntryId = logEntry.Id,
                Timestamp = EnsureUtc(ecsFields.Timestamp),
                AgentId = ecsFields.AgentId,
                AgentName = ecsFields.AgentName,
                HostName = ecsFields.HostName,
                HostIp = ecsFields.HostIp,
                UserName = ecsFields.UserName,
                UserId = ecsFields.UserId,
                UserDomain = ecsFields.UserDomain,
                ProcessName = ecsFields.ProcessName,
                ProcessId = ecsFields.ProcessId,
                ProcessPath = ecsFields.ProcessPath,
                ProcessCommandLine = ecsFields.ProcessCommandLine,
                ProcessHash = ecsFields.ProcessHash,
                ParentProcessName = ecsFields.ParentProcessName,
                ParentProcessId = ecsFields.ParentProcessId,
                SourceIp = ecsFields.SourceIp,
                SourcePort = ecsFields.SourcePort,
                DestinationIp = ecsFields.DestinationIp,
                DestinationPort = ecsFields.DestinationPort,
                Protocol = ecsFields.Protocol,
                EventAction = ecsFields.EventAction,
                EventCategory = ecsFields.EventCategory,
                EventType = ecsFields.EventType,
                EventOutcome = ecsFields.EventOutcome,
                EventCode = ecsFields.EventCode,
                FilePath = ecsFields.FilePath,
                FileName = ecsFields.FileName,
                FileHash = ecsFields.FileHash,
                FileSize = ecsFields.FileSize,
                SiemRuleId = ecsFields.SiemRuleId,
                SiemTechniqueId = ecsFields.SiemTechniqueId,
                SiemConfidence = ecsFields.SiemConfidence,
                SiemSeverity = ecsFields.SiemSeverity,
                SiemCorrelationId = ecsFields.SiemCorrelationId,
                MetadataJson = ecsFields.Metadata != null 
                    ? System.Text.Json.JsonSerializer.Serialize(ecsFields.Metadata) 
                    : null,
                CreatedAt = DateTime.UtcNow
            };

            await _normalizedLogRepository.AddAsync(normalizedLog, cancellationToken);

            // Update log entry
            logEntry.NormalizedFields = ecsFields;
            logEntry.IsNormalized = true;
            logEntry.NormalizedAt = DateTime.UtcNow;
            await _logRepository.UpdateAsync(logEntry, cancellationToken);

            // Publish normalization event
            await _mediator.Publish(new LogNormalizedEvent { LogEntry = logEntry }, cancellationToken);

            _logger.LogDebug("Log {LogId} normalized and stored", logEntry.Id);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error normalizing and storing log {LogId}", notification.LogEntry.Id);
        }
    }
}
