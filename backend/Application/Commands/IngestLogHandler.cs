using MediatR;
using Microsoft.Extensions.Logging;
using Backend.Application.Commands;
using Backend.Domain.Entities;
using Backend.Domain.Events;
using Backend.Domain.Interfaces;
using Backend.Workers;

namespace Backend.Application.Handlers;

public class IngestLogHandler : IRequestHandler<IngestLogCommand, IngestLogResult>
{
    private readonly ILogRepository _logRepository;
    private readonly IMediator _mediator;
    private readonly LogNormalizationWorker _normalizationWorker;
    private readonly ILogger<IngestLogHandler> _logger;

    public IngestLogHandler(
        ILogRepository logRepository,
        IMediator mediator,
        LogNormalizationWorker normalizationWorker,
        ILogger<IngestLogHandler> logger)
    {
        _logRepository = logRepository;
        _mediator = mediator;
        _normalizationWorker = normalizationWorker;
        _logger = logger;
    }

    public async Task<IngestLogResult> Handle(IngestLogCommand request, CancellationToken cancellationToken)
    {
        try
        {
            var logEntry = new LogEntry
            {
                AgentId = request.AgentId,
                Timestamp = request.Timestamp,
                ReceivedAt = DateTime.UtcNow,
                RawMessage = request.Message,
                Source = request.Source,
                Category = request.Category,
                EventId = request.EventId,
                RawProperties = request.Properties != null 
                    ? System.Text.Json.JsonSerializer.Serialize(request.Properties) 
                    : null
            };

            await _logRepository.AddAsync(logEntry, cancellationToken);
            
            // Publish event for downstream processing
            await _mediator.Publish(new LogIngestedEvent { LogEntry = logEntry }, cancellationToken);
            
            // Enqueue for normalization
            await _normalizationWorker.EnqueueLogAsync(logEntry, cancellationToken);

            _logger.LogDebug("Log ingested: {LogId} from agent {AgentId}", logEntry.Id, request.AgentId);

            return new IngestLogResult
            {
                LogId = logEntry.Id,
                Success = true
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error ingesting log from agent {AgentId}", request.AgentId);
            return new IngestLogResult
            {
                Success = false,
                ErrorMessage = ex.Message
            };
        }
    }
}
