using System.Threading.Channels;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Backend.Domain.Entities;
using Backend.Domain.Events;
using Backend.Domain.Interfaces;
using Backend.Infrastructure.Normalizers;
using Backend.Infrastructure.Data.Repositories;
using Backend.Workers;
using MediatR;

namespace Backend.Workers;

public class LogNormalizationWorker : BackgroundService
{
    private readonly IServiceProvider _serviceProvider;
    private readonly ILogger<LogNormalizationWorker> _logger;
    private readonly Channel<LogEntry> _channel;
    private const int BatchSize = 100;
    private const int BatchTimeoutSeconds = 5;

    public LogNormalizationWorker(
        IServiceProvider serviceProvider,
        ILogger<LogNormalizationWorker> logger)
    {
        _serviceProvider = serviceProvider;
        _logger = logger;
        
        var options = new BoundedChannelOptions(10000)
        {
            FullMode = BoundedChannelFullMode.Wait
        };
        _channel = Channel.CreateBounded<LogEntry>(options);
    }

    public async Task EnqueueLogAsync(LogEntry logEntry, CancellationToken cancellationToken = default)
    {
        await _channel.Writer.WriteAsync(logEntry, cancellationToken);
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation("Log normalization worker started");

        while (!stoppingToken.IsCancellationRequested)
        {
            try
            {
                var batch = new List<LogEntry>();
                var batchTimeout = DateTime.UtcNow.AddSeconds(BatchTimeoutSeconds);

                // Collect batch
                while (batch.Count < BatchSize && DateTime.UtcNow < batchTimeout)
                {
                    if (_channel.Reader.TryRead(out var logEntry))
                    {
                        batch.Add(logEntry);
                    }
                    else
                    {
                        await Task.Delay(100, stoppingToken);
                    }
                }

                if (batch.Count > 0)
                {
                    await ProcessBatchAsync(batch, stoppingToken);
                }
                else
                {
                    await Task.Delay(1000, stoppingToken);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error in log normalization worker");
                await Task.Delay(5000, stoppingToken);
            }
        }

        _logger.LogInformation("Log normalization worker stopped");
    }

    private async Task ProcessBatchAsync(List<LogEntry> batch, CancellationToken cancellationToken)
    {
        using var scope = _serviceProvider.CreateScope();
        var normalizer = scope.ServiceProvider.GetRequiredService<ILogNormalizer>();
        var logRepository = scope.ServiceProvider.GetRequiredService<ILogRepository>();
        var normalizedLogRepository = scope.ServiceProvider.GetRequiredService<INormalizedLogRepository>();
        var mediator = scope.ServiceProvider.GetRequiredService<IMediator>();

        try
        {
            var normalizedFields = await normalizer.NormalizeBatchAsync(batch, cancellationToken);
            var normalizedList = normalizedFields.ToList();
            var normalizedLogsToSave = new List<NormalizedLog>();

            for (int i = 0; i < batch.Count && i < normalizedList.Count; i++)
            {
                var logEntry = batch[i];
                var ecsFields = normalizedList[i];

                if (ecsFields != null)
                {
                    // Check if normalized log already exists
                    var existingNormalizedLog = await normalizedLogRepository.GetByLogEntryIdAsync(logEntry.Id, cancellationToken);
                    
                    if (existingNormalizedLog == null)
                    {
                        // Create new normalized log entry
                        var normalizedLog = new NormalizedLog
                        {
                            Id = Guid.NewGuid().ToString(),
                            LogEntryId = logEntry.Id,
                            Timestamp = ecsFields.Timestamp,
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

                        normalizedLogsToSave.Add(normalizedLog);
                    }
                    else
                    {
                        // Update existing normalized log
                        existingNormalizedLog.Timestamp = ecsFields.Timestamp;
                        existingNormalizedLog.SourceIp = ecsFields.SourceIp;
                        existingNormalizedLog.DestinationIp = ecsFields.DestinationIp;
                        existingNormalizedLog.EventType = ecsFields.EventType;
                        existingNormalizedLog.EventAction = ecsFields.EventAction;
                        existingNormalizedLog.EventCategory = ecsFields.EventCategory;
                        existingNormalizedLog.SiemSeverity = ecsFields.SiemSeverity;
                        existingNormalizedLog.UserName = ecsFields.UserName;
                        existingNormalizedLog.ProcessName = ecsFields.ProcessName;
                        existingNormalizedLog.ProcessId = ecsFields.ProcessId;
                        existingNormalizedLog.Protocol = ecsFields.Protocol;
                        existingNormalizedLog.MetadataJson = ecsFields.Metadata != null
                            ? System.Text.Json.JsonSerializer.Serialize(ecsFields.Metadata)
                            : null;

                        await normalizedLogRepository.UpdateAsync(existingNormalizedLog, cancellationToken);
                    }

                    // Update log entry
                    logEntry.NormalizedFields = ecsFields;
                    logEntry.IsNormalized = true;
                    logEntry.NormalizedAt = DateTime.UtcNow;

                    await logRepository.UpdateAsync(logEntry, cancellationToken);

                    // Publish normalization event
                    await mediator.Publish(new LogNormalizedEvent { LogEntry = logEntry }, cancellationToken);

                    // Enqueue for correlation processing
                    var correlationWorker = scope.ServiceProvider.GetService<CorrelationWorker>();
                    if (correlationWorker != null)
                    {
                        await correlationWorker.EnqueueLogAsync(logEntry, cancellationToken);
                    }
                }
            }

            // Bulk save new normalized logs
            if (normalizedLogsToSave.Count > 0)
            {
                await normalizedLogRepository.AddRangeAsync(normalizedLogsToSave, cancellationToken);
                _logger.LogDebug("Saved {Count} new normalized log entries", normalizedLogsToSave.Count);
            }

            _logger.LogDebug("Normalized {Count} log entries", batch.Count);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error processing normalization batch");
        }
    }
}
