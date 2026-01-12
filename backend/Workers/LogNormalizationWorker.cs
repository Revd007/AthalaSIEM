using System.Threading.Channels;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Backend.Domain.Entities;
using Backend.Domain.Events;
using Backend.Domain.Interfaces;
using Backend.Infrastructure.Normalizers;
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
        var mediator = scope.ServiceProvider.GetRequiredService<IMediator>();

        try
        {
            var normalizedFields = await normalizer.NormalizeBatchAsync(batch, cancellationToken);
            var normalizedList = normalizedFields.ToList();

            for (int i = 0; i < batch.Count && i < normalizedList.Count; i++)
            {
                var logEntry = batch[i];
                var ecsFields = normalizedList[i];

                if (ecsFields != null)
                {
                    logEntry.NormalizedFields = ecsFields;
                    logEntry.IsNormalized = true;
                    logEntry.NormalizedAt = DateTime.UtcNow;

                    await logRepository.UpdateAsync(logEntry, cancellationToken);

                    // Publish normalization event
                    await mediator.Publish(new LogNormalizedEvent { LogEntry = logEntry }, cancellationToken);
                }
            }

            _logger.LogDebug("Normalized {Count} log entries", batch.Count);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error processing normalization batch");
        }
    }
}
