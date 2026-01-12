using System.Threading.Channels;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Backend.Domain.Entities;
using Backend.Domain.Events;
using Backend.Domain.Interfaces;
using Backend.Infrastructure.Detection;
using MediatR;
using Backend.Infrastructure.Detection.RuleEngine;

namespace Backend.Workers;

public class DetectionWorker : BackgroundService
{
    private readonly IServiceProvider _serviceProvider;
    private readonly ILogger<DetectionWorker> _logger;
    private const int BatchSize = 50;
    private const int BatchTimeoutSeconds = 3;

    public DetectionWorker(
        IServiceProvider serviceProvider,
        ILogger<DetectionWorker> logger)
    {
        _serviceProvider = serviceProvider;
        _logger = logger;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation("Detection worker started");

        while (!stoppingToken.IsCancellationRequested)
        {
            try
            {
                using var scope = _serviceProvider.CreateScope();
                var logRepository = scope.ServiceProvider.GetRequiredService<ILogRepository>();
                var detectionEngine = scope.ServiceProvider.GetRequiredService<IDetectionEngine>();

                // Get normalized, unprocessed logs
                var logs = await logRepository.GetUnprocessedAsync(BatchSize, stoppingToken);
                var logList = logs.ToList();

                if (logList.Count == 0)
                {
                    await Task.Delay(5000, stoppingToken);
                    continue;
                }

                // Run detection on batch
                var results = await detectionEngine.DetectBatchAsync(logList, stoppingToken);

                // Process detection results
                foreach (var result in results.Where(r => r.Matched))
                {
                    await ProcessDetectionResultAsync(result, scope, stoppingToken);
                }

                // Mark logs as processed
                foreach (var log in logList)
                {
                    log.Processed = true;
                    log.ProcessedAt = DateTime.UtcNow;
                    if (results.Any(r => r.LogEntry.Id == log.Id && r.Matched))
                    {
                        var matchedResults = results.Where(r => r.LogEntry.Id == log.Id && r.Matched).ToList();
                        log.MatchedRuleIds.AddRange(matchedResults.Select(r => r.Rule.Id));
                        log.TechniqueIds.AddRange(matchedResults.SelectMany(r => r.Rule.TechniqueIds));
                    }
                    await logRepository.UpdateAsync(log, stoppingToken);
                }

                _logger.LogDebug("Processed {Count} logs, {MatchCount} detections", 
                    logList.Count, results.Count(r => r.Matched));
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error in detection worker");
                await Task.Delay(5000, stoppingToken);
            }
        }

        _logger.LogInformation("Detection worker stopped");
    }

    private async Task ProcessDetectionResultAsync(
        DetectionResult result,
        IServiceScope scope,
        CancellationToken cancellationToken)
    {
        var mediator = scope.ServiceProvider.GetRequiredService<IMediator>();

        // Publish detection fired event
        await mediator.Publish(new DetectionFiredEvent
        {
            Rule = result.Rule,
            LogEntry = result.LogEntry,
            MatchContext = result.MatchContext,
            DetectedAt = DateTime.UtcNow
        }, cancellationToken);

        _logger.LogInformation("Detection fired: Rule {RuleId} on log {LogId}", 
            result.Rule.Id, result.LogEntry.Id);
    }
}
