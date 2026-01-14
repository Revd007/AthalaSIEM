using System.Threading.Channels;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Backend.Domain.Entities;
using Backend.Infrastructure.Correlation;
using Backend.Data.Repositories;
using Backend.Services;

namespace Backend.Workers;

/// <summary>
/// Background worker that processes normalized logs through correlation engine
/// Detects patterns like "5 failed logins = Brute Force"
/// </summary>
public class CorrelationWorker : BackgroundService
{
    private readonly IServiceProvider _serviceProvider;
    private readonly ILogger<CorrelationWorker> _logger;
    private readonly Channel<LogEntry> _channel;
    private const int BatchSize = 50;

    public CorrelationWorker(
        IServiceProvider serviceProvider,
        ILogger<CorrelationWorker> logger)
    {
        _serviceProvider = serviceProvider;
        _logger = logger;
        
        var options = new BoundedChannelOptions(5000)
        {
            FullMode = BoundedChannelFullMode.Wait
        };
        _channel = Channel.CreateBounded<LogEntry>(options);
    }

    public async Task EnqueueLogAsync(LogEntry logEntry, CancellationToken cancellationToken = default)
    {
        // Only process normalized logs
        if (logEntry.IsNormalized && logEntry.NormalizedFields != null)
        {
            await _channel.Writer.WriteAsync(logEntry, cancellationToken);
        }
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation("Correlation worker started");

        while (!stoppingToken.IsCancellationRequested)
        {
            try
            {
                var batch = new List<LogEntry>();
                
                // Collect batch
                while (batch.Count < BatchSize && await _channel.Reader.WaitToReadAsync(stoppingToken))
                {
                    if (_channel.Reader.TryRead(out var logEntry))
                    {
                        batch.Add(logEntry);
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
            catch (OperationCanceledException)
            {
                break;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error in correlation worker");
                await Task.Delay(5000, stoppingToken);
            }
        }

        _logger.LogInformation("Correlation worker stopped");
    }

    private async Task ProcessBatchAsync(List<LogEntry> batch, CancellationToken cancellationToken)
    {
        using var scope = _serviceProvider.CreateScope();
        var ruleEngine = scope.ServiceProvider.GetRequiredService<SimpleRuleEngine>();
        var alertService = scope.ServiceProvider.GetRequiredService<IAlertService>();
        var logRepository = scope.ServiceProvider.GetRequiredService<Backend.Domain.Interfaces.ILogRepository>();

        foreach (var logEntry in batch)
        {
            try
            {
                // Get related logs function
                async Task<List<LogEntry>> GetRelatedLogsAsync(string correlationKey, DateTime startTime, DateTime endTime)
                {
                    // Parse correlation key (format: "ip:1.2.3.4" or "user:username")
                    var parts = correlationKey.Split(':');
                    if (parts.Length != 2)
                        return new List<LogEntry>();

                    var keyType = parts[0];
                    var keyValue = parts[1];

                    // Query normalized logs by correlation key
                    var normalizedLogs = await logRepository.GetNormalizedLogsByFieldAsync(
                        keyType == "ip" ? "SourceIp" : "UserName",
                        keyValue,
                        startTime,
                        endTime,
                        cancellationToken);

                    return normalizedLogs.ToList();
                }

                // Process through rule engine
                var correlationResults = await ruleEngine.ProcessLogAsync(
                    logEntry,
                    GetRelatedLogsAsync,
                    cancellationToken);

                // Generate alerts for correlation results
                foreach (var result in correlationResults)
                {
                    if (result.Metadata.TryGetValue("alert_severity", out var severityObj) &&
                        severityObj is int severity)
                    {
                        await alertService.CreateAlertAsync(new Backend.DTOs.CreateAlertDto
                        {
                            Title = result.RuleName ?? "Correlation Alert",
                            Description = result.RuleDescription ?? "Pattern detected",
                            Severity = severity.ToString(),
                            Source = "CorrelationEngine",
                            AgentId = logEntry.AgentId,
                            LogEntryIds = result.CorrelatedLogs.Select(l => l.Id).ToList(),
                            Metadata = result.Metadata
                        });

                        _logger.LogWarning(
                            "Correlation alert generated: {RuleName} - {Count} events correlated",
                            result.RuleName,
                            result.CorrelatedLogs.Count);
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error processing correlation for log {LogId}", logEntry.Id);
            }
        }
    }
}
