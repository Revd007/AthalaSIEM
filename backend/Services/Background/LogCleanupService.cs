using System;
using System.Threading;
using System.Threading.Tasks;
using Backend.Data;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Configuration;

namespace Backend.Services.Background
{
    public class LogCleanupService : BackgroundService
    {
        private readonly IServiceScopeFactory _scopeFactory;
        private readonly ILogger<LogCleanupService> _logger;
        private readonly IConfiguration _configuration;
        private readonly TimeSpan _cleanupInterval = TimeSpan.FromHours(24);
        private readonly TimeSpan _retentionPeriod;

        public LogCleanupService(
            IServiceScopeFactory scopeFactory,
            IConfiguration configuration,
            ILogger<LogCleanupService> logger)
        {
            _scopeFactory = scopeFactory;
            _configuration = configuration;
            _logger = logger;

            // Get retention period from configuration or use default (90 days)
            var retentionDays = _configuration.GetValue<int>("LogRetentionDays", 90);
            _retentionPeriod = TimeSpan.FromDays(retentionDays);
        }

        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            while (!stoppingToken.IsCancellationRequested)
            {
                try
                {
                    await CleanupLogs(stoppingToken);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error cleaning up logs");
                }

                await Task.Delay(_cleanupInterval, stoppingToken);
            }
        }

        private async Task CleanupLogs(CancellationToken stoppingToken)
        {
            using var scope = _scopeFactory.CreateScope();
            var context = scope.ServiceProvider.GetRequiredService<ApplicationDbContext>();
            var cutoffDate = DateTime.UtcNow - _retentionPeriod;

            // Delete old log entries
            var deletedLogCount = await context.LogEntries
                .Where(l => l.Timestamp < cutoffDate)
                .ExecuteDeleteAsync(stoppingToken);

            if (deletedLogCount > 0)
            {
                _logger.LogInformation("Deleted {Count} log entries older than {CutoffDate}", 
                    deletedLogCount, cutoffDate);
            }

            // Delete old security events
            var deletedEventCount = await context.SecurityEvents
                .Where(e => e.Timestamp < cutoffDate)
                .ExecuteDeleteAsync(stoppingToken);

            if (deletedEventCount > 0)
            {
                _logger.LogInformation("Deleted {Count} security events older than {CutoffDate}", 
                    deletedEventCount, cutoffDate);
            }

            // Delete old agent heartbeats
            var deletedHeartbeatCount = await context.AgentHeartbeats
                .Where(h => h.Timestamp < cutoffDate)
                .ExecuteDeleteAsync(stoppingToken);

            if (deletedHeartbeatCount > 0)
            {
                _logger.LogInformation("Deleted {Count} agent heartbeats older than {CutoffDate}", 
                    deletedHeartbeatCount, cutoffDate);
            }
        }
    }
} 