using System;
using System.Threading;
using System.Threading.Tasks;
using Backend.Data;
using Backend.Models;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Configuration;

namespace Backend.Services.Background
{
    public class AlertCleanupService : BackgroundService
    {
        private readonly IServiceScopeFactory _scopeFactory;
        private readonly ILogger<AlertCleanupService> _logger;
        private readonly IConfiguration _configuration;
        private readonly TimeSpan _cleanupInterval = TimeSpan.FromHours(24);
        private readonly TimeSpan _resolvedRetentionPeriod;
        private readonly TimeSpan _acknowledgedRetentionPeriod;

        public AlertCleanupService(
            IServiceScopeFactory scopeFactory,
            IConfiguration configuration,
            ILogger<AlertCleanupService> logger)
        {
            _scopeFactory = scopeFactory;
            _configuration = configuration;
            _logger = logger;

            // Get retention periods from configuration or use defaults
            var resolvedDays = _configuration.GetValue<int>("AlertRetentionDays:Resolved", 30);
            var acknowledgedDays = _configuration.GetValue<int>("AlertRetentionDays:Acknowledged", 7);

            _resolvedRetentionPeriod = TimeSpan.FromDays(resolvedDays);
            _acknowledgedRetentionPeriod = TimeSpan.FromDays(acknowledgedDays);
        }

        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            while (!stoppingToken.IsCancellationRequested)
            {
                try
                {
                    await CleanupAlerts(stoppingToken);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error cleaning up alerts");
                }

                await Task.Delay(_cleanupInterval, stoppingToken);
            }
        }

        private async Task CleanupAlerts(CancellationToken stoppingToken)
        {
            using var scope = _scopeFactory.CreateScope();
            var context = scope.ServiceProvider.GetRequiredService<ApplicationDbContext>();
            var now = DateTime.UtcNow;

            // Delete resolved alerts
            var resolvedCutoffDate = now - _resolvedRetentionPeriod;
            var deletedResolvedCount = await context.Alert
                .Where(a => a.Status == AlertStatusModels.Resolved && a.ResolvedAt < resolvedCutoffDate)
                .ExecuteDeleteAsync(stoppingToken);

            if (deletedResolvedCount > 0)
            {
                _logger.LogInformation("Deleted {Count} resolved alerts older than {CutoffDate}", 
                    deletedResolvedCount, resolvedCutoffDate);
            }

            // Delete acknowledged alerts
            var acknowledgedCutoffDate = now - _acknowledgedRetentionPeriod;
            var deletedAcknowledgedCount = await context.Alert
                .Where(a => a.Status == AlertStatusModels.Acknowledged && a.AcknowledgedAt < acknowledgedCutoffDate)
                .ExecuteDeleteAsync(stoppingToken);

            if (deletedAcknowledgedCount > 0)
            {
                _logger.LogInformation("Deleted {Count} acknowledged alerts older than {CutoffDate}", 
                    deletedAcknowledgedCount, acknowledgedCutoffDate);
            }
        }
    }
} 