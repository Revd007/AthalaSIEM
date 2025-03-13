using System;
using System.Threading;
using System.Threading.Tasks;
using Backend.Services;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

namespace Backend.Services
{
    /// <summary>
    /// Background service for processing alerts
    /// </summary>
    public class AlertProcessingService : BackgroundService
    {
        private readonly IServiceProvider _serviceProvider;
        private readonly ILogger<AlertProcessingService> _logger;
        private readonly TimeSpan _processInterval = TimeSpan.FromMinutes(5);
        
        /// <summary>
        /// Initializes a new instance of the <see cref="AlertProcessingService"/> class
        /// </summary>
        /// <param name="serviceProvider">The service provider</param>
        /// <param name="logger">The logger</param>
        public AlertProcessingService(IServiceProvider serviceProvider, ILogger<AlertProcessingService> logger)
        {
            _serviceProvider = serviceProvider ?? throw new ArgumentNullException(nameof(serviceProvider));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }
        
        /// <inheritdoc/>
        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            _logger.LogInformation("Alert processing service is starting");
            
            while (!stoppingToken.IsCancellationRequested)
            {
                try
                {
                    await ProcessAlertsAsync();
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error processing alerts");
                }
                
                await Task.Delay(_processInterval, stoppingToken);
            }
            
            _logger.LogInformation("Alert processing service is stopping");
        }
        
        /// <summary>
        /// Processes alerts
        /// </summary>
        private async Task ProcessAlertsAsync()
        {
            using var scope = _serviceProvider.CreateScope();
            var alertService = scope.ServiceProvider.GetRequiredService<IAlertService>();
            
            // Get unresolved alerts
            var unresolvedAlerts = await alertService.GetUnresolvedAlertsAsync(1000, 0);
            
            _logger.LogInformation("Processing {Count} unresolved alerts", unresolvedAlerts.Count());
            
            foreach (var alert in unresolvedAlerts)
            {
                // Skip alerts that are not in "New" status
                if (alert.Status != "New")
                {
                    continue;
                }
                
                // Update alert status to "In Progress"
                await alertService.UpdateAlertStatusAsync(alert.Id, Models.AlertStatusModels.InProgress);
                
                _logger.LogInformation("Alert {AlertId} updated to In Progress", alert.Id);
                
                // In a real implementation, you would perform additional processing here,
                // such as sending notifications, executing automated remediation actions, etc.
            }
        }
    }
}