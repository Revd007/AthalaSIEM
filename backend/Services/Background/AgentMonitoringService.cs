using System;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Backend.Services;
using Backend.Models;
using Backend.DTOs;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

namespace Backend.Services.Background
{
    /// <summary>
    /// Background service for monitoring agents
    /// </summary>
    public class AgentMonitoringService : BackgroundService
    {
        private readonly ILogger<AgentMonitoringService> _logger;
        private readonly TimeSpan _offlineThreshold = TimeSpan.FromMinutes(5);
        private readonly TimeSpan _checkInterval = TimeSpan.FromMinutes(1);
        private readonly IServiceScopeFactory _scopeFactory;
        
        /// <summary>
        /// Initializes a new instance of the <see cref="AgentMonitoringService"/> class
        /// </summary>
        /// <param name="logger">The logger</param>
        /// <param name="scopeFactory">The service scope factory</param>
        public AgentMonitoringService(
            ILogger<AgentMonitoringService> logger,
            IServiceScopeFactory scopeFactory)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _scopeFactory = scopeFactory ?? throw new ArgumentNullException(nameof(scopeFactory));
        }
        
        /// <inheritdoc/>
        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            _logger.LogInformation("Agent monitoring service is starting");
            
            while (!stoppingToken.IsCancellationRequested)
            {
                try
                {
                    await CheckOfflineAgentsAsync();
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error checking offline agents");
                }
                
                await Task.Delay(_checkInterval, stoppingToken);
            }
            
            _logger.LogInformation("Agent monitoring service is stopping");
        }
        
        /// <summary>
        /// Checks for offline agents
        /// </summary>
        private async Task CheckOfflineAgentsAsync()
        {
            using var scope = _scopeFactory.CreateScope();
            var agentService = scope.ServiceProvider.GetRequiredService<IAgentService>();
            
            // Get offline agents
            var offlineAgents = await agentService.GetOfflineAgentsAsync(_offlineThreshold);
            
            foreach (var agent in offlineAgents)
            {
                // Skip agents that are already marked as offline
                if (agent.Status == AgentStatus.Offline)
                {
                    continue;
                }
                
                _logger.LogWarning("Agent {AgentId} ({Hostname}) is offline", agent.Id, agent.Hostname);
                
                // Update agent status to offline
                await agentService.UpdateAgentStatusAsync(agent.Id, AgentStatus.Offline);
                
                // Create an alert
                await CreateAlertForOfflineAgent(agent);
            }
        }

        private async Task CreateAlertForOfflineAgent(AgentModels agent)
        {
            using var scope = _scopeFactory.CreateScope();
            var alertService = scope.ServiceProvider.GetRequiredService<IAlertService>();

            // Deduplicate: jangan buat alert baru jika sudah ada alert "Agent X is offline" untuk agent ini dalam 6 jam terakhir
            var recentQuery = new AlertQueryDto
            {
                AgentId = agent.Id,
                Source = "AgentMonitoring",
                StartTime = DateTime.UtcNow.AddHours(-6),
                EndTime = DateTime.UtcNow,
                Limit = 5
            };
            var existing = await alertService.SearchAlertsAsync(recentQuery);
            if (existing.Items.Any(a => a.Title != null && a.Title.Contains("is offline", StringComparison.OrdinalIgnoreCase)))
            {
                _logger.LogDebug("Skipping duplicate offline alert for agent {AgentId} ({Hostname})", agent.Id, agent.Hostname);
                return;
            }

            var alert = new AlertDto
            {
                AgentId = agent.Id,
                Title = $"Agent {agent.Hostname} is offline",
                Description = $"Agent has not reported in for {_offlineThreshold.TotalMinutes} minutes",
                Severity = AlertSeverityModels.High.ToString(),
                Status = AlertStatusModels.New.ToString(),
                Source = "AgentMonitoring",
                Timestamp = DateTime.UtcNow
            };

            await alertService.CreateAlertAsync(alert);
        }
    }
} 