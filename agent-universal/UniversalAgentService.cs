using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Configuration;
using System.Threading;
using System.Threading.Tasks;
using System;
using System.Net.Http;

namespace AthalaSIEM.UniversalAgent
{
    public class UniversalAgentService : BackgroundService
    {
        private readonly ILogger<UniversalAgentService> _logger;
        private readonly IConfiguration _configuration;

        public UniversalAgentService(ILogger<UniversalAgentService> logger, IConfiguration configuration)
        {
            _logger = logger;
            _configuration = configuration;
        }

        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            _logger.LogInformation("Athala SIEM Universal Agent service started at: {time}", DateTimeOffset.Now);

            var backendUrl = _configuration["BackendUrl"] ?? "http://localhost:9595";
            _logger.LogInformation("Backend URL configured as: {backendUrl}", backendUrl);

            while (!stoppingToken.IsCancellationRequested)
            {
                try
                {
                    // Agent heartbeat
                    _logger.LogInformation("Agent heartbeat at: {time}", DateTimeOffset.Now);
                    
                    // Test connection to backend
                    await TestBackendConnection(backendUrl);
                    
                    // Wait 30 seconds before next heartbeat
                    await Task.Delay(TimeSpan.FromSeconds(30), stoppingToken);
                }
                catch (OperationCanceledException)
                {
                    // Expected when cancellation is requested
                    break;
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error in agent service loop");
                    await Task.Delay(TimeSpan.FromSeconds(10), stoppingToken);
                }
            }
            
            _logger.LogInformation("Athala SIEM Universal Agent service stopped at: {time}", DateTimeOffset.Now);
        }

        private async Task TestBackendConnection(string backendUrl)
        {
            try
            {
                using var client = new HttpClient();
                client.Timeout = TimeSpan.FromSeconds(5);
                
                var response = await client.GetAsync($"{backendUrl}/api/health");
                
                if (response.IsSuccessStatusCode)
                {
                    _logger.LogDebug("Backend connection test successful");
                }
                else
                {
                    _logger.LogWarning("Backend connection test failed with status: {status}", response.StatusCode);
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning("Backend connection test failed: {message}", ex.Message);
            }
        }

        public override async Task StopAsync(CancellationToken stoppingToken)
        {
            _logger.LogInformation("Universal Agent service is stopping...");
            await base.StopAsync(stoppingToken);
        }
    }
} 