using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using AthalaSIEM.Agent.Models;
using AthalaSIEM.Agent.Collectors;
using AthalaSIEM.Agent.Security;
using AthalaSIEM.Agent.Communication;

namespace AthalaSIEM.Agent.Services
{
    /// <summary>
    /// Main agent service that manages the SIEM agent lifecycle
    /// </summary>
    public class SiemAgentService : BackgroundService
    {
        private readonly ILogger<SiemAgentService> _logger;
        private readonly IOptions<AgentSettings> _settings;
        private readonly IAgentIdentityService _identityService;
        private readonly IAgentHealthMonitor _healthMonitor;
        private readonly ILogCollectorFactory _logCollectorFactory;
        private readonly ILogForwarder _logForwarder;
        private readonly List<ILogCollector> _activeCollectors = new();
        private Timer? _heartbeatTimer;
        private Timer? _configRefreshTimer;
        private bool _isRegistered = false;

        /// <summary>
        /// Initializes a new instance of the <see cref="SiemAgentService"/> class
        /// </summary>
        /// <param name="logger">Logger instance</param>
        /// <param name="settings">Agent settings</param>
        /// <param name="identityService">Agent identity service</param>
        /// <param name="healthMonitor">Health monitor</param>
        /// <param name="logCollectorFactory">Log collector factory</param>
        /// <param name="logForwarder">Log forwarder</param>
        public SiemAgentService(
            ILogger<SiemAgentService> logger,
            IOptions<AgentSettings> settings,
            IAgentIdentityService identityService,
            IAgentHealthMonitor healthMonitor,
            ILogCollectorFactory logCollectorFactory,
            ILogForwarder logForwarder)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _settings = settings ?? throw new ArgumentNullException(nameof(settings));
            _identityService = identityService ?? throw new ArgumentNullException(nameof(identityService));
            _healthMonitor = healthMonitor ?? throw new ArgumentNullException(nameof(healthMonitor));
            _logCollectorFactory = logCollectorFactory ?? throw new ArgumentNullException(nameof(logCollectorFactory));
            _logForwarder = logForwarder ?? throw new ArgumentNullException(nameof(logForwarder));
        }

        /// <summary>
        /// Executes the agent service
        /// </summary>
        /// <param name="stoppingToken">Cancellation token</param>
        /// <returns>A task representing the asynchronous operation</returns>
        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            _logger.LogInformation("Athala SIEM Agent starting at: {time}", DateTimeOffset.Now);

            try
            {
                // Register agent if not already registered
                _isRegistered = await _identityService.IsRegisteredAsync();
                if (!_isRegistered)
                {
                    _logger.LogInformation("Agent not registered. Attempting registration...");
                    _isRegistered = await _identityService.RegisterAgentAsync();
                    
                    if (!_isRegistered)
                    {
                        _logger.LogWarning("Agent registration failed. Will retry on next startup.");
                    }
                    else
                    {
                        _logger.LogInformation("Agent registered successfully.");
                    }
                }
                else
                {
                    _logger.LogInformation("Agent already registered.");
                    
                    // Validate API key
                    bool isApiKeyValid = await _identityService.ValidateApiKeyAsync();
                    if (!isApiKeyValid)
                    {
                        _logger.LogWarning("API key validation failed. Attempting to rotate API key...");
                        await _identityService.RotateApiKeyAsync();
                    }
                }

                // Start health monitoring
                _healthMonitor.StartMonitoring();
                _logger.LogInformation("Health monitoring started.");

                // Initialize log collectors
                await InitializeLogCollectorsAsync();

                // Setup heartbeat timer
                SetupHeartbeatTimer();

                // Setup configuration refresh timer
                SetupConfigRefreshTimer();

                // Wait until the service is stopped
                await Task.Delay(Timeout.Infinite, stoppingToken);
            }
            catch (TaskCanceledException)
            {
                // Normal shutdown, no need to log as error
                _logger.LogInformation("Athala SIEM Agent shutting down.");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "An error occurred while running the agent service.");
            }
            finally
            {
                // Cleanup
                _heartbeatTimer?.Dispose();
                _configRefreshTimer?.Dispose();
                
                foreach (var collector in _activeCollectors)
                {
                    try
                    {
                        await collector.StopAsync();
                    }
                    catch (Exception ex)
                    {
                        _logger.LogError(ex, "Error stopping collector {CollectorType}", collector.GetType().Name);
                    }
                }
                
                _logger.LogInformation("Athala SIEM Agent stopped at: {time}", DateTimeOffset.Now);
            }
        }

        /// <summary>
        /// Initializes the log collectors
        /// </summary>
        /// <returns>A task representing the asynchronous operation</returns>
        private async Task InitializeLogCollectorsAsync()
        {
            _logger.LogInformation("Initializing log collectors...");
            
            try
            {
                // Clear any existing collectors
                foreach (var collector in _activeCollectors)
                {
                    await collector.StopAsync();
                }
                _activeCollectors.Clear();

                // Create and start collectors based on settings
                foreach (var collectorSetting in _settings.Value.Collectors)
                {
                    if (!collectorSetting.Enabled)
                    {
                        _logger.LogInformation("Collector {CollectorType} is disabled, skipping.", collectorSetting.Type);
                        continue;
                    }

                    try
                    {
                        var collector = _logCollectorFactory.CreateCollector(collectorSetting);
                        if (collector != null)
                        {
                            // Subscribe to log events
                            collector.LogCollected += async (sender, logData) =>
                            {
                                try
                                {
                                    await _logForwarder.ForwardLogAsync(logData);
                                }
                                catch (Exception ex)
                                {
                                    _logger.LogError(ex, "Error forwarding log from {CollectorType}", collectorSetting.Type);
                                }
                            };

                            // Start the collector
                            await collector.StartAsync();
                            _activeCollectors.Add(collector);
                            _logger.LogInformation("Started collector: {CollectorType}", collectorSetting.Type);
                        }
                        else
                        {
                            _logger.LogWarning("Failed to create collector of type {CollectorType}", collectorSetting.Type);
                        }
                    }
                    catch (Exception ex)
                    {
                        _logger.LogError(ex, "Error initializing collector {CollectorType}", collectorSetting.Type);
                    }
                }

                _logger.LogInformation("Initialized {Count} log collectors.", _activeCollectors.Count);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error initializing log collectors.");
            }
        }

        /// <summary>
        /// Sets up the heartbeat timer
        /// </summary>
        private void SetupHeartbeatTimer()
        {
            var heartbeatInterval = TimeSpan.FromMinutes(_settings.Value.HeartbeatIntervalMinutes);
            _logger.LogInformation("Setting up heartbeat timer with interval: {Interval}", heartbeatInterval);
            
            _heartbeatTimer = new Timer(async _ => await SendHeartbeatAsync(), null, 
                TimeSpan.FromSeconds(30), // Initial delay
                heartbeatInterval);
        }

        /// <summary>
        /// Sets up the configuration refresh timer
        /// </summary>
        private void SetupConfigRefreshTimer()
        {
            var refreshInterval = TimeSpan.FromMinutes(_settings.Value.ConfigRefreshIntervalMinutes);
            _logger.LogInformation("Setting up configuration refresh timer with interval: {Interval}", refreshInterval);
            
            _configRefreshTimer = new Timer(async _ => await RefreshConfigurationAsync(), null, 
                TimeSpan.FromMinutes(1), // Initial delay
                refreshInterval);
        }

        /// <summary>
        /// Sends a heartbeat to the backend
        /// </summary>
        /// <returns>A task representing the asynchronous operation</returns>
        private async Task SendHeartbeatAsync()
        {
            try
            {
                if (!_isRegistered)
                {
                    _logger.LogWarning("Agent not registered. Skipping heartbeat.");
                    return;
                }

                _logger.LogDebug("Sending heartbeat...");
                var healthStatus = await _healthMonitor.GetCurrentHealthStatus();
                var heartbeat = new AgentHeartbeat
                {
                    Timestamp = DateTime.UtcNow,
                    Status = healthStatus.Status.ToString(),
                    Uptime = healthStatus.Uptime,
                    CpuUsage = healthStatus.CpuUsage,
                    MemoryUsage = healthStatus.MemoryUsage,
                    ActiveCollectors = _activeCollectors.Select(c => c.GetType().Name).ToList(),
                    LogsCollected = 0, // TODO: Implement log statistics
                    LogsForwarded = 0  // TODO: Implement log statistics
                };

                await _logForwarder.SendHeartbeatAsync(heartbeat);
                _logger.LogDebug("Heartbeat sent successfully.");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error sending heartbeat.");
            }
        }

        /// <summary>
        /// Refreshes the agent configuration from the backend
        /// </summary>
        /// <returns>A task representing the asynchronous operation</returns>
        private async Task RefreshConfigurationAsync()
        {
            try
            {
                if (!_isRegistered)
                {
                    _logger.LogWarning("Agent not registered. Skipping configuration refresh.");
                    return;
                }

                _logger.LogInformation("Refreshing agent configuration...");
                var newConfig = await _logForwarder.GetAgentConfigurationAsync();
                
                if (newConfig != null)
                {
                    _logger.LogInformation("Received updated configuration. Reinitializing collectors...");
                    // TODO: Update local configuration
                    await InitializeLogCollectorsAsync();
                }
                else
                {
                    _logger.LogInformation("No configuration changes detected.");
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error refreshing configuration.");
            }
        }
    }
} 