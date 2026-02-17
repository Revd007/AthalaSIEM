using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security.Principal;
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
            _logger.LogInformation("═══════════════════════════════════════════════════════════");
            _logger.LogInformation("Athala SIEM Agent starting at: {time}", DateTimeOffset.Now);
            _logger.LogInformation("═══════════════════════════════════════════════════════════");

            // Check admin privileges (Windows only)
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                try
                {
                    using var identity = WindowsIdentity.GetCurrent();
                    var principal = new WindowsPrincipal(identity);
                    var isAdmin = principal.IsInRole(WindowsBuiltInRole.Administrator);
                    if (isAdmin)
                    {
                        _logger.LogInformation("Running with Administrator privileges. All event logs accessible.");
                    }
                    else
                    {
                        _logger.LogWarning(
                            "Running WITHOUT Administrator privileges. Security event log will NOT be accessible. " +
                            "To collect Security logs, run as Administrator or install as a Windows Service with LocalSystem account.");
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogDebug(ex, "Could not determine privilege level");
                }
            }

            try
            {
                _logger.LogInformation("Step 1: Checking agent identity status...");
                
                // Force reload identity from disk so we never use stale in-memory state
                var hasValidIdentity = await _identityService.HasValidIdentityAsync();
                _isRegistered = await _identityService.IsRegisteredAsync();
                
                _logger.LogInformation("Step 1 Result: HasValidIdentity={HasValid}, IsRegistered={IsRegistered}",
                    hasValidIdentity, _isRegistered);

                if (!_isRegistered)
                {
                    _logger.LogInformation("═══════════════════════════════════════════════════════════");
                    _logger.LogInformation("Step 2: Agent is NOT registered. Starting registration flow...");
                    _logger.LogInformation("═══════════════════════════════════════════════════════════");
                    await AttemptRegistrationAsync();
                    
                    // Re-check after registration attempt
                    _isRegistered = await _identityService.IsRegisteredAsync();
                    _logger.LogInformation("Step 2 Result: After registration attempt, IsRegistered={IsRegistered}", _isRegistered);
                }
                else
                {
                    _logger.LogInformation("Agent already registered. Validating API key...");
                    bool isApiKeyValid = await _identityService.ValidateApiKeyAsync();
                    if (!isApiKeyValid)
                    {
                        _logger.LogWarning("API key validation failed. Attempting to rotate API key...");
                        bool rotated = await _identityService.RotateApiKeyAsync();
                        if (!rotated)
                        {
                            // RotateApiKey clears identity when it fails, so check again
                            _isRegistered = await _identityService.IsRegisteredAsync();
                            _logger.LogInformation("After failed rotation: IsRegistered={IsRegistered}", _isRegistered);
                            if (!_isRegistered)
                            {
                                _logger.LogInformation("Identity cleared after rotation failure. Attempting re-registration...");
                                await AttemptRegistrationAsync();
                            }
                        }
                        else
                        {
                            _logger.LogInformation("API key rotated successfully.");
                        }
                    }
                    else
                    {
                        _logger.LogInformation("API key is valid.");
                    }
                }

                // Ready gate: do not start collectors until agent is registered
                const int registrationRetrySeconds = 30;
                const int maxRegistrationRetries = 20; // ~10 minutes
                int registrationRetries = 0;

                while (!_isRegistered && !stoppingToken.IsCancellationRequested)
                {
                    _logger.LogWarning(
                        "Agent not registered. Collectors will not start until registration succeeds. Retry in {Seconds}s (attempt {Attempt}/{Max}).",
                        registrationRetrySeconds, registrationRetries + 1, maxRegistrationRetries);
                    try
                    {
                        await Task.Delay(TimeSpan.FromSeconds(registrationRetrySeconds), stoppingToken);
                    }
                    catch (OperationCanceledException)
                    {
                        break;
                    }
                    await AttemptRegistrationAsync();
                    _isRegistered = await _identityService.IsRegisteredAsync();
                    registrationRetries++;
                    if (registrationRetries >= maxRegistrationRetries)
                    {
                        _logger.LogError("Registration failed after {Count} attempts. Agent will keep retrying every {Seconds}s until service is stopped.",
                            maxRegistrationRetries, registrationRetrySeconds);
                        registrationRetries = 0; // Continue retrying indefinitely
                    }
                }

                if (!_isRegistered)
                {
                    _logger.LogWarning("Service stopping before registration completed. No collectors started.");
                    return;
                }

                _logger.LogInformation("Agent registered. Starting health monitoring and log collectors.");

                // Start health monitoring
                _healthMonitor.StartMonitoring();
                _logger.LogInformation("Health monitoring started.");

                // Initialize log collectors (FileIntegrity, WindowsEventLog, etc.)
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
        /// Attempts to register the agent with the backend (with token first, then without)
        /// </summary>
        private async Task AttemptRegistrationAsync()
        {
            _logger.LogInformation("=== STARTING AGENT REGISTRATION PROCESS ===");
            
            try
            {
                // Try with deployment token first
                var token = _settings.Value?.DeploymentToken?.Trim();
                _logger.LogInformation("Checking deployment token: {HasToken}", !string.IsNullOrEmpty(token) ? "PRESENT in config" : "NOT SET in config");
                
                if (!string.IsNullOrEmpty(token))
                {
                    _logger.LogInformation("Attempting registration with deployment token (length: {TokenLength} chars)...", token.Length);
                    try
                    {
                        var result = await _identityService.RegisterWithTokenAsync(token);
                        _isRegistered = result.Success;
                        
                        if (_isRegistered)
                        {
                            _logger.LogInformation("✓ SUCCESS: Agent registered with token. AgentId: {AgentId}", result.AgentId);
                            
                            // Verify registration persisted
                            var verifyRegistered = await _identityService.IsRegisteredAsync();
                            _logger.LogInformation("Registration verification: IsRegistered={IsRegistered}", verifyRegistered);
                            
                            if (!verifyRegistered)
                            {
                                _logger.LogError("⚠ WARNING: Registration succeeded but IsRegisteredAsync() returned false. Identity may not have been saved.");
                            }
                            
                            return;
                        }
                        else
                        {
                            _logger.LogWarning("✗ FAILED: Registration with token failed. Error: {ErrorMessage}", result.Message);
                        }
                    }
                    catch (Exception tokenEx)
                    {
                        _logger.LogError(tokenEx, "✗ EXCEPTION during token registration: {ErrorMessage}", tokenEx.Message);
                    }
                }
                else
                {
                    _logger.LogInformation("No deployment token configured, skipping token-based registration.");
                }

                // Fallback: register without token
                _logger.LogInformation("Attempting registration WITHOUT token (standard registration)...");
                try
                {
                    var fallbackResult = await _identityService.RegisterAgentAsync();
                    _isRegistered = fallbackResult.Success;
                    
                    if (_isRegistered)
                    {
                        _logger.LogInformation("✓ SUCCESS: Agent registered without token. AgentId: {AgentId}", fallbackResult.AgentId);
                        
                        // Verify registration persisted
                        var verifyRegistered = await _identityService.IsRegisteredAsync();
                        _logger.LogInformation("Registration verification: IsRegistered={IsRegistered}", verifyRegistered);
                        
                        if (!verifyRegistered)
                        {
                            _logger.LogError("⚠ WARNING: Registration succeeded but IsRegisteredAsync() returned false. Identity may not have been saved.");
                        }
                    }
                    else
                    {
                        _logger.LogError("✗ FAILED: Agent registration failed. Error: {ErrorMessage}. Agent will retry on next startup.", fallbackResult.Message);
                    }
                }
                catch (Exception fallbackEx)
                {
                    _logger.LogError(fallbackEx, "✗ EXCEPTION during standard registration: {ErrorMessage}", fallbackEx.Message);
                    _isRegistered = false;
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "✗ CRITICAL EXCEPTION in registration process: {ErrorMessage}", ex.Message);
                _isRegistered = false;
            }
            finally
            {
                _logger.LogInformation("=== REGISTRATION PROCESS COMPLETED. Final status: IsRegistered={IsRegistered} ===", _isRegistered);
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

                // Resolve collector list: use settings, or default Windows collectors if empty on Windows
                var collectorList = _settings.Value?.Collectors ?? new List<CollectorSettings>();
                _logger.LogInformation("Collectors from config: {Count} configured", collectorList.Count);
                if (collectorList.Count == 0 && RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                {
                    _logger.LogInformation("No collectors configured. Using default Windows Event Log collector.");
                    collectorList = new List<CollectorSettings>
                    {
                        new CollectorSettings
                        {
                            Type = "WindowsEventLog",
                            Enabled = true,
                            IntervalSeconds = 10,
                            Properties = new Dictionary<string, string>
                            {
                                ["EventLogs"] = "Application,System,Security",
                                ["CollectionMode"] = "Polling",
                                ["MaxEvents"] = "100"
                            }
                        }
                    };
                }

                // Create and start collectors based on settings
                foreach (var collectorSetting in collectorList)
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
                    _logger.LogDebug("Agent not registered. Skipping heartbeat. Registration will be retried on config refresh.");
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
                    // Silently retry registration instead of just logging a warning
                    _logger.LogDebug("Agent not registered. Attempting re-registration before config refresh...");
                    _isRegistered = await _identityService.IsRegisteredAsync();
                    if (!_isRegistered)
                    {
                        await AttemptRegistrationAsync();
                        _isRegistered = await _identityService.IsRegisteredAsync();
                    }

                    if (!_isRegistered)
                    {
                        _logger.LogWarning("Agent still not registered. Skipping configuration refresh. Will retry next cycle.");
                        return;
                    }
                }

                _logger.LogInformation("Refreshing agent configuration...");
                var newConfig = await _logForwarder.GetAgentConfigurationAsync();
                
                if (newConfig != null)
                {
                    _logger.LogInformation("Received updated configuration. Reinitializing collectors...");
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