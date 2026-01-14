using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Configuration;
using System.Threading;
using System.Threading.Tasks;
using System;
using System.Net.Http;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using AthalaSIEM.UniversalAgent.Core;
using AthalaSIEM.UniversalAgent.Services;
using AthalaSIEM.UniversalAgent.Models;
using AthalaSIEM.Agent.Collectors;
using AthalaSIEM.UniversalAgent.Core.Collectors;

namespace AthalaSIEM.UniversalAgent
{
    /// <summary>
    /// Enhanced Universal Agent Service following ManageEngine EventLog Analyzer architecture
    /// Orchestrates the complete SIEM agent pipeline: Collection → Processing → Communication
    /// </summary>
    public class UniversalAgentService : BackgroundService
    {
        private readonly ILogger<UniversalAgentService> _logger;
        private readonly ILoggerFactory _loggerFactory;
        private readonly IConfiguration _configuration;
        private readonly CollectorManager _collectorManager;
        private readonly LogProcessor _logProcessor;
        private readonly BackendCommunicationService _communicationService;
        private readonly WindowsAuthenticationService _authenticationService;
        private readonly FIMConfigurationService _fimConfigService;
        private readonly Timer _statusTimer;

        private DateTime _startTime;
        private bool _isInitialized;

        public UniversalAgentService(
            ILogger<UniversalAgentService> logger, 
            ILoggerFactory loggerFactory,
            IConfiguration configuration,
            CollectorManager collectorManager,
            LogProcessor logProcessor,
            BackendCommunicationService communicationService,
            WindowsAuthenticationService authenticationService,
            FIMConfigurationService fimConfigService)
        {
            _logger = logger;
            _loggerFactory = loggerFactory;
            _configuration = configuration;
            _collectorManager = collectorManager;
            _logProcessor = logProcessor;
            _communicationService = communicationService;
            _authenticationService = authenticationService;
            _fimConfigService = fimConfigService;

            // Setup status reporting timer (every 5 minutes)
            _statusTimer = new Timer(ReportStatus, null, TimeSpan.FromMinutes(5), TimeSpan.FromMinutes(5));
        }

        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            _startTime = DateTime.UtcNow;
            _logger.LogInformation("🛡️ Athala SIEM Universal Agent starting up at: {Time}", _startTime);

            try
            {
                // Initialize the complete ManageEngine-style pipeline
                await InitializeAgentPipelineAsync();

                if (!_isInitialized)
                {
                    _logger.LogError("Failed to initialize agent pipeline, stopping service");
                    return;
                }

                _logger.LogInformation("✅ Agent pipeline initialized successfully, starting collection");

                // Start the main agent loop
                await RunAgentMainLoopAsync(stoppingToken);
            }
            catch (OperationCanceledException)
            {
                _logger.LogInformation("Agent service cancellation requested");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Fatal error in agent service");
            }
            finally
            {
                await ShutdownAgentPipelineAsync();
                _logger.LogInformation("🛡️ Athala SIEM Universal Agent stopped at: {Time}", DateTime.UtcNow);
            }
        }

        /// <summary>
        /// Initialize the complete agent pipeline following ManageEngine architecture
        /// </summary>
        private async Task InitializeAgentPipelineAsync()
        {
            try
            {
                _logger.LogInformation("Initializing agent pipeline...");

                // Step 1: Initialize Windows Authentication (CRITICAL for SIEM operations)
                Models.AuthenticationStatus authStatus;
                if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                {
                    _logger.LogInformation("🔐 Step 1: Initializing Windows Authentication...");
                    var authInitialized = await _authenticationService.InitializeAsync();
                    if (!authInitialized)
                    {
                        _logger.LogError("❌ Windows authentication initialization failed - Cannot proceed");
                        throw new InvalidOperationException("Windows authentication failed");
                    }

                    // Check if we have required privileges
                    authStatus = _authenticationService.GetAuthenticationStatus();
                    if (!authStatus.HasAdminPrivileges)
                    {
                        _logger.LogWarning("⚠️ Running without Administrator privileges - SIEM functionality will be limited");
                        _authenticationService.LogAuthenticationGuidance();
                    
                        // Continue but with warnings - some collectors will be disabled
                    }
                    else
                    {
                        _logger.LogInformation("✅ Administrator privileges confirmed - Full SIEM functionality available");
                    }
                }
                else
                {
                    // On non-Windows platforms, create a default authentication status
                    authStatus = new Models.AuthenticationStatus
                    {
                        HasAdminPrivileges = false,
                        IsAuthenticated = false,
                        CurrentUser = Environment.UserName,
                        ServiceAccount = Environment.UserName
                    };
                    _logger.LogInformation("🔐 Skipping Windows Authentication - not running on Windows");
                }

                // Step 2: Initialize backend communication
                _logger.LogInformation("🔗 Step 2: Initializing backend communication...");
                var commInitialized = await _communicationService.InitializeAsync();
                if (!commInitialized)
                {
                    _logger.LogError("Failed to initialize backend communication");
                    throw new InvalidOperationException("Backend communication initialization failed");
                }

                // Step 3: Initialize log processor
                _logger.LogInformation("⚙️ Step 3: Initializing log processor...");
                var processorInitialized = await _logProcessor.InitializeAsync();
                if (!processorInitialized)
                {
                    _logger.LogError("Failed to initialize log processor");
                    throw new InvalidOperationException("Log processor initialization failed");
                }

                // Step 4: Register and initialize collectors (with authentication context)
                _logger.LogInformation("📊 Step 4: Registering collectors with authentication context...");
                await RegisterCollectorsAsync();

                // Step 5: Setup event handlers
                _logger.LogInformation("🔗 Step 5: Setting up event handlers...");
                SetupEventHandlers();

                _isInitialized = true;
                _logger.LogInformation("✅ Agent pipeline initialized successfully with Windows authentication");
                
                // Log final authentication status
                LogAuthenticationSummary(authStatus);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Failed to initialize agent pipeline");
                throw;
            }
        }

        /// <summary>
        /// Logs authentication summary for operational visibility
        /// </summary>
        private void LogAuthenticationSummary(AuthenticationStatus authStatus)
        {
            _logger.LogInformation("🔐 WINDOWS AUTHENTICATION SUMMARY:");
            _logger.LogInformation("User: {User}", authStatus.CurrentUser);
            _logger.LogInformation("Authenticated: {IsAuthenticated}", authStatus.IsAuthenticated ? "YES" : "NO");
            _logger.LogInformation("Administrator: {HasAdmin}", authStatus.HasAdminPrivileges ? "YES" : "NO");
            _logger.LogInformation("Security Log Access: {CanAccess}", authStatus.CanAccessSecurityLog ? "AVAILABLE" : "UNAVAILABLE");
            _logger.LogInformation("Registry Access: {CanAccess}", authStatus.CanAccessRegistry ? "FULL" : "LIMITED");
            _logger.LogInformation("File System Access: {CanAccess}", authStatus.CanAccessFileSystem ? "AVAILABLE" : "LIMITED");
            
            if (authStatus.RequiresElevation)
            {
                _logger.LogWarning("⚠️ ELEVATION REQUIRED for full SIEM functionality");
            }
        }

        /// <summary>
        /// Register all available collectors based on platform and configuration
        /// </summary>
        private async Task RegisterCollectorsAsync()
        {
            try
            {
                var collectorConfigs = GetCollectorConfigurations();
                
                // Get authentication status (Windows only)
                var authStatus = RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
                    ? _authenticationService.GetAuthenticationStatus()
                    : new Models.AuthenticationStatus { HasAdminPrivileges = false };
                
                _logger.LogInformation("Registering {Count} collectors with authentication context", collectorConfigs.Count);

                foreach (var config in collectorConfigs)
                {
                    if (!config.Enabled)
                    {
                        _logger.LogDebug("Skipping disabled collector: {Type}", config.Type);
                        continue;
                    }

                    try
                    {
                        ILogCollector? collector = config.Type.ToLowerInvariant() switch
                        {
                            "windowseventlog" when System.OperatingSystem.IsWindows() => new WindowsEventLogCollector(_loggerFactory.CreateLogger<WindowsEventLogCollector>()),
                            "fileintegrity" => CreateFileIntegrityCollector(),
                            "windowsregistry" when System.OperatingSystem.IsWindows() => new WindowsRegistryCollector(_loggerFactory.CreateLogger<WindowsRegistryCollector>()),
                            "linuxsyslog" when System.OperatingSystem.IsLinux() => new LinuxSyslogCollector(_loggerFactory.CreateLogger<LinuxSyslogCollector>()),
                            // Add more collectors as needed
                            // "iis" => new IISLogCollector(),
                            _ => null
                        };

                        if (collector != null)
                        {
                            // Check authentication requirements
                            if (RequiresAdminPrivileges(config.Type) && !authStatus.HasAdminPrivileges)
                            {
                                _logger.LogWarning("❌ Collector {Type} requires Administrator privileges but not available - SKIPPING", config.Type);
                                continue;
                            }

                            // Debug: Log configuration being passed to collector
                            _logger.LogDebug("🔍 Passing configuration to {Type} collector:", config.Type);
                            foreach (var prop in config.Properties)
                            {
                                _logger.LogDebug("🔍   {Key} = {Value} (Type: {Type})", 
                                    prop.Key, prop.Value, prop.Value?.GetType().Name);
                            }

                            // Pass the Properties dictionary as the configuration
                            var success = await _collectorManager.RegisterCollectorAsync(collector, config.Properties);
                            if (success)
                            {
                                _logger.LogInformation("✅ Registered collector: {Type}", config.Type);
                            }
                            else
                            {
                                _logger.LogWarning("❌ Failed to register collector: {Type}", config.Type);
                            }
                        }
                        else
                        {
                            _logger.LogWarning("❌ Unsupported collector type: {Type}", config.Type);
                        }
                    }
                    catch (Exception ex)
                    {
                        _logger.LogError(ex, "❌ Error registering collector {Type}", config.Type);
                    }
                }

                _logger.LogInformation("Collector registration completed. Active collectors: {Count}", _collectorManager.ActiveCollectors);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error during collector registration");
                throw;
            }
        }

        /// <summary>
        /// Creates FileIntegrityCollector with FIMConfigurationService dependency
        /// </summary>
        private FileIntegrityCollector CreateFileIntegrityCollector()
        {
            return new FileIntegrityCollector(
                _loggerFactory.CreateLogger<FileIntegrityCollector>(),
                _fimConfigService);
        }

        /// <summary>
        /// Checks if a collector type requires Administrator privileges
        /// </summary>
        private bool RequiresAdminPrivileges(string collectorType)
        {
            return collectorType.ToLowerInvariant() switch
            {
                "windowseventlog" => true,  // Security log requires admin
                "windowsregistry" => true,  // Registry monitoring requires admin
                "fileintegrity" => false,  // Can work with limited privileges
                _ => false
            };
        }

        /// <summary>
        /// Setup event handlers for the processing pipeline
        /// </summary>
        private void SetupEventHandlers()
        {
            // CollectorManager events
            _collectorManager.LogsCollected += OnLogsCollected;
            _collectorManager.CollectorError += OnCollectorError;

            // LogProcessor events
            _logProcessor.LogProcessed += OnLogProcessed;
            _logProcessor.CorrelationDetected += OnCorrelationDetected;

            // Communication service events
            _communicationService.LogsSent += OnLogsSent;
            _communicationService.CommunicationError += OnCommunicationError;
            _communicationService.ConnectionStatusChanged += OnConnectionStatusChanged;
        }

        /// <summary>
        /// Main agent loop following ManageEngine's continuous processing pattern
        /// </summary>
        private async Task RunAgentMainLoopAsync(CancellationToken stoppingToken)
        {
            // Start all collectors
            await _collectorManager.StartAllCollectorsAsync();

            // Main processing loop
            while (!stoppingToken.IsCancellationRequested)
            {
                try
                {
                    // Get aggregated logs from all collectors
                    var logs = await _collectorManager.GetAggregatedLogsAsync();
                    
                    if (logs.Any())
                    {
                        // Process logs through the pipeline
                        var processedBatch = await _logProcessor.ProcessLogBatchAsync(logs);
                        
                        // Queue processed logs for sending to backend
                        _communicationService.QueueLogs(processedBatch.ProcessedLogs);
                        
                        _logger.LogDebug("Processed {Count} logs in batch", processedBatch.ProcessedLogs.Count);
                    }

                    // Wait before next processing cycle
                    await Task.Delay(TimeSpan.FromSeconds(10), stoppingToken);
                }
                catch (OperationCanceledException)
                {
                    break;
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error in agent main loop");
                    await Task.Delay(TimeSpan.FromSeconds(30), stoppingToken);
                }
            }
        }

        /// <summary>
        /// Shutdown agent pipeline gracefully
        /// </summary>
        private async Task ShutdownAgentPipelineAsync()
        {
            try
            {
                _logger.LogInformation("Shutting down agent pipeline...");

                // Stop collectors
                await _collectorManager.StopAllCollectorsAsync();

                // Flush remaining logs
                await _communicationService.FlushLogsAsync();

                // Dispose resources
                await _collectorManager.DisposeAsync();
                await _logProcessor.DisposeAsync();
                await _communicationService.DisposeAsync();

                _logger.LogInformation("Agent pipeline shutdown complete");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error shutting down agent pipeline");
            }
        }

        #region Event Handlers

        private void OnLogsCollected(object? sender, LogCollectedEventArgs e)
        {
            _logger.LogDebug("Collected {Count} logs from {Source}", e.Logs.Count(), e.Source);
        }

        private void OnCollectorError(object? sender, CollectorErrorEventArgs e)
        {
            _logger.LogError(e.Exception, "Collector error from {CollectorName}: {Message}", 
                e.CollectorName, e.Message);
        }

        private void OnLogProcessed(object? sender, LogProcessedEventArgs e)
        {
            _logger.LogDebug("Processed batch of {Count} logs", e.ProcessedBatch.TotalProcessed);
        }

        private void OnCorrelationDetected(object? sender, CorrelationDetectedEventArgs e)
        {
            _logger.LogWarning("🚨 Security correlation detected: {CorrelationName} - {Description}",
                e.Correlation.Name, e.Correlation.Description);
        }

        private void OnLogsSent(object? sender, LogsSentEventArgs e)
        {
            _logger.LogDebug("Sent {Count} logs to backend", e.LogCount);
        }

        private void OnCommunicationError(object? sender, CommunicationErrorEventArgs e)
        {
            _logger.LogError("Communication error: {Message} (affected {Count} logs)", 
                e.ErrorMessage, e.LogCount);
        }

        private void OnConnectionStatusChanged(object? sender, ConnectionStatusChangedEventArgs e)
        {
            _logger.LogInformation("Backend connection status: {Status} - {Message}", 
                e.IsConnected ? "CONNECTED" : "DISCONNECTED", e.StatusMessage);
        }

        #endregion

        #region Configuration Management

        private List<CollectorConfiguration> GetCollectorConfigurations()
        {
            var configs = new List<CollectorConfiguration>();
            
            // Get collector configurations from appsettings.json
            var collectorsConfig = _configuration.GetSection("Collectors");
            
            if (collectorsConfig.Exists())
            {
                // Parse configurations manually to preserve array types
                foreach (var collectorSection in collectorsConfig.GetChildren())
                {
                    var config = new CollectorConfiguration();
                    config.Type = collectorSection.GetValue<string>("Type") ?? "";
                    config.Enabled = collectorSection.GetValue<bool>("Enabled", true);
                    
                    // Get properties section and preserve array types
                    var propertiesSection = collectorSection.GetSection("Properties");
                    if (propertiesSection.Exists())
                    {
                        foreach (var property in propertiesSection.GetChildren())
                        {
                            var key = property.Key;
                            var value = property.Value;
                            
                            // Check if this is an array property
                            if (property.GetChildren().Any())
                            {
                                // This is an array - convert to string array
                                var arrayValues = property.GetChildren().Select(c => c.Value).Where(v => !string.IsNullOrEmpty(v)).ToArray();
                                config.Properties[key] = arrayValues;
                            }
                            else
                            {
                                // This is a simple value
                                config.Properties[key] = value ?? "";
                            }
                        }
                    }
                    
                    configs.Add(config);
                }
            }
            else
            {
                
                // Default configuration for Windows Event Log (REQUIRES ADMIN for Security logs!)
                configs.Add(new CollectorConfiguration
                {
                    Type = "WindowsEventLog",
                    Enabled = true,
                    Properties = new Dictionary<string, object>
                    {
                        ["LogSources"] = new[] { "Security", "System", "Application" },
                        ["CollectAllEvents"] = true,
                        ["EnableSecurityFiltering"] = false
                    }
                });

                // Add File Integrity Monitoring (FIM)
                configs.Add(new CollectorConfiguration
                {
                    Type = "FileIntegrity",
                    Enabled = true,
                    Properties = new Dictionary<string, object>
                    {
                        ["MonitoredPaths"] = @"C:\Windows\System32\drivers,C:\Windows\System32\config,C:\Program Files\AthalaSIEM,C:\inetpub\wwwroot",
                        ["RealTimeMonitoring"] = "true",
                        ["ScanIntervalMinutes"] = "30"
                    }
                });

                // Add Windows Registry Monitoring
                configs.Add(new CollectorConfiguration
                {
                    Type = "WindowsRegistry",
                    Enabled = true,
                    Properties = new Dictionary<string, object>
                    {
                        ["ScanIntervalMinutes"] = "10",
                        ["EnableThreatDetection"] = "true"
                    }
                });
            }

            return configs.Where(c => c.Enabled).ToList();
        }

        #endregion

        #region Status Reporting

        private async void ReportStatus(object? state)
        {
            try
            {
                var uptime = DateTime.UtcNow - _startTime;
                var collectorHealth = await _collectorManager.GetHealthStatusAsync();
                var communicationHealth = _communicationService.GetHealthStatus();

                _logger.LogInformation(
                    "📊 Agent Status - Uptime: {Uptime}, Active Collectors: {ActiveCollectors}/{TotalCollectors}, " +
                    "Total Logs: {TotalLogs}, Queued: {QueuedLogs}, Connected: {Connected}",
                    uptime.ToString(@"dd\.hh\:mm\:ss"),
                    collectorHealth.ActiveCollectors,
                    collectorHealth.TotalCollectors,
                    collectorHealth.TotalLogsCollected,
                    communicationHealth.QueuedLogs,
                    communicationHealth.IsConnected);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error reporting agent status");
            }
        }

        #endregion

        public override async Task StopAsync(CancellationToken stoppingToken)
        {
            _logger.LogInformation("🛑 Universal Agent service stopping...");
            _statusTimer?.Dispose();
            await base.StopAsync(stoppingToken);
        }
    }

    #region Supporting Classes

    // CollectorConfiguration moved to Program.cs to avoid duplication

    #endregion
}
