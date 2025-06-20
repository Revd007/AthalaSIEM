using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Configuration;
using System.Threading;
using System.Threading.Tasks;
using System;
using System.Net.Http;
using System.Collections.Generic;
using System.Linq;
using AthalaSIEM.Agent.Core;
using AthalaSIEM.UniversalAgent.Services;
using AthalaSIEM.Agent.Collectors;

namespace AthalaSIEM.UniversalAgent
{
    /// <summary>
    /// Enhanced Universal Agent Service following ManageEngine EventLog Analyzer architecture
    /// Orchestrates the complete SIEM agent pipeline: Collection → Processing → Communication
    /// </summary>
    public class UniversalAgentService : BackgroundService
    {
        private readonly ILogger<UniversalAgentService> _logger;
        private readonly IConfiguration _configuration;
        private readonly CollectorManager _collectorManager;
        private readonly LogProcessor _logProcessor;
        private readonly BackendCommunicationService _communicationService;
        private readonly Timer _statusTimer;

        private DateTime _startTime;
        private bool _isInitialized;

        public UniversalAgentService(
            ILogger<UniversalAgentService> logger, 
            IConfiguration configuration,
            CollectorManager collectorManager,
            LogProcessor logProcessor,
            BackendCommunicationService communicationService)
        {
            _logger = logger;
            _configuration = configuration;
            _collectorManager = collectorManager;
            _logProcessor = logProcessor;
            _communicationService = communicationService;

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

                // Step 1: Initialize backend communication
                _logger.LogInformation("🔗 Initializing backend communication...");
                var communicationInitialized = await _communicationService.InitializeAsync();
                if (!communicationInitialized)
                {
                    _logger.LogError("Failed to initialize backend communication");
                    return;
                }

                // Step 2: Register and configure collectors (ManageEngine multi-source pattern)
                _logger.LogInformation("📊 Registering log collectors...");
                await RegisterCollectorsAsync();

                // Step 3: Setup event handlers for the processing pipeline
                SetupEventHandlers();

                _isInitialized = true;
                _logger.LogInformation("✅ Agent pipeline initialized successfully");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error initializing agent pipeline");
                _isInitialized = false;
            }
        }

        /// <summary>
        /// Register all available collectors based on platform and configuration
        /// </summary>
        private async Task RegisterCollectorsAsync()
        {
            var collectorConfigs = GetCollectorConfigurations();

            foreach (var config in collectorConfigs)
            {
                try
                {
                    ILogCollector? collector = config.Type.ToLowerInvariant() switch
                    {
                        "windowseventlog" => new WindowsEventLogCollector(),
                        // Add more collectors as needed
                        // "syslog" => new SyslogCollector(),
                        // "iis" => new IISLogCollector(),
                        _ => null
                    };

                    if (collector != null)
                    {
                        var registered = await _collectorManager.RegisterCollectorAsync(collector, config.Properties);
                        if (registered)
                        {
                            _logger.LogInformation("✅ Registered collector: {CollectorName}", collector.CollectorName);
                        }
                        else
                        {
                            _logger.LogWarning("❌ Failed to register collector: {CollectorName}", collector.CollectorName);
                        }
                    }
                    else
                    {
                        _logger.LogWarning("Unknown collector type: {Type}", config.Type);
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error registering collector: {Type}", config.Type);
                }
            }
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
                configs = collectorsConfig.Get<List<CollectorConfiguration>>() ?? new List<CollectorConfiguration>();
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