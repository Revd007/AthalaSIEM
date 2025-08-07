using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using AthalaSIEM.UniversalAgent.Models;

namespace AthalaSIEM.UniversalAgent.Core
{
    /// <summary>
    /// Central manager for all log collectors following ManageEngine EventLog Analyzer pattern
    /// Orchestrates multiple collectors, manages their lifecycle, and coordinates data flow
    /// </summary>
    public class CollectorManager : IAsyncDisposable
    {
                private readonly ILogger<CollectorManager> _logger;
        private readonly IConfiguration _configuration;
        private readonly Dictionary<string, ILogCollector> _collectors = new();
        private readonly Dictionary<string, Task> _collectorTasks = new();
        private readonly CancellationTokenSource _cancellationTokenSource = new();
        private readonly List<LogEntry> _aggregatedLogs = new();
        private readonly object _logLock = new();

        public bool IsRunning { get; private set; }
        public int ActiveCollectors => _collectors.Count(c => c.Value.IsActive);
        public long TotalLogsCollected => _collectors.Values.Sum(c => c.LogsCollected);

        public event EventHandler<LogCollectedEventArgs>? LogsCollected;
        public event EventHandler<CollectorErrorEventArgs>? CollectorError;

        public CollectorManager(ILogger<CollectorManager> logger, IConfiguration configuration)
        {
            _logger = logger;
            _configuration = configuration;
        }

        /// <summary>
        /// Register a new log collector (following ManageEngine's multi-source pattern)
        /// </summary>
        public async Task<bool> RegisterCollectorAsync(ILogCollector collector, Dictionary<string, object> config)
        {
            try
            {
                _logger.LogInformation("Registering collector: {CollectorName}", collector.CollectorName);

                // Initialize collector with config
                var initialized = await collector.InitializeAsync(config);
                if (!initialized)
                {
                    _logger.LogError("Failed to initialize collector: {CollectorName}", collector.CollectorName);
                    return false;
                }

                // Subscribe to collector events
                collector.LogCollected += OnCollectorLogCollected;
                collector.CollectionError += OnCollectorError;

                // Register collector
                _collectors[collector.CollectorName] = collector;
                
                _logger.LogInformation("Successfully registered collector: {CollectorName}", collector.CollectorName);
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error registering collector: {CollectorName}", collector.CollectorName);
                return false;
            }
        }

        /// <summary>
        /// Start all registered collectors (ManageEngine parallel collection pattern)
        /// </summary>
        public Task StartAllCollectorsAsync()
        {
            try
            {
                _logger.LogInformation("Starting {Count} collectors", _collectors.Count);
                IsRunning = true;

                var startTasks = new List<Task>();

                foreach (var collector in _collectors.Values)
                {
                    var task = Task.Run(async () =>
                    {
                        try
                        {
                            await collector.StartCollectionAsync(_cancellationTokenSource.Token);
                        }
                        catch (Exception ex)
                        {
                            _logger.LogError(ex, "Error in collector: {CollectorName}", collector.CollectorName);
                        }
                    }, _cancellationTokenSource.Token);

                    _collectorTasks[collector.CollectorName] = task;
                    startTasks.Add(task);
                }

                // Don't wait for all tasks to complete (they run continuously)
                _logger.LogInformation("All collectors started successfully");
                
                return Task.CompletedTask;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error starting collectors");
                throw;
            }
        }

        /// <summary>
        /// Stop all collectors gracefully
        /// </summary>
        public async Task StopAllCollectorsAsync()
        {
            try
            {
                _logger.LogInformation("Stopping all collectors");
                IsRunning = false;

                // Cancel all operations (check if not disposed)
                if (!_cancellationTokenSource.IsCancellationRequested)
                {
                    _cancellationTokenSource.Cancel();
                }

                // Stop each collector
                var stopTasks = _collectors.Values.Select(c => c.StopCollectionAsync());
                await Task.WhenAll(stopTasks);

                // Wait for all collector tasks to complete (with timeout)
                var timeout = Task.Delay(TimeSpan.FromSeconds(30));
                var allTasks = Task.WhenAll(_collectorTasks.Values);
                
                await Task.WhenAny(allTasks, timeout);

                _logger.LogInformation("All collectors stopped");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error stopping collectors");
            }
        }

        /// <summary>
        /// Get aggregated logs from all collectors (ManageEngine batch processing pattern)
        /// </summary>
        public async Task<IEnumerable<LogEntry>> GetAggregatedLogsAsync(int maxBatchSize = 1000)
        {
            var allLogs = new List<LogEntry>();

            // Collect from all active collectors
            foreach (var collector in _collectors.Values.Where(c => c.IsActive))
            {
                try
                {
                    var collectorLogs = await collector.GetLogsAsync(maxBatchSize / _collectors.Count);
                    allLogs.AddRange(collectorLogs);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error getting logs from collector: {CollectorName}", collector.CollectorName);
                }
            }

            // Also return aggregated logs
            lock (_logLock)
            {
                var aggregatedToReturn = _aggregatedLogs.Take(maxBatchSize - allLogs.Count).ToList();
                _aggregatedLogs.RemoveRange(0, aggregatedToReturn.Count);
                allLogs.AddRange(aggregatedToReturn);
            }

            // Sort by timestamp (ManageEngine correlation pattern)
            return allLogs.OrderBy(log => log.Timestamp);
        }

        /// <summary>
        /// Get health status of all collectors (ManageEngine monitoring pattern)
        /// </summary>
        public async Task<CollectorManagerHealth> GetHealthStatusAsync()
        {
            var health = new CollectorManagerHealth
            {
                IsHealthy = IsRunning,
                TotalCollectors = _collectors.Count,
                ActiveCollectors = ActiveCollectors,
                TotalLogsCollected = TotalLogsCollected,
                LastHealthCheck = DateTime.UtcNow
            };

            foreach (var collector in _collectors.Values)
            {
                try
                {
                    var collectorHealth = await collector.GetHealthAsync();
                    health.CollectorHealthStatuses[collector.CollectorName] = collectorHealth;
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error getting health from collector: {CollectorName}", collector.CollectorName);
                    health.CollectorHealthStatuses[collector.CollectorName] = new CollectorHealth
                    {
                        IsHealthy = false,
                        Status = "Error",
                        Errors = new List<string> { ex.Message }
                    };
                }
            }

            health.IsHealthy = health.CollectorHealthStatuses.Values.All(h => h.IsHealthy);
            return health;
        }

        /// <summary>
        /// Handle logs collected from individual collectors
        /// </summary>
        private void OnCollectorLogCollected(object? sender, LogCollectedEventArgs e)
        {
            try
            {
                // Aggregate logs for batch processing (ManageEngine pattern)
                lock (_logLock)
                {
                    _aggregatedLogs.AddRange(e.Logs);
                    
                    // Prevent memory overflow using configurable limits
                    var maxAggregatedLogs = _configuration.GetValue<int>("Processing:CollectorLimits:MaxAggregatedLogs", 10000);
                    var removalCount = _configuration.GetValue<int>("Processing:CollectorLimits:AggregatedLogsRemovalCount", 5000);
                    
                    if (_aggregatedLogs.Count > maxAggregatedLogs)
                    {
                        _aggregatedLogs.RemoveRange(0, removalCount);
                        _logger.LogDebug("Aggregated logs limit reached in OnCollectorLogCollected. Removed {RemovalCount} oldest logs. Max={MaxLogs}", 
                            removalCount, maxAggregatedLogs);
                    }
                }

                // Forward to subscribers
                LogsCollected?.Invoke(this, e);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error handling collected logs");
            }
        }

        /// <summary>
        /// Handle collector errors
        /// </summary>
        private void OnCollectorError(object? sender, LogCollectionErrorEventArgs e)
        {
            _logger.LogError(e.Exception, "Collector error from {Source}: {Message}", e.Source, e.Message);
            
            CollectorError?.Invoke(this, new CollectorErrorEventArgs
            {
                CollectorName = e.Source,
                Exception = e.Exception,
                Message = e.Message,
                ErrorTime = e.ErrorTime
            });
        }

        public async ValueTask DisposeAsync()
        {
            await StopAllCollectorsAsync();
            
            foreach (var collector in _collectors.Values)
            {
                await collector.DisposeAsync();
            }
            
            _cancellationTokenSource?.Dispose();
            _collectors.Clear();
            _collectorTasks.Clear();
        }
    }

    /// <summary>
    /// Health status for the entire collector manager
    /// </summary>
    public class CollectorManagerHealth
    {
        public bool IsHealthy { get; set; }
        public int TotalCollectors { get; set; }
        public int ActiveCollectors { get; set; }
        public long TotalLogsCollected { get; set; }
        public DateTime LastHealthCheck { get; set; }
        public Dictionary<string, CollectorHealth> CollectorHealthStatuses { get; set; } = new();
        public TimeSpan Uptime { get; set; }
    }

    /// <summary>
    /// Event arguments for collector manager errors
    /// </summary>
    public class CollectorErrorEventArgs : EventArgs
    {
        public string CollectorName { get; set; } = "";
        public Exception Exception { get; set; } = new();
        public string Message { get; set; } = "";
        public DateTime ErrorTime { get; set; } = DateTime.UtcNow;
    }
} 
