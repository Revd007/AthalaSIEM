using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using AthalaSIEM.Agent.Models;
using AthalaSIEM.Agent.Security;

namespace AthalaSIEM.Agent.Communication
{
    /// <summary>
    /// Processes log batches with adaptive sizing, compression, and retries
    /// </summary>
    public class LogBatchProcessor : ILogBatchProcessor, IDisposable
    {
        private readonly ILogger<LogBatchProcessor> _logger;
        private readonly ILogForwarder _logForwarder;
        private readonly IAgentIdentityService _identityService;
        private readonly IEncryptionService _encryptionService;
        private readonly AgentSettings _settings;
        
        private readonly ConcurrentQueue<NormalizedLogEntry> _logQueue = new ConcurrentQueue<NormalizedLogEntry>();
        private readonly ConcurrentDictionary<string, BatchStatistics> _batchStatistics = new ConcurrentDictionary<string, BatchStatistics>();
        private readonly CancellationTokenSource _cts = new CancellationTokenSource();
        private readonly SemaphoreSlim _processingSemaphore = new SemaphoreSlim(1, 1);
        private readonly Timer _flushTimer;
        
        private bool _isDisposed;
        private Task? _processingTask;
        private DateTime _lastFlushTime = DateTime.UtcNow;
        private int _totalLogsQueued;
        private int _totalLogsSent;
        private int _totalBatchesSent;
        private int _failedBatches;
        
        /// <summary>
        /// Initializes a new instance of the <see cref="LogBatchProcessor"/> class
        /// </summary>
        /// <param name="logger">The logger</param>
        /// <param name="logForwarder">The log forwarder</param>
        /// <param name="identityService">The agent identity service</param>
        /// <param name="encryptionService">The encryption service</param>
        /// <param name="settings">The agent settings</param>
        public LogBatchProcessor(
            ILogger<LogBatchProcessor> logger,
            ILogForwarder logForwarder,
            IAgentIdentityService identityService,
            IEncryptionService encryptionService,
            IOptions<AgentSettings> settings)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _logForwarder = logForwarder ?? throw new ArgumentNullException(nameof(logForwarder));
            _identityService = identityService ?? throw new ArgumentNullException(nameof(identityService));
            _encryptionService = encryptionService ?? throw new ArgumentNullException(nameof(encryptionService));
            _settings = settings?.Value ?? throw new ArgumentNullException(nameof(settings));
            
            // Initialize timer for periodic flushing
            var flushIntervalMs = Math.Min(_settings.MaxLogBatchIntervalSeconds * 1000, 60000); // Ensure max 60 seconds
            _flushTimer = new Timer(FlushCallback, null, flushIntervalMs, flushIntervalMs);
            
            // Start the processing task
            _processingTask = Task.Run(ProcessQueueAsync);
            
            _logger.LogInformation(
                "Log batch processor initialized. BatchSize: {BatchSize}, BufferSize: {BufferSize}, FlushInterval: {FlushInterval}s", 
                _settings.LogBatchSize, 
                _settings.MaxLogBufferSize, 
                _settings.MaxLogBatchIntervalSeconds);
        }
        
        /// <summary>
        /// Adds a log entry to the batch queue
        /// </summary>
        /// <param name="logEntry">The log entry to add</param>
        public void AddLog(NormalizedLogEntry logEntry)
        {
            if (logEntry == null)
                throw new ArgumentNullException(nameof(logEntry));
            
            if (_isDisposed)
                throw new ObjectDisposedException(nameof(LogBatchProcessor));
            
            // Add log to queue
            _logQueue.Enqueue(logEntry);
            Interlocked.Increment(ref _totalLogsQueued);
            
            // Check if queue is at capacity and trigger processing
            if (_logQueue.Count >= _settings.LogBatchSize)
            {
                Task.Run(async () => await TriggerProcessingAsync());
            }
        }
        
        /// <summary>
        /// Adds multiple log entries to the batch queue
        /// </summary>
        /// <param name="logEntries">The log entries to add</param>
        public void AddLogs(IEnumerable<NormalizedLogEntry> logEntries)
        {
            if (logEntries == null)
                throw new ArgumentNullException(nameof(logEntries));
            
            if (_isDisposed)
                throw new ObjectDisposedException(nameof(LogBatchProcessor));
            
            int added = 0;
            foreach (var logEntry in logEntries)
            {
                if (logEntry != null)
                {
                    _logQueue.Enqueue(logEntry);
                    added++;
                }
            }
            
            Interlocked.Add(ref _totalLogsQueued, added);
            
            if (_logQueue.Count >= _settings.LogBatchSize)
            {
                Task.Run(async () => await TriggerProcessingAsync());
            }
        }
        
        /// <summary>
        /// Flushes the log queue
        /// </summary>
        /// <returns>The number of logs sent</returns>
        public async Task<int> FlushAsync()
        {
            return await TriggerProcessingAsync(forceFlush: true);
        }
        
        /// <summary>
        /// Gets batch statistics
        /// </summary>
        /// <returns>Dictionary of batch statistics by source</returns>
        public IDictionary<string, BatchStatistics> GetBatchStatistics()
        {
            return new Dictionary<string, BatchStatistics>(_batchStatistics);
        }
        
        /// <summary>
        /// Gets the current queue status
        /// </summary>
        /// <returns>Queue status information</returns>
        public QueueStatus GetQueueStatus()
        {
            return new QueueStatus
            {
                QueuedLogs = _logQueue.Count,
                TotalLogsQueued = _totalLogsQueued,
                TotalLogsSent = _totalLogsSent,
                TotalBatchesSent = _totalBatchesSent,
                FailedBatches = _failedBatches,
                LastFlushTime = _lastFlushTime
            };
        }
        
        /// <summary>
        /// Resets batch statistics
        /// </summary>
        public void ResetStatistics()
        {
            _batchStatistics.Clear();
            _logger.LogInformation("Batch statistics reset");
        }
        
        /// <summary>
        /// Triggers processing of the log queue
        /// </summary>
        /// <param name="forceFlush">Whether to force flush regardless of batch size</param>
        /// <returns>The number of logs processed</returns>
        private async Task<int> TriggerProcessingAsync(bool forceFlush = false)
        {
            if (_isDisposed)
                return 0;
            
            // If we're not forcing a flush and the queue is empty, do nothing
            if (!forceFlush && _logQueue.Count == 0)
                return 0;
            
            // If processing is already in progress and we're not forcing a flush, do nothing
            if (!forceFlush && !await _processingSemaphore.WaitAsync(0))
                return 0;
            
            // If we're forcing a flush, wait for the semaphore
            if (forceFlush)
                await _processingSemaphore.WaitAsync();
            
            try
            {
                return await SendBatchAsync(forceFlush);
            }
            finally
            {
                _processingSemaphore.Release();
            }
        }
        
        /// <summary>
        /// Main queue processing loop
        /// </summary>
        private async Task ProcessQueueAsync()
        {
            while (!_cts.IsCancellationRequested)
            {
                try
                {
                    // Check if enough time has passed since the last flush
                    var timeSinceLastFlush = DateTime.UtcNow - _lastFlushTime;
                    if (timeSinceLastFlush.TotalSeconds >= _settings.MaxLogBatchIntervalSeconds && _logQueue.Count > 0)
                    {
                        await TriggerProcessingAsync(forceFlush: true);
                    }
                    
                    // Check if we have enough logs to send a batch
                    if (_logQueue.Count >= _settings.LogBatchSize)
                    {
                        await TriggerProcessingAsync();
                    }
                    
                    // Avoid CPU spinning
                    await Task.Delay(100);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error in log processing loop");
                    await Task.Delay(1000);
                }
            }
        }
        
        /// <summary>
        /// Sends a batch of logs
        /// </summary>
        /// <param name="forceFlush">Whether to force flush regardless of batch size</param>
        /// <returns>The number of logs sent</returns>
        private async Task<int> SendBatchAsync(bool forceFlush)
        {
            // If the queue is empty, nothing to do
            if (_logQueue.Count == 0)
                return 0;
            
            // Determine optimal batch size
            int batchSize = CalculateOptimalBatchSize(forceFlush);
            
            // Dequeue logs
            var logs = new List<NormalizedLogEntry>(batchSize);
            int dequeued = 0;
            
            while (dequeued < batchSize && _logQueue.TryDequeue(out var log))
            {
                logs.Add(log);
                dequeued++;
            }
            
            if (logs.Count == 0)
                return 0;
            
            // Try to send the batch
            try
            {
                string agentId = await _identityService.GetAgentIdAsync();
                
                // Create batch with metadata
                var batch = new LogBatch
                {
                    BatchId = Guid.NewGuid().ToString(),
                    AgentId = agentId,
                    CreatedAt = DateTime.UtcNow,
                    BatchSize = logs.Count,
                    Logs = logs
                };
                
                // Send the batch and get result
                bool success = await SendBatchWithRetriesAsync(batch);
                
                if (success)
                {
                    // Update statistics
                    Interlocked.Add(ref _totalLogsSent, logs.Count);
                    Interlocked.Increment(ref _totalBatchesSent);
                    _lastFlushTime = DateTime.UtcNow;
                    
                    // Update source statistics
                    UpdateBatchStatistics(logs);
                    
                    _logger.LogDebug("Sent batch of {Count} logs. BatchId: {BatchId}", logs.Count, batch.BatchId);
                    return logs.Count;
                }
                else
                {
                    // Failed to send even after retries
                    Interlocked.Increment(ref _failedBatches);
                    _logger.LogWarning("Failed to send batch of {Count} logs after retries. BatchId: {BatchId}", logs.Count, batch.BatchId);
                    
                    // Put the logs back in the queue if they're important enough (optimize)
                    if (ShouldRetainFailedLogs(logs))
                    {
                        foreach (var log in logs.Where(l => IsHighPriorityLog(l)))
                        {
                            _logQueue.Enqueue(log);
                        }
                        
                        _logger.LogInformation("Re-queued {Count} high-priority logs", logs.Count(l => IsHighPriorityLog(l)));
                    }
                    
                    return 0;
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error sending log batch of {Count} logs", logs.Count);
                Interlocked.Increment(ref _failedBatches);
                return 0;
            }
        }
        
        /// <summary>
        /// Sends a batch with retries
        /// </summary>
        /// <param name="batch">The batch to send</param>
        /// <returns>True if successful, false otherwise</returns>
        private async Task<bool> SendBatchWithRetriesAsync(LogBatch batch)
        {
            int retries = 0;
            bool success = false;
            
            while (!success && retries < _settings.MaxRetries)
            {
                try
                {
                    if (retries > 0)
                    {
                        _logger.LogDebug("Retrying batch send. Attempt {Retry}/{MaxRetries}. BatchId: {BatchId}", 
                            retries + 1, _settings.MaxRetries, batch.BatchId);
                    }
                    
                    await _logForwarder.ForwardLogBatchAsync(batch.Logs.ToArray());
                    success = true;
                }
                catch (Exception ex)
                {
                    retries++;
                    if (retries >= _settings.MaxRetries)
                    {
                        _logger.LogError(ex, "Failed to send batch after {Retries} retries. BatchId: {BatchId}", 
                            retries, batch.BatchId);
                        break;
                    }
                    
                    int delay = CalculateRetryDelay(retries);
                    _logger.LogWarning(ex, "Error sending batch. Retrying in {Delay}ms. Attempt {Retry}/{MaxRetries}. BatchId: {BatchId}", 
                        delay, retries, _settings.MaxRetries, batch.BatchId);
                    
                    await Task.Delay(delay);
                }
            }
            
            return success;
        }
        
        /// <summary>
        /// Calculates the optimal batch size
        /// </summary>
        /// <param name="forceFlush">Whether to force flush regardless of batch size</param>
        /// <returns>The optimal batch size</returns>
        private int CalculateOptimalBatchSize(bool forceFlush)
        {
            // If forced flush, take all logs but respect max buffer size
            if (forceFlush)
            {
                return Math.Min(_logQueue.Count, _settings.MaxLogBufferSize);
            }
            
            // Start with default batch size
            int batchSize = _settings.LogBatchSize;
            
            // If queue is getting large, increase batch size
            if (_logQueue.Count > _settings.LogBatchSize * 2)
            {
                double fillRatio = (double)_logQueue.Count / _settings.MaxLogBufferSize;
                batchSize = (int)(_settings.LogBatchSize * Math.Min(5, 1 + fillRatio * 4));
            }
            
            // Ensure we don't exceed the queue size or max buffer size
            return Math.Min(Math.Min(batchSize, _logQueue.Count), _settings.MaxLogBufferSize);
        }
        
        /// <summary>
        /// Calculates the retry delay using exponential backoff
        /// </summary>
        /// <param name="retryCount">The retry count</param>
        /// <returns>The delay in milliseconds</returns>
        private int CalculateRetryDelay(int retryCount)
        {
            // Exponential backoff with jitter
            var random = new Random();
            int maxDelay = Math.Min(60000, _settings.RetryDelaySeconds * 1000 * (int)Math.Pow(2, retryCount - 1));
            return random.Next(maxDelay / 2, maxDelay);
        }
        
        /// <summary>
        /// Updates batch statistics
        /// </summary>
        /// <param name="logs">The logs that were sent</param>
        private void UpdateBatchStatistics(IList<NormalizedLogEntry> logs)
        {
            // Group logs by source
            foreach (var group in logs.GroupBy(l => l.Source ?? "Unknown"))
            {
                string source = group.Key;
                int count = group.Count();
                
                _batchStatistics.AddOrUpdate(
                    source,
                    // Add new entry
                    _ => new BatchStatistics
                    {
                        Source = source,
                        TotalLogs = count,
                        BatchesSent = 1,
                        LastSent = DateTime.UtcNow
                    },
                    // Update existing entry
                    (_, stats) =>
                    {
                        stats.TotalLogs += count;
                        stats.BatchesSent++;
                        stats.LastSent = DateTime.UtcNow;
                        return stats;
                    });
            }
        }
        
        /// <summary>
        /// Checks if failed logs should be retained
        /// </summary>
        /// <param name="logs">The logs to check</param>
        /// <returns>True if the logs should be retained</returns>
        private bool ShouldRetainFailedLogs(IList<NormalizedLogEntry> logs)
        {
            // Decide based on:
            // 1. Are there high priority logs (Error/Critical)?
            // 2. Is the queue not already too full?
            bool hasHighPriorityLogs = logs.Any(IsHighPriorityLog);
            bool queueHasSpace = _logQueue.Count < _settings.MaxLogBufferSize * 0.9;
            
            return hasHighPriorityLogs && queueHasSpace;
        }
        
        /// <summary>
        /// Checks if a log is high priority
        /// </summary>
        /// <param name="log">The log to check</param>
        /// <returns>True if the log is high priority</returns>
        private bool IsHighPriorityLog(NormalizedLogEntry log)
        {
            return log.Severity?.Equals("Error", StringComparison.OrdinalIgnoreCase) == true ||
                   log.Severity?.Equals("Critical", StringComparison.OrdinalIgnoreCase) == true ||
                   log.Severity?.Equals("Alert", StringComparison.OrdinalIgnoreCase) == true ||
                   log.Severity?.Equals("Emergency", StringComparison.OrdinalIgnoreCase) == true;
        }
        
        /// <summary>
        /// Callback for the flush timer
        /// </summary>
        /// <param name="state">Timer state</param>
        private void FlushCallback(object? state)
        {
            if (_isDisposed)
                return;
            
            var timeSinceLastFlush = DateTime.UtcNow - _lastFlushTime;
            if (timeSinceLastFlush.TotalSeconds >= _settings.MaxLogBatchIntervalSeconds && _logQueue.Count > 0)
            {
                Task.Run(async () => await TriggerProcessingAsync(forceFlush: true));
            }
        }
        
        /// <summary>
        /// Disposes resources
        /// </summary>
        public void Dispose()
        {
            if (_isDisposed)
                return;
            
            _isDisposed = true;
            
            // Cancel and wait for processing task
            _cts.Cancel();
            _flushTimer.Dispose();
            
            // Final flush
            Task.Run(async () => await FlushAsync()).Wait(5000);
            
            // Clean up resources
            _processingTask?.Wait(1000);
            _processingSemaphore.Dispose();
            _cts.Dispose();
            
            _logger.LogInformation("Log batch processor disposed. Total logs queued: {TotalQueued}, sent: {TotalSent}, batches sent: {BatchesSent}, failed: {FailedBatches}",
                _totalLogsQueued, _totalLogsSent, _totalBatchesSent, _failedBatches);
        }
    }
    
    /// <summary>
    /// Interface for log batch processing
    /// </summary>
    public interface ILogBatchProcessor : IDisposable
    {
        /// <summary>
        /// Adds a log entry to the batch queue
        /// </summary>
        /// <param name="logEntry">The log entry to add</param>
        void AddLog(NormalizedLogEntry logEntry);
        
        /// <summary>
        /// Adds multiple log entries to the batch queue
        /// </summary>
        /// <param name="logEntries">The log entries to add</param>
        void AddLogs(IEnumerable<NormalizedLogEntry> logEntries);
        
        /// <summary>
        /// Flushes the log queue
        /// </summary>
        /// <returns>The number of logs sent</returns>
        Task<int> FlushAsync();
        
        /// <summary>
        /// Gets batch statistics
        /// </summary>
        /// <returns>Dictionary of batch statistics by source</returns>
        IDictionary<string, BatchStatistics> GetBatchStatistics();
        
        /// <summary>
        /// Gets the current queue status
        /// </summary>
        /// <returns>Queue status information</returns>
        QueueStatus GetQueueStatus();
        
        /// <summary>
        /// Resets batch statistics
        /// </summary>
        void ResetStatistics();
    }
    
    /// <summary>
    /// Statistics for a batch source
    /// </summary>
    public class BatchStatistics
    {
        /// <summary>
        /// Gets or sets the source name
        /// </summary>
        public string Source { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the total number of logs sent
        /// </summary>
        public int TotalLogs { get; set; }
        
        /// <summary>
        /// Gets or sets the total number of batches sent
        /// </summary>
        public int BatchesSent { get; set; }
        
        /// <summary>
        /// Gets or sets the time of the last batch sent
        /// </summary>
        public DateTime LastSent { get; set; }
    }
    
    /// <summary>
    /// Status of the log queue
    /// </summary>
    public class QueueStatus
    {
        /// <summary>
        /// Gets or sets the number of logs currently in the queue
        /// </summary>
        public int QueuedLogs { get; set; }
        
        /// <summary>
        /// Gets or sets the total number of logs that have been queued
        /// </summary>
        public int TotalLogsQueued { get; set; }
        
        /// <summary>
        /// Gets or sets the total number of logs that have been sent
        /// </summary>
        public int TotalLogsSent { get; set; }
        
        /// <summary>
        /// Gets or sets the total number of batches that have been sent
        /// </summary>
        public int TotalBatchesSent { get; set; }
        
        /// <summary>
        /// Gets or sets the number of batches that have failed to send
        /// </summary>
        public int FailedBatches { get; set; }
        
        /// <summary>
        /// Gets or sets the time of the last flush
        /// </summary>
        public DateTime LastFlushTime { get; set; }
    }
    
    /// <summary>
    /// Represents a batch of logs
    /// </summary>
    public class LogBatch
    {
        /// <summary>
        /// Gets or sets the batch ID
        /// </summary>
        public string BatchId { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the agent ID
        /// </summary>
        public string AgentId { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the time the batch was created
        /// </summary>
        public DateTime CreatedAt { get; set; }
        
        /// <summary>
        /// Gets or sets the number of logs in the batch
        /// </summary>
        public int BatchSize { get; set; }
        
        /// <summary>
        /// Gets or sets the logs in the batch
        /// </summary>
        public IList<NormalizedLogEntry> Logs { get; set; } = new List<NormalizedLogEntry>();
    }
} 