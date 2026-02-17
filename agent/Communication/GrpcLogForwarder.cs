using AthalaSIEM.Agent.Models;
using AthalaSIEM.Agent.Security;
using Google.Protobuf;
using Grpc.Core;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

namespace AthalaSIEM.Agent.Communication
{
    /// <summary>
    /// Handles log forwarding to the backend using gRPC
    /// </summary>
    public class GrpcLogForwarder : ILogForwarder, IDisposable
    {
        private readonly ILogger<GrpcLogForwarder> _logger;
        private readonly SiemService.SiemServiceClient _client;
        private readonly IEncryptionService _encryptionService;
        private readonly IAgentIdentityService _identityService;
        private readonly AgentSettings _settings;
        
        private readonly ConcurrentQueue<NormalizedLogEntry> _logBuffer = new ConcurrentQueue<NormalizedLogEntry>();
        private readonly CancellationTokenSource _cts = new CancellationTokenSource();
        private readonly SemaphoreSlim _sendingSemaphore = new SemaphoreSlim(1, 1);
        
        private bool _isDisposed;
        private Task? _batchingTask;
        private int _logsCollected;
        private int _logsForwarded;
        private int _logsPending;
        
        /// <summary>
        /// Initializes a new instance of the <see cref="GrpcLogForwarder"/> class
        /// </summary>
        /// <param name="logger">Logger instance</param>
        /// <param name="client">gRPC client</param>
        /// <param name="encryptionService">Encryption service</param>
        /// <param name="identityService">Agent identity service</param>
        /// <param name="settings">Agent settings</param>
        public GrpcLogForwarder(
            ILogger<GrpcLogForwarder> logger,
            SiemService.SiemServiceClient client,
            IEncryptionService encryptionService,
            IAgentIdentityService identityService,
            IOptions<AgentSettings> settings)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _client = client ?? throw new ArgumentNullException(nameof(client));
            _encryptionService = encryptionService ?? throw new ArgumentNullException(nameof(encryptionService));
            _identityService = identityService ?? throw new ArgumentNullException(nameof(identityService));
            _settings = settings?.Value ?? throw new ArgumentNullException(nameof(settings));
            
            // Configure TLS and auth
            ConfigureSecureChannel();
            
            // Start the batching task
            _batchingTask = Task.Run(ProcessLogBatchesAsync);
        }
        
        /// <summary>
        /// Logs gRPC channel configuration (insecure or TLS). Channel is created by DI with ChannelCredentials.Insecure when UseInsecureGrpcChannel is true.
        /// </summary>
        private void ConfigureSecureChannel()
        {
            try
            {
                if (_settings.UseInsecureGrpcChannel)
                {
                    _logger.LogInformation("gRPC using insecure channel (no TLS). Backend: {BackendGrpcUrl}", _settings.BackendGrpcUrl ?? _settings.BackendApiUrl);
                    return;
                }

                if (_settings.UseMutualTls)
                {
                    _logger.LogInformation("Using mutual TLS for secure communication");
                    if (!File.Exists(_settings.ClientCertificatePath))
                        _logger.LogWarning("Client certificate not found at {Path}", _settings.ClientCertificatePath);
                    if (!File.Exists(_settings.ServerCaCertificatePath))
                        _logger.LogWarning("Server CA certificate not found at {Path}", _settings.ServerCaCertificatePath);
                }
                else
                {
                    _logger.LogWarning("Mutual TLS is disabled. Communication security may be reduced.");
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error configuring gRPC channel");
            }
        }
        
        /// <summary>
        /// Forwards a normalized log entry to the backend
        /// </summary>
        /// <param name="logEntry">The normalized log entry to forward</param>
        /// <returns>A task representing the asynchronous operation</returns>
        public Task ForwardLogAsync(NormalizedLogEntry logEntry)
        {
            if (logEntry == null)
                throw new ArgumentNullException(nameof(logEntry));
                
            // Add to buffer for batching
            _logBuffer.Enqueue(logEntry);
            Interlocked.Increment(ref _logsCollected);
            Interlocked.Increment(ref _logsPending);
            
            // If buffer exceeds batch size, trigger sending
            if (_logBuffer.Count >= _settings.LogBatchSize && _sendingSemaphore.CurrentCount > 0)
            {
                _ = Task.Run(async () =>
                {
                    await SendLogBatchAsync();
                });
            }
            
            return Task.CompletedTask;
        }
        
        /// <summary>
        /// Forwards a batch of normalized log entries to the backend
        /// </summary>
        /// <param name="logEntries">The batch of log entries to forward</param>
        /// <returns>A task representing the asynchronous operation</returns>
        public async Task ForwardLogBatchAsync(NormalizedLogEntry[] logEntries)
        {
            if (logEntries == null)
                throw new ArgumentNullException(nameof(logEntries));
                
            if (logEntries.Length == 0)
                return;
                
            // Add all entries to the buffer
            foreach (var entry in logEntries)
            {
                _logBuffer.Enqueue(entry);
                Interlocked.Increment(ref _logsCollected);
                Interlocked.Increment(ref _logsPending);
            }
            
            // If we've received a full batch, send immediately
            if (logEntries.Length >= _settings.LogBatchSize)
            {
                await SendLogBatchAsync();
            }
        }
        
        /// <summary>
        /// Sends a heartbeat to the backend
        /// </summary>
        /// <param name="heartbeatData">The heartbeat data to send</param>
        /// <returns>A task representing the asynchronous operation</returns>
        public async Task SendHeartbeatAsync(AgentHeartbeat heartbeatData)
        {
            if (heartbeatData == null)
                throw new ArgumentNullException(nameof(heartbeatData));
                
            try
            {
                // Ensure we have the agent ID
                if (string.IsNullOrEmpty(heartbeatData.AgentId))
                {
                    heartbeatData.AgentId = await _identityService.GetAgentIdAsync();
                }
                
                // Update stats
                heartbeatData.LogsCollected = _logsCollected;
                heartbeatData.LogsForwarded = _logsForwarded;
                heartbeatData.LogsPending = _logsPending;
                
                // Reset counters after sending
                Interlocked.Exchange(ref _logsCollected, 0);
                
                // Convert to gRPC request
                var request = new HeartbeatRequest
                {
                    AgentId = heartbeatData.AgentId.ToString(),
                    Status = heartbeatData.Status,
                    Timestamp = heartbeatData.Timestamp.ToString("o"), // ISO 8601 format
                    UptimeHours = heartbeatData.Uptime / 3600.0, // Convert seconds to hours
                    CpuUsage = heartbeatData.CpuUsage ?? 0.0,
                    MemoryUsage = heartbeatData.MemoryUsage ?? 0.0,
                    ActiveCollectors = (int)heartbeatData.LogsCollected,
                    LogsCollected = (int)heartbeatData.LogsCollected,
                    LogsForwarded = (int)heartbeatData.LogsForwarded
                };
                
                // Add API key if available
                if (!string.IsNullOrEmpty(await _identityService.GetApiKeyAsync()))
                {
                    request.ApiKey = await _identityService.GetApiKeyAsync();
                }
                
                // Add additional details - check if the property exists first
                var detailsType = request.GetType();
                var detailsProperty = detailsType.GetProperty("AdditionalDetails");
                
                if (detailsProperty != null)
                {
                    // If the property exists, use reflection to get its value and add items
                    var detailsDict = detailsProperty.GetValue(request) as Google.Protobuf.Collections.MapField<string, string>;
                    if (detailsDict != null)
                    {
                        foreach (var detail in heartbeatData.AdditionalDetails)
                        {
                            detailsDict[detail.Key] = detail.Value;
                        }
                    }
                }
                
                // Send the heartbeat
                var response = await _client.SendHeartbeatAsync(request);
                
                // If configuration has changed, refresh it
                if (response.ConfigurationChanged)
                {
                    _logger.LogInformation("Backend indicated configuration has changed, will refresh");
                    _ = Task.Run(async () => await GetAgentConfigurationAsync());
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error sending heartbeat");
                // We don't rethrow as heartbeat failures shouldn't stop the agent
            }
        }
        
        /// <summary>
        /// Retrieves the latest agent configuration from the backend
        /// </summary>
        /// <returns>The updated agent configuration</returns>
        public async Task<AgentSettings> GetAgentConfigurationAsync()
        {
            try
            {
                string agentId = await _identityService.GetAgentIdAsync();
                
                var request = new GetAgentConfigurationRequest
                {
                    AgentId = agentId.ToString()
                };
                
                var response = await _client.GetAgentConfigurationAsync(request);
                
                if (response.Success && response.ConfigurationChanged)
                {
                    _logger.LogInformation("Retrieved updated configuration from server");
                    
                    // Parse configuration from JSON
                    if (!string.IsNullOrEmpty(response.ConfigurationJson))
                    {
                        try
                        {
                            var newSettings = JsonSerializer.Deserialize<AgentSettings>(response.ConfigurationJson, new JsonSerializerOptions
                            {
                                PropertyNameCaseInsensitive = true
                            });
                            
                            if (newSettings != null)
                            {
                                return newSettings;
                            }
                        }
                        catch (Exception ex)
                        {
                            _logger.LogError(ex, "Error deserializing configuration JSON");
                        }
                    }
                }
                
                _logger.LogInformation("No configuration changes from server");
                return _settings; // Return current settings
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error retrieving agent configuration");
                return _settings; // Return current settings on error
            }
        }
        
        /// <summary>
        /// Sends system metrics to the backend
        /// </summary>
        /// <param name="metrics">The system metrics to send</param>
        /// <returns>A task representing the asynchronous operation</returns>
        public async Task SendSystemMetricsAsync(SystemMetrics metrics)
        {
            try
            {
                // Add API key to headers
                var metadata = await GetAuthorizationMetadataAsync();
                
                // Convert to gRPC request
                var request = new SystemMetricsRequest();
                
                // Create and add CPU metrics
                var cpuData = new CpuMetricsData
                {
                    Usage = metrics.Cpu?.Usage ?? 0.0,
                    NumberOfCores = metrics.Cpu?.NumberOfCores ?? 0
                };
                request.Cpu = cpuData;
                
                // Create and add Memory metrics
                var memoryData = new MemoryMetricsData
                {
                    UsedPercentage = metrics.Memory?.UsedPercentage ?? 0.0,
                    AvailableBytes = metrics.Memory?.AvailableBytes ?? 0,
                    ProcessUsedBytes = metrics.Memory?.ProcessUsedBytes ?? 0,
                    ProcessPrivateBytes = metrics.Memory?.ProcessPrivateBytes ?? 0
                };
                request.Memory = memoryData;
                
                // Create and add Network metrics
                var networkData = new NetworkMetricsData
                {
                    TotalBytesSent = metrics.Network?.TotalBytesSent ?? 0,
                    TotalBytesReceived = metrics.Network?.TotalBytesReceived ?? 0
                };
                
                // Add network interfaces
                if (metrics.Network?.Interfaces != null)
                {
                    foreach (var intf in metrics.Network.Interfaces)
                    {
                        var interfaceData = new NetworkInterfaceMeasurement
                        {
                            InterfaceName = intf.Name,
                            BytesReceived = intf.BytesReceived,
                            BytesSent = intf.BytesSent
                        };
                        networkData.Interfaces.Add(interfaceData);
                    }
                }
                request.Network = networkData;
                
                // Add disk metrics
                if (metrics.Disk?.Drives != null)
                {
                    foreach (var drive in metrics.Disk.Drives)
                    {
                        var diskData = new DiskMetricsData
                        {
                            DriveName = drive.Name,
                            AvailableBytes = drive.AvailableBytes,
                            UsedPercentage = drive.UsedPercentage
                        };
                        request.Disks.Add(diskData);
                    }
                }
                
                // Add process metrics
                if (metrics.Process?.MemoryUsageProcesses != null)
                {
                    foreach (var process in metrics.Process.MemoryUsageProcesses)
                    {
                        var processData = new ProcessMetricsData
                        {
                            ProcessName = process.Name,
                            MemoryUsage = process.MemoryUsageBytes
                        };
                        request.Processes.Add(processData);
                    }
                }
                
                // Send the metrics
                var response = await _client.SendSystemMetricsAsync(request, metadata);
                
                if (response.Success)
                {
                    _logger.LogDebug("System metrics sent successfully");
                }
                else
                {
                    _logger.LogWarning("Failed to send system metrics: {Message}", response.Message);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error sending system metrics");
                throw;
            }
        }
        
        /// <summary>
        /// Sends a health report to the backend
        /// </summary>
        /// <param name="healthReport">The health report to send</param>
        /// <returns>A task representing the asynchronous operation</returns>
        public async Task SendHealthReportAsync(AgentHealthReport healthReport)
        {
            try
            {
                // Add API key to headers
                var metadata = await GetAuthorizationMetadataAsync();
                
                // Get agent ID
                string agentId = await _identityService.GetAgentIdAsync();
                
                // Create health report request
                var request = new HealthReportRequest
                {
                    AgentId = agentId,
                    ApiKey = await _identityService.GetApiKeyAsync(),
                    Timestamp = healthReport.Timestamp.ToString("o"),  // ISO 8601 format
                    OverallStatus = healthReport.Status,
                    UptimeHours = healthReport.Uptime / 3600.0  // Convert seconds to hours
                };
                
                // Add component statuses
                foreach (var componentStatus in healthReport.Components)
                {
                    request.ComponentStatuses.Add(componentStatus.Name, new ComponentStatus
                    {
                        Status = componentStatus.Status,
                        Message = componentStatus.Message
                    });
                }
                
                // Add diagnostics
                foreach (var diag in healthReport.Diagnostics)
                {
                    request.Diagnostics.Add($"{diag.Key}: {diag.Value}");
                }
                
                // Send the health report
                var response = await _client.SendHealthReportAsync(request, metadata);
                
                if (response.Success)
                {
                    _logger.LogDebug("Health report sent successfully");
                }
                else
                {
                    _logger.LogWarning("Failed to send health report: {Message}", response.Message);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error sending health report");
                throw;
            }
        }
        
        /// <summary>
        /// Processes batches of logs in the background
        /// </summary>
        private async Task ProcessLogBatchesAsync()
        {
            while (!_cts.Token.IsCancellationRequested)
            {
                try
                {
                    // Wait for the max batch interval or until cancelled
                    await Task.Delay(TimeSpan.FromSeconds(_settings.MaxLogBatchIntervalSeconds), _cts.Token);
                    
                    // Send any pending logs
                    if (_logBuffer.Count > 0)
                    {
                        await SendLogBatchAsync();
                    }
                }
                catch (TaskCanceledException)
                {
                    // Expected when cancellation is requested
                    break;
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error in log batch processing");
                    
                    // Brief pause to avoid tight loop on errors
                    await Task.Delay(1000);
                }
            }
        }
        
        /// <summary>
        /// Sends a batch of logs to the backend
        /// </summary>
        private async Task SendLogBatchAsync()
        {
            // Use semaphore to ensure only one send operation at a time
            if (!await _sendingSemaphore.WaitAsync(0))
                return;
                
            try
            {
                // Dequeue up to batch size logs
                var batch = new List<NormalizedLogEntry>();
                int batchSize = Math.Min(_logBuffer.Count, _settings.LogBatchSize);
                
                for (int i = 0; i < batchSize; i++)
                {
                    if (_logBuffer.TryDequeue(out var log))
                    {
                        batch.Add(log);
                        Interlocked.Decrement(ref _logsPending);
                    }
                    else
                    {
                        break;
                    }
                }
                
                if (batch.Count == 0)
                    return;
                    
                _logger.LogDebug("Sending batch of {Count} logs", batch.Count);
                
                // Create batch with agent ID
                string agentId = await _identityService.GetAgentIdAsync();
                var logBatch = new LogBatch
                {
                    BatchId = Guid.NewGuid().ToString(),
                    AgentId = agentId.ToString(),
                    CreatedAt = DateTime.UtcNow,
                    BatchSize = batch.Count,
                    Logs = batch
                };
                
                // Convert to gRPC format
                var request = new LogBatchRequest
                {
                    AgentId = logBatch.AgentId,
                    ApiKey = await _identityService.GetApiKeyAsync(),
                    Encrypted = _settings.EncryptLogs,
                    Compressed = _settings.UseCompression
                };
                
                // Determine if we should compress/encrypt
                if (_settings.UseCompression || _settings.EncryptLogs)
                {
                    // Serialize logs to JSON
                    var json = JsonSerializer.Serialize(batch);
                    byte[] data = System.Text.Encoding.UTF8.GetBytes(json);
                    
                    // Compress if enabled
                    if (_settings.UseCompression)
                    {
                        data = CompressData(data);
                    }
                    
                    // Encrypt if enabled
                    if (_settings.EncryptLogs)
                    {
                        byte[] key = await GetEncryptionKeyAsync();
                        data = _encryptionService.Encrypt(data, key);
                    }
                    
                    // Convert each log entry to gRPC format
                    foreach (var log in batch)
                    {
                        var logEntry = ConvertToLogEntry(log);
                        request.Logs.Add(logEntry);
                    }
                }
                else
                {
                    // Add logs directly if no compression/encryption
                    foreach (var log in batch)
                    {
                        var logEntry = ConvertToLogEntry(log);
                        request.Logs.Add(logEntry);
                    }
                }
                
                // Send the batch
                var response = await _client.ForwardLogsAsync(request);
                
                // Update forwarded count
                int accepted = response.AcceptedCount;
                Interlocked.Add(ref _logsForwarded, accepted);
                
                if (response.RejectedCount > 0)
                {
                    _logger.LogWarning("Server rejected {Count} logs from batch", response.RejectedCount);
                }
                
                _logger.LogDebug("Successfully sent {Count} logs", accepted);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error sending log batch");
                if (ex is RpcException rpcEx && rpcEx.Status.Detail != null &&
                    rpcEx.Status.Detail.Contains("HTTP_1_1_REQUIRED", StringComparison.OrdinalIgnoreCase))
                    _logger.LogWarning(
                        "Backend or proxy is rejecting HTTP/2. gRPC requires HTTP/2: ensure the backend URL supports HTTP/2 and no proxy is forcing HTTP/1.1.");
            }
            finally
            {
                _sendingSemaphore.Release();
            }
        }
        
        /// <summary>
        /// Gets the encryption key from the backend
        /// </summary>
        private async Task<byte[]> GetEncryptionKeyAsync()
        {
            // In a real implementation, you'd get this from secure storage or the backend
            // This is a simplified approach for demonstration
            string apiKey = await _identityService.GetApiKeyAsync();
            return System.Text.Encoding.UTF8.GetBytes(apiKey);
        }
        
        /// <summary>
        /// Compresses data using GZip
        /// </summary>
        private byte[] CompressData(byte[] data)
        {
            using var memoryStream = new MemoryStream();
            using (var gzipStream = new GZipStream(memoryStream, System.IO.Compression.CompressionLevel.Fastest))
            {
                gzipStream.Write(data, 0, data.Length);
            }
            return memoryStream.ToArray();
        }
        
        /// <summary>
        /// Disposes resources
        /// </summary>
        public void Dispose()
        {
            if (_isDisposed)
                return;
                
            _cts.Cancel();
            _cts.Dispose();
            _sendingSemaphore.Dispose();
            
            // Wait for batching task to complete
            if (_batchingTask != null)
            {
                try
                {
                    _batchingTask.Wait(TimeSpan.FromSeconds(5));
                }
                catch
                {
                    // Ignore exceptions during disposal
                }
            }
            
            _isDisposed = true;
        }

        private async Task<LogBatchResponse> ForwardLogBatchInternalAsync(LogBatch batch)
        {
            // Add API key to headers
            var metadata = await GetAuthorizationMetadataAsync();
            
            // Create log batch request
            var request = new LogBatchRequest
            {
                AgentId = batch.AgentId,
                ApiKey = await _identityService.GetApiKeyAsync(),
                Encrypted = false,  // Set based on settings, not from batch
                Compressed = false  // Set based on settings, not from batch
            };
            
            if (batch.Logs != null && batch.Logs.Count > 0)
            {
                foreach (var log in batch.Logs)
                {
                    var logEntry = new LogEntry
                    {
                        Id = log.Id,
                        Timestamp = log.Timestamp.ToString("o"),  // ISO 8601 format
                        Source = log.Source,
                        SourceType = log.SourceType,
                        LogLevel = log.Severity,
                        Message = log.Message,
                        Hash = log.Hash
                    };
                    
                    // Add metadata
                    foreach (var meta in log.AdditionalFields)
                    {
                        logEntry.Metadata.Add(meta.Key, meta.Value);
                    }
                    
                    request.Logs.Add(logEntry);
                }
            }
            
            try
            {
                // Send the batch
                var response = await _client.ForwardLogsAsync(request, metadata);
                return response;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error forwarding log batch");
                return new LogBatchResponse
                {
                    Success = false,
                    Message = $"Error: {ex.Message}"
                };
            }
        }

        /// <summary>
        /// Gets the authorization metadata for gRPC calls
        /// </summary>
        /// <returns>A metadata object with the API key</returns>
        private async Task<Grpc.Core.Metadata> GetAuthorizationMetadataAsync()
        {
            var metadata = new Grpc.Core.Metadata();
            string apiKey = await _identityService.GetApiKeyAsync();
            
            if (!string.IsNullOrEmpty(apiKey))
            {
                metadata.Add("Authorization", $"Bearer {apiKey}");
            }
            
            return metadata;
        }

        private LogEntry ConvertToLogEntry(NormalizedLogEntry log)
        {
            var logEntry = new LogEntry
            {
                Id = log.Id,
                Timestamp = log.Timestamp.ToString("o"),  // ISO 8601 format
                Source = log.Source,
                SourceType = log.SourceType,
                LogLevel = log.Severity,
                Message = log.Message,
                Hash = log.Hash
            };
            
            // Add metadata
            foreach (var meta in log.AdditionalFields)
            {
                logEntry.Metadata.Add(meta.Key, meta.Value);
            }
            
            return logEntry;
        }
    }
} 