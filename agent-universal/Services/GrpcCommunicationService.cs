using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Configuration;
using AthalaSIEM.UniversalAgent.Models;
using AthalaSIEM.UniversalAgent.Services.Interfaces;
using AthalaSIEM.Agent;
using Grpc.Net.Client;
using Grpc.Core;
using LocalLogEntry = AthalaSIEM.UniversalAgent.Models.LogEntry;
using GrpcLogEntry = AthalaSIEM.Agent.LogEntry;

namespace AthalaSIEM.UniversalAgent.Services
{
    /// <summary>
    /// Production-grade gRPC communication service for AthalaSIEM backend
    /// Provides high-performance streaming communication using Protocol Buffers
    /// Supports graceful fallback to HTTP if gRPC is unavailable
    /// </summary>
    public class GrpcCommunicationService : IAsyncDisposable, IBackendCommunicationService
    {
        private readonly ILogger<GrpcCommunicationService> _logger;
        private readonly IConfiguration _configuration;
        private readonly Timer _heartbeatTimer;
        private readonly Timer _batchTimer;
        private readonly Queue<LocalLogEntry> _logQueue = new();
        private readonly object _queueLock = new();
        private readonly SemaphoreSlim _sendSemaphore = new(1, 1);
        private readonly CancellationTokenSource _cancellationTokenSource = new();

        private string _serverUrl = "";
        private string _grpcUrl = "";
        private string _agentId = "";
        private string _apiKey = "";
        private int _batchSize;
        private int _batchIntervalSeconds;
        private int _heartbeatIntervalSeconds;
        private bool _isConnected;
        private DateTime _lastSuccessfulSend;
        private GrpcChannel? _channel;
        private SiemService.SiemServiceClient? _client;
        private AsyncDuplexStreamingCall<HeartbeatRequest, HeartbeatResponse>? _heartbeatStream;
        private AsyncClientStreamingCall<GrpcLogEntry, LogBatchResponse>? _logStream;
        private Task? _heartbeatStreamTask;
        private Task? _logStreamTask;

        public bool IsConnected => _isConnected && _channel?.State == ConnectivityState.Ready;
        public long QueuedLogs => _logQueue.Count;
        public DateTime LastSuccessfulSend => _lastSuccessfulSend;
        public long TotalLogsSent { get; private set; }
        public long TotalSendErrors { get; private set; }

        public event EventHandler<LogsSentEventArgs>? LogsSent;
        public event EventHandler<CommunicationErrorEventArgs>? CommunicationError;
        public event EventHandler<ConnectionStatusChangedEventArgs>? ConnectionStatusChanged;

        public GrpcCommunicationService(
            ILogger<GrpcCommunicationService> logger,
            IConfiguration configuration)
        {
            _logger = logger;
            _configuration = configuration;

            LoadConfiguration();

            // Setup timers
            _heartbeatIntervalSeconds = _configuration.GetValue<int>("Agent:HeartbeatIntervalSeconds", 30);
            _heartbeatTimer = new Timer(SendHeartbeat, null, TimeSpan.FromSeconds(_heartbeatIntervalSeconds), TimeSpan.FromSeconds(_heartbeatIntervalSeconds));
            _batchTimer = new Timer(ProcessLogBatch, null, 
                TimeSpan.FromSeconds(_batchIntervalSeconds), 
                TimeSpan.FromSeconds(_batchIntervalSeconds));
        }

        public async Task<bool> InitializeAsync()
        {
            try
            {
                _logger.LogInformation("Initializing gRPC communication to: {GrpcUrl}", _grpcUrl);
                
                // Create gRPC channel with proper configuration
                var channelOptions = new GrpcChannelOptions
                {
                    MaxReceiveMessageSize = 100 * 1024 * 1024, // 100MB
                    MaxSendMessageSize = 100 * 1024 * 1024,    // 100MB
                    HttpHandler = new System.Net.Http.SocketsHttpHandler
                    {
                        KeepAlivePingDelay = TimeSpan.FromSeconds(60),
                        KeepAlivePingTimeout = TimeSpan.FromSeconds(30),
                        PooledConnectionIdleTimeout = TimeSpan.FromMinutes(5),
                        EnableMultipleHttp2Connections = true
                    }
                };

                _channel = GrpcChannel.ForAddress(_grpcUrl, channelOptions);
                _client = new SiemService.SiemServiceClient(_channel);

                // Test connection
                if (!await TestConnectionAsync())
                {
                    _logger.LogWarning("gRPC connection test failed, will retry");
                    return false;
                }

                // Initialize streaming connections
                await InitializeStreamsAsync();

                _isConnected = true;
                _lastSuccessfulSend = DateTime.UtcNow;
                
                ConnectionStatusChanged?.Invoke(this, new ConnectionStatusChangedEventArgs
                {
                    IsConnected = true,
                    StatusMessage = "Connected to SIEM backend via gRPC streaming"
                });

                _logger.LogInformation("gRPC communication service initialized successfully");
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to initialize gRPC communication service");
                _isConnected = false;
                
                CommunicationError?.Invoke(this, new CommunicationErrorEventArgs
                {
                    ErrorMessage = $"gRPC initialization failed: {ex.Message}",
                    ErrorTime = DateTime.UtcNow
                });
                
                return false;
            }
        }

        private Task InitializeStreamsAsync()
        {
            try
            {
                if (_client == null) throw new InvalidOperationException("gRPC client not initialized");

                // Initialize heartbeat streaming
                _heartbeatStream = _client.StreamHeartbeat(GetCallOptions());
                _heartbeatStreamTask = Task.Run(async () => await ProcessHeartbeatStreamAsync(_cancellationTokenSource.Token));

                // Initialize log streaming
                _logStream = _client.StreamLogs(GetCallOptions());
                _logStreamTask = Task.Run(async () => await ProcessLogStreamAsync(_cancellationTokenSource.Token));

                _logger.LogInformation("gRPC streams initialized");
                return Task.CompletedTask;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to initialize gRPC streams");
                return Task.FromException(ex);
            }
        }

        private CallOptions GetCallOptions()
        {
            var headers = new Metadata();
            if (!string.IsNullOrEmpty(_apiKey))
            {
                headers.Add("x-api-key", _apiKey);
            }
            headers.Add("x-agent-id", _agentId);
            return new CallOptions(headers: headers);
        }

        private async Task ProcessHeartbeatStreamAsync(CancellationToken cancellationToken)
        {
            try
            {
                await foreach (var response in _heartbeatStream!.ResponseStream.ReadAllAsync(cancellationToken))
                {
                    if (response.ConfigurationChanged)
                    {
                        _logger.LogInformation("Configuration changed, version: {Version}", response.ConfigVersion);
                        // Trigger configuration refresh
                        ConfigurationUpdated?.Invoke(this, new BackendConfigurationUpdatedEventArgs());
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error processing heartbeat stream");
                _isConnected = false;
            }
        }

        private async Task ProcessLogStreamAsync(CancellationToken cancellationToken)
        {
            try
            {
                // Wait for log stream response
                var response = await _logStream!.ResponseAsync;
                _logger.LogInformation("Log stream established, accepted: {Accepted}, rejected: {Rejected}", 
                    response.AcceptedCount, response.RejectedCount);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error processing log stream");
            }
        }

        public void QueueLog(LocalLogEntry log)
        {
            if (log == null) return;

            lock (_queueLock)
            {
                _logQueue.Enqueue(log);
                
                var maxQueueSize = _configuration.GetValue<int>("GrpcCommunication:MaxQueueSize", 10000);
                if (_logQueue.Count > maxQueueSize)
                {
                    _logQueue.Dequeue();
                    _logger.LogWarning("Log queue overflow, oldest log discarded");
                }
            }
        }

        public void QueueLogs(IEnumerable<LocalLogEntry> logs)
        {
            foreach (var log in logs)
            {
                QueueLog(log);
            }
        }

        public async Task<bool> FlushLogsAsync()
        {
            return await ProcessLogBatch(true);
        }

        public async Task<bool> TestConnectionAsync()
        {
            try
            {
                if (_client == null)
                {
                    _logger.LogWarning("gRPC client not initialized");
                    return false;
                }

                // Test with a simple heartbeat
                var testRequest = new AthalaSIEM.Agent.HeartbeatRequest
                {
                    AgentId = _agentId,
                    ApiKey = _apiKey,
                    Timestamp = DateTime.UtcNow.ToString("O"),
                    Status = "Healthy",
                    CpuUsage = 0,
                    MemoryUsage = 0
                };

                var response = await _client.SendHeartbeatAsync(testRequest, GetCallOptions());
                
                _logger.LogInformation("gRPC connection test successful");
                return response.Success;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "gRPC connection test failed");
                return false;
            }
        }

        public CommunicationHealth GetHealthStatus()
        {
            return new CommunicationHealth
            {
                IsConnected = IsConnected,
                ManagerUrl = _grpcUrl,
                QueuedLogs = QueuedLogs,
                TotalLogsSent = TotalLogsSent,
                TotalSendErrors = TotalSendErrors,
                LastSuccessfulSend = _lastSuccessfulSend,
                LastHealthCheck = DateTime.UtcNow
            };
        }

        private void LoadConfiguration()
        {
            var managerIP = _configuration["SiemManager:ManagerIP"];
            if (string.IsNullOrEmpty(managerIP))
            {
                _logger.LogError("SiemManager:ManagerIP is REQUIRED and not configured!");
                throw new InvalidOperationException("SiemManager:ManagerIP configuration is required.");
            }
            
            var managerPort = _configuration.GetValue<int>("SiemManager:ManagerPort");
            if (managerPort == 0)
            {
                _logger.LogError("SiemManager:ManagerPort is REQUIRED and not configured!");
                throw new InvalidOperationException("SiemManager:ManagerPort configuration is required.");
            }

            var useHTTPS = _configuration.GetValue<bool>("SiemManager:UseHTTPS", false);
            var protocol = useHTTPS ? "https" : "http";
            
            _serverUrl = $"{protocol}://{managerIP}:{managerPort}";
            
            // gRPC uses same host but typically same port (HTTP/2) or dedicated port
            var grpcPort = _configuration.GetValue<int>("SiemManager:GrpcPort", managerPort);
            _grpcUrl = $"{protocol}://{managerIP}:{grpcPort}";
            
            _agentId = _configuration["Agent:Id"] ?? Environment.MachineName;
            _apiKey = _configuration["Agent:ApiKey"] ?? "";
            _batchSize = _configuration.GetValue<int>("Agent:BatchSize", 100);
            _batchIntervalSeconds = _configuration.GetValue<int>("Agent:BatchIntervalSeconds", 30);
            
            _logger.LogInformation("gRPC Configuration - Server: {ServerUrl}, gRPC: {GrpcUrl}, Batch: {BatchSize}, Interval: {Interval}s", 
                _serverUrl, _grpcUrl, _batchSize, _batchIntervalSeconds);
        }

        private async void SendHeartbeat(object? state)
        {
            try
            {
                if (_client == null || !_isConnected) return;

                var metrics = GetSystemMetrics();
                var heartbeat = new AthalaSIEM.Agent.HeartbeatRequest
                {
                    AgentId = _agentId,
                    ApiKey = _apiKey,
                    Timestamp = DateTime.UtcNow.ToString("O"),
                    Status = "Healthy",
                    UptimeHours = (DateTime.UtcNow - Process.GetCurrentProcess().StartTime.ToUniversalTime()).TotalHours,
                    CpuUsage = metrics.CpuUsage,
                    MemoryUsage = metrics.MemoryUsage,
                    DiskUsage = metrics.DiskUsage,
                    IpAddress = GetLocalIpAddress(),
                    ActiveCollectors = 1, // TODO: Get from collector manager
                    LogsCollected = (int)TotalLogsSent,
                    LogsForwarded = (int)TotalLogsSent
                };

                // Send via streaming if available, otherwise unary
                if (_heartbeatStream != null)
                {
                    await _heartbeatStream.RequestStream.WriteAsync(heartbeat);
                }
                else if (_client != null)
                {
                    var response = await _client.SendHeartbeatAsync(heartbeat, GetCallOptions());
                    if (!response.Success)
                    {
                        _logger.LogWarning("Heartbeat failed: {Message}", response.Message);
                    }
                }

                _lastSuccessfulSend = DateTime.UtcNow;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error sending heartbeat via gRPC");
                _isConnected = false;
            }
        }

        private async void ProcessLogBatch(object? state)
        {
            await ProcessLogBatch(false);
        }

        private async Task<bool> ProcessLogBatch(bool forceFlush)
        {
            if (!_isConnected || _client == null) return false;
            
            List<LocalLogEntry> logsToSend;
            
            lock (_queueLock)
            {
                if (_logQueue.Count == 0) return true;

                var count = forceFlush ? _logQueue.Count : Math.Min(_batchSize, _logQueue.Count);
                logsToSend = new List<LocalLogEntry>();
                
                for (int i = 0; i < count && _logQueue.Count > 0; i++)
                {
                    logsToSend.Add(_logQueue.Dequeue());
                }
            }

            if (logsToSend.Count == 0) return true;

            await _sendSemaphore.WaitAsync();
            try
            {
                return await SendLogBatchAsync(logsToSend);
            }
            finally
            {
                _sendSemaphore.Release();
            }
        }

        private async Task<bool> SendLogBatchAsync(List<LocalLogEntry> logs)
        {
            try
            {
                if (_client == null)
                {
                    throw new InvalidOperationException("gRPC client not initialized");
                }

                // Convert local LogEntry to gRPC LogEntry
                var grpcLogs = logs.Select(log => new GrpcLogEntry
                {
                    Id = log.Id ?? Guid.NewGuid().ToString(),
                    Timestamp = log.Timestamp.ToString("O"),
                    Source = log.Source ?? "",
                    SourceType = log.Category ?? log.Source ?? "",
                    LogLevel = log.Level ?? "Information",
                    Message = log.Message ?? "",
                    Hash = ComputeLogHash(log),
                    Metadata = { log.Properties?.ToDictionary(kvp => kvp.Key, kvp => kvp.Value?.ToString() ?? "") ?? new Dictionary<string, string>() }
                }).ToList();

                // Use streaming if available, otherwise batch unary
                if (_logStream != null)
                {
                    foreach (var log in grpcLogs)
                    {
                        await _logStream.RequestStream.WriteAsync(log);
                    }
                }
                else if (_client != null)
                {
                    var request = new LogBatchRequest
                    {
                        AgentId = _agentId,
                        ApiKey = _apiKey,
                        Logs = { grpcLogs }
                    };

                    var response = await _client.ForwardLogsAsync(request, GetCallOptions());
                    if (!response.Success)
                    {
                        throw new Exception($"Log batch failed: {response.Message}");
                    }
                }
                
                TotalLogsSent += logs.Count;
                _lastSuccessfulSend = DateTime.UtcNow;
                
                _logger.LogDebug("Successfully sent {LogCount} logs via gRPC", logs.Count);
                
                LogsSent?.Invoke(this, new LogsSentEventArgs
                {
                    LogCount = logs.Count,
                    SentAt = DateTime.UtcNow
                });
                
                return true;
            }
            catch (Exception ex)
            {
                TotalSendErrors++;
                _logger.LogError(ex, "Failed to send log batch via gRPC");
                
                // Re-queue logs on failure
                lock (_queueLock)
                {
                    foreach (var log in logs.AsEnumerable().Reverse())
                    {
                        _logQueue.Enqueue(log);
                    }
                }
                
                CommunicationError?.Invoke(this, new CommunicationErrorEventArgs
                {
                    ErrorMessage = ex.Message,
                    LogCount = logs.Count,
                    ErrorTime = DateTime.UtcNow
                });
                
                return false;
            }
        }

        private (double CpuUsage, double MemoryUsage, double DiskUsage) GetSystemMetrics()
        {
            try
            {
                var process = Process.GetCurrentProcess();
                var cpuUsage = 0.0; // TODO: Implement CPU usage calculation
                var memoryUsage = (double)process.WorkingSet64 / (1024 * 1024 * 1024) * 100; // GB to percentage
                var diskUsage = 0.0; // TODO: Implement disk usage calculation

                return (cpuUsage, memoryUsage, diskUsage);
            }
            catch
            {
                return (0, 0, 0);
            }
        }

        private string GetLocalIpAddress()
        {
            try
            {
                var host = System.Net.Dns.GetHostEntry(System.Net.Dns.GetHostName());
                var localIp = host.AddressList.FirstOrDefault(ip => 
                    ip.AddressFamily == System.Net.Sockets.AddressFamily.InterNetwork &&
                    !System.Net.IPAddress.IsLoopback(ip));
                return localIp?.ToString() ?? "127.0.0.1";
            }
            catch
            {
                return "127.0.0.1";
            }
        }

        private string ComputeLogHash(LocalLogEntry log)
        {
            try
            {
                var content = $"{log.Timestamp:yyyy-MM-ddTHH:mm:ss.fffZ}|{log.Source}|{log.Message}";
                using var sha256 = System.Security.Cryptography.SHA256.Create();
                var hash = sha256.ComputeHash(System.Text.Encoding.UTF8.GetBytes(content));
                return Convert.ToHexString(hash)[..16];
            }
            catch
            {
                return Guid.NewGuid().ToString("N")[..16];
            }
        }

        public Task<bool> TryAutoDeploymentAsync(string backendUrl)
        {
            // gRPC doesn't support auto-deployment, use HTTP fallback
            _logger.LogWarning("Auto-deployment not supported via gRPC, use HTTP fallback");
            return Task.FromResult(false);
        }

        public event EventHandler<BackendConfigurationUpdatedEventArgs>? ConfigurationUpdated;

        public Task<List<LocalLogEntry>> LoadArchivedLogsAsync(DateTime fromDate, DateTime toDate)
        {
            // gRPC doesn't support archive loading, use HTTP fallback
            _logger.LogWarning("Archive loading not supported via gRPC, use HTTP fallback");
            return Task.FromResult(new List<LocalLogEntry>());
        }

        public async ValueTask DisposeAsync()
        {
            _cancellationTokenSource.Cancel();
            
            try
            {
                if (_heartbeatStream != null)
                {
                    await _heartbeatStream.RequestStream.CompleteAsync();
                    _heartbeatStream.Dispose();
                }

                if (_logStream != null)
                {
                    await _logStream.RequestStream.CompleteAsync();
                    _logStream.Dispose();
                }

                if (_heartbeatStreamTask != null)
                {
                    await _heartbeatStreamTask;
                }

                if (_logStreamTask != null)
                {
                    await _logStreamTask;
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error disposing gRPC streams");
            }

            _heartbeatTimer?.Dispose();
            _batchTimer?.Dispose();
            _channel?.Dispose();
            _cancellationTokenSource.Dispose();
        }
    }
}
