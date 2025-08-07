using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using System.Linq;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Configuration;
using AthalaSIEM.UniversalAgent.Models;
using Grpc.Net.Client;
using Grpc.Core;
// using AthalaSIEM.Agent;
// using GrpcLogEntry = AthalaSIEM.Agent.LogEntry;
using LocalLogEntry = AthalaSIEM.UniversalAgent.Models.LogEntry;

namespace AthalaSIEM.UniversalAgent.Services
{
    /// <summary>
    /// gRPC communication service for AthalaSIEM backend
    /// Provides high-performance communication using Protocol Buffers
    /// </summary>
    public class GrpcCommunicationService : IAsyncDisposable
    {
        private readonly ILogger<GrpcCommunicationService> _logger;
        private readonly IConfiguration _configuration;
        private readonly Timer _heartbeatTimer;
        private readonly Timer _batchTimer;
        private readonly Queue<LocalLogEntry> _logQueue = new();
        private readonly object _queueLock = new();
        private readonly SemaphoreSlim _sendSemaphore = new(1, 1);

        private string _serverUrl = "";
        private string _agentId = "";
        private string _apiKey = "";
        private int _batchSize;
        private int _batchIntervalSeconds;
        private bool _isConnected;
        private DateTime _lastSuccessfulSend;
        private GrpcChannel? _channel;
        private object? _client = null; // Temporary until proto is generated

        public bool IsConnected => _isConnected;
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
            _heartbeatTimer = new Timer(SendHeartbeat, null, TimeSpan.FromMinutes(1), TimeSpan.FromMinutes(1));
            _batchTimer = new Timer(ProcessLogBatch, null, 
                TimeSpan.FromSeconds(_batchIntervalSeconds), 
                TimeSpan.FromSeconds(_batchIntervalSeconds));
        }

        public async Task<bool> InitializeAsync()
        {
            try
            {
                _logger.LogInformation("Initializing gRPC communication to: {ServerUrl}", _serverUrl);
                
                // Create gRPC channel
                _channel = GrpcChannel.ForAddress(_serverUrl);
                // _client = new SiemService.SiemServiceClient(_channel);
                
                // Test connection and register agent
                await RegisterAgentAsync();
                
                _isConnected = true;
                _lastSuccessfulSend = DateTime.UtcNow;
                
                ConnectionStatusChanged?.Invoke(this, new ConnectionStatusChangedEventArgs
                {
                    IsConnected = true,
                    StatusMessage = "Connected to SIEM backend via gRPC"
                });

                _logger.LogInformation("gRPC communication service initialized successfully");
                await Task.CompletedTask;
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

        public void QueueLog(LocalLogEntry log)
        {
            if (log == null) return;

            lock (_queueLock)
            {
                _logQueue.Enqueue(log);
                
                // Prevent memory overflow
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

                // TODO: Implement when proto is generated
                // var request = new HeartbeatRequest { ... };
                // var response = await _client.SendHeartbeatAsync(request);
                
                _logger.LogInformation("gRPC connection test - channel ready: {IsReady}", _channel?.State);
                await Task.CompletedTask;
                return _channel?.State == ConnectivityState.Ready;
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
                IsConnected = _isConnected,
                ManagerUrl = _serverUrl,
                QueuedLogs = QueuedLogs,
                TotalLogsSent = TotalLogsSent,
                TotalSendErrors = TotalSendErrors,
                LastSuccessfulSend = _lastSuccessfulSend,
                LastHealthCheck = DateTime.UtcNow
            };
        }

        private void LoadConfiguration()
        {
            // Get server URL from configuration
            var managerIP = _configuration["SiemManager:ManagerIP"];
            if (string.IsNullOrEmpty(managerIP))
            {
                _logger.LogError("❌ SiemManager:ManagerIP is REQUIRED and not configured!");
                throw new InvalidOperationException("SiemManager:ManagerIP configuration is required.");
            }
            
            var managerPort = _configuration.GetValue<int>("SiemManager:ManagerPort");
            if (managerPort == 0)
            {
                _logger.LogError("❌ SiemManager:ManagerPort is REQUIRED and not configured!");
                throw new InvalidOperationException("SiemManager:ManagerPort configuration is required.");
            }
            var useHTTPS = _configuration.GetValue<bool>("SiemManager:UseHTTPS", false);
            var protocol = useHTTPS ? "https" : "http";
            
            _serverUrl = $"{protocol}://{managerIP}:{managerPort}";
            _agentId = _configuration["Agent:Id"] ?? Environment.MachineName;
            _apiKey = _configuration["Agent:ApiKey"] ?? "";
            _batchSize = _configuration.GetValue<int>("Agent:BatchSize", 100);
            _batchIntervalSeconds = _configuration.GetValue<int>("Agent:BatchIntervalSeconds", 30);
            
            _logger.LogInformation("gRPC Configuration loaded - Server: {ServerUrl}, Batch: {BatchSize}, Interval: {Interval}s", 
                _serverUrl, _batchSize, _batchIntervalSeconds);
        }

        private async Task RegisterAgentAsync()
        {
            try
            {
                // TODO: Implement when proto is generated
                _logger.LogInformation("Agent registration placeholder - gRPC proto not yet generated");
                await Task.Delay(100); // Placeholder
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to register agent via gRPC");
                throw;
            }
        }

        private async void SendHeartbeat(object? state)
        {
            try
            {
                if (_client == null || !_isConnected) return;

                // TODO: Implement when proto is generated  
                _logger.LogDebug("gRPC heartbeat placeholder");
                await Task.CompletedTask;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error sending heartbeat via gRPC");
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

                // TODO: Implement when proto is generated
                // Simulate successful send for now
                await Task.Delay(10);
                
                TotalLogsSent += logs.Count;
                _lastSuccessfulSend = DateTime.UtcNow;
                
                _logger.LogDebug("Successfully sent {LogCount} logs via gRPC (simulated)", logs.Count);
                
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
                return Convert.ToHexString(hash)[..16]; // First 16 chars
            }
            catch
            {
                return Guid.NewGuid().ToString("N")[..16];
            }
        }

        public ValueTask DisposeAsync()
        {
            _heartbeatTimer?.Dispose();
            _batchTimer?.Dispose();
            _channel?.Dispose();
            return ValueTask.CompletedTask;
        }
    }


} 
