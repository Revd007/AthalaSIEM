using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using System.Linq;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Configuration;
using AthalaSIEM.UniversalAgent.Models;
using AthalaSIEM.Agent.Core;

namespace AthalaSIEM.UniversalAgent.Services
{
    /// <summary>
    /// Backend communication service following ManageEngine EventLog Analyzer pattern
    /// Handles secure communication with AthalaSIEM backend API
    /// Implements batch processing, retry logic, and health monitoring
    /// </summary>
    public class BackendCommunicationService : IAsyncDisposable
    {
        private readonly ILogger<BackendCommunicationService> _logger;
        private readonly IConfiguration _configuration;
        private readonly HttpClient _httpClient;
        private readonly Timer _heartbeatTimer;
        private readonly Timer _batchTimer;
        private readonly Queue<LogEntry> _logQueue = new();
        private readonly object _queueLock = new();
        private readonly SemaphoreSlim _sendSemaphore = new(1, 1);

        private string _backendUrl = "";
        private string _agentId = "";
        private string _apiKey = "";
        private int _batchSize;
        private int _batchIntervalSeconds;
        private bool _isConnected;
        private DateTime _lastSuccessfulSend;

        public bool IsConnected => _isConnected;
        public long QueuedLogs => _logQueue.Count;
        public DateTime LastSuccessfulSend => _lastSuccessfulSend;
        public long TotalLogsSent { get; private set; }
        public long TotalSendErrors { get; private set; }

        public event EventHandler<LogsSentEventArgs>? LogsSent;
        public event EventHandler<CommunicationErrorEventArgs>? CommunicationError;
        public event EventHandler<ConnectionStatusChangedEventArgs>? ConnectionStatusChanged;

        public BackendCommunicationService(
            ILogger<BackendCommunicationService> logger,
            IConfiguration configuration,
            HttpClient httpClient)
        {
            _logger = logger;
            _configuration = configuration;
            _httpClient = httpClient;

            LoadConfiguration();
            ConfigureHttpClient();

            // Setup timers
            _heartbeatTimer = new Timer(SendHeartbeat, null, TimeSpan.FromMinutes(1), TimeSpan.FromMinutes(1));
            _batchTimer = new Timer(ProcessLogBatch, null, 
                TimeSpan.FromSeconds(_batchIntervalSeconds), 
                TimeSpan.FromSeconds(_batchIntervalSeconds));
        }

        /// <summary>
        /// Initialize connection with backend server
        /// </summary>
        public async Task<bool> InitializeAsync()
        {
            try
            {
                _logger.LogInformation("Initializing connection to backend: {BackendUrl}", _backendUrl);

                // Test connection
                var isHealthy = await TestConnectionAsync();
                if (isHealthy)
                {
                    // Register agent with backend
                    await RegisterAgentAsync();
                    
                    _isConnected = true;
                    _lastSuccessfulSend = DateTime.UtcNow;
                    
                    ConnectionStatusChanged?.Invoke(this, new ConnectionStatusChangedEventArgs
                    {
                        IsConnected = true,
                        StatusMessage = "Connected to backend successfully"
                    });

                    _logger.LogInformation("Successfully connected to backend");
                    return true;
                }
                else
                {
                    _logger.LogError("Failed to connect to backend");
                    return false;
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error initializing backend connection");
                return false;
            }
        }

        /// <summary>
        /// Queue log entry for batch sending (ManageEngine batch pattern)
        /// </summary>
        public void QueueLog(LogEntry log)
        {
            lock (_queueLock)
            {
                _logQueue.Enqueue(log);
                
                // Prevent memory overflow
                if (_logQueue.Count > 50000)
                {
                    // Remove oldest 10000 logs
                    for (int i = 0; i < 10000; i++)
                    {
                        if (_logQueue.Count > 0)
                            _logQueue.Dequeue();
                    }
                    
                    _logger.LogWarning("Log queue overflow, removed oldest logs");
                }
            }
        }

        /// <summary>
        /// Queue multiple logs for batch sending
        /// </summary>
        public void QueueLogs(IEnumerable<LogEntry> logs)
        {
            lock (_queueLock)
            {
                foreach (var log in logs)
                {
                    _logQueue.Enqueue(log);
                }
            }
        }

        /// <summary>
        /// Force send queued logs immediately
        /// </summary>
        public async Task<bool> FlushLogsAsync()
        {
            return await ProcessLogBatch(forceFlush: true);
        }

        /// <summary>
        /// Test connection to backend API
        /// </summary>
        public async Task<bool> TestConnectionAsync()
        {
            try
            {
                var response = await _httpClient.GetAsync($"{_backendUrl}/api/health");
                var isHealthy = response.IsSuccessStatusCode;
                
                if (isHealthy != _isConnected)
                {
                    _isConnected = isHealthy;
                    ConnectionStatusChanged?.Invoke(this, new ConnectionStatusChangedEventArgs
                    {
                        IsConnected = isHealthy,
                        StatusMessage = isHealthy ? "Connection restored" : "Connection lost"
                    });
                }

                return isHealthy;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error testing backend connection");
                
                if (_isConnected)
                {
                    _isConnected = false;
                    ConnectionStatusChanged?.Invoke(this, new ConnectionStatusChangedEventArgs
                    {
                        IsConnected = false,
                        StatusMessage = $"Connection error: {ex.Message}"
                    });
                }
                
                return false;
            }
        }

        /// <summary>
        /// Get communication health status
        /// </summary>
        public CommunicationHealth GetHealthStatus()
        {
            return new CommunicationHealth
            {
                IsConnected = _isConnected,
                BackendUrl = _backendUrl,
                QueuedLogs = QueuedLogs,
                TotalLogsSent = TotalLogsSent,
                TotalSendErrors = TotalSendErrors,
                LastSuccessfulSend = _lastSuccessfulSend,
                LastHealthCheck = DateTime.UtcNow
            };
        }

        #region Private Methods

        private void LoadConfiguration()
        {
            _backendUrl = _configuration["BackendApiUrl"] ?? "http://localhost:9595";
            _agentId = _configuration["Agent:Id"] ?? Environment.MachineName;
            _apiKey = _configuration["Agent:ApiKey"] ?? "";
            _batchSize = _configuration.GetValue<int>("Agent:BatchSize", 100);
            _batchIntervalSeconds = _configuration.GetValue<int>("Agent:BatchIntervalSeconds", 30);
        }

        private void ConfigureHttpClient()
        {
            _httpClient.Timeout = TimeSpan.FromMinutes(2);
            _httpClient.DefaultRequestHeaders.Add("User-Agent", "AthalaSIEM-UniversalAgent/1.0");
            
            if (!string.IsNullOrEmpty(_apiKey))
            {
                _httpClient.DefaultRequestHeaders.Add("X-API-Key", _apiKey);
            }
        }

        private async Task RegisterAgentAsync()
        {
            try
            {
                var registrationData = new
                {
                    AgentId = _agentId,
                    AgentName = _configuration["Agent:Name"] ?? _agentId,
                    Version = "1.0.0",
                    Platform = Environment.OSVersion.Platform.ToString(),
                    MachineName = Environment.MachineName,
                    RegisteredAt = DateTime.UtcNow
                };

                var json = JsonSerializer.Serialize(registrationData);
                var content = new StringContent(json, Encoding.UTF8, "application/json");

                var response = await _httpClient.PostAsync($"{_backendUrl}/api/agents/register", content);
                
                if (response.IsSuccessStatusCode)
                {
                    _logger.LogInformation("Agent registered successfully");
                }
                else
                {
                    _logger.LogWarning("Agent registration failed: {StatusCode}", response.StatusCode);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error registering agent");
            }
        }

        private async void SendHeartbeat(object? state)
        {
            try
            {
                await TestConnectionAsync();

                if (_isConnected)
                {
                    var heartbeatData = new
                    {
                        AgentId = _agentId,
                        Timestamp = DateTime.UtcNow,
                        Status = "Healthy",
                        QueuedLogs = QueuedLogs,
                        TotalLogsSent = TotalLogsSent
                    };

                    var json = JsonSerializer.Serialize(heartbeatData);
                    var content = new StringContent(json, Encoding.UTF8, "application/json");

                    var response = await _httpClient.PostAsync($"{_backendUrl}/api/agents/heartbeat", content);
                    
                    if (!response.IsSuccessStatusCode)
                    {
                        _logger.LogWarning("Heartbeat failed: {StatusCode}", response.StatusCode);
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error sending heartbeat");
            }
        }

        private async void ProcessLogBatch(object? state)
        {
            await ProcessLogBatch(forceFlush: false);
        }

        private async Task<bool> ProcessLogBatch(bool forceFlush)
        {
            if (!_isConnected && !forceFlush)
                return false;

            await _sendSemaphore.WaitAsync();
            
            try
            {
                List<LogEntry> logsToSend;
                
                lock (_queueLock)
                {
                    if (_logQueue.Count == 0)
                        return true;

                    var countToTake = forceFlush ? _logQueue.Count : Math.Min(_batchSize, _logQueue.Count);
                    logsToSend = new List<LogEntry>();
                    
                    for (int i = 0; i < countToTake && _logQueue.Count > 0; i++)
                    {
                        logsToSend.Add(_logQueue.Dequeue());
                    }
                }

                if (logsToSend.Count > 0)
                {
                    var success = await SendLogBatchAsync(logsToSend);
                    
                    if (success)
                    {
                        TotalLogsSent += logsToSend.Count;
                        _lastSuccessfulSend = DateTime.UtcNow;
                        
                        LogsSent?.Invoke(this, new LogsSentEventArgs
                        {
                            LogCount = logsToSend.Count,
                            SentAt = DateTime.UtcNow
                        });

                        _logger.LogDebug("Sent {Count} logs to backend", logsToSend.Count);
                    }
                    else
                    {
                        // Re-queue logs on failure
                        lock (_queueLock)
                        {
                            foreach (var log in logsToSend.AsEnumerable().Reverse())
                            {
                                // Add back to front of queue
                                var tempQueue = new Queue<LogEntry>();
                                tempQueue.Enqueue(log);
                                
                                while (_logQueue.Count > 0)
                                {
                                    tempQueue.Enqueue(_logQueue.Dequeue());
                                }
                                
                                _logQueue.Clear();
                                while (tempQueue.Count > 0)
                                {
                                    _logQueue.Enqueue(tempQueue.Dequeue());
                                }
                            }
                        }
                    }
                    
                    return success;
                }
                
                return true;
            }
            finally
            {
                _sendSemaphore.Release();
            }
        }

        private async Task<bool> SendLogBatchAsync(List<LogEntry> logs)
        {
            try
            {
                var logBatch = new
                {
                    AgentId = _agentId,
                    BatchId = Guid.NewGuid().ToString(),
                    Timestamp = DateTime.UtcNow,
                    Logs = logs
                };

                var json = JsonSerializer.Serialize(logBatch, new JsonSerializerOptions
                {
                    PropertyNamingPolicy = JsonNamingPolicy.CamelCase
                });
                
                var content = new StringContent(json, Encoding.UTF8, "application/json");

                var response = await _httpClient.PostAsync($"{_backendUrl}/api/logs/batch", content);
                
                if (response.IsSuccessStatusCode)
                {
                    return true;
                }
                else
                {
                    TotalSendErrors++;
                    _logger.LogError("Failed to send log batch: {StatusCode} - {Content}", 
                        response.StatusCode, await response.Content.ReadAsStringAsync());
                    
                    CommunicationError?.Invoke(this, new CommunicationErrorEventArgs
                    {
                        ErrorMessage = $"HTTP {response.StatusCode}",
                        LogCount = logs.Count,
                        ErrorTime = DateTime.UtcNow
                    });
                    
                    return false;
                }
            }
            catch (Exception ex)
            {
                TotalSendErrors++;
                _logger.LogError(ex, "Error sending log batch");
                
                CommunicationError?.Invoke(this, new CommunicationErrorEventArgs
                {
                    ErrorMessage = ex.Message,
                    LogCount = logs.Count,
                    ErrorTime = DateTime.UtcNow
                });
                
                return false;
            }
        }

        #endregion

        public async ValueTask DisposeAsync()
        {
            _heartbeatTimer?.Dispose();
            _batchTimer?.Dispose();
            _sendSemaphore?.Dispose();
            _httpClient?.Dispose();
        }
    }

    #region Supporting Classes

    public class CommunicationHealth
    {
        public bool IsConnected { get; set; }
        public string BackendUrl { get; set; } = "";
        public long QueuedLogs { get; set; }
        public long TotalLogsSent { get; set; }
        public long TotalSendErrors { get; set; }
        public DateTime LastSuccessfulSend { get; set; }
        public DateTime LastHealthCheck { get; set; }
    }

    public class LogsSentEventArgs : EventArgs
    {
        public int LogCount { get; set; }
        public DateTime SentAt { get; set; }
    }

    public class CommunicationErrorEventArgs : EventArgs
    {
        public string ErrorMessage { get; set; } = "";
        public int LogCount { get; set; }
        public DateTime ErrorTime { get; set; }
    }

    public class ConnectionStatusChangedEventArgs : EventArgs
    {
        public bool IsConnected { get; set; }
        public string StatusMessage { get; set; } = "";
    }

    #endregion
} 