using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using System.Linq;
using System.ComponentModel.DataAnnotations;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Configuration;
using AthalaSIEM.UniversalAgent.Models;
using AthalaSIEM.UniversalAgent.Services.Interfaces;
using AthalaSIEM.Agent.Core;
using static AthalaSIEM.UniversalAgent.Models.Constants;

namespace AthalaSIEM.UniversalAgent.Services
{
    /// <summary>
    /// Backend communication service following ManageEngine EventLog Analyzer pattern
    /// Handles secure communication with AthalaSIEM backend API
    /// Implements batch processing, retry logic, and health monitoring
    /// </summary>
    public sealed class BackendCommunicationService : IBackendCommunicationService
    {
        private readonly ILogger<BackendCommunicationService> _logger;
        private readonly IConfiguration _configuration;
        private readonly HttpClient _httpClient;
        private readonly Timer _heartbeatTimer;
        private readonly Timer _batchTimer;
        private readonly Queue<LogEntry> _logQueue = new();
        private readonly object _queueLock = new();
        private readonly SemaphoreSlim _sendSemaphore = new(1, 1);

        private string _managerUrl = "";
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
                _logger.LogInformation("Initializing connection to SIEM Manager: {ManagerUrl}", _managerUrl);

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

                    _logger.LogInformation("Successfully connected to SIEM Manager");
                    return true;
                }
                else
                {
                    _logger.LogError("Failed to connect to SIEM Manager");
                    return false;
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error initializing SIEM Manager connection");
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
        /// Test connection to SIEM Manager API
        /// </summary>
        public async Task<bool> TestConnectionAsync()
        {
            try
            {
                var response = await _httpClient.GetAsync($"{_managerUrl}/api/health");
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
                _logger.LogError(ex, "Error testing SIEM Manager connection");
                
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
                ManagerUrl = _managerUrl,
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
            // Build Manager URL from IP and Port (SIEM standard)
            var managerIP = _configuration["SiemManager:ManagerIP"] ?? "192.168.1.100";
            var managerPort = _configuration.GetValue<int>("SiemManager:ManagerPort", 9595);
            var useHTTPS = _configuration.GetValue<bool>("SiemManager:UseHTTPS", false);
            var protocol = useHTTPS ? "https" : "http";
            
            _managerUrl = $"{protocol}://{managerIP}:{managerPort}";
            _agentId = string.IsNullOrEmpty(_configuration[ConfigurationKeys.AgentId]) ? Environment.MachineName : _configuration[ConfigurationKeys.AgentId];
            _apiKey = _configuration[ConfigurationKeys.ApiKey] ?? "";
            _batchSize = _configuration.GetValue<int>(ConfigurationKeys.BatchSize, Defaults.BatchSize);
            _batchIntervalSeconds = _configuration.GetValue<int>(ConfigurationKeys.BatchIntervalSeconds, Defaults.BatchIntervalSeconds);

            // Validate configuration
            if (_batchSize < Validation.MinBatchSize || _batchSize > Validation.MaxBatchSize)
            {
                _logger.LogWarning("Invalid batch size {BatchSize}, using default {Default}", _batchSize, Defaults.BatchSize);
                _batchSize = Defaults.BatchSize;
            }

            if (_batchIntervalSeconds < Validation.MinIntervalSeconds || _batchIntervalSeconds > Validation.MaxIntervalSeconds)
            {
                _logger.LogWarning("Invalid batch interval {Interval}, using default {Default}", _batchIntervalSeconds, Defaults.BatchIntervalSeconds);
                _batchIntervalSeconds = Defaults.BatchIntervalSeconds;
            }
        }

        private void ConfigureHttpClient()
        {
            // Set base address for the HTTP client
            _httpClient.BaseAddress = new Uri(_managerUrl);
            
            // Configure timeout
            _httpClient.Timeout = TimeSpan.FromMilliseconds(Constants.Timeouts.HttpRequestTimeout);
            
            // Set default headers
            _httpClient.DefaultRequestHeaders.Add(Constants.Headers.UserAgent, $"AthalaSIEM-Agent/{Constants.Defaults.AgentVersion}");
            _httpClient.DefaultRequestHeaders.Accept.Add(
                new System.Net.Http.Headers.MediaTypeWithQualityHeaderValue(Constants.ContentTypes.ApplicationJson));

            // Add API key if available
            if (!string.IsNullOrEmpty(_apiKey))
            {
                _httpClient.DefaultRequestHeaders.Remove(Constants.Headers.ApiKey);
                _httpClient.DefaultRequestHeaders.Add(Constants.Headers.ApiKey, _apiKey);
            }

            _logger.LogDebug("HTTP client configured with base address: {BaseAddress}", _managerUrl);
        }

        private async Task RegisterAgentAsync()
        {
            try
            {
                _logger.LogInformation("Registering agent with SIEM Manager...");

                var registrationRequest = new AgentRegistrationRequest
                {
                    DeploymentToken = _configuration.GetValue<string>("Agent:RegistrationKey") ?? Constants.Defaults.RegistrationKey,
                    Hostname = Environment.MachineName,
                    IpAddress = GetLocalIpAddress(),
                    Platform = Environment.OSVersion.Platform.ToString(),
                    OsVersion = Environment.OSVersion.ToString(),
                    Version = Constants.Defaults.AgentVersion,
                    Capabilities = new List<string> { "WindowsEventLog", "FileIntegrity", "Registry", "LogProcessing" }
                };

                if (!registrationRequest.IsValid())
                {
                    _logger.LogError("Invalid registration request data");
                    return;
                }

                var json = JsonSerializer.Serialize(registrationRequest);
                var content = new StringContent(json, Encoding.UTF8, Constants.ContentTypes.ApplicationJson);

                var response = await _httpClient.PostAsync(Constants.ApiEndpoints.AgentRegistration, content);
                var responseContent = await response.Content.ReadAsStringAsync();

                if (response.IsSuccessStatusCode)
                {
                    var registrationResponse = JsonSerializer.Deserialize<AgentRegistrationResponse>(responseContent);
                    if (registrationResponse != null && registrationResponse.IsValid())
                    {
                        _agentId = registrationResponse.AgentId;
                        _apiKey = registrationResponse.ApiKey;
                        
                        // Update HTTP client with new API key
                        _httpClient.DefaultRequestHeaders.Remove(Constants.Headers.ApiKey);
                        _httpClient.DefaultRequestHeaders.Add(Constants.Headers.ApiKey, _apiKey);
                        
                        _logger.LogInformation("Agent registered successfully with ID: {AgentId}", _agentId);
                        
                        // Update connection status
                        _isConnected = true;
                        ConnectionStatusChanged?.Invoke(this, new ConnectionStatusChangedEventArgs
                        {
                            IsConnected = true,
                            StatusMessage = "Agent registered and connected successfully",
                            StatusTime = DateTime.UtcNow
                        });
                    }
                    else
                    {
                        _logger.LogError("Invalid registration response received");
                    }
                }
                else
                {
                    _logger.LogWarning("Agent registration failed: {StatusCode} - {Content}", 
                        response.StatusCode, responseContent);
                    
                    // If registration fails but we have existing credentials, try to use them
                    var existingApiKey = _configuration.GetValue<string>("Agent:ApiKey");
                    var existingAgentId = _configuration.GetValue<string>("Agent:Id");
                    
                    if (!string.IsNullOrEmpty(existingApiKey) && !string.IsNullOrEmpty(existingAgentId))
                    {
                        _logger.LogInformation("Using existing agent credentials for authentication");
                        _apiKey = existingApiKey;
                        _agentId = existingAgentId;
                        
                        // Update HTTP client with existing API key
                        _httpClient.DefaultRequestHeaders.Remove(Constants.Headers.ApiKey);
                        _httpClient.DefaultRequestHeaders.Add(Constants.Headers.ApiKey, _apiKey);
                        
                        _isConnected = true;
                        ConnectionStatusChanged?.Invoke(this, new ConnectionStatusChangedEventArgs
                        {
                            IsConnected = true,
                            StatusMessage = "Using existing agent credentials",
                            StatusTime = DateTime.UtcNow
                        });
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error during agent registration");
                
                // Try to use existing credentials as fallback
                var existingApiKey = _configuration.GetValue<string>("Agent:ApiKey");
                var existingAgentId = _configuration.GetValue<string>("Agent:Id");
                
                if (!string.IsNullOrEmpty(existingApiKey) && !string.IsNullOrEmpty(existingAgentId))
                {
                    _logger.LogInformation("Registration failed, falling back to existing credentials");
                    _apiKey = existingApiKey;
                    _agentId = existingAgentId;
                    
                    // Update HTTP client with existing API key
                    _httpClient.DefaultRequestHeaders.Remove(Constants.Headers.ApiKey);
                    _httpClient.DefaultRequestHeaders.Add(Constants.Headers.ApiKey, _apiKey);
                    
                    _isConnected = true;
                }
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

                    var response = await _httpClient.PostAsync($"{_managerUrl}{string.Format(ApiEndpoints.Heartbeat, _agentId)}", content);
                    
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
                            SentAt = DateTime.UtcNow,
                            ProcessingDuration = TimeSpan.FromMilliseconds(100), // Could be measured
                            BatchSize = logsToSend.Count
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

                var response = await _httpClient.PostAsync($"{_managerUrl}/api/logs/batch", content);
                
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
                        ErrorTime = DateTime.UtcNow,
                        ErrorCategory = ErrorCategories.NetworkError,
                        IsRetryable = true
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
                    ErrorTime = DateTime.UtcNow,
                    Exception = ex,
                    ErrorCategory = ErrorCategories.NetworkError,
                    IsRetryable = true
                });
                
                return false;
            }
        }

        private string GetLocalIpAddress()
        {
            try
            {
                var host = System.Net.Dns.GetHostEntry(System.Net.Dns.GetHostName());
                foreach (var ip in host.AddressList)
                {
                    if (ip.AddressFamily == System.Net.Sockets.AddressFamily.InterNetwork)
                    {
                        return ip.ToString();
                    }
                }
                return "127.0.0.1";
            }
            catch
            {
                return "127.0.0.1";
            }
        }

        #endregion

        public ValueTask DisposeAsync()
        {
            _heartbeatTimer?.Dispose();
            _batchTimer?.Dispose();
            _sendSemaphore?.Dispose();
            _httpClient?.Dispose();
            
            return ValueTask.CompletedTask;
        }
    }
} 