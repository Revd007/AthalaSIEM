using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
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
    /// Enhanced HTTP communication service with backend configuration support.
    /// Handles registration, heartbeat, log forwarding, and real-time configuration updates.
    /// </summary>
    public sealed class BackendCommunicationService : IBackendCommunicationService
    {
        private readonly ILogger<BackendCommunicationService> _logger;
        private readonly IConfiguration _configuration;
        private readonly HttpClient _httpClient;
        private readonly Timer _heartbeatTimer;
        private readonly Timer _batchTimer;
        private readonly Timer _archivalTimer;
        private readonly Timer _configUpdateTimer;
        private readonly Queue<LogEntry> _logQueue = new();
        private readonly object _queueLock = new();
        private readonly SemaphoreSlim _sendSemaphore = new(1, 1);
        private readonly string _archiveDirectory;
        
        // Configurable values - no more hardcoding
        private readonly int _retentionDays;
        private readonly int _maxQueueSize;
        private readonly int _heartbeatIntervalMinutes;
        private readonly int _batchIntervalSeconds;
        private readonly int _configUpdateIntervalMinutes;
        private readonly int _archivalIntervalHours;
        private readonly string _fallbackLocalIp;

        private string _managerUrl = "";
        private string _agentId = "";
        private string _apiKey = "";
        private int _batchSize;
        private bool _isConnected;
        private DateTime _lastSuccessfulSend;
        private DateTime _lastConfigUpdate;
        private string _configurationVersion = "";

        public bool IsConnected => _isConnected;
        public long QueuedLogs => _logQueue.Count;
        public DateTime LastSuccessfulSend => _lastSuccessfulSend;
        public long TotalLogsSent { get; private set; }
        public long TotalSendErrors { get; private set; }

        public event EventHandler<LogsSentEventArgs>? LogsSent;
        public event EventHandler<CommunicationErrorEventArgs>? CommunicationError;
        public event EventHandler<ConnectionStatusChangedEventArgs>? ConnectionStatusChanged;

        // Backend configuration event
        public event EventHandler<BackendConfigurationUpdatedEventArgs>? ConfigurationUpdated;

        public BackendCommunicationService(
            ILogger<BackendCommunicationService> logger,
            IConfiguration configuration,
            HttpClient httpClient)
        {
            _logger = logger;
            _configuration = configuration;
            _httpClient = httpClient;

            // Load all configurable values from appsettings - NO HARDCODING
            _retentionDays = _configuration.GetValue<int>("Communication:RetentionDays", 90);
            _maxQueueSize = _configuration.GetValue<int>("Communication:MaxQueueSize", 50000);
            _heartbeatIntervalMinutes = _configuration.GetValue<int>("Communication:HeartbeatIntervalMinutes", 1);
            _batchIntervalSeconds = _configuration.GetValue<int>("Communication:BatchIntervalSeconds", 30);
            _configUpdateIntervalMinutes = _configuration.GetValue<int>("Communication:ConfigUpdateIntervalMinutes", 30);
            _archivalIntervalHours = _configuration.GetValue<int>("Communication:ArchivalIntervalHours", 24);
            _fallbackLocalIp = _configuration.GetValue<string>("Communication:FallbackLocalIp") ?? "127.0.0.1";

            // Create archive directory
            var baseDir = Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location) ?? Environment.CurrentDirectory;
            var archiveDir = _configuration.GetValue<string>("Communication:ArchiveDirectory") ?? "LogArchive";
            _archiveDirectory = Path.Combine(baseDir, archiveDir);
            Directory.CreateDirectory(_archiveDirectory);

            // Initialize timers with configurable intervals
            var heartbeatInterval = TimeSpan.FromMinutes(_heartbeatIntervalMinutes);
            var batchInterval = TimeSpan.FromSeconds(_batchIntervalSeconds);
            var archivalInterval = TimeSpan.FromHours(_archivalIntervalHours);
            var configUpdateInterval = TimeSpan.FromMinutes(_configUpdateIntervalMinutes);

            _heartbeatTimer = new Timer(SendHeartbeat, null, heartbeatInterval, heartbeatInterval);
            _batchTimer = new Timer(ProcessLogBatch, null, batchInterval, batchInterval);
            _archivalTimer = new Timer(ArchiveLogs, null, archivalInterval, archivalInterval);
            _configUpdateTimer = new Timer(FetchBackendConfiguration, null, configUpdateInterval, configUpdateInterval);

            _logger.LogInformation("Communication service initialized with configurable values - RetentionDays: {RetentionDays}, MaxQueueSize: {MaxQueueSize}, HeartbeatInterval: {HeartbeatInterval}min",
                _retentionDays, _maxQueueSize, _heartbeatIntervalMinutes);
        }

        /// <summary>
        /// Initialize connection with backend server
        /// </summary>
        public async Task<bool> InitializeAsync()
        {
            try
            {
                LoadConfiguration();
                ConfigureHttpClient();
                await RegisterAgentAsync();

                _logger.LogInformation("Backend communication service initialized successfully - Connected: {IsConnected}", _isConnected);
                return _isConnected;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to initialize backend communication service");
                return false;
            }
        }

        /// <summary>
        /// Queue log entry for batch sending (ManageEngine batch pattern)
        /// </summary>
        public void QueueLog(LogEntry log)
        {
            if (log == null) return;

            lock (_queueLock)
            {
                // Check queue size limit
                if (_logQueue.Count >= _maxQueueSize)
                {
                    // Remove configurable percentage of oldest logs to prevent memory issues
                    var removalPercentage = _configuration.GetValue<double>("Communication:QueueRemovalPercentage", 0.25);
                    var logsToRemove = (int)(_logQueue.Count * removalPercentage);
                    
                    for (int i = 0; i < logsToRemove; i++)
                    {
                        _logQueue.Dequeue();
                    }

                    _logger.LogWarning("Queue size limit reached ({MaxSize}), removed {RemovedCount} oldest logs ({Percentage:P0})",
                        _maxQueueSize, logsToRemove, removalPercentage);
                }

                _logQueue.Enqueue(log);
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
            // REMOVED HARDCODED IP - Backend configuration is now REQUIRED
            var managerIP = _configuration["SiemManager:ManagerIP"];
            var managerPort = _configuration.GetValue<int>("SiemManager:ManagerPort");
            if (managerPort == 0)
            {
                _logger.LogError("❌ SiemManager:ManagerPort is REQUIRED and not configured! Please provide your backend server port.");
                throw new InvalidOperationException("SiemManager:ManagerPort configuration is required. Please specify your backend server port.");
            }
            var useHTTPS = _configuration.GetValue<bool>("SiemManager:UseHTTPS", false);
            var protocol = useHTTPS ? "https" : "http";

            // Validate that Manager IP is provided - NO MORE DEFAULTS
            if (string.IsNullOrWhiteSpace(managerIP))
            {
                _logger.LogError("❌ SiemManager:ManagerIP is REQUIRED and not configured! Please provide your backend server IP.");
                throw new InvalidOperationException("SiemManager:ManagerIP configuration is required. Please specify your backend server IP address.");
            }
            
            _managerUrl = $"{protocol}://{managerIP}:{managerPort}";
            _agentId = string.IsNullOrEmpty(_configuration[ConfigurationKeys.AgentId]) ? Environment.MachineName : _configuration[ConfigurationKeys.AgentId] ?? Environment.MachineName;
            _apiKey = _configuration[ConfigurationKeys.ApiKey] ?? "";
            _batchSize = _configuration.GetValue<int>(ConfigurationKeys.BatchSize, Defaults.BatchSize);

            _logger.LogInformation("✅ Configuration loaded - Manager URL: {ManagerUrl}, AgentId: {AgentId}, ApiKey: {ApiKey}, BatchSize: {BatchSize}",
                _managerUrl, _agentId, string.IsNullOrEmpty(_apiKey) ? "NOT SET" : "SET", _batchSize);

            // Validate configuration with configurable limits
            var minBatchSize = _configuration.GetValue<int>("Validation:MinBatchSize", Validation.MinBatchSize);
            var maxBatchSize = _configuration.GetValue<int>("Validation:MaxBatchSize", Validation.MaxBatchSize);

            if (_batchSize < minBatchSize || _batchSize > maxBatchSize)
            {
                _logger.LogWarning("Invalid batch size {BatchSize}, using default {Default} (range: {Min}-{Max})", 
                    _batchSize, Defaults.BatchSize, minBatchSize, maxBatchSize);
                _batchSize = Defaults.BatchSize;
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
                    DeploymentToken = _configuration.GetValue<string>("Agent:DeploymentToken") ?? "",
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

                _logger.LogDebug("Sending registration request to: {Url}", Constants.ApiEndpoints.AgentRegistration);
                _logger.LogDebug("Registration request payload: {Json}", json);

                var response = await _httpClient.PostAsync(Constants.ApiEndpoints.AgentRegistration, content);
                var responseContent = await response.Content.ReadAsStringAsync();
                
                _logger.LogDebug("Registration response: {StatusCode} - {Content}", response.StatusCode, responseContent);
                
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
                        
                        _logger.LogInformation("Agent registered successfully with ID: {AgentId}, ApiKey: {ApiKey}", 
                            _agentId, string.IsNullOrEmpty(_apiKey) ? "NOT SET" : "SET");
                        
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

                if (_isConnected && !string.IsNullOrEmpty(_agentId) && !string.IsNullOrEmpty(_apiKey))
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
                        var errorContent = await response.Content.ReadAsStringAsync();
                        _logger.LogWarning("Heartbeat failed: {StatusCode} - {Content}", response.StatusCode, errorContent);
                        
                        if (response.StatusCode == System.Net.HttpStatusCode.Unauthorized)
                        {
                            _logger.LogError("Heartbeat authentication failed. AgentId: {AgentId}, ApiKey: {ApiKey}", 
                                _agentId, string.IsNullOrEmpty(_apiKey) ? "NOT SET" : "SET");
                        }
                    }
                }
                else
                {
                    _logger.LogDebug("Heartbeat skipped: Connection: {Connected}, AgentId: {AgentId}, ApiKey: {ApiKey}", 
                        _isConnected, 
                        string.IsNullOrEmpty(_agentId) ? "NOT SET" : "SET", 
                        string.IsNullOrEmpty(_apiKey) ? "NOT SET" : "SET");
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
                // Ensure agent is properly authenticated before sending logs
                if (string.IsNullOrEmpty(_agentId))
                {
                    _logger.LogWarning("Cannot send logs: AgentId is not set. Agent may not be registered.");
                    return false;
                }

                if (string.IsNullOrEmpty(_apiKey))
                {
                    _logger.LogWarning("Cannot send logs: ApiKey is not set. Agent may not be authenticated.");
                    return false;
                }

                // Sanitize logs before sending - ensure ProcessId is valid integer or null
                var sanitizedLogs = logs.Select(log => 
                {
                    // Create a copy with cleaned ProcessId
                    var cleanLog = new LogEntry
                    {
                        Id = log.Id, // Preserve unique log ID
                        Timestamp = log.Timestamp,
                        Source = log.Source,
                        Level = log.Level,
                        Message = log.Message,
                        EventId = log.EventId,
                        Category = log.Category,
                        SecurityRelevance = log.SecurityRelevance,
                        Properties = new Dictionary<string, object>(log.Properties),
                        CollectorType = log.CollectorType,
                        AgentId = log.AgentId,
                        CollectionTime = log.CollectionTime,
                        ComputerName = log.ComputerName,
                        Username = log.Username,
                        ProcessName = log.ProcessName,
                        ProcessId = SanitizeProcessId(log.ProcessId, log.Properties),
                        IpAddress = log.IpAddress,
                        LogHash = log.LogHash,
                        SearchIndex = log.SearchIndex
                    };
                    return cleanLog;
                }).ToList();

                // Backend expects AgentId at top level with Logs array
                var logBatchData = new
                {
                    AgentId = _agentId,
                    BatchId = Guid.NewGuid().ToString(),
                    BatchTimestamp = DateTime.UtcNow,
                    Logs = sanitizedLogs
                };

                var json = JsonSerializer.Serialize(logBatchData, new JsonSerializerOptions
                {
                    PropertyNamingPolicy = null, // Keep PascalCase for .NET backend
                    WriteIndented = false,
                    DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
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
                    var errorContent = await response.Content.ReadAsStringAsync();
                    _logger.LogError("Failed to send log batch: {StatusCode} - {Content}", 
                        response.StatusCode, errorContent);
                    
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

        /// <summary>
        /// Sanitizes ProcessId to ensure it's a valid integer or null
        /// </summary>
        private int? SanitizeProcessId(int? originalProcessId, Dictionary<string, object> properties)
        {
            // If original ProcessId is valid, return it
            if (originalProcessId.HasValue && originalProcessId.Value > 0)
                return originalProcessId;

            // Try to extract ProcessId from properties
            if (properties != null)
            {
                foreach (var prop in properties)
                {
                    if (prop.Key.ToLower().Contains("processid") || prop.Key.ToLower().Contains("process_id"))
                    {
                        if (prop.Value != null && int.TryParse(prop.Value.ToString(), out var processId) && processId > 0)
                        {
                            return processId;
                        }
                    }
                }
            }

            // Return null if no valid ProcessId found
            return null;
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
                return _fallbackLocalIp;
            }
            catch
            {
                return _fallbackLocalIp;
            }
        }

        /// <summary>
        /// Archives old logs and cleans up expired archive files (90-day retention)
        /// </summary>
        private async void ArchiveLogs(object? state)
        {
            try
            {
                await Task.Run(() =>
                {
                    var cutoffDate = DateTime.UtcNow.AddDays(-_retentionDays);
                    var deletedFiles = 0;
                    var archivedSize = 0L;

                    // Clean up old archive files
                    if (Directory.Exists(_archiveDirectory))
                    {
                        var archiveFiles = Directory.GetFiles(_archiveDirectory, "*.json");
                        
                        foreach (var file in archiveFiles)
                        {
                            var fileInfo = new FileInfo(file);
                            if (fileInfo.CreationTimeUtc < cutoffDate)
                            {
                                archivedSize += fileInfo.Length;
                                File.Delete(file);
                                deletedFiles++;
                            }
                        }
                    }

                    if (deletedFiles > 0)
                    {
                        _logger.LogInformation("Archive cleanup: Deleted {FileCount} files ({SizeMB:F2} MB) older than {Days} days",
                            deletedFiles, archivedSize / 1024.0 / 1024.0, _retentionDays);
                    }
                });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error during log archive cleanup");
            }
        }

        /// <summary>
        /// Archives logs to file system for later retrieval
        /// </summary>
        private async Task ArchiveLogsToFile(List<LogEntry> logs)
        {
            if (logs == null || !logs.Any()) return;

            try
            {
                var timestamp = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss");
                var fileName = $"archived_logs_{timestamp}_{Guid.NewGuid():N}.json";
                var filePath = Path.Combine(_archiveDirectory, fileName);

                var archiveData = new
                {
                    ArchivedAt = DateTime.UtcNow,
                    AgentId = _agentId,
                    LogCount = logs.Count,
                    Logs = logs
                };

                var json = JsonSerializer.Serialize(archiveData, new JsonSerializerOptions
                {
                    WriteIndented = true,
                    PropertyNamingPolicy = null
                });

                await File.WriteAllTextAsync(filePath, json);

                _logger.LogInformation("Archived {LogCount} logs to {FileName} ({SizeKB:F2} KB)",
                    logs.Count, fileName, json.Length / 1024.0);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error archiving logs to file");
            }
        }

        /// <summary>
        /// Loads archived logs from file system
        /// </summary>
        public async Task<List<LogEntry>> LoadArchivedLogsAsync(DateTime fromDate, DateTime toDate)
        {
            var result = new List<LogEntry>();

            try
            {
                if (!Directory.Exists(_archiveDirectory))
                    return result;

                var archiveFiles = Directory.GetFiles(_archiveDirectory, "*.json");
                
                foreach (var file in archiveFiles)
                {
                    var fileInfo = new FileInfo(file);
                    
                    // Check if file is within date range
                    if (fileInfo.CreationTimeUtc >= fromDate && fileInfo.CreationTimeUtc <= toDate)
                    {
                        try
                        {
                            var json = await File.ReadAllTextAsync(file);
                            var archiveData = JsonSerializer.Deserialize<JsonElement>(json);
                            
                            if (archiveData.TryGetProperty("Logs", out var logsProperty))
                            {
                                var logs = JsonSerializer.Deserialize<List<LogEntry>>(logsProperty.GetRawText());
                                if (logs != null)
                                {
                                    result.AddRange(logs);
                                }
                            }
                        }
                        catch (Exception ex)
                        {
                            _logger.LogWarning(ex, "Error reading archive file: {FileName}", file);
                        }
                    }
                }

                _logger.LogInformation("Loaded {LogCount} archived logs from {FileCount} files between {FromDate} and {ToDate}",
                    result.Count, archiveFiles.Length, fromDate.ToString("yyyy-MM-dd"), toDate.ToString("yyyy-MM-dd"));
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error loading archived logs");
            }

            return result;
        }

        /// <summary>
        /// Attempts automatic token deployment by fetching token from backend.
        /// This enables plug-and-play installation experience.
        /// </summary>
        /// <param name="backendUrl">Backend URL provided during installation.</param>
        /// <returns>True if token was successfully obtained.</returns>
        public async Task<bool> TryAutoDeploymentAsync(string backendUrl)
        {
            try
            {
                _logger.LogInformation("Attempting automatic token deployment from backend: {BackendUrl}", backendUrl);
                
                if (string.IsNullOrWhiteSpace(backendUrl))
                {
                    _logger.LogWarning("Backend URL not provided for automatic token deployment");
                    return false;
                }

                // Update manager URL for token fetch
                _managerUrl = backendUrl.TrimEnd('/');
                ConfigureHttpClient();

                // Fetch deployment token from backend
                var tokenRequest = new
                {
                    hostname = Environment.MachineName,
                    ipAddress = GetLocalIpAddress(),
                    platform = Environment.OSVersion.Platform.ToString(),
                    osVersion = Environment.OSVersion.VersionString,
                    requestTime = DateTime.UtcNow
                };

                var json = System.Text.Json.JsonSerializer.Serialize(tokenRequest);
                var content = new StringContent(json, Encoding.UTF8, Constants.ContentTypes.ApplicationJson);

                _logger.LogDebug("Requesting deployment token from: {Endpoint}", Constants.ApiEndpoints.GetDeploymentToken);
                var response = await _httpClient.PostAsync(Constants.ApiEndpoints.GetDeploymentToken, content);

                if (response.IsSuccessStatusCode)
                {
                    var responseContent = await response.Content.ReadAsStringAsync();
                    var tokenResponse = System.Text.Json.JsonSerializer.Deserialize<DeploymentTokenResponse>(responseContent);

                    if (tokenResponse != null && !string.IsNullOrEmpty(tokenResponse.Token))
                    {
                        // Update configuration with received token
                        _configuration["Agent:RegistrationKey"] = tokenResponse.Token;
                        _configuration["Agent:ManagerUrl"] = backendUrl;
                        
                        _logger.LogInformation("✅ Automatic token deployment successful! Token expires: {Expiry}", tokenResponse.ExpiresAt);
                        return true;
                    }
                }
                else
                {
                    var errorContent = await response.Content.ReadAsStringAsync();
                    _logger.LogError("Automatic token deployment failed: {StatusCode} - {Content}", response.StatusCode, errorContent);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error during automatic token deployment");
            }

            return false;
        }

        /// <summary>
        /// Fetches configuration from backend including Event IDs, FIM paths, and detection thresholds.
        /// This replaces hardcoded values with dynamic backend-controlled configuration.
        /// </summary>
        /// <param name="state">Timer state (not used).</param>
        private async void FetchBackendConfiguration(object? state)
        {
            if (!_isConnected || string.IsNullOrEmpty(_agentId) || string.IsNullOrEmpty(_apiKey))
            {
                _logger.LogDebug("Skipping backend configuration fetch - not connected or authenticated");
                return;
            }

            try
            {
                _logger.LogDebug("Fetching configuration from backend...");

                // Fetch all configuration types
                var configTasks = new List<Task<BackendConfigResult>>
                {
                    FetchConfigurationAsync(Constants.BackendConfig.ConfigurationTypeEventFiltering),
                    FetchConfigurationAsync(Constants.BackendConfig.ConfigurationTypeFIM),
                    FetchConfigurationAsync(Constants.BackendConfig.ConfigurationTypeDetectionThresholds),
                    FetchConfigurationAsync(Constants.BackendConfig.ConfigurationTypeMonitoring)
                };

                var results = await Task.WhenAll(configTasks);
                var updatedConfigs = new List<BackendConfigResult>();

                foreach (var result in results)
                {
                    if (result.Success)
                    {
                        updatedConfigs.Add(result);
                    }
                    else
                    {
                        _logger.LogWarning("Failed to fetch {ConfigType} configuration: {Error}", result.ConfigType, result.Error);
                    }
                }

                if (updatedConfigs.Any())
                {
                    _lastConfigUpdate = DateTime.UtcNow;
                    
                    // Fire configuration updated event
                    ConfigurationUpdated?.Invoke(this, new BackendConfigurationUpdatedEventArgs
                    {
                        UpdatedConfigurations = updatedConfigs,
                        UpdateTime = _lastConfigUpdate,
                        ConfigurationVersion = _configurationVersion
                    });

                    _logger.LogInformation("Backend configuration updated successfully - {ConfigCount} configurations fetched", updatedConfigs.Count);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error fetching backend configuration");
            }
        }

        /// <summary>
        /// Fetches a specific configuration type from backend.
        /// </summary>
        /// <param name="configType">Type of configuration to fetch.</param>
        /// <returns>Configuration fetch result.</returns>
        private async Task<BackendConfigResult> FetchConfigurationAsync(string configType)
        {
            try
            {
                var endpoint = configType switch
                {
                    Constants.BackendConfig.ConfigurationTypeEventFiltering => string.Format(Constants.ApiEndpoints.GetEventFilteringRules, _agentId),
                    Constants.BackendConfig.ConfigurationTypeFIM => string.Format(Constants.ApiEndpoints.GetFIMConfiguration, _agentId),
                    Constants.BackendConfig.ConfigurationTypeDetectionThresholds => string.Format(Constants.ApiEndpoints.GetDetectionThresholds, _agentId),
                    _ => string.Format(Constants.ApiEndpoints.AgentConfiguration, _agentId)
                };

                _logger.LogDebug("Fetching {ConfigType} from: {Endpoint}", configType, endpoint);

                var response = await _httpClient.GetAsync(endpoint);
                
                if (response.IsSuccessStatusCode)
                {
                    var content = await response.Content.ReadAsStringAsync();
                    var configData = System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, object>>(content);

                    return new BackendConfigResult
                    {
                        Success = true,
                        ConfigType = configType,
                        Configuration = configData ?? new Dictionary<string, object>(),
                        FetchTime = DateTime.UtcNow
                    };
                }
                else
                {
                    var errorContent = await response.Content.ReadAsStringAsync();
                    return new BackendConfigResult
                    {
                        Success = false,
                        ConfigType = configType,
                        Error = $"HTTP {response.StatusCode}: {errorContent}"
                    };
                }
            }
            catch (Exception ex)
            {
                return new BackendConfigResult
                {
                    Success = false,
                    ConfigType = configType,
                    Error = ex.Message
                };
            }
        }

        #endregion

        public ValueTask DisposeAsync()
        {
            try
            {
                _heartbeatTimer?.Dispose();
                _batchTimer?.Dispose();
                _archivalTimer?.Dispose();
                _configUpdateTimer?.Dispose();
                _httpClient?.Dispose();
                _sendSemaphore?.Dispose();
                
                _logger.LogInformation("BackendCommunicationService disposed");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error disposing BackendCommunicationService");
            }
            
            return ValueTask.CompletedTask;
        }
    }

    // NOTE: All models (DeploymentTokenResponse, BackendConfigResult, BackendConfigurationUpdatedEventArgs) 
    // have been moved to AthalaSIEM.UniversalAgent.Models.CommunicationServiceModels.cs for clean architecture separation
} 