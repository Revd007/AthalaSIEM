using AthalaSIEM.Agent.Models;
using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using System.Text;

namespace AthalaSIEM.Agent.Collectors
{
    /// <summary>
    /// Cloud Services Collector for AWS CloudTrail, Azure Activity Logs, GCP Audit Logs
    /// </summary>
    public class CloudServicesCollector : ILogCollector
    {
        private readonly ILogger<CloudServicesCollector> _logger;
        private readonly ILogNormalizer _normalizer;
        private readonly HttpClient _httpClient;
        private bool _isRunning;
        private bool _isPaused;
        private string _errorMessage = string.Empty;
        private CollectorSettings _settings = new();
        
        // Configuration
        private bool _enableAwsCloudTrail = false;
        private bool _enableAzureActivityLogs = false;
        private bool _enableGcpAuditLogs = false;
        private bool _enableAwsS3Logs = false;
        private int _collectionInterval = 300; // 5 minutes
        
        // AWS Configuration
        private string _awsAccessKey = string.Empty;
        private string _awsSecretKey = string.Empty;
        private string _awsRegion = "us-east-1";
        private string _awsS3Bucket = string.Empty;
        private string _awsCloudTrailLogGroup = string.Empty;
        
        // Azure Configuration
        private string _azureTenantId = string.Empty;
        private string _azureClientId = string.Empty;
        private string _azureClientSecret = string.Empty;
        private string _azureSubscriptionId = string.Empty;
        private string _azureResourceGroup = string.Empty;
        
        // GCP Configuration
        private string _gcpProjectId = string.Empty;
        private string _gcpServiceAccountKey = string.Empty;
        private string _gcpLogName = string.Empty;
        
        private CancellationTokenSource? _cancellationTokenSource;

        public event EventHandler<NormalizedLogEntry>? LogCollected;
        public string CollectorType => "CloudServices";
        public bool IsRunning => _isRunning;
        public bool IsPaused => _isPaused;
        public string LastError => _errorMessage;
        public CollectorSettings Settings => _settings;

        public CloudServicesCollector(ILogger<CloudServicesCollector> logger, ILogNormalizer normalizer)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _normalizer = normalizer ?? throw new ArgumentNullException(nameof(normalizer));
            _httpClient = new HttpClient();
        }

        public void Initialize(CollectorSettings settings)
        {
            _settings = settings ?? throw new ArgumentNullException(nameof(settings));
            _logger.LogInformation("Initializing Cloud Services Collector");

            try
            {
                ParseSettings();
                _logger.LogInformation("Cloud Services Collector initialized - AWS: {AWS}, Azure: {Azure}, GCP: {GCP}", 
                    _enableAwsCloudTrail, _enableAzureActivityLogs, _enableGcpAuditLogs);
            }
            catch (Exception ex)
            {
                _errorMessage = ex.Message;
                _logger.LogError(ex, "Failed to initialize Cloud Services Collector");
                throw;
            }
        }

        public async Task StartAsync()
        {
            if (_isRunning) return;

            try
            {
                _logger.LogInformation("Starting Cloud Services Collector");
                _cancellationTokenSource = new CancellationTokenSource();

                if (_enableAwsCloudTrail || _enableAwsS3Logs)
                {
                    _ = Task.Run(() => CollectAwsLogsAsync(_cancellationTokenSource.Token));
                }

                if (_enableAzureActivityLogs)
                {
                    _ = Task.Run(() => CollectAzureLogsAsync(_cancellationTokenSource.Token));
                }

                if (_enableGcpAuditLogs)
                {
                    _ = Task.Run(() => CollectGcpLogsAsync(_cancellationTokenSource.Token));
                }

                _isRunning = true;
                _isPaused = false;
                _errorMessage = string.Empty;

                _logger.LogInformation("Cloud Services Collector started successfully");
            }
            catch (Exception ex)
            {
                _errorMessage = ex.Message;
                _logger.LogError(ex, "Failed to start Cloud Services Collector");
                throw;
            }
        }

        public async Task StopAsync()
        {
            if (!_isRunning) return;

            try
            {
                _logger.LogInformation("Stopping Cloud Services Collector");
                
                _cancellationTokenSource?.Cancel();
                _isRunning = false;
                
                _logger.LogInformation("Cloud Services Collector stopped");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error stopping Cloud Services Collector");
            }
        }

        public void Pause()
        {
            _isPaused = true;
            _logger.LogInformation("Cloud Services Collector paused");
        }

        public void Resume()
        {
            _isPaused = false;
            _logger.LogInformation("Cloud Services Collector resumed");
        }

        private void ParseSettings()
        {
            // AWS Settings
            if (_settings.Properties.ContainsKey("EnableAwsCloudTrail"))
            {
                bool.TryParse(_settings.Properties["EnableAwsCloudTrail"], out _enableAwsCloudTrail);
            }

            if (_settings.Properties.ContainsKey("EnableAwsS3Logs"))
            {
                bool.TryParse(_settings.Properties["EnableAwsS3Logs"], out _enableAwsS3Logs);
            }

            if (_settings.Properties.ContainsKey("AwsAccessKey"))
            {
                _awsAccessKey = _settings.Properties["AwsAccessKey"];
            }

            if (_settings.Properties.ContainsKey("AwsSecretKey"))
            {
                _awsSecretKey = _settings.Properties["AwsSecretKey"];
            }

            if (_settings.Properties.ContainsKey("AwsRegion"))
            {
                _awsRegion = _settings.Properties["AwsRegion"];
            }

            if (_settings.Properties.ContainsKey("AwsS3Bucket"))
            {
                _awsS3Bucket = _settings.Properties["AwsS3Bucket"];
            }

            // Azure Settings
            if (_settings.Properties.ContainsKey("EnableAzureActivityLogs"))
            {
                bool.TryParse(_settings.Properties["EnableAzureActivityLogs"], out _enableAzureActivityLogs);
            }

            if (_settings.Properties.ContainsKey("AzureTenantId"))
            {
                _azureTenantId = _settings.Properties["AzureTenantId"];
            }

            if (_settings.Properties.ContainsKey("AzureClientId"))
            {
                _azureClientId = _settings.Properties["AzureClientId"];
            }

            if (_settings.Properties.ContainsKey("AzureClientSecret"))
            {
                _azureClientSecret = _settings.Properties["AzureClientSecret"];
            }

            if (_settings.Properties.ContainsKey("AzureSubscriptionId"))
            {
                _azureSubscriptionId = _settings.Properties["AzureSubscriptionId"];
            }

            // GCP Settings
            if (_settings.Properties.ContainsKey("EnableGcpAuditLogs"))
            {
                bool.TryParse(_settings.Properties["EnableGcpAuditLogs"], out _enableGcpAuditLogs);
            }

            if (_settings.Properties.ContainsKey("GcpProjectId"))
            {
                _gcpProjectId = _settings.Properties["GcpProjectId"];
            }

            if (_settings.Properties.ContainsKey("GcpServiceAccountKey"))
            {
                _gcpServiceAccountKey = _settings.Properties["GcpServiceAccountKey"];
            }

            // General Settings
            if (_settings.Properties.ContainsKey("CollectionInterval"))
            {
                int.TryParse(_settings.Properties["CollectionInterval"], out _collectionInterval);
            }
        }

        private async Task CollectAwsLogsAsync(CancellationToken cancellationToken)
        {
            _logger.LogInformation("Starting AWS logs collection");

            while (_isRunning && !cancellationToken.IsCancellationRequested)
            {
                try
                {
                    if (!_isPaused)
                    {
                        if (_enableAwsCloudTrail)
                        {
                            await CollectAwsCloudTrailLogs();
                        }

                        if (_enableAwsS3Logs)
                        {
                            await CollectAwsS3Logs();
                        }
                    }

                    await Task.Delay(TimeSpan.FromSeconds(_collectionInterval), cancellationToken);
                }
                catch (OperationCanceledException)
                {
                    break;
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error collecting AWS logs");
                    await Task.Delay(TimeSpan.FromSeconds(60), cancellationToken);
                }
            }
        }

        private async Task CollectAzureLogsAsync(CancellationToken cancellationToken)
        {
            _logger.LogInformation("Starting Azure logs collection");

            while (_isRunning && !cancellationToken.IsCancellationRequested)
            {
                try
                {
                    if (!_isPaused)
                    {
                        await CollectAzureActivityLogs();
                    }

                    await Task.Delay(TimeSpan.FromSeconds(_collectionInterval), cancellationToken);
                }
                catch (OperationCanceledException)
                {
                    break;
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error collecting Azure logs");
                    await Task.Delay(TimeSpan.FromSeconds(60), cancellationToken);
                }
            }
        }

        private async Task CollectGcpLogsAsync(CancellationToken cancellationToken)
        {
            _logger.LogInformation("Starting GCP logs collection");

            while (_isRunning && !cancellationToken.IsCancellationRequested)
            {
                try
                {
                    if (!_isPaused)
                    {
                        await CollectGcpAuditLogs();
                    }

                    await Task.Delay(TimeSpan.FromSeconds(_collectionInterval), cancellationToken);
                }
                catch (OperationCanceledException)
                {
                    break;
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error collecting GCP logs");
                    await Task.Delay(TimeSpan.FromSeconds(60), cancellationToken);
                }
            }
        }

        private async Task CollectAwsCloudTrailLogs()
        {
            try
            {
                _logger.LogInformation("Collecting AWS CloudTrail logs");

                // Simulate AWS CloudTrail API call
                var endTime = DateTime.UtcNow;
                var startTime = endTime.AddMinutes(-_collectionInterval / 60);

                // In real implementation, use AWS SDK to call CloudTrail LookupEvents
                var mockEvents = GenerateMockAwsCloudTrailEvents(startTime, endTime);

                foreach (var awsEvent in mockEvents)
                {
                    var logEntry = ParseAwsCloudTrailEvent(awsEvent);
                    if (logEntry != null)
                    {
                        var normalizedEntry = _normalizer.NormalizeLogEntry(logEntry);
                        LogCollected?.Invoke(this, normalizedEntry);
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error collecting AWS CloudTrail logs");
            }
        }

        private async Task CollectAwsS3Logs()
        {
            try
            {
                _logger.LogInformation("Collecting AWS S3 access logs from bucket: {Bucket}", _awsS3Bucket);

                // In real implementation, use AWS SDK to list and download S3 access logs
                // For now, simulate some S3 access log events
                var mockS3Events = GenerateMockAwsS3Events();

                foreach (var s3Event in mockS3Events)
                {
                    var logEntry = ParseAwsS3Event(s3Event);
                    if (logEntry != null)
                    {
                        var normalizedEntry = _normalizer.NormalizeLogEntry(logEntry);
                        LogCollected?.Invoke(this, normalizedEntry);
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error collecting AWS S3 logs");
            }
        }

        private async Task CollectAzureActivityLogs()
        {
            try
            {
                _logger.LogInformation("Collecting Azure Activity logs");

                // In real implementation, use Azure SDK to call Activity Log API
                var mockEvents = GenerateMockAzureActivityEvents();

                foreach (var azureEvent in mockEvents)
                {
                    var logEntry = ParseAzureActivityEvent(azureEvent);
                    if (logEntry != null)
                    {
                        var normalizedEntry = _normalizer.NormalizeLogEntry(logEntry);
                        LogCollected?.Invoke(this, normalizedEntry);
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error collecting Azure Activity logs");
            }
        }

        private async Task CollectGcpAuditLogs()
        {
            try
            {
                _logger.LogInformation("Collecting GCP Audit logs for project: {Project}", _gcpProjectId);

                // In real implementation, use GCP SDK to call Cloud Logging API
                var mockEvents = GenerateMockGcpAuditEvents();

                foreach (var gcpEvent in mockEvents)
                {
                    var logEntry = ParseGcpAuditEvent(gcpEvent);
                    if (logEntry != null)
                    {
                        var normalizedEntry = _normalizer.NormalizeLogEntry(logEntry);
                        LogCollected?.Invoke(this, normalizedEntry);
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error collecting GCP Audit logs");
            }
        }

        private NormalizedLogEntry? ParseAwsCloudTrailEvent(dynamic awsEvent)
        {
            try
            {
                return new NormalizedLogEntry
                {
                    Id = Guid.NewGuid().ToString(),
                    Timestamp = DateTime.Parse(awsEvent.eventTime.ToString()),
                    Level = "Information",
                    Source = $"AWS/CloudTrail/{awsEvent.awsRegion}",
                    Category = "CloudTrail",
                    EventId = awsEvent.eventName.ToString(),
                    Message = $"AWS API call: {awsEvent.eventName} by {awsEvent.userIdentity?.userName ?? "Unknown"}",
                    Details = JsonSerializer.Serialize(awsEvent),
                    Tags = new List<string> { "aws", "cloudtrail", awsEvent.eventSource.ToString(), awsEvent.eventName.ToString().ToLower() },
                    Severity = DetermineAwsSeverity(awsEvent.eventName.ToString())
                };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error parsing AWS CloudTrail event");
                return null;
            }
        }

        private NormalizedLogEntry? ParseAwsS3Event(dynamic s3Event)
        {
            try
            {
                return new NormalizedLogEntry
                {
                    Id = Guid.NewGuid().ToString(),
                    Timestamp = DateTime.Parse(s3Event.timestamp.ToString()),
                    Level = "Information",
                    Source = $"AWS/S3/{s3Event.bucket}",
                    Category = "S3Access",
                    EventId = "S3_ACCESS",
                    Message = $"S3 {s3Event.operation}: {s3Event.key} from {s3Event.remoteIp}",
                    Details = JsonSerializer.Serialize(s3Event),
                    Tags = new List<string> { "aws", "s3", s3Event.operation.ToString().ToLower(), s3Event.bucket.ToString() },
                    Severity = s3Event.httpStatus >= 400 ? "Medium" : "Low"
                };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error parsing AWS S3 event");
                return null;
            }
        }

        private NormalizedLogEntry? ParseAzureActivityEvent(dynamic azureEvent)
        {
            try
            {
                return new NormalizedLogEntry
                {
                    Id = Guid.NewGuid().ToString(),
                    Timestamp = DateTime.Parse(azureEvent.eventTimestamp.ToString()),
                    Level = azureEvent.level.ToString() == "Error" ? "Error" : "Information",
                    Source = $"Azure/Activity/{azureEvent.resourceGroupName}",
                    Category = "AzureActivity",
                    EventId = azureEvent.operationName.ToString(),
                    Message = $"Azure operation: {azureEvent.operationName} on {azureEvent.resourceId}",
                    Details = JsonSerializer.Serialize(azureEvent),
                    Tags = new List<string> { "azure", "activity", azureEvent.category.ToString().ToLower(), azureEvent.resourceType.ToString().ToLower() },
                    Severity = azureEvent.level.ToString() == "Error" ? "High" : "Low"
                };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error parsing Azure Activity event");
                return null;
            }
        }

        private NormalizedLogEntry? ParseGcpAuditEvent(dynamic gcpEvent)
        {
            try
            {
                return new NormalizedLogEntry
                {
                    Id = Guid.NewGuid().ToString(),
                    Timestamp = DateTime.Parse(gcpEvent.timestamp.ToString()),
                    Level = gcpEvent.severity.ToString() == "ERROR" ? "Error" : "Information",
                    Source = $"GCP/Audit/{gcpEvent.resource?.type ?? "unknown"}",
                    Category = "GCPAudit",
                    EventId = gcpEvent.protoPayload?.methodName?.ToString() ?? "GCP_AUDIT",
                    Message = $"GCP API call: {gcpEvent.protoPayload?.methodName} by {gcpEvent.protoPayload?.authenticationInfo?.principalEmail}",
                    Details = JsonSerializer.Serialize(gcpEvent),
                    Tags = new List<string> { "gcp", "audit", gcpEvent.logName.ToString().Split('/').Last(), gcpEvent.resource?.type?.ToString() ?? "unknown" },
                    Severity = gcpEvent.severity.ToString() == "ERROR" ? "High" : "Low"
                };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error parsing GCP Audit event");
                return null;
            }
        }

        private string DetermineAwsSeverity(string eventName)
        {
            var highRiskEvents = new[] { "DeleteBucket", "DeleteUser", "DeleteRole", "PutBucketPolicy", "CreateUser", "AttachUserPolicy" };
            var mediumRiskEvents = new[] { "AssumeRole", "GetSessionToken", "CreateAccessKey", "UpdateLoginProfile" };

            if (highRiskEvents.Contains(eventName)) return "High";
            if (mediumRiskEvents.Contains(eventName)) return "Medium";
            return "Low";
        }

        // Mock data generators for testing
        private List<dynamic> GenerateMockAwsCloudTrailEvents(DateTime startTime, DateTime endTime)
        {
            return new List<dynamic>
            {
                new {
                    eventTime = DateTime.UtcNow.ToString("o"),
                    eventName = "AssumeRole",
                    eventSource = "sts.amazonaws.com",
                    awsRegion = _awsRegion,
                    userIdentity = new { userName = "testuser@example.com", type = "IAMUser" },
                    sourceIPAddress = "203.0.113.1"
                }
            };
        }

        private List<dynamic> GenerateMockAwsS3Events()
        {
            return new List<dynamic>
            {
                new {
                    timestamp = DateTime.UtcNow.ToString("o"),
                    bucket = _awsS3Bucket,
                    key = "logs/access.log",
                    operation = "GET",
                    remoteIp = "203.0.113.1",
                    httpStatus = 200,
                    userAgent = "Mozilla/5.0"
                }
            };
        }

        private List<dynamic> GenerateMockAzureActivityEvents()
        {
            return new List<dynamic>
            {
                new {
                    eventTimestamp = DateTime.UtcNow.ToString("o"),
                    operationName = "Microsoft.Compute/virtualMachines/write",
                    category = "Administrative",
                    level = "Informational",
                    resourceGroupName = _azureResourceGroup,
                    resourceId = "/subscriptions/sub1/resourceGroups/rg1/providers/Microsoft.Compute/virtualMachines/vm1",
                    resourceType = "Microsoft.Compute/virtualMachines",
                    caller = "user@example.com"
                }
            };
        }

        private List<dynamic> GenerateMockGcpAuditEvents()
        {
            return new List<dynamic>
            {
                new {
                    timestamp = DateTime.UtcNow.ToString("o"),
                    severity = "INFO",
                    logName = $"projects/{_gcpProjectId}/logs/cloudaudit.googleapis.com%2Factivity",
                    resource = new { type = "gce_instance" },
                    protoPayload = new {
                        methodName = "compute.instances.insert",
                        authenticationInfo = new { principalEmail = "user@example.com" }
                    }
                }
            };
        }

        public void Dispose()
        {
            StopAsync().Wait();
            _httpClient?.Dispose();
            _cancellationTokenSource?.Dispose();
        }
    }
} 