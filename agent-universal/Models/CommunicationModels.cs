using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Text.Json.Serialization;

namespace AthalaSIEM.UniversalAgent.Models
{
    /// <summary>
    /// Response model for agent registration with the SIEM backend.
    /// Contains authentication and configuration information.
    /// </summary>
    public sealed class AgentRegistrationResponse
    {
        [Required]
        [JsonPropertyName("agentId")]
        public string AgentId { get; init; } = string.Empty;

        [Required]
        [JsonPropertyName("apiKey")]
        public string ApiKey { get; init; } = string.Empty;

        [JsonPropertyName("backendUrl")]
        public string BackendUrl { get; init; } = string.Empty;

        [JsonPropertyName("configuration")]
        public string Configuration { get; init; } = string.Empty;

        [Range(30, 3600)]
        [JsonPropertyName("updateIntervalSeconds")]
        public int UpdateIntervalSeconds { get; init; } = 300;

        [Range(30, 3600)]
        [JsonPropertyName("heartbeatIntervalSeconds")]
        public int HeartbeatIntervalSeconds { get; init; } = 60;

        /// <summary>
        /// Validates the registration response data.
        /// </summary>
        /// <returns>True if all required fields are valid.</returns>
        public bool IsValid()
        {
            return !string.IsNullOrWhiteSpace(AgentId) &&
                   !string.IsNullOrWhiteSpace(ApiKey) &&
                   UpdateIntervalSeconds is >= 30 and <= 3600 &&
                   HeartbeatIntervalSeconds is >= 30 and <= 3600;
        }
    }

    /// <summary>
    /// Request model for agent registration with deployment token.
    /// </summary>
    public sealed class AgentRegistrationRequest
    {
        [Required]
        [JsonPropertyName("deploymentToken")]
        public string DeploymentToken { get; init; } = string.Empty;

        [Required]
        [JsonPropertyName("hostname")]
        public string Hostname { get; init; } = string.Empty;

        [Required]
        [JsonPropertyName("ipAddress")]
        public string IpAddress { get; init; } = string.Empty;

        [Required]
        [JsonPropertyName("platform")]
        public string Platform { get; init; } = string.Empty;

        [JsonPropertyName("osVersion")]
        public string OsVersion { get; init; } = string.Empty;

        [JsonPropertyName("version")]
        public string Version { get; init; } = "1.0.0";

        [JsonPropertyName("capabilities")]
        public List<string> Capabilities { get; init; } = new();

        /// <summary>
        /// Validates the registration request data.
        /// </summary>
        /// <returns>True if all required fields are valid.</returns>
        public bool IsValid()
        {
            return !string.IsNullOrWhiteSpace(DeploymentToken) &&
                   !string.IsNullOrWhiteSpace(Hostname) &&
                   !string.IsNullOrWhiteSpace(IpAddress) &&
                   !string.IsNullOrWhiteSpace(Platform);
        }
    }

    // NOTE: CommunicationHealth model has been moved to 
    // AthalaSIEM.UniversalAgent.Models.CommunicationServiceModels.cs for clean architecture separation



    /// <summary>
    /// Represents configuration update from backend.
    /// </summary>
    public class BackendConfigurationUpdate
    {
        public string ConfigurationType { get; set; } = "";
        public Dictionary<string, object> Configuration { get; set; } = new();
        public DateTime UpdateTime { get; set; }
        public bool RequiresRestart { get; set; }
        public string UpdateReason { get; set; } = "";
    }

    /// <summary>
    /// Represents auto-deployment request.
    /// </summary>
    public class AutoDeploymentRequest
    {
        public string BackendUrl { get; set; } = "";
        public string InstallerId { get; set; } = "";
        public string InstallerVersion { get; set; } = "";
        public string MachineName { get; set; } = "";
        public string UserName { get; set; } = "";
        public string OperatingSystem { get; set; } = "";
        public string Architecture { get; set; } = "";
        public DateTime RequestTime { get; set; } = DateTime.UtcNow;
    }

    /// <summary>
    /// Represents auto-deployment response.
    /// </summary>
    public class AutoDeploymentResponse
    {
        public bool Success { get; set; }
        public string Message { get; set; } = "";
        public string DeploymentToken { get; set; } = "";
        public string ApiKey { get; set; } = "";
        public string AgentId { get; set; } = "";
        public Dictionary<string, object> InitialConfiguration { get; set; } = new();
        public DateTime TokenExpiration { get; set; }
        public bool RequiresAdditionalSetup { get; set; }
    }

    // NOTE: BackendConfigurationUpdatedEventArgs has been moved to 
    // AthalaSIEM.UniversalAgent.Models.CommunicationServiceModels.cs for clean architecture separation

    /// <summary>
    /// Event arguments for auto-deployment events.
    /// </summary>
    public class AutoDeploymentEventArgs : EventArgs
    {
        public bool Success { get; set; }
        public string Message { get; set; } = "";
        public string DeploymentToken { get; set; } = "";
        public string ApiKey { get; set; } = "";
        public string Error { get; set; } = "";
        public DateTime EventTime { get; set; } = DateTime.UtcNow;
    }

    /// <summary>
    /// Event arguments for configuration fetch events.
    /// </summary>
    public class ConfigurationFetchEventArgs : EventArgs
    {
        public string ConfigurationType { get; set; } = "";
        public bool Success { get; set; }
        public Dictionary<string, object> Configuration { get; set; } = new();
        public string Error { get; set; } = "";
        public DateTime FetchTime { get; set; } = DateTime.UtcNow;
    }

    /// <summary>
    /// Represents a request to fetch configuration from backend.
    /// </summary>
    public class ConfigurationFetchRequest
    {
        public string AgentId { get; set; } = "";
        public string ConfigurationType { get; set; } = "";
        public string ApiKey { get; set; } = "";
        public Dictionary<string, object> CurrentConfiguration { get; set; } = new();
        public DateTime LastUpdated { get; set; }
        public string AgentVersion { get; set; } = "";
    }

    /// <summary>
    /// Represents a response from backend configuration fetch.
    /// </summary>
    public class ConfigurationFetchResponse
    {
        public bool Success { get; set; }
        public string Message { get; set; } = "";
        public Dictionary<string, object> Configuration { get; set; } = new();
        public DateTime ConfigurationTime { get; set; }
        public bool HasUpdates { get; set; }
        public bool RequiresRestart { get; set; }
        public string ConfigurationVersion { get; set; } = "";
    }

    /// <summary>
    /// Represents enterprise-grade Event ID configuration from backend.
    /// </summary>
    public class EventIdConfiguration
    {
        public List<string> MonitoredEventIds { get; set; } = new();
        public List<string> ExcludedEventIds { get; set; } = new();
        public Dictionary<string, int> EventIdPriorities { get; set; } = new();
        public Dictionary<string, string> EventIdCategories { get; set; } = new();
        public bool AllowAllEventIds { get; set; } = false;
        public bool EnableCustomFiltering { get; set; } = true;
        public DateTime ConfigurationTime { get; set; } = DateTime.UtcNow;
    }

    /// <summary>
    /// Represents File Integrity Monitoring configuration from backend.
    /// </summary>
    public class LegacyFIMConfiguration
    {
        public List<string> MonitoredPaths { get; set; } = new();
        public List<string> ExcludedPaths { get; set; } = new();
        public List<string> FileExtensions { get; set; } = new();
        public int ScanIntervalMinutes { get; set; } = 60;
        public bool MonitorSubdirectories { get; set; } = true;
        public bool MonitorSystemFiles { get; set; } = true;
        public bool EnableRealTimeMonitoring { get; set; } = true;
        public int MaxFileSize { get; set; } = 1024 * 1024; // 1MB
        public DateTime ConfigurationTime { get; set; } = DateTime.UtcNow;
    }

    /// <summary>
    /// Represents detection thresholds configuration from backend.
    /// </summary>
    public class DetectionThresholdsConfiguration
    {
        public Dictionary<string, int> Thresholds { get; set; } = new();
        public int TimeWindowMinutes { get; set; } = 15;
        public bool EnableAnomalyDetection { get; set; } = false;
        public bool EnableMachineLearning { get; set; } = false;
        public DateTime ConfigurationTime { get; set; } = DateTime.UtcNow;
    }

    /// <summary>
    /// Represents monitoring settings configuration from backend.
    /// </summary>
    public class MonitoringConfiguration
    {
        public int HeartbeatIntervalMinutes { get; set; } = 1;
        public int ConfigurationRefreshMinutes { get; set; } = 30;
        public int LogBatchSize { get; set; } = 100;
        public int LogBatchIntervalSeconds { get; set; } = 30;
        public bool EnableCompression { get; set; } = true;
        public bool EnableEncryption { get; set; } = false;
        public int MaxRetryAttempts { get; set; } = 3;
        public int RetryDelaySeconds { get; set; } = 10;
        public DateTime ConfigurationTime { get; set; } = DateTime.UtcNow;
    }

    /// <summary>
    /// Represents complete agent configuration from backend.
    /// </summary>
    public class AgentConfiguration
    {
        public EventIdConfiguration EventIds { get; set; } = new();
        public LegacyFIMConfiguration FileIntegrity { get; set; } = new();
        public DetectionThresholdsConfiguration DetectionThresholds { get; set; } = new();
        public MonitoringConfiguration Monitoring { get; set; } = new();
        public Dictionary<string, object> CustomSettings { get; set; } = new();
        public string ConfigurationVersion { get; set; } = "";
        public DateTime ConfigurationTime { get; set; } = DateTime.UtcNow;
        public DateTime ExpirationTime { get; set; } = DateTime.UtcNow.AddHours(24);
    }

    /// <summary>
    /// Represents deployment status for enterprise installations.
    /// </summary>
    public class DeploymentStatus
    {
        public string Status { get; set; } = ""; // "Pending", "InProgress", "Completed", "Failed"
        public string Message { get; set; } = "";
        public int ProgressPercentage { get; set; } = 0;
        public DateTime StartTime { get; set; }
        public DateTime? EndTime { get; set; }
        public List<string> Steps { get; set; } = new();
        public List<string> Errors { get; set; } = new();
        public Dictionary<string, object> Metadata { get; set; } = new();
    }

    /// <summary>
    /// Represents enterprise search request for backend.
    /// </summary>
    public class EnterpriseSearchRequest
    {
        public string SearchQuery { get; set; } = "";
        public List<string> EventIds { get; set; } = new();
        public DateTime StartTime { get; set; }
        public DateTime EndTime { get; set; }
        public string AgentId { get; set; } = "";
        public string LogLevel { get; set; } = "";
        public string Category { get; set; } = "";
        public int MaxResults { get; set; } = 1000;
        public int PageNumber { get; set; } = 1;
        public int PageSize { get; set; } = 100;
        public Dictionary<string, object> Filters { get; set; } = new();
    }

    /// <summary>
    /// Represents enterprise search response from backend.
    /// </summary>
    public class EnterpriseSearchResponse
    {
        public bool Success { get; set; }
        public string Message { get; set; } = "";
        public List<LogEntry> Results { get; set; } = new();
        public int TotalResults { get; set; } = 0;
        public int CurrentPage { get; set; } = 1;
        public int TotalPages { get; set; } = 0;
        public Dictionary<string, object> Aggregations { get; set; } = new();
        public DateTime SearchTime { get; set; } = DateTime.UtcNow;
        public int ExecutionTimeMs { get; set; } = 0;
    }
} 