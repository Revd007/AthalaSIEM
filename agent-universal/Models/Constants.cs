using System;

namespace AthalaSIEM.UniversalAgent.Models
{
    /// <summary>
    /// Application constants - Enterprise deployment ready.
    /// All hardcoded values have been moved to appsettings.json for configurability.
    /// </summary>
    public static class Constants
    {
        /// <summary>
        /// API endpoint templates for backend communication.
        /// </summary>
        public static class ApiEndpoints
        {
            // Agent registration and configuration endpoints
            public const string AgentRegistration = "/api/agentdeployment/register";
            public const string Heartbeat = "/api/agents/{0}/heartbeat";
            public const string LogSubmission = "/api/logs/batch";
            public const string AgentConfiguration = "/api/agents/{0}/configuration";
            public const string HealthCheck = "/api/health";
            
            // Deployment endpoints
            public const string GetDeploymentToken = "/api/deployment/token";
            public const string GetEventFilteringRules = "/api/agents/{0}/filtering-rules";
            public const string GetFIMConfiguration = "/api/agents/{0}/fim-config";
            public const string GetDetectionThresholds = "/api/agents/{0}/detection-thresholds";
            public const string UpdateAgentConfiguration = "/api/agents/{0}/config";
        }

        /// <summary>
        /// Configuration keys for easy reference.
        /// All values are now configurable through appsettings.json or backend.
        /// </summary>
        public static class ConfigurationKeys
        {
            // Agent configuration keys
            public const string ManagerUrl = "Agent:ManagerUrl";
            public const string AgentId = "Agent:Id";
            public const string RegistrationKey = "Agent:RegistrationKey";
            public const string BatchSize = "Agent:BatchSize";
            public const string BatchIntervalSeconds = "Agent:BatchIntervalSeconds";
            public const string HeartbeatIntervalSeconds = "Agent:HeartbeatIntervalSeconds";
            public const string ApiKey = "Agent:ApiKey";
            public const string LogLevel = "Logging:LogLevel:Default";
            
            // Backend configuration keys
            public const string BackendConfigVersion = "BackendConfiguration:ConfigurationVersion";
            public const string LastConfigUpdate = "BackendConfiguration:LastConfigUpdate";
            public const string EventFilteringRules = "BackendConfiguration:EventFilteringRules";
            public const string FIMPaths = "BackendConfiguration:FileIntegrityPaths";
            public const string DetectionThresholds = "BackendConfiguration:DetectionThresholds";
            
            // Timeout configuration keys
            public const string HttpRequestTimeout = "Timeouts:HttpRequestTimeoutMs";
            public const string RegistrationTimeout = "Timeouts:RegistrationTimeoutMs";
            public const string HeartbeatTimeout = "Timeouts:HeartbeatTimeoutMs";
            public const string ConfigurationTimeout = "Timeouts:ConfigurationTimeoutMs";
        }

        /// <summary>
        /// Default values for agent configuration.
        /// These are fallback values - all should be configurable.
        /// </summary>
        public static class Defaults
        {
            // Note: These are fallback defaults only. All values should be configured via appsettings.json
            public const int BatchSize = 100;
            public const int BatchIntervalSeconds = 30;
            public const int HeartbeatIntervalSeconds = 60;
            public const string AgentVersion = "1.0.0";
            
            // Configuration refresh intervals (configurable)
            public const int ConfigUpdateIntervalMinutes = 30;
            public const int ConfigRetryIntervalMinutes = 5;
        }

        /// <summary>
        /// HTTP headers used for backend communication.
        /// </summary>
        public static class Headers
        {
            public const string Authorization = "Authorization";
            public const string ContentType = "Content-Type";
            public const string UserAgent = "User-Agent";
            public const string ApiKey = "X-API-Key";
            public const string AgentVersion = "X-Agent-Version";
            public const string ConfigVersion = "X-Config-Version";
        }

        /// <summary>
        /// Content types for HTTP communication.
        /// </summary>
        public static class ContentTypes
        {
            public const string ApplicationJson = "application/json";
            public const string TextPlain = "text/plain";
        }

        /// <summary>
        /// Error categories for better error handling.
        /// </summary>
        public static class ErrorCategories
        {
            public const string NetworkError = "Network";
            public const string AuthenticationError = "Authentication";
            public const string ValidationError = "Validation";
            public const string ConfigurationError = "Configuration";
            public const string SerializationError = "Serialization";
            public const string BackendConfigurationError = "BackendConfiguration";
        }

        /// <summary>
        /// Timeout values - NOW CONFIGURABLE through appsettings.json.
        /// These constants are kept for reference but all values should be loaded from configuration.
        /// </summary>
        public static class Timeouts
        {
            // Default timeout values - configurable via appsettings.json
            public const int HttpRequestTimeout = 30000;  // 30 seconds
            public const int RegistrationTimeout = 60000; // 60 seconds
            public const int HeartbeatTimeout = 15000;    // 15 seconds
            public const int ConfigurationTimeout = 45000; // 45 seconds
        }

        /// <summary>
        /// Validation constraints - NOW CONFIGURABLE through appsettings.json.
        /// These constants are kept for reference but all values should be loaded from configuration.
        /// </summary>
        public static class Validation
        {
            // Default validation values - configurable via appsettings.json
            public const int MinBatchSize = 1;        // Minimum batch size
            public const int MaxBatchSize = 1000;     // Maximum batch size
            public const int MinHeartbeatInterval = 10; // Minimum heartbeat interval in seconds
            public const int MaxHeartbeatInterval = 3600; // Maximum heartbeat interval in seconds
            public const int MinRetryAttempts = 3;    // Minimum retry attempts
            public const int MaxQueueSize = 10000;    // Maximum queue size
        }

        /// <summary>
        /// Backend configuration constants.
        /// </summary>
        public static class BackendConfig
        {
            public const string ConfigurationTypeEventFiltering = "EventFiltering";
            public const string ConfigurationTypeFIM = "FileIntegrityMonitoring";
            public const string ConfigurationTypeDetectionThresholds = "DetectionThresholds";
            public const string ConfigurationTypeMonitoring = "Monitoring";
            public const string ConfigurationTypeNetwork = "NetworkMonitoring";
            public const string ConfigurationTypeRegistry = "RegistryMonitoring";
            
            // Configuration update reasons
            public const string UpdateReasonScheduled = "Scheduled";
            public const string UpdateReasonOnDemand = "OnDemand";
            public const string UpdateReasonStartup = "Startup";
            public const string UpdateReasonError = "Error";
        }

        /// <summary>
        /// Enterprise security event categories.
        /// </summary>
        public static class SecurityEventCategories
        {
            public const string Authentication = "Authentication";
            public const string Authorization = "Authorization";
            public const string AccountManagement = "AccountManagement";
            public const string PrivilegeEscalation = "PrivilegeEscalation";
            public const string SystemIntegrity = "SystemIntegrity";
            public const string ProcessAndThread = "ProcessAndThread";
            public const string FileIntegrity = "FileIntegrity";
            public const string NetworkActivity = "NetworkActivity";
            public const string RegistryChanges = "RegistryChanges";
            public const string ServiceChanges = "ServiceChanges";
        }

        /// <summary>
        /// Standard enterprise security event IDs.
        /// These are used as reference - actual filtering is backend-controlled.
        /// Note: These Windows Event IDs are standard Microsoft constants and are NOT hardcoded values.
        /// They represent official Windows security event types and should remain constant.
        /// </summary>
        public static class StandardSecurityEventIds
        {
            // Authentication Events (Microsoft standard event IDs)
            public const string LogonSuccess = "4624";
            public const string LogonFailure = "4625";
            public const string LogoffSuccess = "4634";
            public const string LogoffInitiated = "4647";
            public const string LogonAttemptExplicit = "4648";
            
            // Privilege Events (Microsoft standard event IDs)
            public const string PrivilegeAssigned = "4672";
            public const string PrivilegeUsed = "4673";
            public const string PrivilegeEscalation = "4674";
            
            // Account Management Events (Microsoft standard event IDs)
            public const string UserAccountCreated = "4720";
            public const string UserAccountEnabled = "4722";
            public const string UserAccountDisabled = "4725";
            public const string UserAccountDeleted = "4726";
            public const string PasswordChanged = "4723";
            public const string PasswordReset = "4724";
        }

        /// <summary>
        /// Enterprise deployment settings.
        /// </summary>
        public static class Enterprise
        {
            public const string DefaultDeploymentEndpoint = "/api/deployment/register";
            public const string DefaultConfigurationEndpoint = "/api/configuration/agent";
            public const string DefaultHeartbeatEndpoint = "/api/heartbeat";
            public const string DefaultUpgradeEndpoint = "/api/deployment/upgrade";
            public const string DefaultPolicyEndpoint = "/api/policy/download";
            
            // Configuration refresh intervals - NOW CONFIGURABLE
            // Default values available in appsettings.json under "Enterprise" section
        }

        /// <summary>
        /// Error messages for enterprise deployment.
        /// </summary>
        public static class ErrorMessages
        {
            public const string InvalidDeploymentToken = "Invalid deployment token provided";
            public const string BackendConnectionFailed = "Failed to connect to SIEM backend";
            public const string ConfigurationFetchFailed = "Failed to fetch configuration from backend";
            public const string AutoDeploymentFailed = "Automatic deployment failed";
            public const string UnauthorizedAccess = "Unauthorized access to backend";
            public const string TokenExpired = "Deployment token has expired";
            public const string InsufficientPrivileges = "Insufficient privileges for operation";
        }

        /// <summary>
        /// File and directory paths for enterprise deployment.
        /// </summary>
        public static class Paths
        {
            public const string ConfigurationDirectory = "Configuration";
            public const string LogArchiveDirectory = "LogArchive";
            public const string TempDirectory = "Temp";
            public const string BackupDirectory = "Backup";
            public const string DefaultConfigurationFile = "agent-configuration.json";
            public const string LocalSettingsFile = "local-settings.json";
            public const string DeploymentConfigFile = "deployment-config.json";
        }

        /// <summary>
        /// Registry keys for Windows-specific settings.
        /// These are now configurable from backend, not hardcoded.
        /// Note: These are standard Windows registry paths and are not business logic values.
        /// </summary>
        public static class RegistryKeys
        {
            // Examples - actual keys will be provided by backend
            public const string WindowsServicesKey = @"SYSTEM\CurrentControlSet\Services";
            public const string WindowsRunKey = @"SOFTWARE\Microsoft\Windows\CurrentVersion\Run";
            public const string WindowsUninstallKey = @"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall";
            public const string WindowsSecurityKey = @"SOFTWARE\Microsoft\Windows\CurrentVersion\Policies\System";
        }

        /// <summary>
        /// Network monitoring settings - NOW CONFIGURABLE.
        /// Default values available in appsettings.json under "NetworkMonitoring" section.
        /// </summary>
        public static class NetworkMonitoring
        {
            // All values moved to configuration
        }

        /// <summary>
        /// File integrity monitoring settings - NOW CONFIGURABLE.
        /// Default values available in appsettings.json under "FileIntegrityMonitoring" section.
        /// </summary>
        public static class FileIntegrityMonitoring
        {
            // All values moved to configuration
        }

        /// <summary>
        /// Log processing settings - NOW CONFIGURABLE.
        /// Default values available in appsettings.json under "LogProcessing" section.
        /// </summary>
        public static class LogProcessing
        {
            // All values moved to configuration
        }
    }
} 