namespace AthalaSIEM.UniversalAgent.Models
{
    /// <summary>
    /// Application constants to avoid magic strings and hardcoded values.
    /// </summary>
    public static class Constants
    {
        /// <summary>
        /// API endpoint constants.
        /// </summary>
        public static class ApiEndpoints
        {
            public const string AgentRegistration = "/api/agentdeployment/register";
            public const string Heartbeat = "/api/agents/{0}/heartbeat";
            public const string LogSubmission = "/api/logs/batch";
            public const string AgentConfiguration = "/api/agents/{0}/configuration";
            public const string HealthCheck = "/api/health";
        }

        /// <summary>
        /// Configuration keys.
        /// </summary>
        public static class ConfigurationKeys
        {
            public const string ManagerUrl = "Agent:ManagerUrl";
            public const string AgentId = "Agent:Id";
            public const string RegistrationKey = "Agent:RegistrationKey";
            public const string BatchSize = "Agent:BatchSize";
            public const string BatchIntervalSeconds = "Agent:BatchIntervalSeconds";
            public const string HeartbeatIntervalSeconds = "Agent:HeartbeatIntervalSeconds";
            public const string ApiKey = "Agent:ApiKey";
            public const string LogLevel = "Logging:LogLevel:Default";
        }

        /// <summary>
        /// Default values for configuration.
        /// </summary>
        public static class Defaults
        {
            public const int BatchSize = 100;
            public const int BatchIntervalSeconds = 30;
            public const int HeartbeatIntervalSeconds = 60;
            public const string RegistrationKey = "athala-siem-agent-registration-2025";
            public const string DefaultManagerUrl = "http://localhost:9595";
            public const string AgentVersion = "1.0.0";
        }

        /// <summary>
        /// HTTP headers.
        /// </summary>
        public static class Headers
        {
            public const string Authorization = "Authorization";
            public const string ContentType = "Content-Type";
            public const string UserAgent = "User-Agent";
            public const string ApiKey = "X-API-Key";
        }

        /// <summary>
        /// Content types.
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
        }

        /// <summary>
        /// Timeout values in milliseconds.
        /// </summary>
        public static class Timeouts
        {
            public const int HttpRequestTimeout = 30000; // 30 seconds
            public const int RegistrationTimeout = 60000; // 60 seconds
            public const int HeartbeatTimeout = 15000; // 15 seconds
        }

        /// <summary>
        /// Validation constraints.
        /// </summary>
        public static class Validation
        {
            public const int MinBatchSize = 1;
            public const int MaxBatchSize = 1000;
            public const int MinIntervalSeconds = 10;
            public const int MaxIntervalSeconds = 3600;
            public const int MaxRetryAttempts = 3;
            public const int MaxQueueSize = 10000;
        }
    }
} 