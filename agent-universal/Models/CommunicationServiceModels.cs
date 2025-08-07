using System;
using System.Collections.Generic;

namespace AthalaSIEM.UniversalAgent.Models
{
    /// <summary>
    /// Represents a deployment token response from the backend.
    /// </summary>
    public sealed class DeploymentTokenResponse
    {
        /// <summary>
        /// Gets or sets the deployment token.
        /// </summary>
        public string Token { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets when the token expires.
        /// </summary>
        public DateTime ExpiresAt { get; set; }

        /// <summary>
        /// Gets or sets the agent version this token is for.
        /// </summary>
        public string AgentVersion { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the initial configuration for the agent.
        /// </summary>
        public Dictionary<string, object> InitialConfiguration { get; set; } = new();
    }

    /// <summary>
    /// Represents the result of fetching configuration from the backend.
    /// </summary>
    public sealed class BackendConfigResult
    {
        /// <summary>
        /// Gets or sets whether the configuration fetch was successful.
        /// </summary>
        public bool Success { get; set; }

        /// <summary>
        /// Gets or sets the type of configuration fetched.
        /// </summary>
        public string ConfigType { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the configuration data.
        /// </summary>
        public Dictionary<string, object> Configuration { get; set; } = new();

        /// <summary>
        /// Gets or sets the error message if the fetch failed.
        /// </summary>
        public string Error { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets when the configuration was fetched.
        /// </summary>
        public DateTime FetchTime { get; set; }
    }

    /// <summary>
    /// Event arguments for backend configuration updates.
    /// </summary>
    public sealed class BackendConfigurationUpdatedEventArgs : EventArgs
    {
        /// <summary>
        /// Gets or sets the updated configurations.
        /// </summary>
        public List<BackendConfigResult> UpdatedConfigurations { get; set; } = new();

        /// <summary>
        /// Gets or sets when the update occurred.
        /// </summary>
        public DateTime UpdateTime { get; set; }

        /// <summary>
        /// Gets or sets the configuration version.
        /// </summary>
        public string ConfigurationVersion { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the reason for the update.
        /// </summary>
        public string UpdateReason { get; set; } = Constants.BackendConfig.UpdateReasonScheduled;
    }

    /// <summary>
    /// Represents authentication status for Windows authentication service.
    /// </summary>
    public class AuthenticationStatus
    {
        /// <summary>
        /// Gets or sets whether the agent is authenticated.
        /// </summary>
        public bool IsAuthenticated { get; set; }

        /// <summary>
        /// Gets or sets whether the agent has administrative privileges.
        /// </summary>
        public bool HasAdminPrivileges { get; set; }

        /// <summary>
        /// Gets or sets the current user name.
        /// </summary>
        public string CurrentUser { get; set; } = "";

        /// <summary>
        /// Gets or sets the service account name.
        /// </summary>
        public string ServiceAccount { get; set; } = "";

        /// <summary>
        /// Gets or sets when authentication was performed.
        /// </summary>
        public DateTime AuthenticationTime { get; set; }

        /// <summary>
        /// Gets or sets whether the agent can access the security log.
        /// </summary>
        public bool CanAccessSecurityLog { get; set; }

        /// <summary>
        /// Gets or sets whether the agent can access the registry.
        /// </summary>
        public bool CanAccessRegistry { get; set; }

        /// <summary>
        /// Gets or sets whether the agent can access the file system.
        /// </summary>
        public bool CanAccessFileSystem { get; set; }

        /// <summary>
        /// Gets or sets whether the agent requires elevation.
        /// </summary>
        public bool RequiresElevation { get; set; }
    }

    /// <summary>
    /// Event arguments for logs sent to backend.
    /// </summary>
    public class LogsSentEventArgs : EventArgs
    {
        /// <summary>
        /// Gets or sets the number of logs sent.
        /// </summary>
        public int LogCount { get; set; }

        /// <summary>
        /// Gets or sets when the logs were sent.
        /// </summary>
        public DateTime SentAt { get; set; }

        /// <summary>
        /// Gets or sets the batch size.
        /// </summary>
        public int BatchSize { get; set; }

        /// <summary>
        /// Gets or sets whether the send was successful.
        /// </summary>
        public bool Success { get; set; }

        /// <summary>
        /// Gets or sets the processing duration for this batch.
        /// </summary>
        public TimeSpan ProcessingDuration { get; set; }
    }

    /// <summary>
    /// Event arguments for communication errors.
    /// </summary>
    public class CommunicationErrorEventArgs : EventArgs
    {
        /// <summary>
        /// Gets or sets the error message.
        /// </summary>
        public string Message { get; set; } = "";

        /// <summary>
        /// Gets or sets the error message (legacy compatibility).
        /// </summary>
        public string ErrorMessage 
        { 
            get => Message; 
            set => Message = value; 
        }

        /// <summary>
        /// Gets or sets the exception that occurred.
        /// </summary>
        public Exception? Exception { get; set; }

        /// <summary>
        /// Gets or sets when the error occurred.
        /// </summary>
        public DateTime OccurredAt { get; set; }

        /// <summary>
        /// Gets or sets when the error occurred (legacy compatibility).
        /// </summary>
        public DateTime ErrorTime 
        { 
            get => OccurredAt; 
            set => OccurredAt = value; 
        }

        /// <summary>
        /// Gets or sets the error category.
        /// </summary>
        public string Category { get; set; } = "";

        /// <summary>
        /// Gets or sets the error category (legacy compatibility).
        /// </summary>
        public string ErrorCategory 
        { 
            get => Category; 
            set => Category = value; 
        }

        /// <summary>
        /// Gets or sets the log count related to this error.
        /// </summary>
        public int LogCount { get; set; }

        /// <summary>
        /// Gets or sets whether this error is retryable.
        /// </summary>
        public bool IsRetryable { get; set; } = true;
    }

    /// <summary>
    /// Event arguments for connection status changes.
    /// </summary>
    public class ConnectionStatusChangedEventArgs : EventArgs
    {
        /// <summary>
        /// Gets or sets whether the connection is active.
        /// </summary>
        public bool IsConnected { get; set; }

        /// <summary>
        /// Gets or sets when the status changed.
        /// </summary>
        public DateTime StatusChangeTime { get; set; }

        /// <summary>
        /// Gets or sets when the status changed (legacy compatibility).
        /// </summary>
        public DateTime StatusTime 
        { 
            get => StatusChangeTime; 
            set => StatusChangeTime = value; 
        }

        /// <summary>
        /// Gets or sets the previous connection status.
        /// </summary>
        public bool PreviousStatus { get; set; }

        /// <summary>
        /// Gets or sets additional context about the status change.
        /// </summary>
        public string Context { get; set; } = "";

        /// <summary>
        /// Gets or sets the status message (legacy compatibility).
        /// </summary>
        public string StatusMessage 
        { 
            get => Context; 
            set => Context = value; 
        }
    }

    /// <summary>
    /// Represents the health status of communication services.
    /// </summary>
    public class CommunicationHealth
    {
        /// <summary>
        /// Gets or sets whether the communication service is healthy.
        /// </summary>
        public bool IsHealthy { get; set; }

        /// <summary>
        /// Gets or sets whether the service is connected to the backend.
        /// </summary>
        public bool IsConnected { get; set; }

        /// <summary>
        /// Gets or sets the manager URL being used for communication.
        /// </summary>
        public string ManagerUrl { get; set; } = "";

        /// <summary>
        /// Gets or sets the number of queued logs.
        /// </summary>
        public long QueuedLogs { get; set; }

        /// <summary>
        /// Gets or sets the total number of logs sent.
        /// </summary>
        public long TotalLogsSent { get; set; }

        /// <summary>
        /// Gets or sets the total number of send errors.
        /// </summary>
        public long TotalSendErrors { get; set; }

        /// <summary>
        /// Gets or sets when the last successful send occurred.
        /// </summary>
        public DateTime LastSuccessfulSend { get; set; }

        /// <summary>
        /// Gets or sets when this health check was performed.
        /// </summary>
        public DateTime LastHealthCheck { get; set; }

        /// <summary>
        /// Gets or sets additional health metrics.
        /// </summary>
        public Dictionary<string, object> Metrics { get; set; } = new();
    }
} 
