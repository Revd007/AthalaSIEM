using System;
using System.Collections.Generic;
using System.Threading;

namespace AthalaSIEM.UniversalAgent.Models
{
    /// <summary>
    /// Represents a threat trigger that initiates an active response.
    /// </summary>
    public class ThreatTrigger
    {
        /// <summary>
        /// Gets or sets the unique identifier for this threat trigger.
        /// </summary>
        public string Id { get; set; } = "";

        /// <summary>
        /// Gets or sets the type of trigger (e.g., "BruteForce", "Malware", "Suspicious Activity").
        /// </summary>
        public string TriggerType { get; set; } = "";

        /// <summary>
        /// Gets or sets the description of the threat.
        /// </summary>
        public string Description { get; set; } = "";

        /// <summary>
        /// Gets or sets the severity level (Critical, High, Medium, Low).
        /// </summary>
        public string Severity { get; set; } = "";

        /// <summary>
        /// Gets or sets when the threat was detected.
        /// </summary>
        public DateTime DetectedAt { get; set; } = DateTime.UtcNow;

        /// <summary>
        /// Gets or sets additional properties related to the threat.
        /// </summary>
        public Dictionary<string, object> Properties { get; set; } = new();
    }

    /// <summary>
    /// Represents a response policy configuration from the backend.
    /// </summary>
    public class ResponsePolicy
    {
        /// <summary>
        /// Gets or sets the unique identifier for this policy.
        /// </summary>
        public string Id { get; set; } = "";

        /// <summary>
        /// Gets or sets the name of the policy.
        /// </summary>
        public string Name { get; set; } = "";

        /// <summary>
        /// Gets or sets the trigger type this policy responds to.
        /// </summary>
        public string TriggerType { get; set; } = "";

        /// <summary>
        /// Gets or sets the list of response actions to execute.
        /// </summary>
        public List<ResponseType> ResponseActions { get; set; } = new();

        /// <summary>
        /// Gets or sets whether this policy is enabled.
        /// </summary>
        public bool Enabled { get; set; } = true;

        /// <summary>
        /// Gets or sets the conditions that must be met for this policy to trigger.
        /// </summary>
        public Dictionary<string, object> Conditions { get; set; } = new();
    }

    /// <summary>
    /// Represents a response action to be executed.
    /// </summary>
    public class ResponseAction
    {
        /// <summary>
        /// Gets or sets the unique identifier for this action.
        /// </summary>
        public string Id { get; set; } = "";

        /// <summary>
        /// Gets or sets the threat trigger that initiated this response.
        /// </summary>
        public ThreatTrigger Trigger { get; set; } = new();

        /// <summary>
        /// Gets or sets the type of response to execute.
        /// </summary>
        public ResponseType ResponseType { get; set; }

        /// <summary>
        /// Gets or sets the parameters for the response action.
        /// </summary>
        public Dictionary<string, object> Parameters { get; set; } = new();

        /// <summary>
        /// Gets or sets when this action was queued.
        /// </summary>
        public DateTime QueuedAt { get; set; }

        /// <summary>
        /// Gets or sets when this action started executing.
        /// </summary>
        public DateTime? StartedAt { get; set; }

        /// <summary>
        /// Gets or sets when this action completed.
        /// </summary>
        public DateTime? CompletedAt { get; set; }

        /// <summary>
        /// Gets or sets the current status of this action.
        /// </summary>
        public ResponseStatus Status { get; set; }

        /// <summary>
        /// Gets or sets the result of the action execution.
        /// </summary>
        public ResponseResult? Result { get; set; }
    }

    /// <summary>
    /// Represents the result of a response action execution.
    /// </summary>
    public class ResponseResult
    {
        /// <summary>
        /// Gets or sets whether the action was successful.
        /// </summary>
        public bool Success { get; set; }

        /// <summary>
        /// Gets or sets the success or informational message.
        /// </summary>
        public string Message { get; set; } = "";

        /// <summary>
        /// Gets or sets the error message if the action failed.
        /// </summary>
        public string Error { get; set; } = "";

        /// <summary>
        /// Gets or sets additional details about the execution.
        /// </summary>
        public Dictionary<string, object> Details { get; set; } = new();
    }

    /// <summary>
    /// Represents an active response execution in progress.
    /// </summary>
    public class ResponseExecution
    {
        /// <summary>
        /// Gets or sets the response action being executed.
        /// </summary>
        public ResponseAction Action { get; set; } = new();

        /// <summary>
        /// Gets or sets when the execution started.
        /// </summary>
        public DateTime StartTime { get; set; }

        /// <summary>
        /// Gets or sets the cancellation token for this execution.
        /// </summary>
        public CancellationTokenSource CancellationToken { get; set; } = new();
    }

    /// <summary>
    /// Represents the health status of the Active Response Service.
    /// </summary>
    public class ActiveResponseHealth
    {
        /// <summary>
        /// Gets or sets whether the service is active.
        /// </summary>
        public bool IsActive { get; set; }

        /// <summary>
        /// Gets or sets the number of queued responses.
        /// </summary>
        public int QueuedResponses { get; set; }

        /// <summary>
        /// Gets or sets the number of active responses.
        /// </summary>
        public int ActiveResponses { get; set; }

        /// <summary>
        /// Gets or sets the total number of responses executed.
        /// </summary>
        public long TotalResponsesExecuted { get; set; }

        /// <summary>
        /// Gets or sets the total number of response failures.
        /// </summary>
        public long TotalResponseFailures { get; set; }

        /// <summary>
        /// Gets or sets the number of loaded policies.
        /// </summary>
        public int LoadedPolicies { get; set; }

        /// <summary>
        /// Gets or sets the maximum concurrent responses allowed.
        /// </summary>
        public int MaxConcurrentResponses { get; set; }

        /// <summary>
        /// Gets or sets when this health check was performed.
        /// </summary>
        public DateTime LastHealthCheck { get; set; }
    }

    /// <summary>
    /// Event arguments for response execution completion.
    /// </summary>
    public class ResponseExecutedEventArgs : EventArgs
    {
        /// <summary>
        /// Gets or sets the executed action.
        /// </summary>
        public ResponseAction Action { get; set; } = new();

        /// <summary>
        /// Gets or sets the execution result.
        /// </summary>
        public ResponseResult Result { get; set; } = new();

        /// <summary>
        /// Gets or sets when the execution completed.
        /// </summary>
        public DateTime ExecutionTime { get; set; }
    }

    /// <summary>
    /// Event arguments for response execution errors.
    /// </summary>
    public class ResponseErrorEventArgs : EventArgs
    {
        /// <summary>
        /// Gets or sets the action that failed.
        /// </summary>
        public ResponseAction Action { get; set; } = new();

        /// <summary>
        /// Gets or sets the error message.
        /// </summary>
        public string Error { get; set; } = "";

        /// <summary>
        /// Gets or sets when the error occurred.
        /// </summary>
        public DateTime ErrorTime { get; set; }
    }

    /// <summary>
    /// Types of responses that can be executed.
    /// </summary>
    public enum ResponseType
    {
        BlockIpAddress,
        TerminateProcess,
        QuarantineFile,
        DisableUserAccount,
        IsolateHost,
        CustomScript,
        SendAlert
    }

    /// <summary>
    /// Status of a response action.
    /// </summary>
    public enum ResponseStatus
    {
        Queued,
        Executing,
        Completed,
        Failed,
        Cancelled
    }
} 
