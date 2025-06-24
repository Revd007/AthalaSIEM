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

    /// <summary>
    /// Health status model for backend communication service.
    /// </summary>
    public sealed class CommunicationHealth
    {
        public bool IsConnected { get; init; }
        public string ManagerUrl { get; init; } = string.Empty;
        public long QueuedLogs { get; init; }
        public long TotalLogsSent { get; init; }
        public long TotalSendErrors { get; init; }
        public DateTime LastSuccessfulSend { get; init; }
        public DateTime LastHealthCheck { get; init; } = DateTime.UtcNow;

        /// <summary>
        /// Calculates the health score based on connection status and error rate.
        /// </summary>
        /// <returns>Health score between 0.0 and 1.0.</returns>
        public double GetHealthScore()
        {
            if (!IsConnected) return 0.0;
            
            var totalOperations = TotalLogsSent + TotalSendErrors;
            if (totalOperations == 0) return 1.0;
            
            var successRate = (double)TotalLogsSent / totalOperations;
            return Math.Max(0.0, Math.Min(1.0, successRate));
        }
    }
} 