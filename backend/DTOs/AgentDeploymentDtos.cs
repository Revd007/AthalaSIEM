using System;
using System.Collections.Generic;

namespace Backend.DTOs
{
    public class CreateAgentDeploymentTokenRequest
    {
        public string Name { get; set; } = string.Empty;
        public string Description { get; set; } = string.Empty;
        public string PlatformType { get; set; } = string.Empty;
        public DateTime? ExpiresAt { get; set; }
        public int? MaxUsage { get; set; }
        public Dictionary<string, object> Configuration { get; set; } = new();
    }

    public class AgentDeploymentTokenDto
    {
        public string Id { get; set; } = string.Empty;
        public string Name { get; set; } = string.Empty;
        public string Description { get; set; } = string.Empty;
        public string PlatformType { get; set; } = string.Empty;
        public string? Token { get; set; }
        public DateTime? ExpiresAt { get; set; }
        public bool IsActive { get; set; }
        public int UsageCount { get; set; }
        public int? MaxUsage { get; set; }
        public DateTime CreatedAt { get; set; }
        public string? CreatedBy { get; set; }
        public DateTime? LastUsed { get; set; }
    }

    public class AgentDeploymentScriptResponse
    {
        public string Platform { get; set; } = string.Empty;
        public string Script { get; set; } = string.Empty;
        public List<string> Instructions { get; set; } = new();
        public object ConfigurationTemplate { get; set; } = new();
        public List<string> Prerequisites { get; set; } = new();
    }

    public class AgentRegistrationRequest
    {
        public string DeploymentToken { get; set; } = string.Empty;
        public string Hostname { get; set; } = string.Empty;
        public string IpAddress { get; set; } = string.Empty;
        public string Platform { get; set; } = string.Empty;
        public string OsVersion { get; set; } = string.Empty;
        public string AgentVersion { get; set; } = string.Empty;
        public Dictionary<string, string> SystemInfo { get; set; } = new();
    }

    public class AgentRegistrationResponse
    {
        public string AgentId { get; set; } = string.Empty;
        public string ApiKey { get; set; } = string.Empty;
        public string BackendUrl { get; set; } = string.Empty;
        public string Configuration { get; set; } = string.Empty;
        public int UpdateIntervalSeconds { get; set; }
        public int HeartbeatIntervalSeconds { get; set; }
    }

    public class AgentConfigurationResponse
    {
        public string AgentId { get; set; } = string.Empty;
        public string Configuration { get; set; } = string.Empty;
        public DateTime LastUpdated { get; set; }
        public bool RequiresRestart { get; set; }
    }

    public class UpdateAgentConfigurationRequest
    {
        public Dictionary<string, object> Configuration { get; set; } = new();
        public bool RequiresRestart { get; set; } = false;
    }

    public class DeploymentStatistics
    {
        public int TotalTokens { get; set; }
        public int ActiveTokens { get; set; }
        public int TotalDeployments { get; set; }
        public int OnlineAgents { get; set; }
        public int OfflineAgents { get; set; }
        public List<PlatformCount> PlatformDistribution { get; set; } = new();
        public List<DeploymentTrend> RecentDeployments { get; set; } = new();
    }

    public class PlatformCount
    {
        public string Platform { get; set; } = string.Empty;
        public int Count { get; set; }
    }

    public class DeploymentTrend
    {
        public DateTime Date { get; set; }
        public int Count { get; set; }
    }

    public class GenerateInstallerRequest
    {
        public string TokenId { get; set; } = string.Empty;
        public string Platform { get; set; } = string.Empty;
        public Dictionary<string, object>? CustomConfiguration { get; set; }
    }

    public class AgentInstallerResponse
    {
        public string Platform { get; set; } = string.Empty;
        public string InstallerUrl { get; set; } = string.Empty;
        public string ChecksumSha256 { get; set; } = string.Empty;
        public DateTime ExpiresAt { get; set; }
        public List<string> Instructions { get; set; } = new();
    }
}
