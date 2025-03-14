using System;
using System.ComponentModel.DataAnnotations;
using System.Text.Json;

namespace AthalaSIEM.Backend.Models
{
    public class AgentDeploymentToken
    {
        [Key]
        public string Token { get; set; } = string.Empty;

        public string CreatedById { get; set; } = string.Empty;

        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

        public DateTime ExpiresAt { get; set; }

        public bool IsUsed { get; set; }

        public DateTime? UsedAt { get; set; }

        public string? UsedByAgentId { get; set; }

        // Pre-configured settings for the agent
        public string IpAddress { get; set; } = string.Empty;
        
        public int Port { get; set; } = 443;
        
        public string? AgentName { get; set; }
        
        public bool UseSSL { get; set; } = true;

        // Store collectors configuration as JSON
        public string CollectorsJson { get; set; } = "[]";

        // Helper methods for collectors
        public void SetCollectors(string[] collectors)
        {
            CollectorsJson = JsonSerializer.Serialize(collectors);
        }

        public string[] GetCollectors()
        {
            try
            {
                return JsonSerializer.Deserialize<string[]>(CollectorsJson) ?? Array.Empty<string>();
            }
            catch
            {
                return Array.Empty<string>();
            }
        }
    }
} 