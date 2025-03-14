using System;
using System.Threading.Tasks;
using AthalaSIEM.Backend.Models;

namespace AthalaSIEM.Backend.Repositories
{
    public interface IAgentDeploymentTokenRepository
    {
        Task<AgentDeploymentToken> CreateTokenAsync(AgentDeploymentToken token);
        Task<AgentDeploymentToken?> GetTokenAsync(string token);
        Task<bool> MarkTokenAsUsedAsync(string token, string agentId);
        Task<bool> DeleteTokenAsync(string token);
        Task CleanupExpiredTokensAsync();
    }
} 