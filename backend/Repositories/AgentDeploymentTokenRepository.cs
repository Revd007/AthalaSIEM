using System;
using System.Threading.Tasks;
using AthalaSIEM.Backend.Models;
using Backend.Data;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging;

namespace AthalaSIEM.Backend.Repositories
{
    public class AgentDeploymentTokenRepository : IAgentDeploymentTokenRepository
    {
        private readonly ApplicationDbContext _context;
        private readonly ILogger<AgentDeploymentTokenRepository> _logger;

        public AgentDeploymentTokenRepository(
            ApplicationDbContext context,
            ILogger<AgentDeploymentTokenRepository> logger)
        {
            _context = context;
            _logger = logger;
        }

        public async Task<AgentDeploymentToken> CreateTokenAsync(AgentDeploymentToken token)
        {
            try
            {
                _context.AgentDeploymentTokens.Add(token);
                await _context.SaveChangesAsync();
                return token;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error creating deployment token");
                throw;
            }
        }

        public async Task<AgentDeploymentToken?> GetTokenAsync(string token)
        {
            try
            {
                return await _context.AgentDeploymentTokens
                    .FirstOrDefaultAsync(t => t.Token == token && !t.IsUsed && t.ExpiresAt > DateTime.UtcNow);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting deployment token");
                throw;
            }
        }

        public async Task<bool> MarkTokenAsUsedAsync(string token, string agentId)
        {
            try
            {
                var deploymentToken = await _context.AgentDeploymentTokens
                    .FirstOrDefaultAsync(t => t.Token == token);

                if (deploymentToken == null)
                    return false;

                deploymentToken.IsUsed = true;
                deploymentToken.UsedAt = DateTime.UtcNow;
                deploymentToken.UsedByAgentId = agentId;

                await _context.SaveChangesAsync();
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error marking token as used");
                throw;
            }
        }

        public async Task<bool> DeleteTokenAsync(string token)
        {
            try
            {
                var deploymentToken = await _context.AgentDeploymentTokens
                    .FirstOrDefaultAsync(t => t.Token == token);

                if (deploymentToken == null)
                    return false;

                _context.AgentDeploymentTokens.Remove(deploymentToken);
                await _context.SaveChangesAsync();
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error deleting deployment token");
                throw;
            }
        }

        public async Task CleanupExpiredTokensAsync()
        {
            try
            {
                // Delete tokens that are either used or expired
                var tokensToDelete = await _context.AgentDeploymentTokens
                    .Where(t => t.IsUsed || t.ExpiresAt <= DateTime.UtcNow)
                    .ToListAsync();

                if (tokensToDelete.Any())
                {
                    _context.AgentDeploymentTokens.RemoveRange(tokensToDelete);
                    await _context.SaveChangesAsync();
                    _logger.LogInformation("Cleaned up {Count} expired deployment tokens", tokensToDelete.Count);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error cleaning up expired tokens");
                throw;
            }
        }
    }
} 