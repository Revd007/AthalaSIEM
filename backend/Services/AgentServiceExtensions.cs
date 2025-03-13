using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Backend.DTOs;
using Backend.Services;

namespace Backend.Services
{
    /// <summary>
    /// Extensions for IAgentService
    /// </summary>
    public static class AgentServiceExtensions
    {
        /// <summary>
        /// Gets agent health reports
        /// </summary>
        /// <param name="agentService">The agent service</param>
        /// <param name="agentId">The agent ID</param>
        /// <param name="count">Maximum number of reports to return</param>
        /// <param name="offset">Offset for pagination</param>
        /// <returns>Paginated health reports</returns>
        public static async Task<PaginatedResult<AgentHealthReportDto>> GetAgentHealthReportsAsync(
            this IAgentService agentService,
            string agentId,
            int count = 10,
            int offset = 0)
        {
            // Implementation depends on the actual service methods
            // This is a placeholder that should be replaced with actual implementation
            var reports = await agentService.GetAgentHealthHistoryAsync(agentId, count, offset);
            
            return new PaginatedResult<AgentHealthReportDto>
            {
                Items = reports.Items,
                TotalCount = reports.TotalCount,
                Page = (offset / count) + 1,
                PageSize = count,
                TotalPages = (int)Math.Ceiling((double)reports.TotalCount / count),
                HasPreviousPage = offset > 0,
                HasNextPage = reports.Items.Count() == count
            };
        }
        
        /// <summary>
        /// Rotates the API key for an agent
        /// </summary>
        /// <param name="agentService">The agent service</param>
        /// <param name="agentId">The agent ID</param>
        /// <returns>The new API key</returns>
        public static async Task<string> RotateApiKeyAsync(
            this IAgentService agentService,
            string agentId)
        {
            // Implementation depends on the actual service methods
            // This is a placeholder that should be replaced with actual implementation
            return await agentService.UpdateAgentApiKeyAsync(agentId);
        }
    }
} 