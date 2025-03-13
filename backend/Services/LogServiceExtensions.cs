using System;
using System.Threading.Tasks;
using Backend.DTOs;
using Backend.Services;

namespace Backend.Services
{
    /// <summary>
    /// Extensions for ILogService
    /// </summary>
    public static class LogServiceExtensions
    {
        /// <summary>
        /// Gets logs by agent ID
        /// </summary>
        /// <param name="logService">The log service</param>
        /// <param name="agentId">The agent ID</param>
        /// <param name="count">Maximum number of logs to return</param>
        /// <param name="offset">Offset for pagination</param>
        /// <returns>Paginated logs</returns>
        public static async Task<PaginatedResult<LogEntryDto>> GetLogsByAgentIdAsync(
            this ILogService logService,
            string agentId,
            int count = 100,
            int offset = 0)
        {
            var query = new LogQueryDto
            {
                AgentId = agentId,
                Limit = count,
                Offset = offset,
                SortField = "Timestamp",
                SortDirection = "desc"
            };
            
            return await logService.SearchLogsAsync(query);
        }
    }
} 