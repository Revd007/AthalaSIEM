using AthalaSIEM.Agent.Models;
using System.Threading.Tasks;

namespace AthalaSIEM.Agent.Communication
{
    /// <summary>
    /// Interface for forwarding logs to the backend
    /// </summary>
    public interface ILogForwarder
    {
        /// <summary>
        /// Forwards a normalized log entry to the backend
        /// </summary>
        /// <param name="logEntry">The log entry to forward</param>
        /// <returns>A task representing the asynchronous operation</returns>
        Task ForwardLogAsync(NormalizedLogEntry logEntry);

        /// <summary>
        /// Forwards a batch of normalized log entries to the backend
        /// </summary>
        /// <param name="logEntries">The log entries to forward</param>
        /// <returns>A task representing the asynchronous operation</returns>
        Task ForwardLogBatchAsync(NormalizedLogEntry[] logEntries);

        /// <summary>
        /// Sends a heartbeat to the backend
        /// </summary>
        /// <param name="heartbeatData">The heartbeat data to send</param>
        /// <returns>A task representing the asynchronous operation</returns>
        Task SendHeartbeatAsync(AgentHeartbeat heartbeatData);

        /// <summary>
        /// Gets agent configuration from the backend
        /// </summary>
        /// <returns>The agent configuration</returns>
        Task<AgentSettings> GetAgentConfigurationAsync();

        /// <summary>
        /// Sends system metrics to the backend
        /// </summary>
        /// <param name="metrics">The system metrics to send</param>
        /// <returns>A task representing the asynchronous operation</returns>
        Task SendSystemMetricsAsync(SystemMetrics metrics);

        /// <summary>
        /// Sends a health report to the backend
        /// </summary>
        /// <param name="healthReport">The health report to send</param>
        /// <returns>A task representing the asynchronous operation</returns>
        Task SendHealthReportAsync(AgentHealthReport healthReport);
    }
} 