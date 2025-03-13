using AthalaSIEM.Agent.Models;
using System.Threading.Tasks;

namespace AthalaSIEM.Agent.Services
{
    /// <summary>
    /// Interface for monitoring agent health metrics
    /// </summary>
    public interface IAgentHealthMonitor
    {
        /// <summary>
        /// Starts health monitoring
        /// </summary>
        void StartMonitoring();

        /// <summary>
        /// Gets the current health status
        /// </summary>
        /// <returns>Current health status</returns>
        Task<AgentHeartbeat> GetCurrentHealthStatus();

        /// <summary>
        /// Gets detailed system metrics
        /// </summary>
        /// <returns>System metrics</returns>
        Task<SystemMetrics> GetSystemMetrics();

        /// <summary>
        /// Generates a health report
        /// </summary>
        /// <returns>Health report</returns>
        Task<AgentHealthReport> GenerateHealthReport();
    }
} 