using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using AthalaSIEM.UniversalAgent.Models;

namespace AthalaSIEM.UniversalAgent.Core
{
    /// <summary>
    /// Main agent service interface that orchestrates log collection, forwarding, and health monitoring.
    /// Follows ManageEngine EventLog Analyzer pattern for centralized agent management.
    /// </summary>
    public interface IAgentService : IAsyncDisposable
    {
        /// <summary>
        /// Gets the unique identifier for this agent instance
        /// </summary>
        string AgentId { get; }

        /// <summary>
        /// Gets the friendly name of this agent
        /// </summary>
        string AgentName { get; }

        /// <summary>
        /// Gets the current status of the agent
        /// </summary>
        AgentStatus Status { get; }

        /// <summary>
        /// Gets the agent version
        /// </summary>
        string Version { get; }

        /// <summary>
        /// Gets the time when the agent was started
        /// </summary>
        DateTime StartTime { get; }

        /// <summary>
        /// Gets the collection of registered log collectors
        /// </summary>
        IReadOnlyList<ILogCollector> Collectors { get; }

        /// <summary>
        /// Initializes the agent service with configuration
        /// </summary>
        /// <param name="config">Agent configuration</param>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>True if initialization was successful</returns>
        Task<bool> InitializeAsync(AgentConfiguration config, CancellationToken cancellationToken = default);

        /// <summary>
        /// Starts the agent service and begins log collection
        /// </summary>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>Task representing the start operation</returns>
        Task StartAsync(CancellationToken cancellationToken = default);

        /// <summary>
        /// Stops the agent service gracefully
        /// </summary>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>Task representing the stop operation</returns>
        Task StopAsync(CancellationToken cancellationToken = default);

        /// <summary>
        /// Registers a log collector with the agent
        /// </summary>
        /// <param name="collector">Log collector to register</param>
        /// <returns>True if registration was successful</returns>
        Task<bool> RegisterCollectorAsync(ILogCollector collector);

        /// <summary>
        /// Unregisters a log collector from the agent
        /// </summary>
        /// <param name="collectorName">Name of the collector to unregister</param>
        /// <returns>True if unregistration was successful</returns>
        Task<bool> UnregisterCollectorAsync(string collectorName);

        /// <summary>
        /// Gets comprehensive health status of the agent and all collectors
        /// </summary>
        /// <returns>Agent health information</returns>
        Task<AgentHealth> GetHealthAsync();

        /// <summary>
        /// Sends a heartbeat to the backend server
        /// </summary>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>True if heartbeat was successful</returns>
        Task<bool> SendHeartbeatAsync(CancellationToken cancellationToken = default);

        /// <summary>
        /// Tests connectivity to the backend server
        /// </summary>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>Connection test result</returns>
        Task<ConnectionTestResult> TestConnectionAsync(CancellationToken cancellationToken = default);

        /// <summary>
        /// Event fired when the agent status changes
        /// </summary>
        event EventHandler<AgentStatusChangedEventArgs> StatusChanged;

        /// <summary>
        /// Event fired when logs are forwarded to the backend
        /// </summary>
        event EventHandler<LogsForwardedEventArgs> LogsForwarded;

        /// <summary>
        /// Event fired when an error occurs in the agent
        /// </summary>
        event EventHandler<AgentErrorEventArgs> AgentError;
    }


} 
