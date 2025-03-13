using Backend.Data;
using Backend.DTOs;
using Backend.Models;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging;

namespace Backend.Services
{
    /// <summary>
    /// Service responsible for handling log-related operations
    /// </summary>
    public class LogService : ILogService
    {
        private readonly ApplicationDbContext _context;
        private readonly ILogger<LogService> _logger;

        public LogService(ApplicationDbContext context, ILogger<LogService> logger)
        {
            _context = context ?? throw new ArgumentNullException(nameof(context));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }

        /// <summary>
        /// Processes a single log entry
        /// </summary>
        /// <param name="logRequest">The log data</param>
        /// <param name="hostname">The source hostname</param>
        /// <param name="ip">The source IP address</param>
        /// <returns>The ID of the created log entry, or null if failed</returns>
        public async Task<string?> IngestLogAsync(LogIngestRequest logRequest, string hostname, string ip)
        {
            if (logRequest == null)
            {
                throw new ArgumentNullException(nameof(logRequest));
            }

            try
            {
                // Find or create an agent for this log source
                var agent = await _context.Agents
                    .FirstOrDefaultAsync(a => a.Hostname == hostname && a.IPAddress == ip);

                if (agent == null)
                {
                    // Create a new agent if one doesn't exist
                    agent = new AgentModels
                    {
                        Id = Guid.NewGuid().ToString(),
                        Hostname = hostname,
                        IPAddress = ip,
                        OperatingSystem = "Unknown", // Default OS
                        Status = AgentStatus.Active,
                        LastHeartbeat = DateTime.UtcNow,
                        CreatedAt = DateTime.UtcNow,
                        Port = 514 // Default port
                    };

                    _context.Agents.Add(agent);
                }

                // Create the security event
                var securityEvent = new SecurityEventModels
                {
                    Id = Guid.NewGuid().ToString(),
                    AgentId = agent.Id,
                    Agent = agent,
                    Timestamp = DateTime.UtcNow,
                    LogSource = logRequest.LogSource,
                    Severity = (AlertSeverityModels)logRequest.Severity,
                    RawLog = logRequest.RawLog
                };

                // Update agent's last heartbeat
                agent.LastHeartbeat = DateTime.UtcNow;

                // Save changes
                _context.SecurityEvents.Add(securityEvent);
                await _context.SaveChangesAsync();

                _logger.LogInformation("Log ingested: {LogId} from {Hostname}", securityEvent.Id, hostname);

                return securityEvent.Id;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error ingesting log from {Hostname}: {Message}", hostname, ex.Message);
                return null;
            }
        }

        /// <summary>
        /// Processes a batch of logs from an agent
        /// </summary>
        /// <param name="logs">The collection of logs</param>
        /// <param name="agentId">The agent ID</param>
        /// <returns>The number of logs processed</returns>
        public async Task<int> ProcessLogBatchAsync(IEnumerable<LogEntryRequest> logs, string agentId)
        {
            if (logs == null)
            {
                throw new ArgumentNullException(nameof(logs));
            }

            try
            {
                var agent = await _context.Agents.FindAsync(agentId);
                if (agent == null)
                {
                    _logger.LogWarning("Attempted to process logs for unknown agent: {AgentId}", agentId);
                    return 0;
                }

                int count = 0;
                var logEntries = new List<LogEntryModels>();

                foreach (var log in logs)
                {
                    var logEntry = new LogEntryModels
                    {
                        Id = Guid.NewGuid().ToString(),
                        AgentId = agentId,
                        Agent = agent,
                        Timestamp = log.Timestamp,
                        Source = log.Source,
                        Level = log.Level,
                        Message = log.Message,
                        MachineName = agent.Hostname,
                        IPAddress = agent.IPAddress,
                        EventId = log.EventId
                    };

                    logEntries.Add(logEntry);
                    count++;
                }

                // Update agent's last heartbeat
                agent.LastHeartbeat = DateTime.UtcNow;

                // Add all log entries
                _context.LogEntries.AddRange(logEntries);
                await _context.SaveChangesAsync();

                _logger.LogInformation("Processed {Count} logs from agent {AgentId}", count, agentId);

                return count;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error processing log batch from agent {AgentId}: {Message}", agentId, ex.Message);
                return 0;
            }
        }

        /// <summary>
        /// Converts LogSeverity enum to Severity enum
        /// </summary>
        public SeverityModels ConvertLogSeverityToSeverity(LogSeverityModels logSeverity)
        {
            return logSeverity switch
            {
                LogSeverityModels.Debug => SeverityModels.Low,
                LogSeverityModels.Information => SeverityModels.Low,
                LogSeverityModels.Warning => SeverityModels.Medium,
                LogSeverityModels.Error => SeverityModels.High,
                LogSeverityModels.Critical => SeverityModels.Critical,
                _ => SeverityModels.Low
            };
        }

        public Task<PaginatedResult<LogEntryDto>> SearchLogsAsync(LogQueryDto query)
        {
            // Implementation of SearchLogsAsync method
            throw new NotImplementedException();
        }

        public Task<LogEntryDto> GetLogByIdAsync(string id)
        {
            // Implementation of GetLogByIdAsync method
            throw new NotImplementedException();
        }

        public Task<LogSummaryDto> GetLogSummaryAsync(DateTime? startTime, DateTime? endTime)
        {
            // Implementation of GetLogSummaryAsync method
            throw new NotImplementedException();
        }

        public Task<byte[]> ExportLogsToCsvAsync(LogQueryDto query)
        {
            // Implementation of ExportLogsToCsvAsync method
            throw new NotImplementedException();
        }

        public Task<byte[]> ExportLogsToJsonAsync(LogQueryDto query)
        {
            // Implementation of ExportLogsToJsonAsync method
            throw new NotImplementedException();
        }
    }

    /// <summary>
    /// Interface for log services
    /// </summary>
    public interface ILogService
    {
        Task<string?> IngestLogAsync(LogIngestRequest logRequest, string hostname, string ip);
        Task<int> ProcessLogBatchAsync(IEnumerable<LogEntryRequest> logs, string agentId);
        SeverityModels ConvertLogSeverityToSeverity(LogSeverityModels logSeverity);

        // Add missing methods
        Task<PaginatedResult<LogEntryDto>> SearchLogsAsync(LogQueryDto query);
        Task<LogEntryDto> GetLogByIdAsync(string id);
        Task<LogSummaryDto> GetLogSummaryAsync(DateTime? startTime, DateTime? endTime);
        Task<byte[]> ExportLogsToCsvAsync(LogQueryDto query);
        Task<byte[]> ExportLogsToJsonAsync(LogQueryDto query);
    }
} 