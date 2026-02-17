using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AthalaSIEM.Agent;
using Backend.Data.Repositories;
using Backend.Models;
using Backend.Hubs;
using Grpc.Core;
using Microsoft.AspNetCore.SignalR;
using Microsoft.Extensions.Logging;
using MediatR;
using Backend.Application.Commands;
using Backend.Domain.Interfaces;
using Backend.Domain.Entities;
using Backend.Domain.Events;
using AthalaSIEM.Backend.Repositories;
using LegacyAgentRepository = Backend.Data.Repositories.ILegacyAgentRepository;
using LegacyLogRepository = Backend.Data.Repositories.ILegacyLogEntryRepository;

namespace Backend.Services
{
    public class SiemService : AthalaSIEM.Agent.SiemService.SiemServiceBase
    {
        private readonly ILogger<SiemService> _logger;
        private readonly LegacyAgentRepository _legacyAgentRepository;
        private readonly LegacyLogRepository _legacyLogRepository;
        private readonly IAgentDeploymentTokenRepository _tokenRepository;
        private readonly IMediator _mediator;
        private readonly IAgentRepository _agentRepository;
        private readonly Backend.Domain.Interfaces.ILogRepository _logRepository;
        private readonly Microsoft.Extensions.Configuration.IConfiguration _configuration;
        private readonly IHubContext<SiemHub> _hubContext;

        public SiemService(
            ILogger<SiemService> logger,
            LegacyAgentRepository legacyAgentRepository,
            LegacyLogRepository legacyLogRepository,
            IAgentDeploymentTokenRepository tokenRepository,
            IMediator mediator,
            IAgentRepository agentRepository,
            ILogRepository logRepository,
            Microsoft.Extensions.Configuration.IConfiguration configuration,
            IHubContext<SiemHub> hubContext)
        {
            _logger = logger;
            _legacyAgentRepository = legacyAgentRepository;
            _legacyLogRepository = legacyLogRepository;
            _tokenRepository = tokenRepository;
            _mediator = mediator;
            _agentRepository = agentRepository;
            _logRepository = logRepository;
            _configuration = configuration;
            _hubContext = hubContext;
        }

        public override async Task<RegisterAgentResponse> RegisterAgent(RegisterAgentRequest request, ServerCallContext context)
        {
            try
            {
                _logger.LogInformation(
                    "Agent registration request received from {Hostname} (OS={OS}, Version={Version}, Type={Type})",
                    request.Hostname, request.OperatingSystem, request.AgentVersion, request.AgentType);

                // Check if an agent with this hostname already exists (re-registration after -CleanIdentity)
                var existingAgents = await _legacyAgentRepository.GetAllAsync();
                var existingAgent = existingAgents.FirstOrDefault(a =>
                    string.Equals(a.Hostname, request.Hostname, StringComparison.OrdinalIgnoreCase));

                if (existingAgent != null)
                {
                    _logger.LogInformation(
                        "Agent with hostname {Hostname} already exists (ID={AgentId}). Updating existing registration.",
                        request.Hostname, existingAgent.Id);

                    // Update existing agent instead of creating a new one
                    existingAgent.IPAddress = request.IpAddress;
                    existingAgent.OperatingSystem = request.OperatingSystem;
                    existingAgent.Version = request.AgentVersion;
                    existingAgent.AgentVersion = request.AgentVersion;
                    existingAgent.Type = request.AgentType == "Windows" ? AgentType.Windows :
                                         request.AgentType == "Linux" ? AgentType.Linux : AgentType.Custom;
                    existingAgent.LastConnected = DateTime.UtcNow;
                    existingAgent.LastHeartbeat = DateTime.UtcNow;
                    existingAgent.Status = Backend.Models.AgentStatus.Online;
                    existingAgent.UpdatedAt = DateTime.UtcNow;

                    // Rotate API key on re-registration for security
                    existingAgent.ApiKey = GenerateApiKey();

                    await _legacyAgentRepository.UpdateAsync(existingAgent);

                    // Also update domain entity
                    var domainAgent = await _agentRepository.GetByIdAsync(existingAgent.Id);
                    if (domainAgent != null)
                    {
                        domainAgent.IpAddress = request.IpAddress;
                        domainAgent.OperatingSystem = request.OperatingSystem;
                        domainAgent.AgentVersion = request.AgentVersion;
                        domainAgent.ApiKey = existingAgent.ApiKey;
                        domainAgent.Status = Domain.Entities.AgentStatus.Online;
                        domainAgent.LastHeartbeat = DateTime.UtcNow;
                        domainAgent.UpdatedAt = DateTime.UtcNow;
                        await _agentRepository.UpdateAsync(domainAgent);
                    }

                    var grpcUrlReReg = _configuration["GrpcServer:Url"] ?? "http://localhost:50051";
                    _logger.LogInformation("Agent {AgentId} re-registered successfully", existingAgent.Id);

                    return new RegisterAgentResponse
                    {
                        Success = true,
                        AgentId = existingAgent.Id,
                        ApiKey = existingAgent.ApiKey,
                        Message = "Agent re-registered successfully",
                        GrpcEndpoint = grpcUrlReReg
                    };
                }

                // New agent: use CQRS command which writes to the domain repository (-> AgentModels table)
                var command = new RegisterAgentCommand
                {
                    Name = request.Hostname,
                    Hostname = request.Hostname,
                    IpAddress = request.IpAddress,
                    OperatingSystem = request.OperatingSystem,
                    AgentVersion = request.AgentVersion,
                    Metadata = request.Metadata?.ToDictionary(kvp => kvp.Key, kvp => (object)kvp.Value)
                };

                var result = await _mediator.Send(command);

                if (!result.Success)
                {
                    _logger.LogError("CQRS registration failed: {Error}", result.ErrorMessage);
                    return new RegisterAgentResponse
                    {
                        Success = false,
                        Message = result.ErrorMessage ?? "Registration failed"
                    };
                }

                // The CQRS handler already wrote AgentModels to the database via Infrastructure.AgentRepository.
                // Update the legacy-specific fields (Type, Version, Status, LastConnected) that the CQRS
                // path doesn't set, by loading the ALREADY TRACKED entity and modifying it in place.
                var tracked = await _legacyAgentRepository.GetByIdAsync(result.AgentId);
                if (tracked != null)
                {
                    tracked.Version = request.AgentVersion;
                    tracked.AgentVersion = request.AgentVersion;
                    tracked.Type = request.AgentType == "Windows" ? AgentType.Windows :
                                   request.AgentType == "Linux" ? AgentType.Linux : AgentType.Custom;
                    tracked.LastConnected = DateTime.UtcNow;
                    tracked.LastHeartbeat = DateTime.UtcNow;
                    tracked.Status = Backend.Models.AgentStatus.Online;
                    tracked.Name = request.Hostname;
                    await _legacyAgentRepository.UpdateAsync(tracked);
                }

                _logger.LogInformation("Agent {AgentId} registered successfully (new)", result.AgentId);

                var grpcUrl = _configuration["GrpcServer:Url"] ?? "http://localhost:50051";

                return new RegisterAgentResponse
                {
                    Success = true,
                    AgentId = result.AgentId,
                    ApiKey = result.ApiKey,
                    Message = "Agent registered successfully",
                    GrpcEndpoint = grpcUrl
                };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error during agent registration from {Hostname}", request.Hostname);
                return new RegisterAgentResponse
                {
                    Success = false,
                    Message = "Registration failed: " + ex.Message
                };
            }
        }

        private static string GenerateApiKey()
        {
            using var rng = System.Security.Cryptography.RandomNumberGenerator.Create();
            var bytes = new byte[32];
            rng.GetBytes(bytes);
            return Convert.ToBase64String(bytes).Replace("+", "-").Replace("/", "_").TrimEnd('=');
        }

        public override async Task<ValidateApiKeyResponse> ValidateApiKey(ValidateApiKeyRequest request, ServerCallContext context)
        {
            try
            {
                _logger.LogDebug("API key validation request for agent {AgentId}", request.AgentId);
                
                var agent = await _legacyAgentRepository.GetByIdAsync(request.AgentId);
                if (agent == null)
                {
                    _logger.LogWarning("Agent {AgentId} not found during API key validation", request.AgentId);
                    return new ValidateApiKeyResponse
                    {
                        Valid = false,
                        Message = "Agent not found"
                    };
                }
                
                var isValid = agent.ApiKey == request.ApiKey;
                
                if (isValid)
                {
                    _logger.LogDebug("API key validation successful for agent {AgentId}", request.AgentId);
                    return new ValidateApiKeyResponse
                    {
                        Valid = true,
                        Message = "API key is valid"
                    };
                }
                
                _logger.LogWarning("Invalid API key provided for agent {AgentId}", request.AgentId);
                return new ValidateApiKeyResponse
                {
                    Valid = false,
                    Message = "Invalid API key"
                };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error validating API key for agent {AgentId}", request.AgentId);
                return new ValidateApiKeyResponse
                {
                    Valid = false,
                    Message = "Validation failed: " + ex.Message
                };
            }
        }

        public override async Task<RotateApiKeyResponse> RotateApiKey(RotateApiKeyRequest request, ServerCallContext context)
        {
            try
            {
                _logger.LogInformation("API key rotation request for agent {AgentId}", request.AgentId);
                
                var agent = await _legacyAgentRepository.GetByIdAsync(request.AgentId);
                if (agent == null)
                {
                    _logger.LogWarning("Agent {AgentId} not found during API key rotation", request.AgentId);
                    return new RotateApiKeyResponse
                    {
                        Success = false,
                        Message = "Agent not found"
                    };
                }
                
                if (agent.ApiKey != request.CurrentApiKey)
                {
                    _logger.LogWarning("Invalid current API key provided for agent {AgentId}", request.AgentId);
                    return new RotateApiKeyResponse
                    {
                        Success = false,
                        Message = "Invalid current API key"
                    };
                }
                
                var newApiKey = Guid.NewGuid().ToString();
                agent.ApiKey = newApiKey;
                agent.LastConnected = DateTime.UtcNow;
                
                await _legacyAgentRepository.UpdateAsync(agent);
                
                _logger.LogInformation("API key rotated successfully for agent {AgentId}", request.AgentId);
                return new RotateApiKeyResponse
                {
                    Success = true,
                    NewApiKey = newApiKey,
                    Message = "API key rotated successfully"
                };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error rotating API key for agent {AgentId}", request.AgentId);
                return new RotateApiKeyResponse
                {
                    Success = false,
                    Message = "Rotation failed: " + ex.Message
                };
            }
        }

        public override async Task<LogBatchResponse> ForwardLogs(LogBatchRequest request, ServerCallContext context)
        {
            try
            {
                _logger.LogDebug("Received log batch from agent {AgentId} with {Count} logs", 
                    request.AgentId, request.Logs.Count);
                
                // Validate agent and API key using domain repository
                var agent = await _agentRepository.GetByIdAsync(request.AgentId);
                if (agent == null || agent.ApiKey != request.ApiKey)
                {
                    _logger.LogWarning("Invalid agent ID or API key for log batch");
                    return new LogBatchResponse
                    {
                        Success = false,
                        Message = "Invalid agent ID or API key"
                    };
                }
                
                // Process logs through new architecture
                var acceptedCount = 0;
                var rejectedCount = 0;
                
                // Extract hostname that applies to all logs in this batch
                var agentHostname = agent.Hostname ?? string.Empty;

                foreach (var log in request.Logs)
                {
                    try
                    {
                        // Extract metadata fields sent by the agent
                        var metadata = log.Metadata;
                        var machineName = metadata?.GetValueOrDefault("machine_name") ?? agentHostname;
                        var logName = metadata?.GetValueOrDefault("log_name") ?? string.Empty;
                        var eventIdStr = metadata?.GetValueOrDefault("event_id") ?? string.Empty;
                        long.TryParse(eventIdStr, out var eventId);

                        // Create domain log entry with ALL fields properly mapped
                        // Ensure all DateTime values are UTC (PostgreSQL requirement)
                        DateTime timestamp;
                        if (DateTime.TryParse(log.Timestamp, out var ts))
                        {
                            // Convert to UTC if not already UTC
                            timestamp = ts.Kind == DateTimeKind.Utc ? ts : ts.ToUniversalTime();
                        }
                        else
                        {
                            timestamp = DateTime.UtcNow;
                        }

                        var logEntry = new Backend.Domain.Entities.LogEntry
                        {
                            Id = log.Id ?? Guid.NewGuid().ToString(),
                            AgentId = request.AgentId,
                            Timestamp = timestamp,
                            ReceivedAt = DateTime.UtcNow,
                            Level = !string.IsNullOrEmpty(log.LogLevel) ? log.LogLevel : "Information",
                            RawMessage = !string.IsNullOrEmpty(log.Message) ? log.Message : "(no message)",
                            Source = !string.IsNullOrEmpty(log.SourceType) ? log.SourceType : log.Source,
                            Category = !string.IsNullOrEmpty(logName) ? logName : log.SourceType,
                            EventId = eventId > 0 ? eventId : null,
                            MachineName = machineName,
                            IPAddress = metadata?.GetValueOrDefault("source_ip") ?? string.Empty,
                            RawProperties = metadata != null && metadata.Count > 0 
                                ? System.Text.Json.JsonSerializer.Serialize(metadata) 
                                : null,
                            Processed = false,
                            IsNormalized = false
                        };
                        
                        // Store raw log entry (MapToModel now carries Level, MachineName, IPAddress)
                        await _logRepository.AddAsync(logEntry);
                        
                        // Publish ingestion event to trigger normalization and detection
                        await _mediator.Publish(new LogIngestedEvent
                        {
                            LogEntry = logEntry
                        });
                        
                        acceptedCount++;
                    }
                    catch (Exception ex)
                    {
                        _logger.LogError(ex, "Error processing log entry {LogId}", log.Id);
                        rejectedCount++;
                    }
                }
                
                // Update agent's last seen time
                agent.LastHeartbeat = DateTime.UtcNow;
                agent.Status = Backend.Domain.Entities.AgentStatus.Online;
                await _agentRepository.UpdateAsync(agent);
                
                // Also update legacy agent for backward compatibility
                var legacyAgent = await _legacyAgentRepository.GetByIdAsync(request.AgentId);
                if (legacyAgent != null)
                {
                    legacyAgent.LastConnected = DateTime.UtcNow;
                    await _legacyAgentRepository.UpdateAsync(legacyAgent);
                }
                
                _logger.LogInformation("Processed log batch from agent {AgentId}: {Accepted} accepted, {Rejected} rejected", 
                    request.AgentId, acceptedCount, rejectedCount);
                
                // Feed the real-time dashboard aggregator with ingestion metadata
                if (acceptedCount > 0)
                {
                    try
                    {
                        // Feed in-memory counters for the DashboardAggregatorWorker
                        var firstLog = request.Logs.FirstOrDefault();
                        Backend.Workers.DashboardAggregatorWorker.RecordIngestion(
                            request.AgentId,
                            acceptedCount,
                            firstLog?.SourceType ?? "Unknown",
                            firstLog?.LogLevel ?? "Information");

                        // Also push a lightweight notification so the frontend can trigger a query refresh
                        await _hubContext.Clients.All.SendAsync("ReceiveLogBatch", new
                        {
                            agentId = request.AgentId,
                            count = acceptedCount,
                            timestamp = DateTime.UtcNow.ToString("o")
                        });
                    }
                    catch (Exception hubEx)
                    {
                        _logger.LogDebug(hubEx, "Non-critical: SignalR broadcast failed");
                    }
                }

                return new LogBatchResponse
                {
                    Success = true,
                    AcceptedCount = acceptedCount,
                    RejectedCount = rejectedCount,
                    Message = $"Processed {acceptedCount} logs, rejected {rejectedCount} logs"
                };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error processing log batch from agent {AgentId}", request.AgentId);
                return new LogBatchResponse
                {
                    Success = false,
                    Message = "Log processing failed: " + ex.Message
                };
            }
        }

        public override async Task<HeartbeatResponse> SendHeartbeat(HeartbeatRequest request, ServerCallContext context)
        {
            try
            {
                _logger.LogDebug("Heartbeat received from agent {AgentId}", request.AgentId);

                // Use CQRS command for heartbeat (updates domain Agent entity)
                var command = new SendHeartbeatCommand
                {
                    AgentId = request.AgentId,
                    ApiKey = request.ApiKey,
                    HealthMetrics = new Dictionary<string, object>
                    {
                        ["cpu_usage"] = request.CpuUsage,
                        ["memory_usage"] = request.MemoryUsage,
                        ["uptime_hours"] = request.UptimeHours,
                        ["status"] = request.Status,
                        ["active_collectors"] = request.ActiveCollectors,
                        ["logs_collected"] = request.LogsCollected,
                        ["logs_forwarded"] = request.LogsForwarded
                    }
                };

                var result = await _mediator.Send(command);

                if (!result.Success)
                {
                    return new HeartbeatResponse
                    {
                        Success = false,
                        Message = result.ErrorMessage ?? "Heartbeat processing failed"
                    };
                }

                // Also update legacy model so the dashboard (which reads AgentModels) shows correct status
                try
                {
                    var legacyAgent = await _legacyAgentRepository.GetByIdAsync(request.AgentId);
                    if (legacyAgent != null)
                    {
                        legacyAgent.LastConnected = DateTime.UtcNow;
                        legacyAgent.LastHeartbeat = DateTime.UtcNow;
                        legacyAgent.Status = Backend.Models.AgentStatus.Online;
                        legacyAgent.CpuUsage = request.CpuUsage;
                        legacyAgent.MemoryUsage = request.MemoryUsage;
                        legacyAgent.UpdatedAt = DateTime.UtcNow;
                        await _legacyAgentRepository.UpdateAsync(legacyAgent);
                    }
                }
                catch (Exception legacyEx)
                {
                    _logger.LogWarning(legacyEx, "Failed to update legacy agent model for heartbeat (non-critical)");
                }

                _logger.LogDebug("Heartbeat processed for agent {AgentId}", request.AgentId);

                return new HeartbeatResponse
                {
                    Success = true,
                    Message = "Heartbeat received",
                    ConfigurationChanged = result.Configuration != null && result.Configuration.Any()
                };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error processing heartbeat from agent {AgentId}", request.AgentId);
                return new HeartbeatResponse
                {
                    Success = false,
                    Message = "Heartbeat processing failed: " + ex.Message
                };
            }
        }

        public override Task<SystemMetricsResponse> SendSystemMetrics(SystemMetricsRequest request, ServerCallContext context)
        {
            // Implement system metrics processing
            return Task.FromResult(new SystemMetricsResponse
            {
                Success = true,
                Message = "System metrics received"
            });
        }

        public override Task<HealthReportResponse> SendHealthReport(HealthReportRequest request, ServerCallContext context)
        {
            // Implement health report processing
            return Task.FromResult(new HealthReportResponse
            {
                Success = true,
                Message = "Health report received"
            });
        }

        public override Task<GetAgentConfigurationResponse> GetAgentConfiguration(GetAgentConfigurationRequest request, ServerCallContext context)
        {
            // Implement configuration retrieval
            return Task.FromResult(new GetAgentConfigurationResponse
            {
                Success = true,
                ConfigurationChanged = false,
                ConfigVersion = "1.0",
                ConfigurationJson = "{}",
                Message = "Configuration retrieved"
            });
        }

        // Streaming RPC implementations
        public override async Task<LogBatchResponse> StreamLogs(IAsyncStreamReader<AthalaSIEM.Agent.LogEntry> requestStream, ServerCallContext context)
        {
            var acceptedCount = 0;
            var rejectedCount = 0;
            string? agentId = null;
            string? apiKey = null;

            try
            {
                // Extract agent ID and API key from metadata
                var metadata = context.RequestHeaders;
                agentId = metadata.FirstOrDefault(m => m.Key == "x-agent-id")?.Value;
                apiKey = metadata.FirstOrDefault(m => m.Key == "x-api-key")?.Value;

                if (string.IsNullOrEmpty(agentId) || string.IsNullOrEmpty(apiKey))
                {
                    _logger.LogWarning("Missing agent ID or API key in gRPC metadata");
                    return new LogBatchResponse
                    {
                        Success = false,
                        Message = "Missing authentication in metadata"
                    };
                }

                // Validate agent
                var agent = await _agentRepository.GetByIdAsync(agentId);
                if (agent == null || agent.ApiKey != apiKey)
                {
                    _logger.LogWarning("Invalid agent ID or API key in gRPC stream");
                    return new LogBatchResponse
                    {
                        Success = false,
                        Message = "Invalid authentication"
                    };
                }

                await foreach (var log in requestStream.ReadAllAsync())
                {
                    try
                    {
                        var logEntry = new Domain.Entities.LogEntry
                        {
                            Id = log.Id ?? Guid.NewGuid().ToString(),
                            AgentId = agentId,
                            Timestamp = DateTime.TryParse(log.Timestamp, out var ts) ? ts : DateTime.UtcNow,
                            ReceivedAt = DateTime.UtcNow,
                            RawMessage = log.Message,
                            Source = log.SourceType ?? log.Source,
                            Category = log.SourceType,
                            RawProperties = log.Metadata != null ? System.Text.Json.JsonSerializer.Serialize(log.Metadata) : null,
                            Processed = false,
                            IsNormalized = false
                        };

                        await _logRepository.AddAsync(logEntry);
                        await _mediator.Publish(new LogIngestedEvent { LogEntry = logEntry });
                        acceptedCount++;
                    }
                    catch (Exception ex)
                    {
                        _logger.LogError(ex, "Error processing log entry in stream");
                        rejectedCount++;
                    }
                }

                // Update agent status
                agent.LastHeartbeat = DateTime.UtcNow;
                agent.Status = Backend.Domain.Entities.AgentStatus.Online;
                await _agentRepository.UpdateAsync(agent);

                _logger.LogInformation("Streamed log batch from agent {AgentId}: {Accepted} accepted, {Rejected} rejected",
                    agentId, acceptedCount, rejectedCount);

                return new LogBatchResponse
                {
                    Success = true,
                    AcceptedCount = acceptedCount,
                    RejectedCount = rejectedCount,
                    Message = $"Processed {acceptedCount} logs, rejected {rejectedCount} logs"
                };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error processing log stream from agent {AgentId}", agentId);
                return new LogBatchResponse
                {
                    Success = false,
                    AcceptedCount = acceptedCount,
                    RejectedCount = rejectedCount,
                    Message = "Stream processing failed: " + ex.Message
                };
            }
        }

        public override async Task StreamHeartbeat(IAsyncStreamReader<HeartbeatRequest> requestStream, IServerStreamWriter<HeartbeatResponse> responseStream, ServerCallContext context)
        {
            try
            {
                var metadata = context.RequestHeaders;
                var agentId = metadata.FirstOrDefault(m => m.Key == "x-agent-id")?.Value;
                var apiKey = metadata.FirstOrDefault(m => m.Key == "x-api-key")?.Value;

                await foreach (var request in requestStream.ReadAllAsync())
                {
                    try
                    {
                        var command = new SendHeartbeatCommand
                        {
                            AgentId = request.AgentId,
                            ApiKey = request.ApiKey,
                            HealthMetrics = new Dictionary<string, object>
                            {
                                ["cpu_usage"] = request.CpuUsage,
                                ["memory_usage"] = request.MemoryUsage,
                                ["disk_usage"] = request.DiskUsage,
                                ["uptime_hours"] = request.UptimeHours,
                                ["status"] = request.Status,
                                ["active_collectors"] = request.ActiveCollectors,
                                ["logs_collected"] = request.LogsCollected,
                                ["logs_forwarded"] = request.LogsForwarded
                            }
                        };

                        var result = await _mediator.Send(command);

                        var response = new HeartbeatResponse
                        {
                            Success = result.Success,
                            Message = result.Success ? "Heartbeat received" : result.ErrorMessage ?? "Heartbeat processing failed",
                            ConfigurationChanged = result.Configuration != null && result.Configuration.Any(),
                            ConfigVersion = result.Configuration?.GetValueOrDefault("version")?.ToString() ?? "1.0"
                        };

                        await responseStream.WriteAsync(response);
                    }
                    catch (Exception ex)
                    {
                        _logger.LogError(ex, "Error processing heartbeat in stream");
                        await responseStream.WriteAsync(new HeartbeatResponse
                        {
                            Success = false,
                            Message = "Heartbeat processing failed: " + ex.Message
                        });
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error in heartbeat stream");
            }
        }

        public override async Task<SystemMetricsResponse> StreamSystemMetrics(IAsyncStreamReader<SystemMetricsRequest> requestStream, ServerCallContext context)
        {
            try
            {
                var metadata = context.RequestHeaders;
                var agentId = metadata.FirstOrDefault(m => m.Key == "x-agent-id")?.Value;

                await foreach (var request in requestStream.ReadAllAsync())
                {
                    // Process system metrics
                    _logger.LogDebug("Received system metrics from agent {AgentId}", request.AgentId);
                }

                return new SystemMetricsResponse
                {
                    Success = true,
                    Message = "System metrics received"
                };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error processing system metrics stream");
                return new SystemMetricsResponse
                {
                    Success = false,
                    Message = "Stream processing failed: " + ex.Message
                };
            }
        }
    }
} 