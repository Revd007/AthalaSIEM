using System;
using System.IO;
using System.Threading.Tasks;
using AthalaSIEM.Agent;
using Backend.Data.Repositories;
using Backend.Models;
using Grpc.Core;
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
        private readonly ILogRepository _logRepository;

        public SiemService(
            ILogger<SiemService> logger,
            LegacyAgentRepository legacyAgentRepository,
            LegacyLogRepository legacyLogRepository,
            IAgentDeploymentTokenRepository tokenRepository,
            IMediator mediator,
            IAgentRepository agentRepository,
            ILogRepository logRepository)
        {
            _logger = logger;
            _legacyAgentRepository = legacyAgentRepository;
            _legacyLogRepository = legacyLogRepository;
            _tokenRepository = tokenRepository;
            _mediator = mediator;
            _agentRepository = agentRepository;
            _logRepository = logRepository;
        }

        public override async Task<RegisterAgentResponse> RegisterAgent(RegisterAgentRequest request, ServerCallContext context)
        {
            try
            {
                _logger.LogInformation("Agent registration request received from {Hostname}", request.Hostname);
                
                // Use CQRS command for registration
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
                    return new RegisterAgentResponse
                    {
                        Success = false,
                        Message = result.ErrorMessage ?? "Registration failed"
                    };
                }
                
                // Also register in legacy repository for backward compatibility
                var legacyAgent = new AgentModels
                {
                    Id = result.AgentId,
                    ApiKey = result.ApiKey,
                    Hostname = request.Hostname,
                    IPAddress = request.IpAddress,
                    OperatingSystem = request.OperatingSystem,
                    Version = request.AgentVersion,
                    Type = request.AgentType == "Windows" ? AgentType.Windows : 
                           request.AgentType == "Linux" ? AgentType.Linux : AgentType.Custom,
                    LastConnected = DateTime.UtcNow,
                    Status = Backend.Models.AgentStatus.Active,
                    CreatedAt = DateTime.UtcNow
                };
                
                await _legacyAgentRepository.AddAgentAsync(legacyAgent);
                
                _logger.LogInformation("Agent {AgentId} registered successfully", result.AgentId);
                
                return new RegisterAgentResponse
                {
                    Success = true,
                    AgentId = result.AgentId,
                    ApiKey = result.ApiKey,
                    Message = "Agent registered successfully"
                };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error during agent registration");
                return new RegisterAgentResponse
                {
                    Success = false,
                    Message = "Registration failed: " + ex.Message
                };
            }
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
                
                foreach (var log in request.Logs)
                {
                    try
                    {
                        // Create domain log entry
                        var logEntry = new Backend.Domain.Entities.LogEntry
                        {
                            Id = log.Id ?? Guid.NewGuid().ToString(),
                            AgentId = request.AgentId,
                            Timestamp = DateTime.TryParse(log.Timestamp, out var ts) ? ts : DateTime.UtcNow,
                            ReceivedAt = DateTime.UtcNow,
                            RawMessage = log.Message,
                            Source = log.SourceType ?? log.Source,
                            Category = log.SourceType,
                            RawProperties = log.Metadata != null ? System.Text.Json.JsonSerializer.Serialize(log.Metadata) : null,
                            Processed = false,
                            IsNormalized = false
                        };
                        
                        // Store raw log entry
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
                
                // Use CQRS command for heartbeat
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
    }
} 