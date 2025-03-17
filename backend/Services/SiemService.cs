using System;
using System.IO;
using System.Threading.Tasks;
using AthalaSIEM.Agent;
using Backend.Data.Repositories;
using Backend.Models;
using Grpc.Core;
using Microsoft.Extensions.Logging;
using AthalaSIEM.Backend.Repositories;
using AthalaSIEM.Backend.Models;

namespace AthalaSIEM.Backend.Services
{
    public class SiemService : Agent.SiemService.SiemServiceBase
    {
        private readonly ILogger<SiemService> _logger;
        private readonly IAgentRepository _agentRepository;
        private readonly ILogEntryRepository _logRepository;
        private readonly IAgentDeploymentTokenRepository _tokenRepository;

        public SiemService(
            ILogger<SiemService> logger,
            IAgentRepository agentRepository,
            ILogEntryRepository logRepository,
            IAgentDeploymentTokenRepository tokenRepository)
        {
            _logger = logger;
            _agentRepository = agentRepository;
            _logRepository = logRepository;
            _tokenRepository = tokenRepository;
        }

        public override async Task<RegisterAgentResponse> RegisterAgent(RegisterAgentRequest request, ServerCallContext context)
        {
            try
            {
                _logger.LogInformation("Agent registration request received from {Hostname}", request.Hostname);
                
                // Check if deployment token was provided
                if (!string.IsNullOrEmpty(request.DeploymentToken))
                {
                    // Validate deployment token
                    var token = await _tokenRepository.GetTokenAsync(request.DeploymentToken);
                    if (token == null)
                    {
                        _logger.LogWarning("Invalid deployment token provided by {Hostname}", request.Hostname);
                        return new RegisterAgentResponse
                        {
                            Success = false,
                            Message = "Invalid deployment token"
                        };
                    }
                    
                    // Use token data for registration
                    var agentId = Guid.NewGuid().ToString();
                    var apiKey = Guid.NewGuid().ToString();
                    
                    // TODO: Create the agent with the pre-configured settings from the token
                    // Use token.ServerUrl, token.Port, token.UseSSL, etc.
                    
                    // Mark token as used
                    await _tokenRepository.MarkTokenAsUsedAsync(request.DeploymentToken, agentId);
                    
                    return new RegisterAgentResponse
                    {
                        Success = true,
                        AgentId = agentId,
                        ApiKey = apiKey,
                        Message = "Agent registered successfully using deployment token"
                    };
                }
                
                // Standard registration without token
                var newAgentId = Guid.NewGuid().ToString();
                var newApiKey = Guid.NewGuid().ToString();
                
                // Add agent to database
                var agent = new AgentModels
                {
                    Id = newAgentId,
                    ApiKey = newApiKey,
                    Hostname = request.Hostname,
                    IPAddress = request.IpAddress,
                    OperatingSystem = request.OperatingSystem,
                    Version = request.AgentVersion,
                    Type = request.AgentType == "Windows" ? AgentType.Windows : 
                           request.AgentType == "Linux" ? AgentType.Linux : AgentType.Custom,
                    LastConnected = DateTime.UtcNow,
                    Status = AgentStatus.Active,
                    CreatedAt = DateTime.UtcNow
                };
                
                await _agentRepository.AddAgentAsync(agent);
                
                _logger.LogInformation("Agent {AgentId} registered successfully", newAgentId);
                
                return new RegisterAgentResponse
                {
                    Success = true,
                    AgentId = newAgentId,
                    ApiKey = newApiKey,
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
                
                var agent = await _agentRepository.GetByIdAsync(request.AgentId);
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
                
                var agent = await _agentRepository.GetByIdAsync(request.AgentId);
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
                
                await _agentRepository.UpdateAsync(agent);
                
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
                
                // Validate agent and API key
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
                
                // Process logs
                var acceptedCount = 0;
                var rejectedCount = 0;
                
                foreach (var log in request.Logs)
                {
                    try
                    {
                        var logEntry = new LogEntryModels
                        {
                            Id = log.Id,
                            AgentId = request.AgentId,
                            Timestamp = DateTime.Parse(log.Timestamp),
                            Source = log.Source + " - " + log.SourceType, // Combine source and sourceType
                            Level = log.LogLevel,
                            Message = log.Message,
                            // Map metadata as needed
                            CreatedAt = DateTime.UtcNow
                        };
                        
                        await _logRepository.AddAsync(logEntry);
                        acceptedCount++;
                    }
                    catch (Exception ex)
                    {
                        _logger.LogError(ex, "Error processing log entry {LogId}", log.Id);
                        rejectedCount++;
                    }
                }
                
                // Update agent's last seen time
                agent.LastConnected = DateTime.UtcNow;
                await _agentRepository.UpdateAsync(agent);
                
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
                
                var agent = await _agentRepository.GetByIdAsync(request.AgentId);
                if (agent == null || agent.ApiKey != request.ApiKey)
                {
                    _logger.LogWarning("Invalid agent ID or API key for heartbeat");
                    return new HeartbeatResponse
                    {
                        Success = false,
                        Message = "Invalid agent ID or API key"
                    };
                }
                
                // Update agent status
                agent.LastConnected = DateTime.UtcNow;
                agent.Status = Enum.TryParse<AgentStatus>(request.Status, true, out var status) ? 
                    status : AgentStatus.Active;
                // Store other heartbeat metrics as needed
                
                await _agentRepository.UpdateAsync(agent);
                
                _logger.LogDebug("Heartbeat processed for agent {AgentId}", request.AgentId);
                
                return new HeartbeatResponse
                {
                    Success = true,
                    Message = "Heartbeat received",
                    ConfigurationChanged = false // Set to true if agent should refresh config
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