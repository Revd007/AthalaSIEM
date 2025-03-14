using AthalaSIEM.Agent.Models;
using Microsoft.Extensions.Logging;
using System;
using System.IO;
using System.Net;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace AthalaSIEM.Agent.Security
{
    /// <summary>
    /// Service for managing agent identity and authentication with the backend
    /// </summary>
    public class AgentIdentityService : IAgentIdentityService
    {
        private readonly ILogger<AgentIdentityService> _logger;
        private readonly SiemService.SiemServiceClient _client;
        private readonly AgentSettings _settings;
        private readonly IEncryptionService _encryptionService;
        private readonly object _identityLock = new object();
        private readonly string _identityFilePath;
        private AgentIdentity _agentIdentity = new AgentIdentity {
            AgentId = string.Empty,
            ApiKey = string.Empty
        };

        public AgentIdentityService(
            ILogger<AgentIdentityService> logger,
            SiemService.SiemServiceClient client,
            AgentSettings settings,
            IEncryptionService encryptionService)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _client = client ?? throw new ArgumentNullException(nameof(client));
            _settings = settings ?? throw new ArgumentNullException(nameof(settings));
            _encryptionService = encryptionService ?? throw new ArgumentNullException(nameof(encryptionService));

            // Set up identity file path in a standard location
            string appDataFolder = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData),
                "AthalaSIEM");
            
            // Ensure the directory exists
            if (!Directory.Exists(appDataFolder))
            {
                Directory.CreateDirectory(appDataFolder);
            }

            _identityFilePath = Path.Combine(appDataFolder, "agent_identity.json");
            
            // Try to load existing identity
            try
            {
                LoadAgentIdentity();
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Failed to load agent identity from {IdentityFilePath}. Will need to register.", _identityFilePath);
            }
        }

        /// <summary>
        /// Checks if the agent is registered with the backend
        /// </summary>
        /// <returns>True if the agent is registered, otherwise false</returns>
        public Task<bool> IsRegisteredAsync()
        {
            return Task.FromResult(_agentIdentity != null);
        }

        /// <summary>
        /// Registers the agent with the backend
        /// </summary>
        /// <returns>True if registration was successful, otherwise false</returns>
        public async Task<bool> RegisterAgentAsync()
        {
            try
            {
                _logger.LogInformation("Registering agent with backend");

                // Prepare registration request
                var request = new RegisterAgentRequest
                {
                    Hostname = Environment.MachineName,
                    IpAddress = await GetLocalIpAddress(),
                    OperatingSystem = GetOperatingSystemDescription(),
                    AgentVersion = GetType().Assembly.GetName().Version?.ToString() ?? "1.0.0",
                    AgentType = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "Windows" : "Linux"
                };

                try
                {
                    // Send registration request
                    var response = await _client.RegisterAgentAsync(request);
                    
                    if (response != null && !string.IsNullOrEmpty(response.AgentId))
                    {
                        // Create new agent identity
                        _agentIdentity = new AgentIdentity
                        {
                            AgentId = response.AgentId,
                            // Use the API key from the response or generate a fallback if empty
                            ApiKey = !string.IsNullOrEmpty(response.ApiKey) ? response.ApiKey : GenerateFallbackApiKey(response.AgentId),
                            RegisteredAt = DateTime.UtcNow,
                            LastSeenAt = DateTime.UtcNow
                        };

                        // Save agent identity
                        SaveAgentIdentity();

                        _logger.LogInformation("Agent registered successfully with ID: {AgentId}", _agentIdentity.AgentId);
                        return true;
                    }
                    else
                    {
                        _logger.LogError("Failed to register agent: Invalid response from backend");
                        return false;
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Error during standard registration, attempting fallback local registration");
                    
                    // Fallback: Create a local identity without server registration
                    var fallbackAgentId = Guid.NewGuid().ToString();
                    _agentIdentity = new AgentIdentity
                    {
                        AgentId = fallbackAgentId,
                        ApiKey = GenerateFallbackApiKey(fallbackAgentId),
                        RegisteredAt = DateTime.UtcNow,
                        LastSeenAt = DateTime.UtcNow
                    };

                    // Save the local identity
                    SaveAgentIdentity();
                    
                    _logger.LogInformation("Agent created locally with fallback ID: {AgentId}", _agentIdentity.AgentId);
                    return true;
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error registering agent");
                return false;
            }
        }

        /// <summary>
        /// Registers the agent with the backend using a deployment token
        /// </summary>
        /// <param name="token">The deployment token</param>
        /// <returns>True if registration was successful, otherwise false</returns>
        public async Task<bool> RegisterWithTokenAsync(string token)
        {
            if (string.IsNullOrEmpty(token))
            {
                _logger.LogError("Failed to register agent: Deployment token is required");
                return false;
            }

            try
            {
                _logger.LogInformation("Registering agent with deployment token");

                // Prepare registration request
                var request = new RegisterAgentRequest
                {
                    Hostname = Environment.MachineName,
                    IpAddress = await GetLocalIpAddress(),
                    OperatingSystem = GetOperatingSystemDescription(),
                    AgentVersion = GetType().Assembly.GetName().Version?.ToString() ?? "1.0.0",
                    AgentType = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "Windows" : "Linux",
                    DeploymentToken = token
                };

                try
                {
                    // Send token registration request
                    var response = await _client.RegisterAgentAsync(request);
                    
                    if (response != null && !string.IsNullOrEmpty(response.AgentId))
                    {
                        // Create new agent identity
                        _agentIdentity = new AgentIdentity
                        {
                            AgentId = response.AgentId,
                            ApiKey = !string.IsNullOrEmpty(response.ApiKey) ? response.ApiKey : GenerateFallbackApiKey(response.AgentId),
                            RegisteredAt = DateTime.UtcNow,
                            LastSeenAt = DateTime.UtcNow
                        };

                        // Save agent identity
                        SaveAgentIdentity();

                        _logger.LogInformation("Agent registered successfully with token. Agent ID: {AgentId}", _agentIdentity.AgentId);
                        return true;
                    }
                    else
                    {
                        _logger.LogError("Failed to register agent with token: Invalid response from backend");
                        return false;
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Failed to register agent with token");
                    return false;
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error registering agent with token");
                return false;
            }
        }

        /// <summary>
        /// Gets the agent's API key
        /// </summary>
        /// <returns>The agent's API key or empty string if not registered</returns>
        public Task<string> GetApiKeyAsync()
        {
            if (_agentIdentity == null)
            {
                _logger.LogWarning("Cannot get API key: Agent is not registered");
                return Task.FromResult(string.Empty);
            }

            return Task.FromResult(_agentIdentity.ApiKey);
        }

        /// <summary>
        /// Gets the agent's ID
        /// </summary>
        /// <returns>The agent's ID or empty string if not registered</returns>
        public Task<string> GetAgentIdAsync()
        {
            if (_agentIdentity == null)
            {
                _logger.LogWarning("Cannot get agent ID: Agent is not registered");
                return Task.FromResult(string.Empty);
            }

            return Task.FromResult(_agentIdentity.AgentId);
        }

        /// <summary>
        /// Validates the API key with the backend
        /// </summary>
        /// <returns>True if the API key is valid, otherwise false</returns>
        public async Task<bool> ValidateApiKeyAsync()
        {
            if (_agentIdentity == null || string.IsNullOrEmpty(_agentIdentity.AgentId) || string.IsNullOrEmpty(_agentIdentity.ApiKey))
            {
                _logger.LogWarning("Cannot validate API key: Agent is not registered or missing required credentials");
                return false;
            }

            try
            {
                var request = new ValidateApiKeyRequest
                {
                    AgentId = _agentIdentity.AgentId,
                    ApiKey = _agentIdentity.ApiKey
                };

                _logger.LogInformation("Attempting to validate API key with server at {url}", _settings.BackendGrpcUrl);
                var response = await _client.ValidateApiKeyAsync(request);
                return response != null && response.Valid;
            }
            catch (Exception ex)
            {
                if (ex is System.Net.Http.HttpRequestException httpEx && httpEx.InnerException != null)
                {
                    _logger.LogError(ex, "SSL/TLS connection error validating API key: {message}. Check that the server URL is correct and SSL is properly configured.", httpEx.InnerException.Message);
                }
                else
                {
                    _logger.LogError(ex, "Error validating API key");
                }
                return false;
            }
        }

        /// <summary>
        /// Rotates the agent's API key
        /// </summary>
        /// <returns>True if rotation was successful, otherwise false</returns>
        public async Task<bool> RotateApiKeyAsync()
        {
            if (_agentIdentity == null)
            {
                _logger.LogWarning("Cannot rotate API key: Agent is not registered");
                return false;
            }

            try
            {
                var request = new RotateApiKeyRequest
                {
                    AgentId = _agentIdentity.AgentId,
                    CurrentApiKey = _agentIdentity.ApiKey
                };

                var response = await _client.RotateApiKeyAsync(request);
                if (response != null && !string.IsNullOrEmpty(response.NewApiKey))
                {
                    lock (_identityLock)
                    {
                        _agentIdentity.ApiKey = response.NewApiKey;
                        _agentIdentity.LastRotatedAt = DateTime.UtcNow;
                    }

                    SaveAgentIdentity();
                    _logger.LogInformation("Agent API key rotated successfully");
                    return true;
                }
                else
                {
                    _logger.LogError("Failed to rotate API key: Invalid response from backend");
                    return false;
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error rotating API key");
                return false;
            }
        }

        private void LoadAgentIdentity()
        {
            if (!File.Exists(_identityFilePath))
            {
                _logger.LogInformation("Agent identity file not found at {IdentityFilePath}", _identityFilePath);
                return;
            }

            try
            {
                // Read encrypted identity file
                byte[] encryptedJson = File.ReadAllBytes(_identityFilePath);
                
                // Decrypt identity
                byte[] decryptedJson = _encryptionService.Decrypt(encryptedJson, Encoding.UTF8.GetBytes(GenerateMachineSpecificKey()));
                string json = Encoding.UTF8.GetString(decryptedJson);
                
                // Deserialize identity
                lock (_identityLock)
                {
                    _agentIdentity = JsonSerializer.Deserialize<AgentIdentity>(json, new JsonSerializerOptions
                    {
                        PropertyNameCaseInsensitive = true
                    }) ?? new AgentIdentity { AgentId = string.Empty, ApiKey = string.Empty };
                }

                if (_agentIdentity != null)
                {
                    _logger.LogInformation("Successfully loaded agent identity with ID: {AgentId}", _agentIdentity.AgentId);
                }
                else
                {
                    _logger.LogWarning("Loaded agent identity is null");
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error loading agent identity from {IdentityFilePath}", _identityFilePath);
                _agentIdentity = new AgentIdentity { AgentId = string.Empty, ApiKey = string.Empty };
                throw;
            }
        }

        private void SaveAgentIdentity()
        {
            if (_agentIdentity == null)
            {
                _logger.LogWarning("Cannot save agent identity: Identity is null");
                return;
            }

            try
            {
                // Serialize identity
                string json;
                lock (_identityLock)
                {
                    json = JsonSerializer.Serialize(_agentIdentity, new JsonSerializerOptions
                    {
                        WriteIndented = true
                    });
                }
                
                // Encrypt identity
                byte[] jsonBytes = Encoding.UTF8.GetBytes(json);
                byte[] encryptedJson = _encryptionService.Encrypt(jsonBytes, Encoding.UTF8.GetBytes(GenerateMachineSpecificKey()));
                
                // Write to file
                File.WriteAllBytes(_identityFilePath, encryptedJson);
                
                _logger.LogDebug("Agent identity saved successfully");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to save agent identity");
            }
        }

        private string GenerateMachineSpecificKey()
        {
            try
            {
                // Create a machine-specific key based on hardware info
                using var sha = SHA256.Create();
                
                // Collect machine-specific identifiers
                StringBuilder sb = new StringBuilder();
                sb.Append(Environment.MachineName);
                sb.Append(Environment.ProcessorCount);
                sb.Append(Environment.OSVersion);
                
                if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                {
                    // Add Windows-specific identifiers
                    sb.Append(Environment.SystemDirectory);
                }
                else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
                {
                    // Add Linux-specific identifiers
                    try
                    {
                        if (File.Exists("/etc/machine-id"))
                        {
                            sb.Append(File.ReadAllText("/etc/machine-id").Trim());
                        }
                    }
                    catch { /* Ignore errors */ }
                }
                
                // Compute hash
                byte[] hashBytes = sha.ComputeHash(Encoding.UTF8.GetBytes(sb.ToString()));
                return Convert.ToBase64String(hashBytes);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error generating machine-specific key, using fallback");
                // Fallback to a fixed key
                return "AthalaSIEM-Agent-FixedKey-NotSecure";
            }
        }

        private async Task<string> GetLocalIpAddress()
        {
            try
            {
                string hostName = Dns.GetHostName();
                var hostEntry = await Dns.GetHostEntryAsync(hostName);
                
                // Find IPv4 address
                foreach (var address in hostEntry.AddressList)
                {
                    if (address.AddressFamily == System.Net.Sockets.AddressFamily.InterNetwork)
                    {
                        return address.ToString();
                    }
                }
                
                return "127.0.0.1";
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting local IP address");
                return "127.0.0.1";
            }
        }

        private string GetOperatingSystemDescription()
        {
            try
            {
                return RuntimeInformation.OSDescription;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting OS description");
                return "Unknown";
            }
        }

        /// <summary>
        /// Generates a fallback API key when the server doesn't provide one
        /// </summary>
        private string GenerateFallbackApiKey(string agentId)
        {
            // Generate a deterministic but secure key based on the agent ID
            using (var hmac = new HMACSHA256(Encoding.UTF8.GetBytes("AthalaSIEM-Local-Key")))
            {
                var hash = hmac.ComputeHash(Encoding.UTF8.GetBytes(agentId));
                return Convert.ToBase64String(hash);
            }
        }
    }

    /// <summary>
    /// Class that holds the agent identity information
    /// </summary>
    internal class AgentIdentity
    {
        /// <summary>
        /// Agent ID assigned by the backend
        /// </summary>
        public required string AgentId { get; set; }
        
        /// <summary>
        /// API key for authentication
        /// </summary>
        public required string ApiKey { get; set; }
        
        /// <summary>
        /// When the agent was registered
        /// </summary>
        public DateTime RegisteredAt { get; set; } = DateTime.UtcNow;
        
        /// <summary>
        /// When the agent was last seen by the backend
        /// </summary>
        public DateTime LastSeenAt { get; set; }
        
        /// <summary>
        /// When the API key was last rotated
        /// </summary>
        public DateTime? LastRotatedAt { get; set; }
    }
} 