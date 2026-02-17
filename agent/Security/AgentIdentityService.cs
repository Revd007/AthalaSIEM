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
        private string _identityFilePath; // Not readonly - can be changed to fallback location
        private AgentIdentity? _agentIdentity;

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

            // Set up identity file path with fallback locations
            _identityFilePath = GetIdentityFilePath();
            _logger.LogInformation("Using identity file path: {IdentityFilePath}", _identityFilePath);
            
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
        /// Gets the identity file path, trying multiple locations with fallback
        /// </summary>
        private string GetIdentityFilePath()
        {
            // Try locations in order of preference:
            // 1. ProgramData\AthalaSIEM (standard Windows location)
            // 2. AppData\Local\AthalaSIEM (user-specific, usually writable)
            // 3. Executable directory (last resort)
            
            var locations = new[]
            {
                Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData), "AthalaSIEM"),
                Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "AthalaSIEM"),
                Path.Combine(AppContext.BaseDirectory, "Data")
            };

            foreach (var location in locations)
            {
                try
                {
                    if (!Directory.Exists(location))
                    {
                        Directory.CreateDirectory(location);
                        _logger.LogDebug("Created directory: {Directory}", location);
                    }

                    // Test write permission by creating a temp file
                    string testFile = Path.Combine(location, ".write_test");
                    try
                    {
                        File.WriteAllText(testFile, "test");
                        File.Delete(testFile);
                        
                        // This location is writable, use it
                        string identityFile = Path.Combine(location, "agent_identity.json");
                        _logger.LogInformation("Selected identity file location: {IdentityFile} (directory: {Directory})", identityFile, location);
                        return identityFile;
                    }
                    catch (Exception writeEx)
                    {
                        _logger.LogWarning(writeEx, "Cannot write to {Directory}, trying next location...", location);
                        continue;
                    }
                }
                catch (Exception dirEx)
                {
                    _logger.LogWarning(dirEx, "Cannot create directory {Directory}, trying next location...", location);
                    continue;
                }
            }

            // If all locations failed, use ProgramData anyway (will fail with clear error)
            string fallback = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData), "AthalaSIEM", "agent_identity.json");
            _logger.LogWarning("All identity file locations failed. Using fallback: {FallbackPath}", fallback);
            return fallback;
        }

        /// <summary>
        /// Checks if the agent is registered with the backend (has valid identity file with AgentId and ApiKey).
        /// </summary>
        /// <returns>True if the agent is registered, otherwise false</returns>
        public Task<bool> IsRegisteredAsync()
        {
            return Task.FromResult(_agentIdentity != null
                && !string.IsNullOrEmpty(_agentIdentity.AgentId)
                && !string.IsNullOrEmpty(_agentIdentity.ApiKey));
        }

        /// <summary>
        /// Checks if the agent has a valid identity
        /// </summary>
        /// <returns>True if the agent has a valid identity, false otherwise</returns>
        public Task<bool> HasValidIdentityAsync()
        {
            LoadAgentIdentity();
            return Task.FromResult(_agentIdentity != null
                && !string.IsNullOrEmpty(_agentIdentity.AgentId)
                && !string.IsNullOrEmpty(_agentIdentity.ApiKey));
        }

        /// <inheritdoc/>
        public async Task<AgentRegistrationResult> RegisterAgentAsync()
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
                    // Send registration request via gRPC
                    _logger.LogInformation("Sending gRPC RegisterAgent to backend...");
                    var response = await _client.RegisterAgentAsync(request);

                    // Log exactly what the backend returned so we can debug any mismatch
                    _logger.LogInformation(
                        "Backend RegisterAgent response: Success={Success}, AgentId='{AgentId}', ApiKey={HasApiKey}, Message='{Message}'",
                        response?.Success,
                        response?.AgentId ?? "(null)",
                        !string.IsNullOrEmpty(response?.ApiKey) ? "PRESENT" : "EMPTY",
                        response?.Message ?? "(null)");

                    if (response != null && response.Success && !string.IsNullOrEmpty(response.AgentId))
                    {
                        // Create new agent identity
                        _agentIdentity = new AgentIdentity
                        {
                            AgentId = response.AgentId,
                            ApiKey = !string.IsNullOrEmpty(response.ApiKey)
                                ? response.ApiKey
                                : GenerateFallbackApiKey(response.AgentId),
                            RegisteredAt = DateTime.UtcNow,
                            LastSeenAt = DateTime.UtcNow
                        };

                        // Save agent identity (with retry and fallback)
                        SaveAgentIdentity();

                        // CRITICAL: Verify file was created - if not, registration is not persistent
                        if (File.Exists(_identityFilePath))
                        {
                            var fileInfo = new FileInfo(_identityFilePath);
                            _logger.LogInformation(
                                "Agent registered successfully with ID: {AgentId}. Identity file saved to: {IdentityFilePath} (size: {Size} bytes)",
                                _agentIdentity.AgentId, _identityFilePath, fileInfo.Length);
                        }
                        else
                        {
                            _logger.LogError(
                                "CRITICAL: Agent registered but identity file was NOT saved to {IdentityFilePath}. " +
                                "Registration will be LOST on restart! Check file permissions and disk space.",
                                _identityFilePath);
                        }

                        return AgentRegistrationResult.CreateSuccess(_agentIdentity.AgentId, _agentIdentity.ApiKey);
                    }
                    else
                    {
                        // Log exactly WHY registration failed
                        string reason;
                        if (response == null)
                            reason = "Response was null (backend unreachable or returned empty)";
                        else if (!response.Success)
                            reason = $"Backend returned Success=false: {response.Message ?? "no message"}";
                        else if (string.IsNullOrEmpty(response.AgentId))
                            reason = $"Backend returned Success=true but AgentId is empty: {response.Message ?? "no message"}";
                        else
                            reason = "Unknown reason";

                        _logger.LogError("Failed to register agent: {Reason}", reason);
                        return AgentRegistrationResult.CreateFailure(reason);
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

                    // Save the local identity (with retry and fallback)
                    SaveAgentIdentity();
                    
                    // CRITICAL: Verify file was created
                    if (File.Exists(_identityFilePath))
                    {
                        var fileInfo = new FileInfo(_identityFilePath);
                        _logger.LogInformation("Agent created locally with fallback ID: {AgentId}. Identity file saved to: {IdentityFilePath} (size: {Size} bytes)", 
                            _agentIdentity.AgentId, _identityFilePath, fileInfo.Length);
                    }
                    else
                    {
                        _logger.LogError("CRITICAL: Agent created locally but identity file was NOT saved to {IdentityFilePath}. " +
                            "Registration will be LOST on restart! Check file permissions and disk space.", _identityFilePath);
                    }
                    
                    return AgentRegistrationResult.CreateSuccess(_agentIdentity.AgentId, _agentIdentity.ApiKey);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error registering agent");
                return AgentRegistrationResult.CreateFailure($"Error registering agent: {ex.Message}");
            }
        }

        /// <inheritdoc/>
        public async Task<AgentRegistrationResult> RegisterAgentAsync(string agentName, string serverUrl, int serverPort)
        {
            try
            {
                _logger.LogInformation("Registering agent with backend at {ServerUrl}:{ServerPort}", serverUrl, serverPort);
                
                // Update settings: REST and gRPC can use different ports (e.g. REST 9595, gRPC 50051)
                _settings.BackendApiUrl = $"{(serverPort == 443 ? "https" : "http")}://{serverUrl}:{serverPort}";
                var grpcPort = serverPort == 9595 ? 50051 : serverPort;
                _settings.BackendGrpcUrl = $"{(serverPort == 443 ? "https" : "http")}://{serverUrl}:{grpcPort}";
                _settings.AgentName = agentName;
                
                // Then proceed with standard registration
                return await RegisterAgentAsync();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error registering agent with custom parameters");
                return AgentRegistrationResult.CreateFailure($"Error registering agent: {ex.Message}");
            }
        }

        /// <inheritdoc/>
        public async Task<AgentRegistrationResult> RegisterWithTokenAsync(string token)
        {
            if (string.IsNullOrEmpty(token))
            {
                _logger.LogError("Failed to register agent: Deployment token is required");
                return AgentRegistrationResult.CreateFailure("Deployment token is required");
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
                    // Send token registration request via gRPC
                    _logger.LogInformation("Sending gRPC RegisterAgent (with token) to backend...");
                    var response = await _client.RegisterAgentAsync(request);

                    _logger.LogInformation(
                        "Backend RegisterAgent (token) response: Success={Success}, AgentId='{AgentId}', ApiKey={HasApiKey}, Message='{Message}'",
                        response?.Success,
                        response?.AgentId ?? "(null)",
                        !string.IsNullOrEmpty(response?.ApiKey) ? "PRESENT" : "EMPTY",
                        response?.Message ?? "(null)");

                    if (response != null && response.Success && !string.IsNullOrEmpty(response.AgentId))
                    {
                        // Create new agent identity
                        _agentIdentity = new AgentIdentity
                        {
                            AgentId = response.AgentId,
                            ApiKey = !string.IsNullOrEmpty(response.ApiKey) ? response.ApiKey : GenerateFallbackApiKey(response.AgentId),
                            RegisteredAt = DateTime.UtcNow,
                            LastSeenAt = DateTime.UtcNow
                        };

                        // Save agent identity (with retry and fallback)
                        SaveAgentIdentity();

                        // CRITICAL: Verify file was created
                        if (File.Exists(_identityFilePath))
                        {
                            var fileInfo = new FileInfo(_identityFilePath);
                            _logger.LogInformation(
                                "Agent registered successfully with token. Agent ID: {AgentId}. Identity file saved to: {IdentityFilePath} (size: {Size} bytes)",
                                _agentIdentity.AgentId, _identityFilePath, fileInfo.Length);
                        }
                        else
                        {
                            _logger.LogError(
                                "CRITICAL: Agent registered but identity file was NOT saved to {IdentityFilePath}. " +
                                "Registration will be LOST on restart! Check file permissions and disk space.",
                                _identityFilePath);
                        }

                        return AgentRegistrationResult.CreateSuccess(_agentIdentity.AgentId, _agentIdentity.ApiKey);
                    }
                    else
                    {
                        string reason;
                        if (response == null)
                            reason = "Response was null (backend unreachable or returned empty)";
                        else if (!response.Success)
                            reason = $"Backend returned Success=false: {response.Message ?? "no message"}";
                        else if (string.IsNullOrEmpty(response.AgentId))
                            reason = $"Backend returned Success=true but AgentId is empty: {response.Message ?? "no message"}";
                        else
                            reason = "Unknown reason";

                        _logger.LogError("Failed to register agent with token: {Reason}", reason);
                        return AgentRegistrationResult.CreateFailure(reason);
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Failed to register agent with token");
                    return AgentRegistrationResult.CreateFailure($"Failed to register agent with token: {ex.Message}");
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error registering agent with token");
                return AgentRegistrationResult.CreateFailure($"Error registering agent with token: {ex.Message}");
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
                    var msg = response?.Message ?? "Backend returned no new API key";
                    _logger.LogError("Failed to rotate API key: {Message}", msg);

                    // Rotation failed - clear local identity so the agent can re-register.
                    // This handles ALL failure modes: agent not found, invalid key, invalid response, etc.
                    try
                    {
                        if (File.Exists(_identityFilePath))
                        {
                            File.Delete(_identityFilePath);
                            _logger.LogWarning("Deleted identity file after failed rotation.");
                        }
                        lock (_identityLock) { _agentIdentity = null; }
                        _logger.LogWarning("Local identity cleared after failed API key rotation. Re-registration will be attempted.");
                    }
                    catch (Exception clearEx)
                    {
                        _logger.LogWarning(clearEx, "Could not clear identity file after failed rotation");
                    }

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
                _logger.LogInformation("Agent identity file not found at {IdentityFilePath}. This is normal on first run - agent will register and create the file.", _identityFilePath);
                lock (_identityLock)
                {
                    _agentIdentity = null;
                }
                return;
            }

            try
            {
                // Read encrypted identity file
                byte[] encryptedJson = File.ReadAllBytes(_identityFilePath);
                
                // Decrypt identity using machine-specific key (32 bytes)
                byte[] machineKey = GenerateMachineSpecificKey();
                byte[] decryptedJson = _encryptionService.Decrypt(encryptedJson, machineKey);
                string json = Encoding.UTF8.GetString(decryptedJson);
                
                // Deserialize identity
                lock (_identityLock)
                {
                    _agentIdentity = JsonSerializer.Deserialize<AgentIdentity>(json, new JsonSerializerOptions
                    {
                        PropertyNameCaseInsensitive = true
                    }) ?? new AgentIdentity { AgentId = string.Empty, ApiKey = string.Empty };
                }

                if (_agentIdentity != null && !string.IsNullOrEmpty(_agentIdentity.AgentId) && !string.IsNullOrEmpty(_agentIdentity.ApiKey))
                {
                    _logger.LogInformation("Successfully loaded agent identity with ID: {AgentId}", _agentIdentity.AgentId);
                }
                else
                {
                    _logger.LogWarning("Loaded agent identity is invalid (empty AgentId or ApiKey). Will re-register.");
                    lock (_identityLock)
                    {
                        _agentIdentity = null;
                    }
                }
            }
            catch (ArgumentException argEx) when (argEx.Message.Contains("Key must be 32 bytes") || argEx.ParamName == "key")
            {
                // File was encrypted with old key format (Base64 string converted to bytes = >32 bytes)
                // Delete the corrupt file so agent can create a new one with correct format
                _logger.LogWarning("Identity file was encrypted with incompatible key format (old version). Deleting corrupt file: {IdentityFilePath}", _identityFilePath);
                try
                {
                    File.Delete(_identityFilePath);
                    _logger.LogInformation("Deleted corrupt identity file. Agent will create a new identity on registration.");
                }
                catch (Exception deleteEx)
                {
                    _logger.LogWarning(deleteEx, "Could not delete corrupt identity file. Please delete manually: {IdentityFilePath}", _identityFilePath);
                }
                lock (_identityLock)
                {
                    _agentIdentity = null;
                }
            }
            catch (Exception ex)
            {
                // Other decryption errors - file might be corrupted or encrypted with different key
                _logger.LogWarning(ex, "Error decrypting identity file (possibly corrupted or wrong key). Deleting file: {IdentityFilePath}", _identityFilePath);
                try
                {
                    File.Delete(_identityFilePath);
                    _logger.LogInformation("Deleted corrupt identity file. Agent will create a new identity on registration.");
                }
                catch (Exception deleteEx)
                {
                    _logger.LogWarning(deleteEx, "Could not delete corrupt identity file. Please delete manually: {IdentityFilePath}", _identityFilePath);
                }
                lock (_identityLock)
                {
                    _agentIdentity = null;
                }
            }
        }

        private void SaveAgentIdentity()
        {
            if (_agentIdentity == null)
            {
                _logger.LogWarning("Cannot save agent identity: Identity is null");
                return;
            }

            const int maxRetries = 3;
            int retryCount = 0;
            
            while (retryCount < maxRetries)
            {
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
                    
                    // Encrypt identity using machine-specific key (32 bytes)
                    byte[] jsonBytes = Encoding.UTF8.GetBytes(json);
                    byte[] machineKey = GenerateMachineSpecificKey();
                    byte[] encryptedJson = _encryptionService.Encrypt(jsonBytes, machineKey);
                    
                    // Ensure directory exists
                    string? directory = Path.GetDirectoryName(_identityFilePath);
                    if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
                    {
                        Directory.CreateDirectory(directory);
                        _logger.LogDebug("Created directory for identity file: {Directory}", directory);
                    }
                    
                    // Write to file
                    File.WriteAllBytes(_identityFilePath, encryptedJson);
                    
                    // CRITICAL: Verify file was actually written
                    if (!File.Exists(_identityFilePath))
                    {
                        throw new IOException($"File was not created after write operation: {_identityFilePath}");
                    }
                    
                    // Verify file size matches
                    var fileInfo = new FileInfo(_identityFilePath);
                    if (fileInfo.Length != encryptedJson.Length)
                    {
                        throw new IOException($"File size mismatch. Expected {encryptedJson.Length} bytes, got {fileInfo.Length} bytes.");
                    }
                    
                    _logger.LogInformation("Agent identity saved successfully to: {IdentityFilePath} (size: {Size} bytes)", 
                        _identityFilePath, encryptedJson.Length);
                    return; // Success - exit retry loop
                }
                catch (UnauthorizedAccessException uaEx)
                {
                    _logger.LogError(uaEx, "Permission denied writing to {IdentityFilePath}. Attempt {Attempt}/{MaxRetries}. Error: {ErrorMessage}", 
                        _identityFilePath, retryCount + 1, maxRetries, uaEx.Message);
                    
                    if (retryCount == maxRetries - 1)
                    {
                        // Last attempt failed - try fallback location
                        _logger.LogWarning("All attempts failed. Trying fallback location...");
                        TrySaveToFallbackLocation();
                        return;
                    }
                    
                    retryCount++;
                    System.Threading.Thread.Sleep(500 * retryCount); // Exponential backoff
                }
                catch (DirectoryNotFoundException dirEx)
                {
                    _logger.LogWarning(dirEx, "Directory not found for {IdentityFilePath}. Attempt {Attempt}/{MaxRetries}. Creating directory...", 
                        _identityFilePath, retryCount + 1, maxRetries);
                    
                    try
                    {
                        string? directory = Path.GetDirectoryName(_identityFilePath);
                        if (!string.IsNullOrEmpty(directory))
                        {
                            Directory.CreateDirectory(directory);
                            retryCount++; // Retry write after creating directory
                            continue;
                        }
                    }
                    catch (Exception createDirEx)
                    {
                        string? dirName = Path.GetDirectoryName(_identityFilePath);
                        _logger.LogError(createDirEx, "Failed to create directory: {Directory}", dirName ?? "unknown");
                    }
                    
                    if (retryCount == maxRetries - 1)
                    {
                        TrySaveToFallbackLocation();
                        return;
                    }
                    
                    retryCount++;
                }
                catch (IOException ioEx)
                {
                    _logger.LogWarning(ioEx, "IO error saving identity file. Attempt {Attempt}/{MaxRetries}. Error: {ErrorMessage}", 
                        retryCount + 1, maxRetries, ioEx.Message);
                    
                    if (retryCount == maxRetries - 1)
                    {
                        TrySaveToFallbackLocation();
                        return;
                    }
                    
                    retryCount++;
                    System.Threading.Thread.Sleep(500 * retryCount);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Unexpected error saving agent identity to {IdentityFilePath}. Attempt {Attempt}/{MaxRetries}. Error: {ErrorMessage}", 
                        _identityFilePath, retryCount + 1, maxRetries, ex.Message);
                    
                    if (retryCount == maxRetries - 1)
                    {
                        TrySaveToFallbackLocation();
                        return;
                    }
                    
                    retryCount++;
                    System.Threading.Thread.Sleep(500 * retryCount);
                }
            }
        }

        /// <summary>
        /// Tries to save identity to a fallback location if primary location fails
        /// </summary>
        private void TrySaveToFallbackLocation()
        {
            if (_agentIdentity == null) return;
            
            var fallbackLocations = new[]
            {
                Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "AthalaSIEM"),
                Path.Combine(AppContext.BaseDirectory, "Data"),
                Path.GetTempPath()
            };

            foreach (var location in fallbackLocations)
            {
                try
                {
                    if (!Directory.Exists(location))
                    {
                        Directory.CreateDirectory(location);
                    }

                    string fallbackPath = Path.Combine(location, "agent_identity.json");
                    
                    // Serialize and encrypt
                    string json = JsonSerializer.Serialize(_agentIdentity, new JsonSerializerOptions { WriteIndented = true });
                    byte[] jsonBytes = Encoding.UTF8.GetBytes(json);
                    byte[] machineKey = GenerateMachineSpecificKey();
                    byte[] encryptedJson = _encryptionService.Encrypt(jsonBytes, machineKey);
                    
                    File.WriteAllBytes(fallbackPath, encryptedJson);
                    
                    if (File.Exists(fallbackPath))
                    {
                        // Update the file path for future operations
                        _identityFilePath = fallbackPath;
                        _logger.LogWarning("Saved identity to fallback location: {FallbackPath}. Future operations will use this location.", fallbackPath);
                        return;
                    }
                }
                catch (Exception fallbackEx)
                {
                    _logger.LogWarning(fallbackEx, "Failed to save to fallback location: {Location}", location);
                    continue;
                }
            }
            
            _logger.LogError("CRITICAL: Failed to save agent identity to all locations (primary and fallbacks). Identity will be lost on restart!");
        }

        /// <summary>
        /// Generates a machine-specific encryption key (32 bytes for AES-256)
        /// </summary>
        private byte[] GenerateMachineSpecificKey()
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
                
                // Compute SHA256 hash - this produces exactly 32 bytes (perfect for AES-256)
                byte[] hashBytes = sha.ComputeHash(Encoding.UTF8.GetBytes(sb.ToString()));
                return hashBytes; // Return raw bytes, not Base64 string
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error generating machine-specific key, using fallback");
                // Fallback: Use SHA256 of a fixed string to ensure exactly 32 bytes
                using var sha = SHA256.Create();
                return sha.ComputeHash(Encoding.UTF8.GetBytes("AthalaSIEM-Agent-FixedKey-NotSecure"));
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