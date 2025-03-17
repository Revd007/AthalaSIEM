using System.Threading;
using System.Threading.Tasks;
using AthalaSIEM.Agent.Models;

namespace AthalaSIEM.Agent.Security
{
    /// <summary>
    /// Interface for managing agent identity and authentication
    /// </summary>
    public interface IAgentIdentityService
    {
        /// <summary>
        /// Checks if the agent is registered with the backend
        /// </summary>
        /// <returns>True if the agent is registered, false otherwise</returns>
        Task<bool> IsRegisteredAsync();
        
        /// <summary>
        /// Checks if the agent has a valid identity
        /// </summary>
        /// <returns>True if the agent has a valid identity, false otherwise</returns>
        Task<bool> HasValidIdentityAsync();
        
        /// <summary>
        /// Registers the agent with the backend
        /// </summary>
        /// <returns>Registration result</returns>
        Task<AgentRegistrationResult> RegisterAgentAsync();
        
        /// <summary>
        /// Registers the agent with the backend with explicit parameters
        /// </summary>
        /// <param name="agentName">The agent name</param>
        /// <param name="serverUrl">The server URL</param>
        /// <param name="serverPort">The server port</param>
        /// <returns>Registration result</returns>
        Task<AgentRegistrationResult> RegisterAgentAsync(string agentName, string serverUrl, int serverPort);
        
        /// <summary>
        /// Registers the agent with the backend using a deployment token
        /// </summary>
        /// <param name="token">The deployment token</param>
        /// <returns>Registration result</returns>
        Task<AgentRegistrationResult> RegisterWithTokenAsync(string token);
        
        /// <summary>
        /// Gets the agent's API key
        /// </summary>
        /// <returns>The agent's API key or null if not registered</returns>
        Task<string> GetApiKeyAsync();
        
        /// <summary>
        /// Gets the agent's ID
        /// </summary>
        /// <returns>The agent's ID or null if not registered</returns>
        Task<string> GetAgentIdAsync();
        
        /// <summary>
        /// Validates the agent's API key with the backend
        /// </summary>
        /// <returns>True if the API key is valid, false otherwise</returns>
        Task<bool> ValidateApiKeyAsync();
        
        /// <summary>
        /// Rotates the agent's API key
        /// </summary>
        /// <returns>True if the API key was rotated successfully, false otherwise</returns>
        Task<bool> RotateApiKeyAsync();
    }
    
    /// <summary>
    /// Interface for encryption services
    /// </summary>
    public interface IEncryptionService
    {
        /// <summary>
        /// Encrypts data using the specified key
        /// </summary>
        /// <param name="data">Data to encrypt</param>
        /// <param name="key">Encryption key</param>
        /// <returns>Encrypted data</returns>
        byte[] Encrypt(byte[] data, byte[] key);
        
        /// <summary>
        /// Decrypts data using the specified key
        /// </summary>
        /// <param name="encryptedData">Data to decrypt</param>
        /// <param name="key">Decryption key</param>
        /// <returns>Decrypted data</returns>
        byte[] Decrypt(byte[] encryptedData, byte[] key);
        
        /// <summary>
        /// Generates a random encryption key
        /// </summary>
        /// <returns>A random encryption key</returns>
        byte[] GenerateKey();
        
        /// <summary>
        /// Computes a hash of the input data
        /// </summary>
        /// <param name="data">Data to hash</param>
        /// <returns>Hash value</returns>
        string ComputeHash(byte[] data);
    }
} 