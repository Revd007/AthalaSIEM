using System;

namespace AthalaSIEM.Agent.Models
{
    /// <summary>
    /// Result of an agent registration operation
    /// </summary>
    public class AgentRegistrationResult
    {
        /// <summary>
        /// Whether the registration was successful
        /// </summary>
        public bool Success { get; set; }
        
        /// <summary>
        /// The agent ID (if registration was successful)
        /// </summary>
        public string AgentId { get; set; } = string.Empty;
        
        /// <summary>
        /// The agent API key (if registration was successful)
        /// </summary>
        public string ApiKey { get; set; } = string.Empty;
        
        /// <summary>
        /// Message describing the result (error message if not successful)
        /// </summary>
        public string Message { get; set; } = string.Empty;
        
        /// <summary>
        /// Creates a successful result
        /// </summary>
        /// <param name="agentId">The agent ID</param>
        /// <param name="apiKey">The API key</param>
        /// <returns>A successful result</returns>
        public static AgentRegistrationResult CreateSuccess(string agentId, string apiKey)
        {
            return new AgentRegistrationResult
            {
                Success = true,
                AgentId = agentId,
                ApiKey = apiKey,
                Message = "Registration successful"
            };
        }
        
        /// <summary>
        /// Creates a failed result
        /// </summary>
        /// <param name="message">The error message</param>
        /// <returns>A failed result</returns>
        public static AgentRegistrationResult CreateFailure(string message)
        {
            return new AgentRegistrationResult
            {
                Success = false,
                Message = message
            };
        }
    }
} 