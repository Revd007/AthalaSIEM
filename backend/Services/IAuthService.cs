using System.Threading.Tasks;
using Backend.Models;
using System.Collections.Generic;

namespace Backend.Services
{
    /// <summary>
    /// Interface for authentication operations
    /// </summary>
    public interface IAuthService
    {
        /// <summary>
        /// Authenticates a user
        /// </summary>
        /// <param name="username">The username</param>
        /// <param name="password">The password</param>
        /// <returns>Authentication result</returns>
        Task<(bool Success, string Token, string RefreshToken, string ErrorMessage)> AuthenticateAsync(string username, string password);
        
        /// <summary>
        /// Validates a token
        /// </summary>
        /// <param name="token">The token to validate</param>
        /// <returns>True if the token is valid, false otherwise</returns>
        Task<bool> ValidateTokenAsync(string token);
        
        /// <summary>
        /// Gets a user from a token
        /// </summary>
        /// <param name="token">The token</param>
        /// <returns>The user if found, null otherwise</returns>
        Task<UserModels?> GetUserFromTokenAsync(string token);
        
        /// <summary>
        /// Generates a JWT token for a user
        /// </summary>
        /// <param name="user">The user</param>
        /// <returns>The JWT token</returns>
        string GenerateJwtToken(UserModels user);
        
        /// <summary>
        /// Refreshes a token
        /// </summary>
        /// <param name="token">The token to refresh</param>
        /// <param name="refreshToken">The refresh token</param>
        /// <returns>Refresh result</returns>
        Task<(bool Success, string Token, string RefreshToken, string ErrorMessage)> RefreshTokenAsync(string token, string refreshToken);
        
        /// <summary>
        /// Revokes a token
        /// </summary>
        /// <param name="userId">The user ID</param>
        /// <returns>True if successful, false otherwise</returns>
        Task<bool> RevokeTokenAsync(string userId);
        
        /// <summary>
        /// Checks if a user is in a role
        /// </summary>
        /// <param name="userId">The user ID</param>
        /// <param name="role">The role</param>
        /// <returns>True if the user is in the role, false otherwise</returns>
        Task<bool> IsInRoleAsync(string userId, string role);
        
        /// <summary>
        /// Checks if a user has a specific role
        /// </summary>
        /// <param name="id">The user ID</param>
        /// <param name="v">The role</param>
        /// <returns>True if the user has the role, false otherwise</returns>
        Task<bool> UserHasRoleAsync(string id, string v);
        
        /// <summary>
        /// Gets the roles for a user
        /// </summary>
        /// <param name="userId">The user ID</param>
        /// <returns>The user's roles</returns>
        Task<IEnumerable<string>> GetUserRolesAsync(string userId);
    }
} 