using System.Collections.Generic;
using System.Threading.Tasks;
using Backend.Models;

namespace Backend.Services
{
    /// <summary>
    /// Service interface for user operations
    /// </summary>
    public interface IUserService
    {
        /// <summary>
        /// Gets all users
        /// </summary>
        /// <returns>All users</returns>
        Task<IEnumerable<UserModels>> GetAllUsersAsync();
        
        /// <summary>
        /// Gets a user by ID
        /// </summary>
        /// <param name="id">The user ID</param>
        /// <returns>The user, or null if not found</returns>
        Task<UserModels?> GetUserByIdAsync(string id);
        
        /// <summary>
        /// Gets a user by username
        /// </summary>
        /// <param name="username">The username</param>
        /// <returns>The user, or null if not found</returns>
        Task<UserModels?> GetUserByUsernameAsync(string username);
        
        /// <summary>
        /// Gets a user by email
        /// </summary>
        /// <param name="email">The email</param>
        /// <returns>The user, or null if not found</returns>
        Task<UserModels?> GetUserByEmailAsync(string email);
        
        /// <summary>
        /// Gets users by role
        /// </summary>
        /// <param name="roleId">The role ID</param>
        /// <returns>The users with the specified role</returns>
        Task<IEnumerable<UserModels>> GetUsersByRoleAsync(string roleId);
        
        /// <summary>
        /// Creates a new user
        /// </summary>
        /// <param name="user">The user to create</param>
        /// <param name="password">The user's password</param>
        /// <returns>The created user</returns>
        Task<UserModels> CreateUserAsync(UserModels user, string password);
        
        /// <summary>
        /// Updates a user
        /// </summary>
        /// <param name="user">The user to update</param>
        /// <returns>The updated user</returns>
        Task<UserModels> UpdateUserAsync(UserModels user);
        
        /// <summary>
        /// Changes a user's password
        /// </summary>
        /// <param name="userId">The user ID</param>
        /// <param name="currentPassword">The current password</param>
        /// <param name="newPassword">The new password</param>
        /// <returns>True if successful, false otherwise</returns>
        Task<bool> ChangePasswordAsync(string userId, string currentPassword, string newPassword);
        
        /// <summary>
        /// Adds a role to a user
        /// </summary>
        /// <param name="userId">The user ID</param>
        /// <param name="roleId">The role ID</param>
        /// <returns>True if successful, false otherwise</returns>
        Task<bool> AddRoleToUserAsync(string userId, string roleId);
        
        /// <summary>
        /// Removes a role from a user
        /// </summary>
        /// <param name="userId">The user ID</param>
        /// <param name="roleId">The role ID</param>
        /// <returns>True if successful, false otherwise</returns>
        Task<bool> RemoveRoleFromUserAsync(string userId, string roleId);
        
        /// <summary>
        /// Deletes a user
        /// </summary>
        /// <param name="id">The user ID</param>
        /// <returns>True if successful, false otherwise</returns>
        Task<bool> DeleteUserAsync(string id);
        
        /// <summary>
        /// Resets a user's password (admin function)
        /// </summary>
        /// <param name="userId">The user ID</param>
        /// <param name="newPassword">The new password</param>
        /// <returns>True if successful, false otherwise</returns>
        Task<bool> AdminResetPasswordAsync(string userId, string newPassword);
    }
} 