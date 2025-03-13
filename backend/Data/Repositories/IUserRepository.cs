using System.Collections.Generic;
using System.Threading.Tasks;
using Backend.Models;
using System.Linq.Expressions;

namespace Backend.Data.Repositories
{
    /// <summary>
    /// Repository interface for user operations
    /// </summary>
    public interface IUserRepository : IRepository<UserModels, string>
    {
        new Task<UserModels?> GetByIdAsync(string id);
        new Task<IEnumerable<UserModels>> GetAllAsync();
        new Task<UserModels> AddAsync(UserModels user);
        new Task<UserModels> UpdateAsync(UserModels user);
        
        /// <summary>
        /// Gets a user by username
        /// </summary>
        /// <param name="username">The username</param>
        /// <returns>The user if found, null otherwise</returns>
        Task<UserModels?> GetByUsernameAsync(string username);
        
        /// <summary>
        /// Gets a user by email
        /// </summary>
        /// <param name="email">The email</param>
        /// <returns>The user if found, null otherwise</returns>
        Task<UserModels?> GetByEmailAsync(string email);
        
        /// <summary>
        /// Gets a user's roles
        /// </summary>
        /// <param name="userId">The user ID</param>
        /// <returns>The user's roles</returns>
        Task<IEnumerable<string>> GetUserRolesAsync(string userId);
        
        /// <summary>
        /// Adds a user to a role
        /// </summary>
        /// <param name="userId">The user ID</param>
        /// <param name="roleName">The role name</param>
        /// <returns>True if successful, false otherwise</returns>
        Task<bool> AddToRoleAsync(string userId, string roleName);
        
        /// <summary>
        /// Removes a user from a role
        /// </summary>
        /// <param name="userId">The user ID</param>
        /// <param name="roleName">The role name</param>
        /// <returns>True if successful, false otherwise</returns>
        Task<bool> RemoveFromRoleAsync(string userId, string roleName);
        
        /// <summary>
        /// Checks if a user is in a role
        /// </summary>
        /// <param name="userId">The user ID</param>
        /// <param name="roleName">The role name</param>
        /// <returns>True if the user is in the role, false otherwise</returns>
        Task<bool> IsInRoleAsync(string userId, string roleName);

        Task<IEnumerable<UserModels>> GetByRoleAsync(string roleId);
        Task DeleteAsync(string id);
        Task AddRoleAsync(string userId, string roleName);
        Task RemoveRoleAsync(string userId, string roleName);
        new Task<IEnumerable<UserModels>> FindAsync(Expression<Func<UserModels, bool>> predicate);
    }
} 