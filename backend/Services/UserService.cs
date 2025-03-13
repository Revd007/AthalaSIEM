using System;
using System.Collections.Generic;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;
using Backend.Data.Repositories;
using Backend.Models;
using Microsoft.Extensions.Logging;

namespace Backend.Services
{
    /// <summary>
    /// Service for user operations
    /// </summary>
    public class UserService : IUserService
    {
        private readonly IUserRepository _userRepository;
        private readonly ILogger<UserService> _logger;
        
        /// <summary>
        /// Initializes a new instance of the <see cref="UserService"/> class
        /// </summary>
        /// <param name="userRepository">The user repository</param>
        /// <param name="logger">The logger</param>
        public UserService(IUserRepository userRepository, ILogger<UserService> logger)
        {
            _userRepository = userRepository ?? throw new ArgumentNullException(nameof(userRepository));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }
        
        /// <inheritdoc/>
        public async Task<IEnumerable<UserModels>> GetAllUsersAsync()
        {
            return await _userRepository.GetAllAsync();
        }
        
        /// <inheritdoc/>
        public async Task<UserModels?> GetUserByIdAsync(string id)
        {
            return await _userRepository.GetByIdAsync(id);
        }
        
        /// <inheritdoc/>
        public async Task<UserModels?> GetUserByUsernameAsync(string username)
        {
            return await _userRepository.GetByUsernameAsync(username);
        }
        
        /// <inheritdoc/>
        public async Task<UserModels?> GetUserByEmailAsync(string email)
        {
            return await _userRepository.GetByEmailAsync(email);
        }
        
        /// <inheritdoc/>
        public async Task<IEnumerable<UserModels>> GetUsersByRoleAsync(string roleId)
        {
            return await _userRepository.GetByRoleAsync(roleId);
        }
        
        /// <inheritdoc/>
        public async Task<UserModels> CreateUserAsync(UserModels user, string password)
        {
            if (user == null)
            {
                throw new ArgumentNullException(nameof(user));
            }
            
            if (string.IsNullOrEmpty(password))
            {
                throw new ArgumentException("Password cannot be empty", nameof(password));
            }
            
            // Check if username or email already exists
            var existingUser = await _userRepository.GetByUsernameAsync(user.Username);
            if (existingUser != null)
            {
                throw new InvalidOperationException($"Username '{user.Username}' is already taken");
            }
            
            existingUser = await _userRepository.GetByEmailAsync(user.Email);
            if (existingUser != null)
            {
                throw new InvalidOperationException($"Email '{user.Email}' is already registered");
            }
            
            // Generate password hash and salt
            CreatePasswordHash(password, out var passwordHash, out var passwordSalt);
            
            // Set user properties
            user.Id = Guid.NewGuid().ToString();
            user.PasswordHash = Convert.ToBase64String(passwordHash);
            user.PasswordSalt = Convert.ToBase64String(passwordSalt);
            user.CreatedAt = DateTime.UtcNow;
            user.UpdatedAt = DateTime.UtcNow;
            
            // Add user to database
            await _userRepository.AddAsync(user);
            
            _logger.LogInformation("User created: {UserId} ({Username})", user.Id, user.Username);
            
            return user;
        }
        
        /// <inheritdoc/>
        public async Task<UserModels> UpdateUserAsync(UserModels user)
        {
            if (user == null)
            {
                throw new ArgumentNullException(nameof(user));
            }
            
            var existingUser = await _userRepository.GetByIdAsync(user.Id);
            if (existingUser == null)
            {
                throw new KeyNotFoundException($"User with ID {user.Id} not found");
            }
            
            // Check if username is being changed and is already taken
            if (existingUser.Username != user.Username)
            {
                var userWithSameUsername = await _userRepository.GetByUsernameAsync(user.Username);
                if (userWithSameUsername != null && userWithSameUsername.Id != user.Id)
                {
                    throw new InvalidOperationException($"Username '{user.Username}' is already taken");
                }
            }
            
            // Check if email is being changed and is already registered
            if (existingUser.Email != user.Email)
            {
                var userWithSameEmail = await _userRepository.GetByEmailAsync(user.Email);
                if (userWithSameEmail != null && userWithSameEmail.Id != user.Id)
                {
                    throw new InvalidOperationException($"Email '{user.Email}' is already registered");
                }
            }
            
            // Update user properties
            existingUser.Username = user.Username;
            existingUser.Email = user.Email;
            existingUser.FirstName = user.FirstName;
            existingUser.LastName = user.LastName;
            existingUser.IsActive = user.IsActive;
            existingUser.UpdatedAt = DateTime.UtcNow;
            
            // Update user in database
            await _userRepository.UpdateAsync(existingUser);
            
            _logger.LogInformation("User updated: {UserId} ({Username})", existingUser.Id, existingUser.Username);
            
            return existingUser;
        }
        
        /// <inheritdoc/>
        public async Task<bool> ChangePasswordAsync(string userId, string currentPassword, string newPassword)
        {
            var user = await _userRepository.GetByIdAsync(userId);
            if (user == null)
            {
                _logger.LogWarning("User with ID {UserId} not found", userId);
                return false;
            }
            
            // Verify current password
            if (!VerifyPasswordHash(currentPassword, Convert.FromBase64String(user.PasswordHash), Convert.FromBase64String(user.PasswordSalt)))
            {
                _logger.LogWarning("Invalid current password for user: {UserId}", userId);
                return false;
            }
            
            // Generate new password hash and salt
            CreatePasswordHash(newPassword, out var passwordHash, out var passwordSalt);
            
            // Update password
            user.PasswordHash = Convert.ToBase64String(passwordHash);
            user.PasswordSalt = Convert.ToBase64String(passwordSalt);
            user.UpdatedAt = DateTime.UtcNow;
            
            // Update user in database
            await _userRepository.UpdateAsync(user);
            
            _logger.LogInformation("Password changed for user: {UserId}", userId);
            
            return true;
        }
        
        /// <inheritdoc/>
        public async Task<bool> AddRoleToUserAsync(string userId, string roleId)
        {
            try
            {
                await _userRepository.AddRoleAsync(userId, roleId);
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error adding role {RoleId} to user {UserId}", roleId, userId);
                return false;
            }
        }
        
        /// <inheritdoc/>
        public async Task<bool> RemoveRoleFromUserAsync(string userId, string roleId)
        {
            try
            {
                await _userRepository.RemoveRoleAsync(userId, roleId);
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error removing role {RoleId} from user {UserId}", roleId, userId);
                return false;
            }
        }
        
        /// <inheritdoc/>
        public async Task<bool> DeleteUserAsync(string id)
        {
            try
            {
                await _userRepository.DeleteByIdAsync(id);
                _logger.LogInformation("User deleted: {UserId}", id);
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error deleting user: {UserId}", id);
                return false;
            }
        }
        
        /// <summary>
        /// Creates a password hash and salt
        /// </summary>
        /// <param name="password">The password</param>
        /// <param name="passwordHash">The generated password hash</param>
        /// <param name="passwordSalt">The generated password salt</param>
        private void CreatePasswordHash(string password, out byte[] passwordHash, out byte[] passwordSalt)
        {
            using var hmac = new HMACSHA512();
            passwordSalt = hmac.Key;
            passwordHash = hmac.ComputeHash(Encoding.UTF8.GetBytes(password));
        }
        
        /// <summary>
        /// Verifies a password against a hash and salt
        /// </summary>
        /// <param name="password">The password to verify</param>
        /// <param name="passwordHash">The password hash</param>
        /// <param name="passwordSalt">The password salt</param>
        /// <returns>True if the password is valid, false otherwise</returns>
        private bool VerifyPasswordHash(string password, byte[] passwordHash, byte[] passwordSalt)
        {
            using var hmac = new HMACSHA512(passwordSalt);
            var computedHash = hmac.ComputeHash(Encoding.UTF8.GetBytes(password));
            
            for (int i = 0; i < computedHash.Length; i++)
            {
                if (computedHash[i] != passwordHash[i])
                {
                    return false;
                }
            }
            
            return true;
        }
    }
} 