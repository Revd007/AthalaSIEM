using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Claims;
using System.Threading.Tasks;
using Backend.Models;
using Backend.Services;
using Backend.DTOs;
using Backend.Data.Repositories;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;

namespace Backend.Controllers
{
    /// <summary>
    /// Controller for user operations
    /// </summary>
    [ApiController]
    [Route("api/[controller]")]
    [Authorize] // Require authentication for all endpoints, but not necessarily Admin role
    public class UsersController : ControllerBase
    {
        private readonly IUserService _userService;
        private readonly IAuthService _authService;
        private readonly IUserRepository _userRepository;
        private readonly ILogger<UsersController> _logger;
        
        /// <summary>
        /// Initializes a new instance of the <see cref="UsersController"/> class
        /// </summary>
        /// <param name="userService">The user service</param>
        /// <param name="authService">The authentication service</param>
        /// <param name="userRepository">The user repository</param>
        /// <param name="logger">The logger</param>
        public UsersController(
            IUserService userService,
            IAuthService authService,
            IUserRepository userRepository,
            ILogger<UsersController> logger)
        {
            _userService = userService ?? throw new ArgumentNullException(nameof(userService));
            _authService = authService ?? throw new ArgumentNullException(nameof(authService));
            _userRepository = userRepository ?? throw new ArgumentNullException(nameof(userRepository));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }
        
        /// <summary>
        /// Gets all users
        /// </summary>
        /// <returns>All users</returns>
        [HttpGet]
        [Authorize(Roles = "Admin")]
        public async Task<ActionResult<IEnumerable<UserDto>>> GetAllUsers()
        {
            try
            {
                var users = await _userService.GetAllUsersAsync();
                var userDtos = new List<UserDto>();
                
                foreach (var user in users)
                {
                    userDtos.Add(await MapToDtoAsync(user));
                }
                
                return Ok(userDtos);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting all users");
                return StatusCode(500, "An error occurred while retrieving users");
            }
        }
        
        /// <summary>
        /// Gets a user by ID
        /// </summary>
        /// <param name="id">The user ID</param>
        /// <returns>The user</returns>
        [HttpGet("{id}")]
        [Authorize(Roles = "Admin")]
        public async Task<ActionResult<UserDto>> GetUserById(string id)
        {
            try
            {
                var user = await _userService.GetUserByIdAsync(id);
                
                if (user == null)
                {
                    return NotFound();
                }
                
                return Ok(await MapToDtoAsync(user));
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting user {UserId}", id);
                return StatusCode(500, "An error occurred while retrieving the user");
            }
        }
        
        /// <summary>
        /// Gets the current user
        /// </summary>
        /// <returns>The current user</returns>
        [HttpGet("me")]
        [Authorize]
        public async Task<ActionResult<UserDto>> GetCurrentUser()
        {
            try
            {
                // Get user ID from claims (more reliable than extracting token)
                var userId = User.FindFirstValue(ClaimTypes.NameIdentifier);
                
                if (string.IsNullOrEmpty(userId))
                {
                    // Fallback: try to get from token
                    var token = HttpContext.Request.Headers["Authorization"].ToString().Replace("Bearer ", "");
                    if (string.IsNullOrEmpty(token))
                    {
                        _logger.LogWarning("GetCurrentUser: No user ID in claims and no token found");
                        return Unauthorized();
                    }
                    
                    var user = await _authService.GetUserFromTokenAsync(token);
                    if (user == null)
                    {
                        _logger.LogWarning("GetCurrentUser: Failed to get user from token");
                        return Unauthorized();
                    }
                    
                    return Ok(await MapToDtoAsync(user));
                }
                
                // Get user by ID from claims
                var userById = await _userService.GetUserByIdAsync(userId);
                if (userById == null)
                {
                    _logger.LogWarning("GetCurrentUser: User with ID {UserId} not found", userId);
                    return Unauthorized();
                }
                
                return Ok(await MapToDtoAsync(userById));
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting current user");
                return StatusCode(500, "An error occurred while retrieving the current user");
            }
        }
        
        /// <summary>
        /// Updates the current user's profile
        /// </summary>
        /// <param name="request">The update user request</param>
        /// <returns>The updated user</returns>
        [HttpPut("me")]
        [Authorize]
        public async Task<ActionResult<UserDto>> UpdateCurrentUser([FromBody] UpdateUserRequestDto request)
        {
            try
            {
                // Get user ID from claims
                var userId = User.FindFirstValue(ClaimTypes.NameIdentifier);
                
                if (string.IsNullOrEmpty(userId))
                {
                    return Unauthorized();
                }
                
                var user = await _userService.GetUserByIdAsync(userId);
                if (user == null)
                {
                    return Unauthorized();
                }
                
                user.Username = request.Username;
                user.Email = request.Email;
                user.FirstName = request.FirstName;
                user.LastName = request.LastName;
                
                var updatedUser = await _userService.UpdateUserAsync(user);
                
                return Ok(await MapToDtoAsync(updatedUser));
            }
            catch (InvalidOperationException ex)
            {
                return BadRequest(new { message = ex.Message });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error updating current user");
                return StatusCode(500, "An error occurred while updating the user");
            }
        }
        
        /// <summary>
        /// Creates a new user
        /// </summary>
        /// <param name="request">The create user request</param>
        /// <returns>The created user</returns>
        [HttpPost]
        [Authorize(Roles = "Admin")]
        public async Task<ActionResult<UserDto>> CreateUser([FromBody] CreateUserRequestDto request)
        {
            try
            {
                var user = new UserModels
                {
                    Username = request.Username,
                    Email = request.Email,
                    FirstName = request.FirstName,
                    LastName = request.LastName,
                    IsActive = true
                };
                
                var createdUser = await _userService.CreateUserAsync(user, request.Password);
                
                // Add roles if specified
                if (request.Roles != null && request.Roles.Count > 0)
                {
                    foreach (var roleId in request.Roles)
                    {
                        await _userService.AddRoleToUserAsync(createdUser.Id, roleId);
                    }
                }
                
                return CreatedAtAction(nameof(GetUserById), new { id = createdUser.Id }, await MapToDtoAsync(createdUser));
            }
            catch (InvalidOperationException ex)
            {
                return BadRequest(new { message = ex.Message });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error creating user");
                return StatusCode(500, "An error occurred while creating the user");
            }
        }
        
        /// <summary>
        /// Updates a user
        /// </summary>
        /// <param name="id">The user ID</param>
        /// <param name="request">The update user request</param>
        /// <returns>The updated user</returns>
        [HttpPut("{id}")]
        [Authorize(Roles = "Admin")]
        public async Task<ActionResult<UserDto>> UpdateUser(string id, [FromBody] UpdateUserRequestDto request)
        {
            try
            {
                var existingUser = await _userService.GetUserByIdAsync(id);
                
                if (existingUser == null)
                {
                    return NotFound();
                }
                
                existingUser.Username = request.Username;
                existingUser.Email = request.Email;
                existingUser.FirstName = request.FirstName;
                existingUser.LastName = request.LastName;
                existingUser.IsActive = request.IsActive;
                
                var updatedUser = await _userService.UpdateUserAsync(existingUser);
                
                return Ok(await MapToDtoAsync(updatedUser));
            }
            catch (KeyNotFoundException)
            {
                return NotFound();
            }
            catch (InvalidOperationException ex)
            {
                return BadRequest(new { message = ex.Message });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error updating user {UserId}", id);
                return StatusCode(500, "An error occurred while updating the user");
            }
        }
        
        /// <summary>
        /// Changes a user's password
        /// </summary>
        /// <param name="id">The user ID</param>
        /// <param name="request">The change password request</param>
        /// <returns>Success or failure</returns>
        [HttpPut("{id}/password")]
        [Authorize(Roles = "Admin")]
        public async Task<ActionResult> ChangePassword(string id, [FromBody] ChangePasswordRequestDto request)
        {
            try
            {
                var result = await _userService.ChangePasswordAsync(id, request.CurrentPassword, request.NewPassword);
                
                if (!result)
                {
                    return BadRequest(new { message = "Invalid current password" });
                }
                
                return Ok(new { message = "Password changed successfully" });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error changing password for user {UserId}", id);
                return StatusCode(500, "An error occurred while changing the password");
            }
        }
        
        /// <summary>
        /// Changes the current user's password
        /// </summary>
        /// <param name="request">The change password request</param>
        /// <returns>Success or failure</returns>
        [HttpPut("me/password")]
        [Authorize]
        public async Task<ActionResult> ChangeMyPassword([FromBody] ChangePasswordRequestDto request)
        {
            try
            {
                // Get user ID from claims
                var userId = User.FindFirstValue(ClaimTypes.NameIdentifier);
                
                if (string.IsNullOrEmpty(userId))
                {
                    return Unauthorized();
                }
                
                var result = await _userService.ChangePasswordAsync(userId, request.CurrentPassword, request.NewPassword);
                
                if (!result)
                {
                    return BadRequest(new { message = "Invalid current password" });
                }
                
                return Ok(new { message = "Password changed successfully" });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error changing password for current user");
                return StatusCode(500, "An error occurred while changing the password");
            }
        }
        
        /// <summary>
        /// Gets the current user's notification settings
        /// </summary>
        /// <returns>Notification settings</returns>
        [HttpGet("me/notifications")]
        [Authorize]
        public ActionResult<object> GetNotificationSettings()
        {
            // Return default notification settings (in production, store in database)
            return Ok(new
            {
                emailAlerts = true,
                pushNotifications = true,
                securityAlerts = true,
                reportNotifications = false,
                maintenanceNotifications = true
            });
        }
        
        /// <summary>
        /// Updates the current user's notification settings
        /// </summary>
        /// <param name="settings">The notification settings</param>
        /// <returns>Success</returns>
        [HttpPut("me/notifications")]
        [Authorize]
        public ActionResult UpdateNotificationSettings([FromBody] object settings)
        {
            // In production, save to database linked to user
            // For now, just return success
            return Ok(new { message = "Notification settings updated" });
        }
        
        /// <summary>
        /// Gets the current user's preferences
        /// </summary>
        /// <returns>User preferences</returns>
        [HttpGet("me/preferences")]
        [Authorize]
        public ActionResult<object> GetPreferences()
        {
            // Return default preferences (in production, store in database)
            return Ok(new
            {
                theme = "system",
                language = "en",
                timezone = "UTC",
                dateFormat = "MM/dd/yyyy",
                timeFormat = "24h"
            });
        }
        
        /// <summary>
        /// Updates the current user's preferences
        /// </summary>
        /// <param name="preferences">The preferences</param>
        /// <returns>Success</returns>
        [HttpPut("me/preferences")]
        [Authorize]
        public ActionResult UpdatePreferences([FromBody] object preferences)
        {
            // In production, save to database linked to user
            // For now, just return success
            return Ok(new { message = "Preferences updated" });
        }
        
        /// <summary>
        /// Adds a role to a user
        /// </summary>
        /// <param name="id">The user ID</param>
        /// <param name="request">The add role request</param>
        /// <returns>Success or failure</returns>
        [HttpPost("{id}/roles")]
        [Authorize(Roles = "Admin")]
        public async Task<ActionResult> AddRoleToUser(string id, [FromBody] AddRoleRequestDto request)
        {
            try
            {
                var result = await _userService.AddRoleToUserAsync(id, request.RoleId);
                
                if (!result)
                {
                    return BadRequest(new { message = "Failed to add role to user" });
                }
                
                return Ok(new { message = "Role added successfully" });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error adding role {RoleId} to user {UserId}", request.RoleId, id);
                return StatusCode(500, "An error occurred while adding the role");
            }
        }
        
        /// <summary>
        /// Removes a role from a user
        /// </summary>
        /// <param name="id">The user ID</param>
        /// <param name="roleId">The role ID</param>
        /// <returns>Success or failure</returns>
        [HttpDelete("{id}/roles/{roleId}")]
        [Authorize(Roles = "Admin")]
        public async Task<ActionResult> RemoveRoleFromUser(string id, string roleId)
        {
            try
            {
                var result = await _userService.RemoveRoleFromUserAsync(id, roleId);
                
                if (!result)
                {
                    return BadRequest(new { message = "Failed to remove role from user" });
                }
                
                return Ok(new { message = "Role removed successfully" });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error removing role {RoleId} from user {UserId}", roleId, id);
                return StatusCode(500, "An error occurred while removing the role");
            }
        }
        
        /// <summary>
        /// Deletes a user
        /// </summary>
        /// <param name="id">The user ID</param>
        /// <returns>No content</returns>
        [HttpDelete("{id}")]
        [Authorize(Roles = "Admin")]
        public async Task<ActionResult> DeleteUser(string id)
        {
            try
            {
                var result = await _userService.DeleteUserAsync(id);
                
                if (!result)
                {
                    return NotFound();
                }
                
                return NoContent();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error deleting user {UserId}", id);
                return StatusCode(500, "An error occurred while deleting the user");
            }
        }
        
        /// <summary>
        /// Resets a user's password (admin only)
        /// </summary>
        /// <param name="id">The user ID</param>
        /// <returns>Temporary password</returns>
        [HttpPost("{id}/reset-password")]
        [Authorize(Roles = "Admin")]
        public async Task<ActionResult<object>> ResetUserPassword(string id)
        {
            try
            {
                var user = await _userService.GetUserByIdAsync(id);
                
                if (user == null)
                {
                    return NotFound(new { message = "User not found" });
                }
                
                // Generate a temporary password
                var tempPassword = GenerateTemporaryPassword();
                
                // Update the user's password
                var result = await _userService.AdminResetPasswordAsync(id, tempPassword);
                
                if (!result)
                {
                    return BadRequest(new { message = "Failed to reset password" });
                }
                
                _logger.LogInformation("Password reset for user {UserId} by admin {AdminId}", 
                    id, User.FindFirst(System.Security.Claims.ClaimTypes.NameIdentifier)?.Value);
                
                return Ok(new { 
                    temporaryPassword = tempPassword,
                    requireChange = true,
                    message = "Password reset successfully. User must change password on next login."
                });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error resetting password for user {UserId}", id);
                return StatusCode(500, new { message = "An error occurred while resetting the password" });
            }
        }
        
        /// <summary>
        /// Generates a temporary password
        /// </summary>
        private static string GenerateTemporaryPassword()
        {
            const string uppercase = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
            const string lowercase = "abcdefghijklmnopqrstuvwxyz";
            const string digits = "0123456789";
            const string special = "!@#$%^&*";
            
            var random = new Random();
            var password = new System.Text.StringBuilder();
            
            // Ensure at least one of each type
            password.Append(uppercase[random.Next(uppercase.Length)]);
            password.Append(lowercase[random.Next(lowercase.Length)]);
            password.Append(digits[random.Next(digits.Length)]);
            password.Append(special[random.Next(special.Length)]);
            
            // Fill the rest randomly
            const string allChars = uppercase + lowercase + digits + special;
            for (int i = 0; i < 8; i++)
            {
                password.Append(allChars[random.Next(allChars.Length)]);
            }
            
            // Shuffle
            var array = password.ToString().ToCharArray();
            for (int i = array.Length - 1; i > 0; i--)
            {
                int j = random.Next(i + 1);
                (array[i], array[j]) = (array[j], array[i]);
            }
            
            return new string(array);
        }
        
        /// <summary>
        /// Maps a user model to a DTO
        /// </summary>
        /// <param name="user">The user model</param>
        /// <returns>The user DTO</returns>
        private async Task<UserDto> MapToDtoAsync(UserModels user)
        {
            var roles = await _userRepository.GetUserRolesAsync(user.Id);
            
            return new UserDto
            {
                Id = user.Id,
                Username = user.Username,
                Email = user.Email,
                FirstName = user.FirstName ?? string.Empty,
                LastName = user.LastName ?? string.Empty,
                IsActive = user.IsActive,
                CreatedAt = user.CreatedAt,
                UpdatedAt = user.UpdatedAt,
                Roles = roles.ToList(),
                TwoFactorEnabled = user.TwoFactorEnabled
            };
        }
    }
} 