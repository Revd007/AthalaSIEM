using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Backend.Models;
using Backend.Services;
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
    [Authorize(Roles = "Admin")]
    public class UsersController : ControllerBase
    {
        private readonly IUserService _userService;
        private readonly IAuthService _authService;
        private readonly ILogger<UsersController> _logger;
        
        /// <summary>
        /// Initializes a new instance of the <see cref="UsersController"/> class
        /// </summary>
        /// <param name="userService">The user service</param>
        /// <param name="authService">The authentication service</param>
        /// <param name="logger">The logger</param>
        public UsersController(
            IUserService userService,
            IAuthService authService,
            ILogger<UsersController> logger)
        {
            _userService = userService ?? throw new ArgumentNullException(nameof(userService));
            _authService = authService ?? throw new ArgumentNullException(nameof(authService));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }
        
        /// <summary>
        /// Gets all users
        /// </summary>
        /// <returns>All users</returns>
        [HttpGet]
        public async Task<ActionResult<IEnumerable<UserDto>>> GetAllUsers()
        {
            try
            {
                var users = await _userService.GetAllUsersAsync();
                var userDtos = new List<UserDto>();
                
                foreach (var user in users)
                {
                    userDtos.Add(MapToDto(user));
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
        public async Task<ActionResult<UserDto>> GetUserById(string id)
        {
            try
            {
                var user = await _userService.GetUserByIdAsync(id);
                
                if (user == null)
                {
                    return NotFound();
                }
                
                return Ok(MapToDto(user));
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
                var token = HttpContext.Request.Headers["Authorization"].ToString().Replace("Bearer ", "");
                var user = await _authService.GetUserFromTokenAsync(token);
                
                if (user == null)
                {
                    return Unauthorized();
                }
                
                return Ok(MapToDto(user));
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting current user");
                return StatusCode(500, "An error occurred while retrieving the current user");
            }
        }
        
        /// <summary>
        /// Creates a new user
        /// </summary>
        /// <param name="request">The create user request</param>
        /// <returns>The created user</returns>
        [HttpPost]
        public async Task<ActionResult<UserDto>> CreateUser(CreateUserRequest request)
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
                
                return CreatedAtAction(nameof(GetUserById), new { id = createdUser.Id }, MapToDto(createdUser));
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
        public async Task<ActionResult<UserDto>> UpdateUser(string id, UpdateUserRequest request)
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
                
                return Ok(MapToDto(updatedUser));
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
        public async Task<ActionResult> ChangePassword(string id, ChangePasswordRequest request)
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
        public async Task<ActionResult> ChangeMyPassword(ChangePasswordRequest request)
        {
            try
            {
                var token = HttpContext.Request.Headers["Authorization"].ToString().Replace("Bearer ", "");
                var user = await _authService.GetUserFromTokenAsync(token);
                
                if (user == null)
                {
                    return Unauthorized();
                }
                
                var result = await _userService.ChangePasswordAsync(user.Id, request.CurrentPassword, request.NewPassword);
                
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
        /// Adds a role to a user
        /// </summary>
        /// <param name="id">The user ID</param>
        /// <param name="request">The add role request</param>
        /// <returns>Success or failure</returns>
        [HttpPost("{id}/roles")]
        public async Task<ActionResult> AddRoleToUser(string id, AddRoleRequest request)
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
        /// Maps a user model to a DTO
        /// </summary>
        /// <param name="user">The user model</param>
        /// <returns>The user DTO</returns>
        private UserDto MapToDto(UserModels user)
        {
            return new UserDto
            {
                Id = user.Id,
                Username = user.Username,
                Email = user.Email,
                FirstName = user.FirstName ?? string.Empty,
                LastName = user.LastName ?? string.Empty,
                IsActive = user.IsActive,
                CreatedAt = user.CreatedAt,
                UpdatedAt = user.UpdatedAt
            };
        }
    }
    
    /// <summary>
    /// User DTO
    /// </summary>
    public class UserDto
    {
        /// <summary>
        /// Gets or sets the ID
        /// </summary>
        public string Id { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the username
        /// </summary>
        public string Username { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the email
        /// </summary>
        public string Email { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the first name
        /// </summary>
        public string FirstName { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the last name
        /// </summary>
        public string LastName { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets a value indicating whether the user is active
        /// </summary>
        public bool IsActive { get; set; }
        
        /// <summary>
        /// Gets or sets the creation timestamp
        /// </summary>
        public DateTime CreatedAt { get; set; }
        
        /// <summary>
        /// Gets or sets the update timestamp
        /// </summary>
        public DateTime UpdatedAt { get; set; }
    }
    
    /// <summary>
    /// Create user request
    /// </summary>
    public class CreateUserRequest
    {
        /// <summary>
        /// Gets or sets the username
        /// </summary>
        public string Username { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the email
        /// </summary>
        public string Email { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the password
        /// </summary>
        public string Password { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the first name
        /// </summary>
        public string FirstName { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the last name
        /// </summary>
        public string LastName { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the roles
        /// </summary>
        public List<string> Roles { get; set; } = new List<string>();
    }
    
    /// <summary>
    /// Update user request
    /// </summary>
    public class UpdateUserRequest
    {
        /// <summary>
        /// Gets or sets the username
        /// </summary>
        public string Username { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the email
        /// </summary>
        public string Email { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the first name
        /// </summary>
        public string FirstName { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the last name
        /// </summary>
        public string LastName { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets a value indicating whether the user is active
        /// </summary>
        public bool IsActive { get; set; }
    }
    
    /// <summary>
    /// Change password request
    /// </summary>
    public class ChangePasswordRequest
    {
        /// <summary>
        /// Gets or sets the current password
        /// </summary>
        public string CurrentPassword { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the new password
        /// </summary>
        public string NewPassword { get; set; } = string.Empty;
    }
    
    /// <summary>
    /// Add role request
    /// </summary>
    public class AddRoleRequest
    {
        /// <summary>
        /// Gets or sets the role ID
        /// </summary>
        public string RoleId { get; set; } = string.Empty;
    }
} 