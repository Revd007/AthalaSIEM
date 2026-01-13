using Microsoft.AspNetCore.Mvc;
using Backend.Models;
using Backend.Data;
using Backend.DTOs;
using System.Linq;
using BCrypt.Net;
using System.Security.Cryptography;
using Microsoft.EntityFrameworkCore;
using System.Security.Claims;
using System.Text;
using Microsoft.IdentityModel.Tokens;
using Microsoft.AspNetCore.Authorization;
using System;
using System.Threading.Tasks;
using Backend.Services;
using Microsoft.Extensions.Logging;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Cors;

namespace Backend.Controllers
{
    /// <summary>
    /// Controller for authentication operations
    /// </summary>
    [ApiController]
    [Route("api/[controller]")]
    [EnableCors("AllowFrontend")]  // Enable CORS for this controller
    public class AuthController : ControllerBase
    {
        private readonly ApplicationDbContext _context;
        private readonly IConfiguration _config;
        private readonly IAuthService _authService;
        private readonly ILogger<AuthController> _logger;

        /// <summary>
        /// Initializes a new instance of the <see cref="AuthController"/> class
        /// </summary>
        /// <param name="context">The application database context</param>
        /// <param name="config">The configuration</param>
        /// <param name="authService">The authentication service</param>
        /// <param name="logger">The logger</param>
        public AuthController(ApplicationDbContext context, IConfiguration config, IAuthService authService, ILogger<AuthController> logger)
        {
            _context = context;
            _config = config;
            _authService = authService ?? throw new ArgumentNullException(nameof(authService));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }

        // Special endpoint just for CORS preflight
        [HttpOptions]
        [Route("{*url}")]
        public IActionResult HandleOptions()
        {
            return Ok();
        }

        [HttpPost("register")]
        [AllowAnonymous]
        [Consumes("application/json")]
        [Produces("application/json")]
        public async Task<IActionResult> Register([FromBody] UserRegisterDto? userDto)
        {
            try
            {
                _logger.LogInformation("Registration attempt for user: {Username}, Email: {Email}", userDto?.Username ?? "null", userDto?.Email ?? "null");
                
                if (userDto == null)
                {
                    _logger.LogWarning("Registration failed: Request body is null");
                    return BadRequest(new { message = "Request body is required" });
                }
                
                if (!ModelState.IsValid)
                {
                    var errors = ModelState
                        .Where(x => x.Value?.Errors.Count > 0)
                        .SelectMany(x => x.Value!.Errors)
                        .Select(x => x.ErrorMessage)
                        .ToList();
                    
                    _logger.LogWarning("Registration failed: Model validation errors: {Errors}", string.Join(", ", errors));
                    return BadRequest(new { message = "Validation failed", errors = errors });
                }
                
                if (string.IsNullOrEmpty(userDto.Username) || string.IsNullOrEmpty(userDto.Password) || string.IsNullOrEmpty(userDto.Email))
                {
                    _logger.LogWarning("Registration failed: Missing required fields. Username: {HasUsername}, Email: {HasEmail}, Password: {HasPassword}", 
                        !string.IsNullOrEmpty(userDto.Username), !string.IsNullOrEmpty(userDto.Email), !string.IsNullOrEmpty(userDto.Password));
                    return BadRequest(new { message = "Username, email, and password are required" });
                }
                
                if (await _context.Users.AnyAsync(u => u.Username == userDto.Username))
                {
                    _logger.LogWarning("Registration failed: Username {Username} already exists", userDto.Username);
                    return BadRequest(new { message = "Username already exists" });
                }
                
                if (await _context.Users.AnyAsync(u => u.Email == userDto.Email))
                {
                    _logger.LogWarning("Registration failed: Email {Email} already exists", userDto.Email);
                    return BadRequest(new { message = "Email already exists" });
                }

                using var hmac = new HMACSHA512();
                var hashBytes = hmac.ComputeHash(Encoding.UTF8.GetBytes(userDto.Password!));
                var saltBytes = hmac.Key;

                // Parse full name if provided
                string? firstName = userDto.FirstName;
                string? lastName = userDto.LastName;
                
                if (string.IsNullOrEmpty(firstName) && !string.IsNullOrEmpty(userDto.FullName))
                {
                    var nameParts = userDto.FullName.Trim().Split(' ', 2);
                    firstName = nameParts[0];
                    if (nameParts.Length > 1)
                    {
                        lastName = nameParts[1];
                    }
                }
                
                var user = new UserModels
                {
                    Username = userDto.Username,
                    Email = userDto.Email,
                    FirstName = firstName,
                    LastName = lastName,
                    PasswordHash = Convert.ToBase64String(hashBytes),
                    PasswordSalt = Convert.ToBase64String(saltBytes),
                    TwoFactorEnabled = userDto.TwoFactorEnabled,
                    IsActive = true,
                    CreatedAt = DateTime.UtcNow,
                    UpdatedAt = DateTime.UtcNow
                };
                
                // Determine if an admin is making this request (must include Authorization header with admin token)
                bool isAdminRequest = false;
                if (Request.Headers.TryGetValue("Authorization", out var authHeader))
                {
                    string token = authHeader.ToString().Replace("Bearer ", "");
                    var requestUser = await _authService.GetUserFromTokenAsync(token);
                    if (requestUser != null && await _authService.IsInRoleAsync(requestUser.Id, RoleModels.DefaultRoles.Admin))
                    {
                        isAdminRequest = true;
                    }
                }
                
                // Add roles if specified and permission allows
                List<string> rolesToAdd = new List<string>();
                
                // Handle both single Role and Roles list from frontend
                if (!string.IsNullOrEmpty(userDto.Role))
                {
                    // Map frontend role values to backend role names
                    var mappedRole = MapRoleName(userDto.Role);
                    if (!string.IsNullOrEmpty(mappedRole))
                    {
                        rolesToAdd.Add(mappedRole);
                    }
                }
                
                if (userDto.Roles != null && userDto.Roles.Count > 0)
                {
                    // Map all roles from frontend format to backend format
                    foreach (var role in userDto.Roles)
                    {
                        var mappedRole = MapRoleName(role);
                        if (!string.IsNullOrEmpty(mappedRole))
                        {
                            rolesToAdd.Add(mappedRole);
                        }
                    }
                }
                
                // Remove duplicates
                rolesToAdd = rolesToAdd.Distinct().ToList();
                
                if (rolesToAdd.Count > 0)
                {
                    // Only allow admin roles to be assigned by an admin
                    if (isAdminRequest)
                    {
                        // Admin can assign any role
                        _logger.LogInformation("Admin user assigning roles: {Roles}", string.Join(", ", rolesToAdd));
                    }
                    else
                    {
                        // Non-admin can only assign User role (filter out admin/operator roles)
                        var filteredRoles = rolesToAdd
                            .Where(r => r != RoleModels.DefaultRoles.Admin && 
                                       r != RoleModels.DefaultRoles.Operator &&
                                       r != "ADMIN" && r != "OPERATOR")
                            .ToList();
                        
                        if (filteredRoles.Count != rolesToAdd.Count)
                        {
                            _logger.LogWarning("Non-admin user attempted to assign restricted roles. Only User role will be assigned.");
                        }
                        
                        rolesToAdd = filteredRoles;
                    }
                }
                
                // Ensure at least User role is assigned if no roles are provided or all provided roles are filtered out
                if (rolesToAdd.Count == 0)
                {
                    rolesToAdd.Add(RoleModels.DefaultRoles.User);
                }
                
                // Add user to database
                _context.Users.Add(user);
                await _context.SaveChangesAsync();
                
                // Add roles to user
                foreach (var roleName in rolesToAdd)
                {
                    var role = await _context.Roles.FirstOrDefaultAsync(r => r.Name == roleName);
                    if (role == null)
                    {
                        role = new RoleModels
                        {
                            Name = roleName,
                            Description = $"Role created during registration",
                            IsSystem = false,
                            CreatedAt = DateTime.UtcNow,
                            UpdatedAt = DateTime.UtcNow
                        };
                        _context.Roles.Add(role);
                        await _context.SaveChangesAsync();
                    }
                    
                    user.UserRoles.Add(new UserRoleModels
                    {
                        UserId = user.Id,
                        RoleId = role.Id
                    });
                }
                
                await _context.SaveChangesAsync();
                _logger.LogInformation("User {Username} registered successfully with roles: {Roles}", userDto.Username, string.Join(", ", rolesToAdd));

                // Reload user with roles to ensure they're included
                await _context.Entry(user).Collection(u => u.UserRoles).LoadAsync();
                foreach (var userRole in user.UserRoles)
                {
                    await _context.Entry(userRole).Reference(ur => ur.Role).LoadAsync();
                }

                // Get roles to return in response
                var userRoles = user.UserRoles.Select(ur => ur.Role.Name).ToList();
                
                return Ok(new 
                { 
                    Id = user.Id, 
                    Username = user.Username, 
                    Email = user.Email, 
                    FirstName = user.FirstName,
                    LastName = user.LastName,
                    Roles = userRoles,
                    Role = userRoles.FirstOrDefault() ?? "User" // Primary role for backward compatibility
                });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error during registration for user {Username}", userDto.Username);
                return StatusCode(500, new { message = "An error occurred during registration", error = ex.Message });
            }
        }

        /// <summary>
        /// Registers a new admin user (requires Admin role)
        /// </summary>
        /// <param name="userDto">The user registration data</param>
        /// <returns>The registered user</returns>
        [HttpPost("register-admin")]
        [Authorize(Roles = "Admin")]
        [ProducesResponseType(StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status400BadRequest)]
        [ProducesResponseType(StatusCodes.Status401Unauthorized)]
        [ProducesResponseType(StatusCodes.Status500InternalServerError)]
        public async Task<IActionResult> RegisterAdmin(UserRegisterDto userDto)
        {
            try
            {
                _logger.LogInformation("Admin registration attempt for user: {Username}", userDto.Username);
                
                if (string.IsNullOrEmpty(userDto.Username) || string.IsNullOrEmpty(userDto.Password) || string.IsNullOrEmpty(userDto.Email))
                {
                    return BadRequest(new { message = "Username, email, and password are required" });
                }
                
                if (await _context.Users.AnyAsync(u => u.Username == userDto.Username))
                {
                    _logger.LogWarning("Admin registration failed: Username {Username} already exists", userDto.Username);
                    return BadRequest(new { message = "Username already exists" });
                }
                
                if (await _context.Users.AnyAsync(u => u.Email == userDto.Email))
                {
                    _logger.LogWarning("Admin registration failed: Email {Email} already exists", userDto.Email);
                    return BadRequest(new { message = "Email already exists" });
                }

                using var hmac = new HMACSHA512();
                var hashBytes = hmac.ComputeHash(Encoding.UTF8.GetBytes(userDto.Password!));
                var saltBytes = hmac.Key;

                var user = new UserModels
                {
                    Username = userDto.Username,
                    Email = userDto.Email,
                    PasswordHash = Convert.ToBase64String(hashBytes),
                    PasswordSalt = Convert.ToBase64String(saltBytes),
                    TwoFactorEnabled = userDto.TwoFactorEnabled,
                    IsActive = true,
                    CreatedAt = DateTime.UtcNow,
                    UpdatedAt = DateTime.UtcNow
                };
                
                // Add user to database
                _context.Users.Add(user);
                await _context.SaveChangesAsync();
                
                // Ensure user has Admin role
                List<string> rolesToAdd = new List<string> { RoleModels.DefaultRoles.Admin };
                
                // Add additional roles if specified
                if (userDto.Roles != null && userDto.Roles.Count > 0)
                {
                    foreach (var role in userDto.Roles)
                    {
                        if (!rolesToAdd.Contains(role))
                        {
                            rolesToAdd.Add(role);
                        }
                    }
                }
                
                // Add roles to user
                foreach (var roleName in rolesToAdd)
                {
                    var role = await _context.Roles.FirstOrDefaultAsync(r => r.Name == roleName);
                    if (role == null)
                    {
                        role = new RoleModels
                        {
                            Name = roleName,
                            Description = $"Role created during admin registration",
                            IsSystem = roleName == RoleModels.DefaultRoles.Admin,
                            CreatedAt = DateTime.UtcNow,
                            UpdatedAt = DateTime.UtcNow
                        };
                        _context.Roles.Add(role);
                        await _context.SaveChangesAsync();
                    }
                    
                    user.UserRoles.Add(new UserRoleModels
                    {
                        UserId = user.Id,
                        RoleId = role.Id
                    });
                }
                
                await _context.SaveChangesAsync();
                _logger.LogInformation("Admin user {Username} registered successfully with roles: {Roles}", userDto.Username, string.Join(", ", rolesToAdd));

                // Reload user with roles to ensure they're included
                await _context.Entry(user).Collection(u => u.UserRoles).LoadAsync();
                foreach (var userRole in user.UserRoles)
                {
                    await _context.Entry(userRole).Reference(ur => ur.Role).LoadAsync();
                }

                // Get roles to return in response
                var userRoles = user.UserRoles.Select(ur => ur.Role.Name).ToList();
                
                return Ok(new 
                { 
                    Id = user.Id, 
                    Username = user.Username, 
                    Email = user.Email, 
                    FirstName = user.FirstName,
                    LastName = user.LastName,
                    Roles = userRoles,
                    Role = userRoles.FirstOrDefault() ?? "User"
                });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error during admin registration for user {Username}", userDto.Username);
                return StatusCode(500, new { message = "An error occurred during admin registration", error = ex.Message });
            }
        }

        /// <summary>
        /// Authenticates a user
        /// </summary>
        /// <param name="loginDto">The login request</param>
        /// <returns>The authentication result</returns>
        [HttpPost("login")]
        [AllowAnonymous]
        [Consumes("application/json")]
        [Produces("application/json")]
        [ProducesResponseType(StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status400BadRequest)]
        [ProducesResponseType(StatusCodes.Status401Unauthorized)]
        public async Task<IActionResult> Login([FromBody] LoginDto? loginDto)
        {
            try
            {
                
                if (loginDto == null)
                {
                    _logger.LogWarning("Login failed: Request body is null");
                    return BadRequest(new { message = "Request body is required" });
                }
                
                if (!ModelState.IsValid)
                {
                    var errors = ModelState
                        .Where(x => x.Value?.Errors.Count > 0)
                        .SelectMany(x => x.Value!.Errors)
                        .Select(x => x.ErrorMessage)
                        .ToList();
                    
                    _logger.LogWarning("Login failed: Model validation errors: {Errors}", string.Join(", ", errors));
                    return BadRequest(new { message = "Validation failed", errors = errors });
                }
                
                _logger.LogInformation("Login attempt for user: {Username}", loginDto.Username);
                
                // Basic validation
                if (string.IsNullOrEmpty(loginDto.Username) || string.IsNullOrEmpty(loginDto.Password))
                {
                    _logger.LogWarning("Login failed: Missing username or password");
                    return BadRequest(new { message = "Username and password are required" });
                }
                
                var result = await _authService.AuthenticateAsync(loginDto.Username, loginDto.Password);
                
                if (!result.Success)
                {
                    _logger.LogWarning("Login failed for user {Username}: {Error}", loginDto.Username, result.ErrorMessage);
                    return Unauthorized(new { message = result.ErrorMessage });
                }
                
                var user = await _authService.GetUserFromTokenAsync(result.Token);
                if (user == null)
                {
                    _logger.LogError("Failed to get user information after successful authentication for {Username}", loginDto.Username);
                    return StatusCode(500, new { message = "Failed to get user information" });
                }

                _logger.LogInformation("Login successful for user {Username}", loginDto.Username);
                
                // Get user roles - ensure they're loaded
                var roles = await _authService.GetUserRolesAsync(user.Id);
                var rolesList = roles.ToList();
                
                _logger.LogInformation("User {Username} has roles: {Roles}", loginDto.Username, string.Join(", ", rolesList));
                
                return Ok(new
                {
                    Token = result.Token,
                    RefreshToken = result.RefreshToken,
                    User = new
                    {
                        Id = user.Id,
                        Username = user.Username,
                        Email = user.Email,
                        FirstName = user.FirstName,
                        LastName = user.LastName,
                        Roles = rolesList, // Include roles as array
                        Role = rolesList.FirstOrDefault() ?? "User" // Include primary role for backward compatibility
                    }
                });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error during login for user {Username}", loginDto.Username);
                return StatusCode(500, new { message = "An error occurred during login", error = ex.Message });
            }
        }

        /// <summary>
        /// Refreshes a token
        /// </summary>
        /// <returns>The refreshed token</returns>
        [HttpPost("refresh")]
        [Authorize]
        public async Task<ActionResult> RefreshToken([FromBody] RefreshTokenRequestDto request)
        {
            try
            {
                if (request == null || string.IsNullOrEmpty(request.RefreshToken))
                {
                    return BadRequest(new { message = "Refresh token is required" });
                }

                var token = HttpContext.Request.Headers["Authorization"].ToString().Replace("Bearer ", "");
                var user = await _authService.GetUserFromTokenAsync(token);
                
                if (user == null)
                {
                    return Unauthorized(new { message = "Invalid token" });
                }

                var result = await _authService.RefreshTokenAsync(token, request.RefreshToken);
                
                if (!result.Success)
                {
                    return Unauthorized(new { message = result.ErrorMessage });
                }
                
                return Ok(new { Token = result.Token, RefreshToken = result.RefreshToken });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error refreshing token");
                return StatusCode(500, "An error occurred while refreshing the token");
            }
        }

        [HttpPost("installer-token")]
        [Authorize]
        public IActionResult GenerateInstallerToken([FromBody] InstallerTokenRequestDto request)
        {
            try
            {
                // Get the current user from the token
                var userId = User.FindFirstValue(ClaimTypes.NameIdentifier);
                if (string.IsNullOrEmpty(userId))
                {
                    return Unauthorized("Invalid token");
                }

                // Get the installer type from the request or use default
                string type = request?.Type ?? "windows";

                // Get JWT secret key from configuration - use either JwtSettings:Secret or Jwt:Key
                var jwtKey = _config["JwtSettings:Secret"];
                if (string.IsNullOrEmpty(jwtKey))
                {
                    jwtKey = _config["Jwt:Key"];
                    if (string.IsNullOrEmpty(jwtKey))
                    {
                        return StatusCode(500, "JWT secret configuration error");
                    }
                }

                // Create token handler and get signing credentials
                var tokenHandler = new System.IdentityModel.Tokens.Jwt.JwtSecurityTokenHandler();
                var key = Encoding.ASCII.GetBytes(jwtKey);
                
                // Set token expiration (15 minutes)
                var expiresAt = DateTime.UtcNow.AddMinutes(15);
                
                // Create token descriptor with claims
                var tokenDescriptor = new SecurityTokenDescriptor
                {
                    Subject = new ClaimsIdentity(new[]
                    {
                        new Claim(ClaimTypes.NameIdentifier, userId),
                        new Claim("purpose", "installer-download"),
                        new Claim("installer-type", type)
                    }),
                    Expires = expiresAt,
                    SigningCredentials = new SigningCredentials(
                        new SymmetricSecurityKey(key),
                        SecurityAlgorithms.HmacSha256)
                };
                
                // Generate the token
                var token = tokenHandler.CreateToken(tokenDescriptor);
                var tokenString = tokenHandler.WriteToken(token);
                
                // Get the base URL for the download
                var baseUrl = $"{Request.Scheme}://{Request.Host}";
                var downloadUrl = $"{baseUrl}/api/agents/download-installer/{type}?token={Uri.EscapeDataString(tokenString)}";
                
                // Return token, expiration, and download URL
                return Ok(new
                {
                    token = tokenString,
                    expiresAt = expiresAt,
                    downloadUrl = downloadUrl
                });
            }
            catch (Exception ex)
            {
                return StatusCode(500, $"Failed to generate installer token: {ex.Message}");
            }
        }

        [HttpGet("secure-download-url")]
        [Authorize]
        public IActionResult GetSecureDownloadUrl()
        {
            try
            {
                // Get the current user from the token
                var userId = User.FindFirstValue(ClaimTypes.NameIdentifier);
                if (string.IsNullOrEmpty(userId))
                {
                    return Unauthorized("Invalid token");
                }

                // Get the download code from configuration
                var downloadCode = _config["InstallerDownloadCode"];
                if (string.IsNullOrEmpty(downloadCode))
                {
                    return StatusCode(500, "Server configuration error");
                }
                
                // Get the base URL for the download
                var baseUrl = $"{Request.Scheme}://{Request.Host}";
                var downloadUrl = $"{baseUrl}/api/agents/secure-download/{downloadCode}";
                
                return Ok(new
                {
                    downloadUrl = downloadUrl
                });
            }
            catch (Exception ex)
            {
                return StatusCode(500, $"Failed to generate secure download URL: {ex.Message}");
            }
        }

        /// <summary>
        /// Validates a token
        /// </summary>
        /// <returns>The validation result</returns>
        [HttpPost("validate")]
        public async Task<ActionResult> ValidateToken([FromBody] ValidateTokenRequestDto request)
        {
            try
            {
                var isValid = await _authService.ValidateTokenAsync(request.Token);
                
                return Ok(new { IsValid = isValid });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error validating token");
                return StatusCode(500, "An error occurred while validating the token");
            }
        }
        
        /// <summary>
        /// Maps frontend role names to backend role names
        /// </summary>
        private string MapRoleName(string? role)
        {
            if (string.IsNullOrEmpty(role))
                return string.Empty;
                
            // Map frontend uppercase role names to backend role names
            return role.ToUpperInvariant() switch
            {
                "ADMIN" => RoleModels.DefaultRoles.Admin,
                "ANALYST" => RoleModels.DefaultRoles.Analyst,
                "OPERATOR" => RoleModels.DefaultRoles.Operator,
                "VIEWER" => RoleModels.DefaultRoles.User, // Viewer maps to User
                "USER" => RoleModels.DefaultRoles.User,
                // If already in correct format, return as-is
                _ => role
            };
        }
    }
} 