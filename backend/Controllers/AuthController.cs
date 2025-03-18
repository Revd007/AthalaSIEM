using Microsoft.AspNetCore.Mvc;
using Backend.Models;
using Backend.Data;
using Backend.DTOs;
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
        public async Task<IActionResult> Register(UserRegisterDto userDto)
        {
            if (await _context.Users.AnyAsync(u => u.Username == userDto.Username))
                return BadRequest("Username already exists");

            using var hmac = new HMACSHA512();
            var hashBytes = hmac.ComputeHash(Encoding.UTF8.GetBytes(userDto.Password!));
            var saltBytes = hmac.Key;

            var user = new UserModels
            {
                Username = userDto.Username,
                Email = userDto.Email,
                PasswordHash = Convert.ToBase64String(hashBytes),
                PasswordSalt = Convert.ToBase64String(saltBytes),
                TwoFactorEnabled = userDto.TwoFactorEnabled
            };

            // Add default role
            var defaultRole = await _context.Roles.FirstOrDefaultAsync(r => r.Name == RoleModels.DefaultRoles.User);
            if (defaultRole == null)
            {
                defaultRole = new RoleModels
                {
                    Name = RoleModels.DefaultRoles.User,
                    Description = "Default user role",
                    IsSystem = true
                };
                _context.Roles.Add(defaultRole);
                await _context.SaveChangesAsync();
            }

            user.UserRoles.Add(new UserRoleModels
            {
                UserId = user.Id,
                RoleId = defaultRole.Id
            });

            _context.Users.Add(user);
            await _context.SaveChangesAsync();

            return Ok(new { user.Id, user.Username, user.Email });
        }

        /// <summary>
        /// Authenticates a user
        /// </summary>
        /// <param name="loginDto">The login request</param>
        /// <returns>The authentication result</returns>
        [HttpPost("login")]
        [ProducesResponseType(StatusCodes.Status200OK)]
        [ProducesResponseType(StatusCodes.Status400BadRequest)]
        [ProducesResponseType(StatusCodes.Status401Unauthorized)]
        public async Task<IActionResult> Login([FromBody] LoginDto loginDto)
        {
            try
            {
                // Ensure CORS headers are added for this response
                var origin = Request.Headers["Origin"].ToString();
                if (!string.IsNullOrEmpty(origin))
                {
                    Response.Headers["Access-Control-Allow-Origin"] = origin;
                    Response.Headers["Access-Control-Allow-Credentials"] = "true";
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
                        LastName = user.LastName
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
        public async Task<ActionResult> RefreshToken()
        {
            try
            {
                var token = HttpContext.Request.Headers["Authorization"].ToString().Replace("Bearer ", "");
                var user = await _authService.GetUserFromTokenAsync(token);
                
                if (user == null || string.IsNullOrEmpty(user.RefreshToken))
                {
                    return Unauthorized(new { message = "Invalid token or refresh token" });
                }

                var result = await _authService.RefreshTokenAsync(token, user.RefreshToken);
                
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
        [AllowAnonymous]
        public async Task<ActionResult> ValidateToken([FromBody] ValidateTokenRequest request)
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
    }

    /// <summary>
    /// Login request
    /// </summary>
    public class LoginDto
    {
        /// <summary>
        /// Gets or sets the username
        /// </summary>
        public string Username { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the password
        /// </summary>
        public string Password { get; set; } = string.Empty;
    }
    
    /// <summary>
    /// Validate token request
    /// </summary>
    public class ValidateTokenRequest
    {
        /// <summary>
        /// Gets or sets the token
        /// </summary>
        public string Token { get; set; } = string.Empty;
    }
} 