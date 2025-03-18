using System;
using System.Collections.Generic;
using System.IdentityModel.Tokens.Jwt;
using System.Linq;
using System.Security.Claims;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;
using Backend.Data;
using Backend.Data.Repositories;
using Backend.Models;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using Microsoft.IdentityModel.Tokens;

namespace Backend.Services
{
    /// <summary>
    /// Service for authentication operations
    /// </summary>
    public class AuthService : IAuthService
    {
        private readonly IUserRepository _userRepository;
        private readonly ApplicationDbContext _context;
        private readonly IConfiguration _configuration;
        private readonly ILogger<AuthService> _logger;
        
        /// <summary>
        /// Initializes a new instance of the <see cref="AuthService"/> class
        /// </summary>
        /// <param name="userRepository">The user repository</param>
        /// <param name="context">The database context</param>
        /// <param name="configuration">The configuration</param>
        /// <param name="logger">The logger</param>
        public AuthService(
            IUserRepository userRepository,
            ApplicationDbContext context,
            IConfiguration configuration,
            ILogger<AuthService> logger)
        {
            _userRepository = userRepository ?? throw new ArgumentNullException(nameof(userRepository));
            _context = context ?? throw new ArgumentNullException(nameof(context));
            _configuration = configuration ?? throw new ArgumentNullException(nameof(configuration));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }
        
        /// <inheritdoc/>
        public async Task<(bool Success, string Token, string RefreshToken, string ErrorMessage)> AuthenticateAsync(string username, string password)
        {
            try
            {
                _logger.LogInformation("AuthenticateAsync: Authenticating user {Username}", username);
                var user = await _userRepository.GetByUsernameAsync(username);
                
                if (user == null)
                {
                    _logger.LogWarning("Authentication failed: User {Username} not found", username);
                    return (false, string.Empty, string.Empty, "Invalid username or password");
                }
                
                if (!user.IsActive)
                {
                    _logger.LogWarning("Authentication failed: User {Username} is inactive", username);
                    return (false, string.Empty, string.Empty, "User account is inactive");
                }
                
                if (!VerifyPasswordHash(password, user.PasswordHash, user.PasswordSalt))
                {
                    _logger.LogWarning("Authentication failed: Invalid password for user {Username}", username);
                    return (false, string.Empty, string.Empty, "Invalid username or password");
                }
                
                // Authentication successful, generate tokens
                _logger.LogInformation("AuthenticateAsync: Password verification successful for user {Username}, generating token", username);
                var token = GenerateJwtToken(user);
                _logger.LogInformation("AuthenticateAsync: JWT token generated for user {Username}", username);
                var refreshToken = GenerateRefreshToken();
                
                // Update user with refresh token
                user.RefreshToken = refreshToken;
                user.RefreshTokenExpiryDate = DateTime.UtcNow.AddDays(7);
                user.LastLoginAt = DateTime.UtcNow;
                
                await _userRepository.UpdateAsync(user);
                
                _logger.LogInformation("User {Username} authenticated successfully", username);
                return (true, token, refreshToken, string.Empty);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error during authentication for user {Username}", username);
                return (false, string.Empty, string.Empty, "An error occurred during authentication");
            }
        }
        
        /// <inheritdoc/>
        public Task<bool> ValidateTokenAsync(string token)
        {
            if (string.IsNullOrEmpty(token))
            {
                return Task.FromResult(false);
            }
            
            try
            {
                var tokenHandler = new JwtSecurityTokenHandler();
                var key = Encoding.ASCII.GetBytes(GetJwtKey());
                
                tokenHandler.ValidateToken(token, new TokenValidationParameters
                {
                    ValidateIssuerSigningKey = true,
                    IssuerSigningKey = new SymmetricSecurityKey(key),
                    ValidateIssuer = false,
                    ValidateAudience = false,
                    ClockSkew = TimeSpan.Zero
                }, out SecurityToken validatedToken);
                
                return Task.FromResult(validatedToken != null);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Token validation failed");
                return Task.FromResult(false);
            }
        }
        
        /// <inheritdoc/>
        public async Task<UserModels?> GetUserFromTokenAsync(string token)
        {
            if (string.IsNullOrEmpty(token))
            {
                _logger.LogWarning("GetUserFromTokenAsync: Token is null or empty");
                return null;
            }
            
            try
            {
                _logger.LogInformation("GetUserFromTokenAsync: Starting to parse token");
                var tokenHandler = new JwtSecurityTokenHandler();
                var key = Encoding.ASCII.GetBytes(GetJwtKey());
                
                var tokenValidationParameters = new TokenValidationParameters
                {
                    ValidateIssuerSigningKey = true,
                    IssuerSigningKey = new SymmetricSecurityKey(key),
                    ValidateIssuer = false,
                    ValidateAudience = false,
                    ClockSkew = TimeSpan.Zero
                };
                
                var principal = tokenHandler.ValidateToken(token, tokenValidationParameters, out var validatedToken);
                _logger.LogInformation("GetUserFromTokenAsync: Token validated successfully");
                
                var userIdClaim = principal.FindFirst(ClaimTypes.NameIdentifier);
                if (userIdClaim == null)
                {
                    _logger.LogWarning("GetUserFromTokenAsync: NameIdentifier claim not found in token");
                    foreach (var claim in principal.Claims)
                    {
                        _logger.LogInformation("Token claim: {Type} = {Value}", claim.Type, claim.Value);
                    }
                    return null;
                }
                
                var userId = userIdClaim.Value;
                _logger.LogInformation("GetUserFromTokenAsync: Found user ID {UserId} in token", userId);
                
                var user = await _userRepository.GetByIdAsync(userId);
                if (user == null)
                {
                    _logger.LogWarning("GetUserFromTokenAsync: User with ID {UserId} not found", userId);
                }
                else
                {
                    _logger.LogInformation("GetUserFromTokenAsync: Successfully retrieved user {Username}", user.Username);
                }
                
                return user;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "GetUserFromTokenAsync: Failed to get user from token");
                return null;
            }
        }
        
        /// <inheritdoc/>
        public string GenerateJwtToken(UserModels user)
        {
            _logger.LogInformation("GenerateJwtToken: Generating token for user {Username}", user.Username);
            
            var tokenHandler = new JwtSecurityTokenHandler();
            var key = Encoding.ASCII.GetBytes(GetJwtKey());
            
            _logger.LogInformation("GenerateJwtToken: Getting roles for user {Username}", user.Username);
            var roles = _userRepository.GetUserRolesAsync(user.Id).Result;
            _logger.LogInformation("GenerateJwtToken: Found {RoleCount} roles for user {Username}", roles.Count(), user.Username);
            
            var claims = new List<Claim>
            {
                new Claim(ClaimTypes.NameIdentifier, user.Id),
                new Claim(ClaimTypes.Name, user.Username),
                new Claim(ClaimTypes.Email, user.Email)
            };
            
            // Add roles as claims
            foreach (var role in roles)
            {
                claims.Add(new Claim(ClaimTypes.Role, role));
                _logger.LogInformation("GenerateJwtToken: Added role {Role} for user {Username}", role, user.Username);
            }
            
            var tokenDescriptor = new SecurityTokenDescriptor
            {
                Subject = new ClaimsIdentity(claims),
                Expires = DateTime.UtcNow.AddHours(1),
                SigningCredentials = new SigningCredentials(new SymmetricSecurityKey(key), SecurityAlgorithms.HmacSha256Signature)
            };
            
            var token = tokenHandler.CreateToken(tokenDescriptor);
            var tokenString = tokenHandler.WriteToken(token);
            
            _logger.LogInformation("GenerateJwtToken: Token successfully generated for user {Username}", user.Username);
            return tokenString;
        }
        
        /// <inheritdoc/>
        public async Task<(bool Success, string Token, string RefreshToken, string ErrorMessage)> RefreshTokenAsync(string token, string refreshToken)
        {
            try
            {
                var user = await GetUserFromTokenAsync(token);
                if (user == null)
                {
                    return (false, string.Empty, string.Empty, "Invalid token");
                }
                
                // Validate refresh token
                if (user.RefreshToken != refreshToken || user.RefreshTokenExpiryDate <= DateTime.UtcNow)
                {
                    return (false, string.Empty, string.Empty, "Invalid or expired refresh token");
                }
                
                // Generate new tokens
                var newToken = GenerateJwtToken(user);
                var newRefreshToken = GenerateRefreshToken();
                
                // Update user with new refresh token
                user.RefreshToken = newRefreshToken;
                user.RefreshTokenExpiryDate = DateTime.UtcNow.AddDays(7);
                
                await _userRepository.UpdateAsync(user);
                
                return (true, newToken, newRefreshToken, string.Empty);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error refreshing token");
                return (false, string.Empty, string.Empty, "An error occurred while refreshing the token");
            }
        }
        
        /// <inheritdoc/>
        public async Task<bool> RevokeTokenAsync(string userId)
        {
            try
            {
                var user = await _userRepository.GetByIdAsync(userId);
                if (user == null)
                {
                    return false;
                }
                
                // Clear refresh token
                user.RefreshToken = null;
                user.RefreshTokenExpiryDate = null;
                
                await _userRepository.UpdateAsync(user);
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error revoking token for user {UserId}", userId);
                return false;
            }
        }
        
        /// <inheritdoc/>
        public async Task<bool> IsInRoleAsync(string userId, string role)
        {
            return await _userRepository.IsInRoleAsync(userId, role);
        }
        
        /// <summary>
        /// Checks if a user has a specific role
        /// </summary>
        /// <param name="userId">The user ID</param>
        /// <param name="roleName">The role name</param>
        /// <returns>True if the user has the role, false otherwise</returns>
        public async Task<bool> UserHasRoleAsync(string userId, string roleName)
        {
            try
            {
                var user = await _userRepository.GetByIdAsync(userId);
                if (user == null)
                {
                    _logger.LogWarning("User not found: {UserId}", userId);
                    return false;
                }

                return user.UserRoles.Any(ur => ur.Role.Name == roleName);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error checking user role: {UserId}, {RoleName}", userId, roleName);
                return false;
            }
        }
        
        private string GenerateRefreshToken()
        {
            var randomNumber = new byte[32];
            using var rng = RandomNumberGenerator.Create();
            rng.GetBytes(randomNumber);
            return Convert.ToBase64String(randomNumber);
        }
        
        /// <summary>
        /// Verifies a password against a hash and salt
        /// </summary>
        /// <param name="password">The password to verify</param>
        /// <param name="passwordHash">The password hash</param>
        /// <param name="passwordSalt">The password salt</param>
        /// <returns>True if the password is valid, false otherwise</returns>
        private bool VerifyPasswordHash(string password, string storedHash, string storedSalt)
        {
            try
            {
                _logger.LogInformation("VerifyPasswordHash: Verifying password");
                
                // Convert stored hash and salt from Base64 to byte arrays
                var hashBytes = Convert.FromBase64String(storedHash);
                var saltBytes = Convert.FromBase64String(storedSalt);
                
                // Use HMACSHA512 with the stored salt to hash the provided password
                using var hmac = new HMACSHA512(saltBytes);
                var computedHash = hmac.ComputeHash(Encoding.UTF8.GetBytes(password));
                
                // Compare the computed hash with the stored hash
                for (int i = 0; i < computedHash.Length; i++)
                {
                    if (i >= hashBytes.Length || computedHash[i] != hashBytes[i])
                    {
                        _logger.LogWarning("VerifyPasswordHash: Password verification failed - hash mismatch");
                        return false;
                    }
                }
                
                _logger.LogInformation("VerifyPasswordHash: Password verified successfully");
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "VerifyPasswordHash: Error during password verification");
                return false;
            }
        }
        
        private string GetJwtKey()
        {
            string key = _configuration["JwtSettings:Secret"];
            if (string.IsNullOrEmpty(key))
            {
                key = _configuration["Jwt:Key"];
                if (string.IsNullOrEmpty(key))
                {
                    throw new InvalidOperationException("JWT Secret is not configured in either JwtSettings:Secret or Jwt:Key");
                }
            }
            _logger.LogInformation("Using JWT key from configuration");
            return key;
        }
    }
} 