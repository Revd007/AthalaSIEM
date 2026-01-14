using System;
using System.Linq;
using System.Security.Claims;
using System.Text.Json;
using System.Threading.Tasks;
using Backend.Data;
using Backend.DTOs;
using Backend.Models;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging;

namespace Backend.Controllers
{
    /// <summary>
    /// Controller for user security hardening settings
    /// </summary>
    [ApiController]
    [Route("api/users/me/hardening")]
    [Authorize]
    public class UserHardeningController : ControllerBase
    {
        private readonly ApplicationDbContext _context;
        private readonly ILogger<UserHardeningController> _logger;

        public UserHardeningController(
            ApplicationDbContext context,
            ILogger<UserHardeningController> logger)
        {
            _context = context ?? throw new ArgumentNullException(nameof(context));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }

        /// <summary>
        /// Gets the current user's security hardening settings
        /// </summary>
        [HttpGet]
        public async Task<ActionResult<UserHardeningSettingsDto>> GetHardeningSettings()
        {
            try
            {
                var userId = User.FindFirstValue(ClaimTypes.NameIdentifier);
                if (string.IsNullOrEmpty(userId))
                {
                    return Unauthorized();
                }

                var settings = await _context.UserSecurityModels
                    .FirstOrDefaultAsync(s => s.UserId == userId);

                if (settings == null)
                {
                    // Return default settings
                    return Ok(new UserHardeningSettingsDto());
                }

                var dto = new UserHardeningSettingsDto
                {
                    MaxConcurrentSessions = settings.MaxConcurrentSessions,
                    SessionTimeoutMinutes = settings.SessionTimeoutMinutes,
                    RequireReauthForSensitive = settings.RequireReauthForSensitive,
                    RestrictLoginByIP = settings.RestrictLoginByIP,
                    AllowedIPAddresses = string.IsNullOrEmpty(settings.AllowedIPAddresses)
                        ? null
                        : JsonSerializer.Deserialize<List<string>>(settings.AllowedIPAddresses),
                    RestrictLoginByTime = settings.RestrictLoginByTime,
                    AllowedTimeWindows = string.IsNullOrEmpty(settings.AllowedTimeWindows)
                        ? null
                        : JsonSerializer.Deserialize<List<TimeWindowDto>>(settings.AllowedTimeWindows),
                    MaxFailedLoginAttempts = settings.MaxFailedLoginAttempts,
                    LockoutDurationMinutes = settings.LockoutDurationMinutes,
                    EnablePasswordExpiration = settings.EnablePasswordExpiration,
                    PasswordExpirationDays = settings.PasswordExpirationDays,
                    PreventPasswordReuse = settings.PreventPasswordReuse,
                    PasswordHistoryCount = settings.PasswordHistoryCount,
                    RequireStrongPassword = settings.RequireStrongPassword,
                    MinPasswordLength = settings.MinPasswordLength,
                    RequireUppercase = settings.RequireUppercase,
                    RequireLowercase = settings.RequireLowercase,
                    RequireDigit = settings.RequireDigit,
                    RequireSpecialChar = settings.RequireSpecialChar,
                    LogAllLoginAttempts = settings.LogAllLoginAttempts,
                    EmailSecurityNotifications = settings.EmailSecurityNotifications
                };

                return Ok(dto);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting user hardening settings");
                return StatusCode(500, new { message = "Error retrieving hardening settings" });
            }
        }

        /// <summary>
        /// Updates the current user's security hardening settings
        /// </summary>
        [HttpPut]
        public async Task<ActionResult<UserHardeningSettingsDto>> UpdateHardeningSettings([FromBody] UserHardeningSettingsDto dto)
        {
            try
            {
                if (!ModelState.IsValid)
                {
                    return BadRequest(ModelState);
                }

                var userId = User.FindFirstValue(ClaimTypes.NameIdentifier);
                if (string.IsNullOrEmpty(userId))
                {
                    return Unauthorized();
                }

                var settings = await _context.UserSecurityModels
                    .FirstOrDefaultAsync(s => s.UserId == userId);

                if (settings == null)
                {
                    settings = new UserSecurityModels
                    {
                        UserId = userId,
                        CreatedAt = DateTime.UtcNow
                    };
                    _context.UserSecurityModels.Add(settings);
                }

                // Update settings
                settings.MaxConcurrentSessions = dto.MaxConcurrentSessions;
                settings.SessionTimeoutMinutes = dto.SessionTimeoutMinutes;
                settings.RequireReauthForSensitive = dto.RequireReauthForSensitive;
                settings.RestrictLoginByIP = dto.RestrictLoginByIP;
                settings.AllowedIPAddresses = dto.AllowedIPAddresses != null
                    ? JsonSerializer.Serialize(dto.AllowedIPAddresses)
                    : null;
                settings.RestrictLoginByTime = dto.RestrictLoginByTime;
                settings.AllowedTimeWindows = dto.AllowedTimeWindows != null
                    ? JsonSerializer.Serialize(dto.AllowedTimeWindows)
                    : null;
                settings.MaxFailedLoginAttempts = dto.MaxFailedLoginAttempts;
                settings.LockoutDurationMinutes = dto.LockoutDurationMinutes;
                settings.EnablePasswordExpiration = dto.EnablePasswordExpiration;
                settings.PasswordExpirationDays = dto.PasswordExpirationDays;
                settings.PreventPasswordReuse = dto.PreventPasswordReuse;
                settings.PasswordHistoryCount = dto.PasswordHistoryCount;
                settings.RequireStrongPassword = dto.RequireStrongPassword;
                settings.MinPasswordLength = dto.MinPasswordLength;
                settings.RequireUppercase = dto.RequireUppercase;
                settings.RequireLowercase = dto.RequireLowercase;
                settings.RequireDigit = dto.RequireDigit;
                settings.RequireSpecialChar = dto.RequireSpecialChar;
                settings.LogAllLoginAttempts = dto.LogAllLoginAttempts;
                settings.EmailSecurityNotifications = dto.EmailSecurityNotifications;
                settings.UpdatedAt = DateTime.UtcNow;

                await _context.SaveChangesAsync();

                _logger.LogInformation("User hardening settings updated for user {UserId}", userId);

                return Ok(dto);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error updating user hardening settings");
                return StatusCode(500, new { message = "Error updating hardening settings" });
            }
        }
    }
}
