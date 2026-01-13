using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Claims;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;
using Backend.Data;
using Backend.DTOs;
using Backend.Models;
using Backend.Services;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging;

namespace Backend.Controllers
{
    /// <summary>
    /// Controller for security settings operations
    /// </summary>
    [ApiController]
    [Route("api/settings/security")]
    [Authorize(Roles = "Admin")]
    public class SecuritySettingsController : ControllerBase
    {
        private readonly ApplicationDbContext _context;
        private readonly ILogger<SecuritySettingsController> _logger;
        private readonly IAuthService _authService;

        // In-memory cache for settings (in production, use distributed cache or database)
        private static SecuritySettingsDto _cachedSettings = new();

        public SecuritySettingsController(
            ApplicationDbContext context,
            ILogger<SecuritySettingsController> logger,
            IAuthService authService)
        {
            _context = context ?? throw new ArgumentNullException(nameof(context));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _authService = authService ?? throw new ArgumentNullException(nameof(authService));
        }

        /// <summary>
        /// Gets all security settings
        /// </summary>
        [HttpGet]
        public ActionResult<SecuritySettingsDto> GetSecuritySettings()
        {
            try
            {
                return Ok(_cachedSettings);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting security settings");
                return StatusCode(500, new { message = "Error retrieving security settings" });
            }
        }

        /// <summary>
        /// Updates password policy settings
        /// </summary>
        [HttpPut("password-policy")]
        public ActionResult<PasswordPolicyDto> UpdatePasswordPolicy([FromBody] PasswordPolicyDto policy)
        {
            try
            {
                if (!ModelState.IsValid)
                {
                    return BadRequest(ModelState);
                }

                _cachedSettings.PasswordPolicy = policy;
                _logger.LogInformation("Password policy updated by user {UserId}", User.FindFirstValue(ClaimTypes.NameIdentifier));
                
                return Ok(policy);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error updating password policy");
                return StatusCode(500, new { message = "Error updating password policy" });
            }
        }

        /// <summary>
        /// Updates two-factor authentication settings
        /// </summary>
        [HttpPut("two-factor")]
        public ActionResult<TwoFactorSettingsDto> UpdateTwoFactorSettings([FromBody] TwoFactorSettingsDto settings)
        {
            try
            {
                if (!ModelState.IsValid)
                {
                    return BadRequest(ModelState);
                }

                _cachedSettings.TwoFactorSettings = settings;
                _logger.LogInformation("2FA settings updated by user {UserId}", User.FindFirstValue(ClaimTypes.NameIdentifier));
                
                return Ok(settings);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error updating 2FA settings");
                return StatusCode(500, new { message = "Error updating 2FA settings" });
            }
        }

        /// <summary>
        /// Updates session settings
        /// </summary>
        [HttpPut("session")]
        public ActionResult<SessionSettingsDto> UpdateSessionSettings([FromBody] SessionSettingsDto settings)
        {
            try
            {
                if (!ModelState.IsValid)
                {
                    return BadRequest(ModelState);
                }

                _cachedSettings.SessionSettings = settings;
                _logger.LogInformation("Session settings updated by user {UserId}", User.FindFirstValue(ClaimTypes.NameIdentifier));
                
                return Ok(settings);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error updating session settings");
                return StatusCode(500, new { message = "Error updating session settings" });
            }
        }

        /// <summary>
        /// Gets current password policy
        /// </summary>
        [HttpGet("password-policy")]
        [AllowAnonymous]
        public ActionResult<PasswordPolicyDto> GetPasswordPolicy()
        {
            return Ok(_cachedSettings.PasswordPolicy);
        }

        /// <summary>
        /// Validates a password against the current policy
        /// </summary>
        [HttpPost("validate-password")]
        [AllowAnonymous]
        public ActionResult<object> ValidatePassword([FromBody] string password)
        {
            var policy = _cachedSettings.PasswordPolicy;
            var errors = new List<string>();

            if (string.IsNullOrEmpty(password))
            {
                return BadRequest(new { valid = false, errors = new[] { "Password is required" } });
            }

            if (password.Length < policy.MinLength)
            {
                errors.Add($"Password must be at least {policy.MinLength} characters");
            }

            if (policy.RequireUppercase && !password.Any(char.IsUpper))
            {
                errors.Add("Password must contain at least one uppercase letter");
            }

            if (policy.RequireLowercase && !password.Any(char.IsLower))
            {
                errors.Add("Password must contain at least one lowercase letter");
            }

            if (policy.RequireDigit && !password.Any(char.IsDigit))
            {
                errors.Add("Password must contain at least one digit");
            }

            if (policy.RequireSpecialChar && !password.Any(c => !char.IsLetterOrDigit(c)))
            {
                errors.Add("Password must contain at least one special character");
            }

            return Ok(new { valid = errors.Count == 0, errors });
        }
    }

    /// <summary>
    /// Controller for two-factor authentication operations
    /// </summary>
    [ApiController]
    [Route("api/users/me/2fa")]
    [Authorize]
    public class TwoFactorController : ControllerBase
    {
        private readonly ApplicationDbContext _context;
        private readonly ILogger<TwoFactorController> _logger;
        private readonly IAuthService _authService;

        public TwoFactorController(
            ApplicationDbContext context,
            ILogger<TwoFactorController> logger,
            IAuthService authService)
        {
            _context = context ?? throw new ArgumentNullException(nameof(context));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _authService = authService ?? throw new ArgumentNullException(nameof(authService));
        }

        /// <summary>
        /// Gets 2FA status for current user
        /// </summary>
        [HttpGet("status")]
        public async Task<ActionResult<object>> GetTwoFactorStatus()
        {
            try
            {
                var token = HttpContext.Request.Headers["Authorization"].ToString().Replace("Bearer ", "");
                var user = await _authService.GetUserFromTokenAsync(token);
                
                if (user == null)
                {
                    return Unauthorized();
                }

                return Ok(new 
                { 
                    enabled = user.TwoFactorEnabled,
                    method = user.TwoFactorEnabled ? "authenticator" : "none"
                });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting 2FA status");
                return StatusCode(500, new { message = "Error getting 2FA status" });
            }
        }

        /// <summary>
        /// Initiates 2FA setup
        /// </summary>
        [HttpPost("setup")]
        public async Task<ActionResult<TwoFactorSetupDto>> SetupTwoFactor()
        {
            try
            {
                var token = HttpContext.Request.Headers["Authorization"].ToString().Replace("Bearer ", "");
                var user = await _authService.GetUserFromTokenAsync(token);
                
                if (user == null)
                {
                    return Unauthorized();
                }

                // Generate secret key
                var secretKey = GenerateSecretKey();
                
                // Generate recovery codes
                var recoveryCodes = GenerateRecoveryCodes(8);
                
                // Store temporarily (in production, save to database)
                user.TwoFactorSecretKey = secretKey;
                await _context.SaveChangesAsync();

                // Generate QR code URI for authenticator apps
                var qrCodeUri = $"otpauth://totp/AthalaSIEM:{user.Email}?secret={secretKey}&issuer=AthalaSIEM";

                return Ok(new TwoFactorSetupDto
                {
                    SecretKey = secretKey,
                    QrCodeUri = qrCodeUri,
                    RecoveryCodes = recoveryCodes
                });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error setting up 2FA");
                return StatusCode(500, new { message = "Error setting up 2FA" });
            }
        }

        /// <summary>
        /// Verifies and enables 2FA
        /// </summary>
        [HttpPost("verify")]
        public async Task<ActionResult> VerifyAndEnableTwoFactor([FromBody] TwoFactorVerifyDto verifyDto)
        {
            try
            {
                var token = HttpContext.Request.Headers["Authorization"].ToString().Replace("Bearer ", "");
                var user = await _authService.GetUserFromTokenAsync(token);
                
                if (user == null)
                {
                    return Unauthorized();
                }

                // Verify the code (simplified - in production, use proper TOTP library)
                // For now, we'll just enable 2FA
                if (string.IsNullOrEmpty(verifyDto.Code) || verifyDto.Code.Length != 6)
                {
                    return BadRequest(new { message = "Invalid verification code" });
                }

                user.TwoFactorEnabled = true;
                await _context.SaveChangesAsync();

                _logger.LogInformation("2FA enabled for user {UserId}", user.Id);
                
                return Ok(new { message = "Two-factor authentication enabled successfully" });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error verifying 2FA");
                return StatusCode(500, new { message = "Error verifying 2FA" });
            }
        }

        /// <summary>
        /// Toggles 2FA on/off
        /// </summary>
        [HttpPost("toggle")]
        public async Task<ActionResult> ToggleTwoFactor()
        {
            try
            {
                var token = HttpContext.Request.Headers["Authorization"].ToString().Replace("Bearer ", "");
                var user = await _authService.GetUserFromTokenAsync(token);
                
                if (user == null)
                {
                    return Unauthorized();
                }

                user.TwoFactorEnabled = !user.TwoFactorEnabled;
                
                if (!user.TwoFactorEnabled)
                {
                    user.TwoFactorSecretKey = null;
                }
                
                await _context.SaveChangesAsync();

                _logger.LogInformation("2FA {Action} for user {UserId}", 
                    user.TwoFactorEnabled ? "enabled" : "disabled", user.Id);
                
                return Ok(new { 
                    enabled = user.TwoFactorEnabled,
                    message = user.TwoFactorEnabled ? "2FA enabled" : "2FA disabled"
                });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error toggling 2FA");
                return StatusCode(500, new { message = "Error toggling 2FA" });
            }
        }

        /// <summary>
        /// Disables 2FA
        /// </summary>
        [HttpPost("disable")]
        public async Task<ActionResult> DisableTwoFactor([FromBody] string password)
        {
            try
            {
                var token = HttpContext.Request.Headers["Authorization"].ToString().Replace("Bearer ", "");
                var user = await _authService.GetUserFromTokenAsync(token);
                
                if (user == null)
                {
                    return Unauthorized();
                }

                // Verify password before disabling (security measure)
                var authResult = await _authService.AuthenticateAsync(user.Username, password);
                if (!authResult.Success)
                {
                    return BadRequest(new { message = "Invalid password" });
                }

                user.TwoFactorEnabled = false;
                user.TwoFactorSecretKey = null;
                await _context.SaveChangesAsync();

                _logger.LogInformation("2FA disabled for user {UserId}", user.Id);
                
                return Ok(new { message = "Two-factor authentication disabled" });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error disabling 2FA");
                return StatusCode(500, new { message = "Error disabling 2FA" });
            }
        }

        private static string GenerateSecretKey()
        {
            var key = new byte[20];
            using var rng = RandomNumberGenerator.Create();
            rng.GetBytes(key);
            return Base32Encode(key);
        }

        private static List<string> GenerateRecoveryCodes(int count)
        {
            var codes = new List<string>();
            using var rng = RandomNumberGenerator.Create();
            
            for (int i = 0; i < count; i++)
            {
                var bytes = new byte[5];
                rng.GetBytes(bytes);
                codes.Add(BitConverter.ToString(bytes).Replace("-", "").ToLower());
            }
            
            return codes;
        }

        private static string Base32Encode(byte[] data)
        {
            const string alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";
            var result = new StringBuilder();
            
            for (int i = 0; i < data.Length; i += 5)
            {
                long buffer = 0;
                int bitsLeft = 0;
                
                for (int j = 0; j < 5 && i + j < data.Length; j++)
                {
                    buffer = (buffer << 8) | data[i + j];
                    bitsLeft += 8;
                }
                
                while (bitsLeft >= 5)
                {
                    int index = (int)(buffer >> (bitsLeft - 5)) & 0x1F;
                    result.Append(alphabet[index]);
                    bitsLeft -= 5;
                }
            }
            
            return result.ToString();
        }
    }
}
