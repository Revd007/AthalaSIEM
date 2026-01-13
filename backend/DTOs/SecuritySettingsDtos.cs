using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;

namespace Backend.DTOs
{
    /// <summary>
    /// DTO for password policy settings
    /// </summary>
    public class PasswordPolicyDto
    {
        [Range(6, 32)]
        public int MinLength { get; set; } = 8;
        
        public bool RequireUppercase { get; set; } = true;
        
        public bool RequireLowercase { get; set; } = true;
        
        public bool RequireDigit { get; set; } = true;
        
        public bool RequireSpecialChar { get; set; } = true;
        
        [Range(0, 365)]
        public int MaxAge { get; set; } = 90;
        
        [Range(0, 24)]
        public int PreventReuse { get; set; } = 5;
        
        [Range(1, 20)]
        public int LockoutThreshold { get; set; } = 5;
        
        [Range(1, 1440)]
        public int LockoutDuration { get; set; } = 30;
    }

    /// <summary>
    /// DTO for two-factor authentication settings
    /// </summary>
    public class TwoFactorSettingsDto
    {
        public bool Required { get; set; } = false;
        
        public List<string> AllowedMethods { get; set; } = new() { "authenticator", "email" };
        
        [Range(0, 30)]
        public int GracePeriodDays { get; set; } = 7;
        
        [Range(0, 90)]
        public int RememberDeviceDays { get; set; } = 30;
    }

    /// <summary>
    /// DTO for session settings
    /// </summary>
    public class SessionSettingsDto
    {
        [Range(5, 1440)]
        public int SessionTimeout { get; set; } = 60;
        
        [Range(1, 10)]
        public int MaxConcurrentSessions { get; set; } = 3;
        
        public bool RequireReauthForSensitive { get; set; } = true;
    }

    /// <summary>
    /// Combined security settings DTO
    /// </summary>
    public class SecuritySettingsDto
    {
        public PasswordPolicyDto PasswordPolicy { get; set; } = new();
        
        public TwoFactorSettingsDto TwoFactorSettings { get; set; } = new();
        
        public SessionSettingsDto SessionSettings { get; set; } = new();
    }

    /// <summary>
    /// DTO for 2FA setup response
    /// </summary>
    public class TwoFactorSetupDto
    {
        public string SecretKey { get; set; } = string.Empty;
        
        public string QrCodeUri { get; set; } = string.Empty;
        
        public List<string> RecoveryCodes { get; set; } = new();
    }

    /// <summary>
    /// DTO for 2FA verification
    /// </summary>
    public class TwoFactorVerifyDto
    {
        [Required]
        public string Code { get; set; } = string.Empty;
        
        public bool RememberDevice { get; set; } = false;
    }

    /// <summary>
    /// DTO for password reset by admin
    /// </summary>
    public class AdminPasswordResetDto
    {
        public string TemporaryPassword { get; set; } = string.Empty;
        
        public bool RequireChange { get; set; } = true;
    }
}
