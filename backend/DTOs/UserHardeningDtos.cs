using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;

namespace Backend.DTOs
{
    /// <summary>
    /// DTO for user security hardening settings
    /// </summary>
    public class UserHardeningSettingsDto
    {
        // Session Management
        [Range(1, 10)]
        public int MaxConcurrentSessions { get; set; } = 3;

        [Range(5, 1440)]
        public int SessionTimeoutMinutes { get; set; } = 60;

        public bool RequireReauthForSensitive { get; set; } = true;

        // Access Restrictions
        public bool RestrictLoginByIP { get; set; } = false;

        public List<string>? AllowedIPAddresses { get; set; }

        public bool RestrictLoginByTime { get; set; } = false;

        public List<TimeWindowDto>? AllowedTimeWindows { get; set; }

        // Login Security
        [Range(1, 20)]
        public int MaxFailedLoginAttempts { get; set; } = 5;

        [Range(1, 1440)]
        public int LockoutDurationMinutes { get; set; } = 30;

        // Password Policy
        public bool EnablePasswordExpiration { get; set; } = false;

        [Range(1, 365)]
        public int PasswordExpirationDays { get; set; } = 90;

        public bool PreventPasswordReuse { get; set; } = true;

        [Range(0, 24)]
        public int PasswordHistoryCount { get; set; } = 5;

        public bool RequireStrongPassword { get; set; } = true;

        [Range(6, 32)]
        public int MinPasswordLength { get; set; } = 8;

        public bool RequireUppercase { get; set; } = true;

        public bool RequireLowercase { get; set; } = true;

        public bool RequireDigit { get; set; } = true;

        public bool RequireSpecialChar { get; set; } = true;

        // Notifications
        public bool LogAllLoginAttempts { get; set; } = true;

        public bool EmailSecurityNotifications { get; set; } = true;
    }

    /// <summary>
    /// DTO for time window restrictions
    /// </summary>
    public class TimeWindowDto
    {
        [Required]
        public string Start { get; set; } = string.Empty; // Format: "HH:mm"

        [Required]
        public string End { get; set; } = string.Empty; // Format: "HH:mm"

        public List<string>? DaysOfWeek { get; set; } // ["Monday", "Tuesday", ...]
    }
}
