using System;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace Backend.Models
{
    /// <summary>
    /// User-specific security hardening settings
    /// </summary>
    [Table("user_security_settings")]
    public class UserSecurityModels
    {
        /// <summary>
        /// Gets or sets the user ID (primary key)
        /// </summary>
        [Key]
        public string UserId { get; set; } = string.Empty;

        /// <summary>
        /// Gets or sets the maximum number of concurrent sessions allowed
        /// </summary>
        [Range(1, 10)]
        public int MaxConcurrentSessions { get; set; } = 3;

        /// <summary>
        /// Gets or sets the session timeout in minutes
        /// </summary>
        [Range(5, 1440)]
        public int SessionTimeoutMinutes { get; set; } = 60;

        /// <summary>
        /// Gets or sets whether to require re-authentication for sensitive operations
        /// </summary>
        public bool RequireReauthForSensitive { get; set; } = true;

        /// <summary>
        /// Gets or sets whether login is restricted to specific IP addresses
        /// </summary>
        public bool RestrictLoginByIP { get; set; } = false;

        /// <summary>
        /// Gets or sets allowed IP addresses (comma-separated or JSON array)
        /// </summary>
        [Column(TypeName = "TEXT")]
        public string? AllowedIPAddresses { get; set; }

        /// <summary>
        /// Gets or sets whether login is restricted to specific time windows
        /// </summary>
        public bool RestrictLoginByTime { get; set; } = false;

        /// <summary>
        /// Gets or sets allowed login time windows (JSON array of {start, end} objects)
        /// </summary>
        [Column(TypeName = "TEXT")]
        public string? AllowedTimeWindows { get; set; }

        /// <summary>
        /// Gets or sets the maximum number of failed login attempts before lockout
        /// </summary>
        [Range(1, 20)]
        public int MaxFailedLoginAttempts { get; set; } = 5;

        /// <summary>
        /// Gets or sets the lockout duration in minutes after failed attempts
        /// </summary>
        [Range(1, 1440)]
        public int LockoutDurationMinutes { get; set; } = 30;

        /// <summary>
        /// Gets or sets whether to enable password expiration
        /// </summary>
        public bool EnablePasswordExpiration { get; set; } = false;

        /// <summary>
        /// Gets or sets the password expiration period in days
        /// </summary>
        [Range(1, 365)]
        public int PasswordExpirationDays { get; set; } = 90;

        /// <summary>
        /// Gets or sets whether to prevent password reuse
        /// </summary>
        public bool PreventPasswordReuse { get; set; } = true;

        /// <summary>
        /// Gets or sets the number of previous passwords to remember
        /// </summary>
        [Range(0, 24)]
        public int PasswordHistoryCount { get; set; } = 5;

        /// <summary>
        /// Gets or sets whether to require strong password
        /// </summary>
        public bool RequireStrongPassword { get; set; } = true;

        /// <summary>
        /// Gets or sets minimum password length
        /// </summary>
        [Range(6, 32)]
        public int MinPasswordLength { get; set; } = 8;

        /// <summary>
        /// Gets or sets whether password must contain uppercase letters
        /// </summary>
        public bool RequireUppercase { get; set; } = true;

        /// <summary>
        /// Gets or sets whether password must contain lowercase letters
        /// </summary>
        public bool RequireLowercase { get; set; } = true;

        /// <summary>
        /// Gets or sets whether password must contain digits
        /// </summary>
        public bool RequireDigit { get; set; } = true;

        /// <summary>
        /// Gets or sets whether password must contain special characters
        /// </summary>
        public bool RequireSpecialChar { get; set; } = true;

        /// <summary>
        /// Gets or sets whether to log all login attempts
        /// </summary>
        public bool LogAllLoginAttempts { get; set; } = true;

        /// <summary>
        /// Gets or sets whether to send email notifications for security events
        /// </summary>
        public bool EmailSecurityNotifications { get; set; } = true;

        /// <summary>
        /// Gets or sets when the settings were created
        /// </summary>
        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

        /// <summary>
        /// Gets or sets when the settings were last updated
        /// </summary>
        public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;

        /// <summary>
        /// Navigation property to user
        /// </summary>
        [ForeignKey("UserId")]
        public UserModels? User { get; set; }
    }
}
