using System;
using System.ComponentModel.DataAnnotations;

namespace AthalaSIEM.Backend.Models
{
    /// <summary>
    /// System-wide configuration settings
    /// </summary>
    public class SystemConfiguration
    {
        /// <summary>
        /// Primary key for the configuration - there should be only one record
        /// </summary>
        [Key]
        public int Id { get; set; } = 1;

        /// <summary>
        /// System-wide secret used for various security operations
        /// </summary>
        public string SystemSecret { get; set; } = string.Empty;

        /// <summary>
        /// When the configuration was last updated
        /// </summary>
        public DateTime LastUpdated { get; set; } = DateTime.UtcNow;

        public string? AdminEmail { get; set; }

        public bool RequireEmailVerification { get; set; } = false;

        public int TokenExpirationHours { get; set; } = 24;

        public int MaxLoginAttempts { get; set; } = 5;

        public int LockoutDurationMinutes { get; set; } = 30;
    }
} 