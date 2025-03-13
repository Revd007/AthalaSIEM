using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Text.Json.Serialization;
using System.Linq;

namespace Backend.Models
{
    /// <summary>
    /// Represents a user in the system
    /// </summary>
    public class UserModels
    {
        /// <summary>
        /// Gets or sets the user ID
        /// </summary>
        [Key]
        public string Id { get; set; } = Guid.NewGuid().ToString();
        
        /// <summary>
        /// Gets or sets the username
        /// </summary>
        [Required]
        [MaxLength(50)]
        public string Username { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the email address
        /// </summary>
        [Required]
        [MaxLength(100)]
        [EmailAddress]
        public string Email { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the first name
        /// </summary>
        [MaxLength(50)]
        public string? FirstName { get; set; }
        
        /// <summary>
        /// Gets or sets the last name
        /// </summary>
        [MaxLength(50)]
        public string? LastName { get; set; }
        
        /// <summary>
        /// Gets or sets the password hash
        /// </summary>
        [Required]
        public string PasswordHash { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the password salt
        /// </summary>
        [JsonIgnore]
        public string PasswordSalt { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets a value indicating whether two-factor authentication is enabled
        /// </summary>
        public bool TwoFactorEnabled { get; set; }
        
        /// <summary>
        /// Gets or sets the two-factor secret key
        /// </summary>
        [JsonIgnore]
        public string? TwoFactorSecretKey { get; set; }
        
        /// <summary>
        /// Gets or sets a value indicating whether the user is active
        /// </summary>
        public bool IsActive { get; set; } = true;
        
        /// <summary>
        /// Gets or sets the timestamp when the user was created
        /// </summary>
        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
        
        /// <summary>
        /// Gets or sets the timestamp when the user was last updated
        /// </summary>
        public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;
        
        /// <summary>
        /// Gets or sets the timestamp when the user last logged in
        /// </summary>
        public DateTime? LastLoginAt { get; set; }
        
        /// <summary>
        /// Gets or sets the refresh token
        /// </summary>
        public string? RefreshToken { get; set; }
        
        /// <summary>
        /// Gets or sets the refresh token expiry date
        /// </summary>
        public DateTime? RefreshTokenExpiryDate { get; set; }
        
        /// <summary>
        /// Gets or sets the user roles
        /// </summary>
        public List<UserRoleModels> UserRoles { get; set; } = new List<UserRoleModels>();
        
        /// <summary>
        /// Gets or sets the dashboards created by the user
        /// </summary>
        public List<DashboardModels> Dashboards { get; set; } = new List<DashboardModels>();
        
        /// <summary>
        /// Gets or sets the reports created by the user
        /// </summary>
        public List<ReportModels> Reports { get; set; } = new List<ReportModels>();

        /// <summary>
        /// Gets the user's role names
        /// </summary>
        [JsonIgnore]
        public IEnumerable<string> Roles => UserRoles.Select(ur => ur.Role.Name);
    }
} 