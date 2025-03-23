using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;

namespace Backend.DTOs
{
    /// <summary>
    /// User registration DTO
    /// </summary>
    public class UserRegisterDto
    {
        /// <summary>
        /// Gets or sets the username
        /// </summary>
        [Required]
        [StringLength(50, MinimumLength = 3)]
        public string? Username { get; set; }
        
        /// <summary>
        /// Gets or sets the email
        /// </summary>
        [Required]
        [EmailAddress]
        [StringLength(100)]
        public string? Email { get; set; }
        
        /// <summary>
        /// Gets or sets the password
        /// </summary>
        [Required]
        [StringLength(100, MinimumLength = 6)]
        public string? Password { get; set; }
        
        /// <summary>
        /// Gets or sets whether two-factor authentication is enabled
        /// </summary>
        public bool TwoFactorEnabled { get; set; }
        
        /// <summary>
        /// Gets or sets the roles for the user
        /// </summary>
        public List<string>? Roles { get; set; }
    }
} 