using System;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace Backend.Models
{
    /// <summary>
    /// Represents a user-role relationship in the system
    /// </summary>
    public class UserRoleModels
    {
        /// <summary>
        /// Gets or sets the user ID
        /// </summary>
        [Required]
        public string UserId { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the role ID
        /// </summary>
        [Required]
        public string RoleId { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the user
        /// </summary>
        [ForeignKey("UserId")]
        public UserModels User { get; set; } = null!;
        
        /// <summary>
        /// Gets or sets the role
        /// </summary>
        [ForeignKey("RoleId")]
        public RoleModels Role { get; set; } = null!;

        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    }
} 