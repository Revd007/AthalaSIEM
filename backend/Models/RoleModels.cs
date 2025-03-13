using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;

namespace Backend.Models
{
    /// <summary>
    /// Represents a role in the system
    /// </summary>
    public class RoleModels
    {
        /// <summary>
        /// Gets or sets the role ID
        /// </summary>
        [Key]
        public string Id { get; set; } = Guid.NewGuid().ToString();
        
        /// <summary>
        /// Gets or sets the role name
        /// </summary>
        [Required]
        [MaxLength(50)]
        public string Name { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the role description
        /// </summary>
        [MaxLength(200)]
        public string? Description { get; set; }
        
        /// <summary>
        /// Gets or sets whether this is a system role
        /// </summary>
        public bool IsSystem { get; set; }
        
        /// <summary>
        /// Gets or sets the timestamp when the role was created
        /// </summary>
        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
        
        /// <summary>
        /// Gets or sets the timestamp when the role was last updated
        /// </summary>
        public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;
        
        /// <summary>
        /// Gets or sets the users in this role
        /// </summary>
        public List<UserRoleModels> UserRoles { get; set; } = new List<UserRoleModels>();

        /// <summary>
        /// Default system roles
        /// </summary>
        public static class DefaultRoles
        {
            public const string Admin = "Admin";
            public const string Operator = "Operator";
            public const string Analyst = "Analyst";
            public const string User = "User";
        }
    }
} 