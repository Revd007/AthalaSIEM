using System.ComponentModel.DataAnnotations;

namespace Backend.DTOs;

/// <summary>
/// Add role request DTO
/// </summary>
public class AddRoleRequestDto
{
    /// <summary>
    /// Gets or sets the role ID
    /// </summary>
    [Required(ErrorMessage = "Role ID is required")]
    public string RoleId { get; set; } = string.Empty;
}
