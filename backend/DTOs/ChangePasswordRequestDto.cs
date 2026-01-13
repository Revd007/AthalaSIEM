using System.ComponentModel.DataAnnotations;

namespace Backend.DTOs;

/// <summary>
/// Change password request DTO
/// </summary>
public class ChangePasswordRequestDto
{
    /// <summary>
    /// Gets or sets the current password
    /// </summary>
    [Required(ErrorMessage = "Current password is required")]
    public string CurrentPassword { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the new password
    /// </summary>
    [Required(ErrorMessage = "New password is required")]
    [StringLength(100, MinimumLength = 6, ErrorMessage = "New password must be between 6 and 100 characters")]
    public string NewPassword { get; set; } = string.Empty;
}
