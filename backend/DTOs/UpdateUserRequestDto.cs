using System.ComponentModel.DataAnnotations;

namespace Backend.DTOs;

/// <summary>
/// Update user request DTO
/// </summary>
public class UpdateUserRequestDto
{
    /// <summary>
    /// Gets or sets the username
    /// </summary>
    [Required(ErrorMessage = "Username is required")]
    [StringLength(50, MinimumLength = 3, ErrorMessage = "Username must be between 3 and 50 characters")]
    public string Username { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the email
    /// </summary>
    [Required(ErrorMessage = "Email is required")]
    [EmailAddress(ErrorMessage = "Invalid email address")]
    [StringLength(100, ErrorMessage = "Email must not exceed 100 characters")]
    public string Email { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the first name
    /// </summary>
    public string? FirstName { get; set; }
    
    /// <summary>
    /// Gets or sets the last name
    /// </summary>
    public string? LastName { get; set; }
    
    /// <summary>
    /// Gets or sets a value indicating whether the user is active
    /// </summary>
    public bool IsActive { get; set; }
}
