namespace Backend.DTOs;

/// <summary>
/// Validate token request DTO
/// </summary>
public class ValidateTokenRequestDto
{
    /// <summary>
    /// Gets or sets the token
    /// </summary>
    public string Token { get; set; } = string.Empty;
}
