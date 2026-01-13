namespace Backend.DTOs;

/// <summary>
/// Refresh token request DTO
/// </summary>
public class RefreshTokenRequestDto
{
    /// <summary>
    /// Gets or sets the refresh token
    /// </summary>
    public string RefreshToken { get; set; } = string.Empty;
}
