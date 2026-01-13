namespace Backend.DTOs
{
    using System.ComponentModel.DataAnnotations;
    using System.Text.Json.Serialization;

    public class LoginDto
    {
        [Required(ErrorMessage = "Username is required")]
        [JsonPropertyName("username")]
        public string Username { get; set; } = string.Empty;
        
        [Required(ErrorMessage = "Password is required")]
        [JsonPropertyName("password")]
        public string Password { get; set; } = string.Empty;
    }
} 