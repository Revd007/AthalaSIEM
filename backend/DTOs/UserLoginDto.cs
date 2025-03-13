namespace Backend.DTOs
{
    using System.ComponentModel.DataAnnotations;

    public class UserLoginDto
    {
        [Required]
        public required string Username { get; set; }
        
        [Required]
        public required string Password { get; set; }
    }
} 