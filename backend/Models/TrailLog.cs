using System;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace AthalaSIEM.Models
{
    public class TrailLog
    {
        [Key]
        public int Id { get; set; }
        
        [Required]
        public string UserId { get; set; }
        
        [Required]
        public string Action { get; set; }
        
        [Required]
        public string Component { get; set; }
        
        [Column(TypeName = "jsonb")]
        public string Details { get; set; }
        
        [Required]
        public DateTime Timestamp { get; set; }
        
        [Required]
        public string UserAgent { get; set; }
        
        public string IpAddress { get; set; }
        
        // Navigation property
        public virtual ApplicationUser User { get; set; }
    }
} 