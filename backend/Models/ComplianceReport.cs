using System.ComponentModel.DataAnnotations;

namespace Backend.Models
{
    public class ComplianceReport
    {
        public Guid Id { get; set; }
        
        [Required]
        public required string Framework { get; set; }
        
        [Required]
        public required string ReportData { get; set; }
        
        public Guid GeneratedById { get; set; }
        public UserModels? GeneratedBy { get; set; }
        public DateTime GeneratedAt { get; set; } = DateTime.UtcNow;
    }
} 