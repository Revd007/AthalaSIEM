using System.ComponentModel.DataAnnotations;

namespace Backend.Models
{
    public enum IOCType { IP, Domain, Hash, URL }

    public class ThreatIntelligence
    {
        public Guid Id { get; set; }
        
        [Required]
        public IOCType Type { get; set; }
        
        [Required]
        public required string Value { get; set; }
        
        [Required]
        public required string Source { get; set; }
        
        public DateTime FirstSeen { get; set; }
        public DateTime LastSeen { get; set; }
        public decimal ConfidenceScore { get; set; }
    }
} 