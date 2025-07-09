using System.ComponentModel.DataAnnotations;
using Backend.Models;
using System;

namespace Backend.DTOs
{
    /// <summary>
    /// DTO for log ingestion
    /// </summary>
    public class LogIngestRequest
    {
        [Required]
        public DateTime Timestamp { get; set; } = DateTime.UtcNow;
        
        [Required]
        public required string LogSource { get; set; }
        
        [Required]
        public SeverityModels Severity { get; set; } = SeverityModels.Low;
        
        [Required]
        public required string RawLog { get; set; }
    }

    /// <summary>
    /// DTO for log batch processing
    /// </summary>
    public class LogEntryRequest
    {
        public DateTime Timestamp { get; set; } = DateTime.UtcNow;
        public required string Source { get; set; }
        public long EventId { get; set; }
        public required string Level { get; set; }
        public required string Message { get; set; }
        public required string MachineName { get; set; }
        public required string IPAddress { get; set; }
    }
} 