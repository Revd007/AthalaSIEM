using System;
using System.Collections.Generic;

namespace Backend.DTOs
{
    public class CreateThreatIndicatorRequest
    {
        public string Type { get; set; } = string.Empty;
        public string Value { get; set; } = string.Empty;
        public string Confidence { get; set; } = "Medium";
        public string Severity { get; set; } = "Medium";
        public string? ThreatType { get; set; }
        public string? MalwareFamily { get; set; }
        public string? Description { get; set; }
        public List<string>? Tags { get; set; }
        public DateTime? ExpiresAt { get; set; }
    }

    public class ThreatSearchRequest
    {
        public string SearchValue { get; set; } = string.Empty;
        public string? IndicatorType { get; set; }
        public DateTime? StartDate { get; set; }
        public DateTime? EndDate { get; set; }
        public bool IncludeEnrichment { get; set; } = true;
    }

    public class CreateWhitelistRequest
    {
        public string Type { get; set; } = string.Empty;
        public string Value { get; set; } = string.Empty;
        public string? Reason { get; set; }
        public DateTime? ExpiresAt { get; set; }
    }
}
