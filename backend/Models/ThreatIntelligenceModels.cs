using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Text.Json.Serialization;

namespace Backend.Models
{
    /// <summary>
    /// Threat Intelligence Feed configuration
    /// </summary>
    public class ThreatIntelligenceFeed
    {
        [Key]
        public string Id { get; set; } = Guid.NewGuid().ToString();

        [Required]
        [MaxLength(100)]
        public string Name { get; set; } = string.Empty;

        [MaxLength(500)]
        public string Description { get; set; } = string.Empty;

        [Required]
        [MaxLength(50)]
        public string FeedType { get; set; } = string.Empty; // MISP, STIX, JSON, CSV, XML

        [Required]
        public string FeedUrl { get; set; } = string.Empty;

        [MaxLength(100)]
        public string ApiKey { get; set; } = string.Empty;

        [MaxLength(100)]
        public string Username { get; set; } = string.Empty;

        [MaxLength(100)]
        public string Password { get; set; } = string.Empty;

        public int UpdateIntervalMinutes { get; set; } = 60;

        public bool IsActive { get; set; } = true;

        public DateTime LastUpdated { get; set; } = DateTime.UtcNow;

        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

        [MaxLength(50)]
        public string? CreatedBy { get; set; }

        public int TotalIndicators { get; set; } = 0;

        public string? LastError { get; set; }

        [MaxLength(20)]
        public string Priority { get; set; } = "Medium"; // High, Medium, Low

        [MaxLength(50)]
        public string Source { get; set; } = string.Empty; // AlienVault, VirusTotal, etc.

        public bool EnableEnrichment { get; set; } = true;

        public string? Configuration { get; set; } // JSON configuration for feed-specific settings
    }

    /// <summary>
    /// Indicator of Compromise (IOC)
    /// </summary>
    public class ThreatIndicator
    {
        [Key]
        public string Id { get; set; } = Guid.NewGuid().ToString();

        [Required]
        [MaxLength(50)]
        public string Type { get; set; } = string.Empty; // IP, Domain, Hash, URL, Email

        [Required]
        [MaxLength(500)]
        public string Value { get; set; } = string.Empty;

        [MaxLength(20)]
        public string Confidence { get; set; } = "Medium"; // High, Medium, Low

        [MaxLength(20)]
        public string Severity { get; set; } = "Medium"; // Critical, High, Medium, Low

        [MaxLength(200)]
        public string? ThreatType { get; set; } = string.Empty; // Malware, Phishing, C2, etc.

        [MaxLength(100)]
        public string? MalwareFamily { get; set; }

        [MaxLength(500)]
        public string? Description { get; set; }

        public string? Tags { get; set; } // JSON array of tags

        public DateTime FirstSeen { get; set; } = DateTime.UtcNow;

        public DateTime LastSeen { get; set; } = DateTime.UtcNow;

        public DateTime? ExpiresAt { get; set; }

        public bool IsActive { get; set; } = true;

        [Required]
        public string FeedId { get; set; } = string.Empty;

        [MaxLength(100)]
        public string? Source { get; set; }

        public string? Context { get; set; } // JSON additional context data

        public int HitCount { get; set; } = 0;

        public DateTime? LastHit { get; set; }

        // Add missing LogEntryId property
        public string? LogEntryId { get; set; }

        // Navigation properties
        [JsonIgnore]
        public virtual ThreatIntelligenceFeed? Feed { get; set; }

        [JsonIgnore]
        public virtual List<ThreatMatch> Matches { get; set; } = new();
    }

    /// <summary>
    /// Threat Match - when an indicator matches log data
    /// </summary>
    public class ThreatMatch
    {
        [Key]
        public string Id { get; set; } = Guid.NewGuid().ToString();

        [Required]
        public string IndicatorId { get; set; } = string.Empty;

        [Required]
        public string LogEntryId { get; set; } = string.Empty;

        [MaxLength(500)]
        public string MatchedValue { get; set; } = string.Empty;

        [MaxLength(100)]
        public string MatchedField { get; set; } = string.Empty; // source_ip, destination_ip, domain, etc.

        [MaxLength(20)]
        public string Confidence { get; set; } = string.Empty;

        [MaxLength(20)]
        public string Severity { get; set; } = string.Empty;

        public DateTime DetectedAt { get; set; } = DateTime.UtcNow;

        public bool IsAcknowledged { get; set; } = false;

        [MaxLength(50)]
        public string? AcknowledgedBy { get; set; }

        public DateTime? AcknowledgedAt { get; set; }

        [MaxLength(500)]
        public string? Notes { get; set; }

        public bool IsFalsePositive { get; set; } = false;

        public string? EnrichmentData { get; set; } // JSON enrichment data

        // Navigation properties
        [JsonIgnore]
        public virtual ThreatIndicator? Indicator { get; set; }

        [JsonIgnore]
        public virtual LogEntryModels? LogEntry { get; set; }
    }

    /// <summary>
    /// Threat Campaign - groups related indicators and matches
    /// </summary>
    public class ThreatCampaign
    {
        [Key]
        public string Id { get; set; } = Guid.NewGuid().ToString();

        [Required]
        [MaxLength(200)]
        public string Name { get; set; } = string.Empty;

        [MaxLength(1000)]
        public string? Description { get; set; }

        [MaxLength(100)]
        public string? Actor { get; set; }

        [MaxLength(50)]
        public string Severity { get; set; } = "Medium";

        public DateTime FirstDetected { get; set; } = DateTime.UtcNow;

        public DateTime LastDetected { get; set; } = DateTime.UtcNow;

        public bool IsActive { get; set; } = true;

        public string? TechniquesUsed { get; set; } // JSON array of MITRE ATT&CK techniques

        public string? TargetedSectors { get; set; } // JSON array of targeted sectors

        public string? Geography { get; set; } // JSON array of targeted countries

        public int IndicatorCount { get; set; } = 0;

        public int MatchCount { get; set; } = 0;

        [MaxLength(50)]
        public string? CreatedBy { get; set; }

        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

        public string? Metadata { get; set; } // JSON additional metadata
    }

    /// <summary>
    /// MITRE ATT&CK Technique tracking
    /// </summary>
    public class AttackTechnique
    {
        [Key]
        public string Id { get; set; } = Guid.NewGuid().ToString();

        [Required]
        [MaxLength(20)]
        public string TechniqueId { get; set; } = string.Empty; // T1001, T1059, etc.

        [Required]
        [MaxLength(200)]
        public string Name { get; set; } = string.Empty;

        [MaxLength(1000)]
        public string? Description { get; set; }

        [MaxLength(50)]
        public string Tactic { get; set; } = string.Empty; // Initial Access, Execution, etc.

        [MaxLength(50)]
        public string Platform { get; set; } = string.Empty; // Windows, Linux, macOS, etc.

        public string? DataSources { get; set; } // JSON array of data sources

        public string? Mitigations { get; set; } // JSON array of mitigations

        public int DetectionCount { get; set; } = 0;

        public DateTime? LastDetected { get; set; }

        public bool IsActive { get; set; } = true;
    }

    /// <summary>
    /// Threat enrichment data from external sources
    /// </summary>
    public class ThreatEnrichment
    {
        [Key]
        public string Id { get; set; } = Guid.NewGuid().ToString();

        [Required]
        [MaxLength(500)]
        public string IndicatorValue { get; set; } = string.Empty;

        [Required]
        [MaxLength(50)]
        public string IndicatorType { get; set; } = string.Empty;

        [Required]
        [MaxLength(100)]
        public string EnrichmentSource { get; set; } = string.Empty; // VirusTotal, PassiveTotal, etc.

        public string EnrichmentData { get; set; } = string.Empty; // JSON enrichment response

        public DateTime EnrichedAt { get; set; } = DateTime.UtcNow;

        public DateTime? ExpiresAt { get; set; }

        [MaxLength(20)]
        public string Status { get; set; } = "Success"; // Success, Failed, Pending

        public string? ErrorMessage { get; set; }

        public bool IsCached { get; set; } = true;
    }

    /// <summary>
    /// Whitelist for known good indicators
    /// </summary>
    public class ThreatWhitelist
    {
        [Key]
        public string Id { get; set; } = Guid.NewGuid().ToString();

        [Required]
        [MaxLength(50)]
        public string Type { get; set; } = string.Empty;

        [Required]
        [MaxLength(500)]
        public string Value { get; set; } = string.Empty;

        [MaxLength(500)]
        public string? Reason { get; set; }

        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

        [MaxLength(50)]
        public string? CreatedBy { get; set; }

        public bool IsActive { get; set; } = true;

        public DateTime? ExpiresAt { get; set; }
    }

    // DTOs for API responses

    /// <summary>
    /// DTO for Threat Intelligence Feed response
    /// </summary>
    public class ThreatIntelligenceFeedDto
    {
        public string Id { get; set; } = string.Empty;
        public string Name { get; set; } = string.Empty;
        public string Description { get; set; } = string.Empty;
        public string FeedType { get; set; } = string.Empty;
        public string FeedUrl { get; set; } = string.Empty;
        public int UpdateIntervalMinutes { get; set; }
        public bool IsActive { get; set; }
        public DateTime LastUpdated { get; set; }
        public int TotalIndicators { get; set; }
        public string? LastError { get; set; }
        public string Priority { get; set; } = string.Empty;
        public string Source { get; set; } = string.Empty;
        public bool EnableEnrichment { get; set; }
    }

    /// <summary>
    /// DTO for Threat Indicator response
    /// </summary>
    public class ThreatIndicatorDto
    {
        public string Id { get; set; } = string.Empty;
        public string Type { get; set; } = string.Empty;
        public string Value { get; set; } = string.Empty;
        public string Confidence { get; set; } = string.Empty;
        public string Severity { get; set; } = string.Empty;
        public string? ThreatType { get; set; }
        public string? MalwareFamily { get; set; }
        public string? Description { get; set; }
        public List<string> Tags { get; set; } = new();
        public DateTime FirstSeen { get; set; }
        public DateTime LastSeen { get; set; }
        public DateTime? ExpiresAt { get; set; }
        public bool IsActive { get; set; }
        public string? Source { get; set; }
        public int HitCount { get; set; }
        public DateTime? LastHit { get; set; }
        public string FeedName { get; set; } = string.Empty;
    }

    /// <summary>
    /// DTO for Threat Match response
    /// </summary>
    public class ThreatMatchDto
    {
        public string Id { get; set; } = string.Empty;
        public string IndicatorId { get; set; } = string.Empty;
        public string LogEntryId { get; set; } = string.Empty;
        public string MatchedValue { get; set; } = string.Empty;
        public string MatchedField { get; set; } = string.Empty;
        public string Confidence { get; set; } = string.Empty;
        public string Severity { get; set; } = string.Empty;
        public DateTime DetectedAt { get; set; }
        public bool IsAcknowledged { get; set; }
        public string? AcknowledgedBy { get; set; }
        public DateTime? AcknowledgedAt { get; set; }
        public bool IsFalsePositive { get; set; }
        public ThreatIndicatorDto? Indicator { get; set; }
        public object? EnrichmentData { get; set; }
        
        // Additional properties used in service
        public string Type { get; set; } = string.Empty;
        public string Value { get; set; } = string.Empty;
        public string Source { get; set; } = string.Empty;
    }

    /// <summary>
    /// DTO for creating Threat Intelligence Feed
    /// </summary>
    public class CreateThreatIntelligenceFeedDto
    {
        [Required]
        [MaxLength(100)]
        public string Name { get; set; } = string.Empty;

        [MaxLength(500)]
        public string Description { get; set; } = string.Empty;

        [Required]
        [MaxLength(50)]
        public string FeedType { get; set; } = string.Empty;

        [Required]
        public string FeedUrl { get; set; } = string.Empty;

        [MaxLength(100)]
        public string? ApiKey { get; set; }

        [MaxLength(100)]
        public string? Username { get; set; }

        [MaxLength(100)]
        public string? Password { get; set; }

        public int UpdateIntervalMinutes { get; set; } = 60;

        public bool IsActive { get; set; } = true;

        [MaxLength(20)]
        public string Priority { get; set; } = "Medium";

        [MaxLength(50)]
        public string Source { get; set; } = string.Empty;

        public bool EnableEnrichment { get; set; } = true;

        public object? Configuration { get; set; }
    }

    /// <summary>
    /// Request for acknowledging threat matches
    /// </summary>
    public class AcknowledgeThreatMatchRequest
    {
        [Required]
        public List<string> MatchIds { get; set; } = new();

        [MaxLength(500)]
        public string? Comments { get; set; }

        public bool MarkAsFalsePositive { get; set; } = false;
    }

    /// <summary>
    /// Threat Intelligence Statistics
    /// </summary>
    public class ThreatIntelligenceStats
    {
        public int TotalFeeds { get; set; }
        public int ActiveFeeds { get; set; }
        public int TotalIndicators { get; set; }
        public int ActiveIndicators { get; set; }
        public int TotalMatches { get; set; }
        public int UnacknowledgedMatches { get; set; }
        public int TodayMatches { get; set; }
        public List<IndicatorTypeCount> IndicatorsByType { get; set; } = new();
        public List<SeverityCount> MatchesBySeverity { get; set; } = new();
        public List<FeedStats> FeedStatistics { get; set; } = new();
        public List<ThreatTrendData> ThreatTrends { get; set; } = new();
    }

    public class IndicatorTypeCount
    {
        public string Type { get; set; } = string.Empty;
        public int Count { get; set; }
    }

    public class SeverityCount
    {
        public string Severity { get; set; } = string.Empty;
        public int Count { get; set; }
    }

    public class FeedStats
    {
        public string FeedName { get; set; } = string.Empty;
        public int IndicatorCount { get; set; }
        public int MatchCount { get; set; }
        public DateTime LastUpdated { get; set; }
        public string Status { get; set; } = string.Empty;
    }

    public class ThreatTrendData
    {
        public DateTime Date { get; set; }
        public int MatchCount { get; set; }
        public int NewIndicators { get; set; }
    }

    // Additional classes needed by services
    public class ThreatAnalysisResult
    {
        public string LogEntryId { get; set; } = string.Empty;
        public DateTime Timestamp { get; set; }
        public ThreatLevel ThreatLevel { get; set; }
        public double ThreatScore { get; set; }
        public List<ThreatIndicatorMatch> Indicators { get; set; } = new();
        public Dictionary<string, object> CollectorSpecificAnalysis { get; set; } = new();
    }

    public class CollectorThreatSummary
    {
        public string CollectorType { get; set; } = string.Empty;
        public DateTime AnalysisPeriod { get; set; }
        public int TotalLogs { get; set; }
        public Dictionary<ThreatLevel, int> ThreatsByLevel { get; set; } = new();
        public List<string> TopThreatIndicators { get; set; } = new();
        public List<string> RecommendedActions { get; set; } = new();
    }

    public class ThreatCorrelation
    {
        public string Pattern { get; set; } = string.Empty;
        public int Occurrences { get; set; }
        public TimeSpan TimeWindow { get; set; }
        public DateTime FirstSeen { get; set; }
        public DateTime LastSeen { get; set; }
        public List<string> CollectorsInvolved { get; set; } = new();
        public Dictionary<string, int> SeverityDistribution { get; set; } = new();
        public List<string> RecommendedActions { get; set; } = new();
    }

    public class CollectorThreatProfile
    {
        public string[] HighRiskPatterns { get; set; } = Array.Empty<string>();
        public string[] SuspiciousActivities { get; set; } = Array.Empty<string>();
        public double ThreatScoreMultiplier { get; set; } = 1.0;
    }

    public class ThreatIndicatorMatch
    {
        public string Type { get; set; } = string.Empty;
        public string Value { get; set; } = string.Empty;
        public string Source { get; set; } = string.Empty;
        public double Confidence { get; set; }
        public string Severity { get; set; } = "Medium";
    }

    public class ThreatSearchRequest
    {
        public string SearchValue { get; set; } = string.Empty;
        public string? IndicatorType { get; set; }
        public DateTime? StartDate { get; set; }
        public DateTime? EndDate { get; set; }
        public bool IncludeEnrichment { get; set; } = true;
    }

    public enum ThreatLevel
    {
        None = 0,
        Low = 1,
        Medium = 2,
        High = 3,
        Critical = 4
    }
}
