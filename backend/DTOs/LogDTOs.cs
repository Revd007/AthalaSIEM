using System.ComponentModel.DataAnnotations;
using Backend.Models;
using System;
using System.Collections.Generic;

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

    /// <summary>
    /// Log entry data transfer object
    /// </summary>
    public class LogEntryDto
    {
        /// <summary>
        /// Gets or sets the log entry ID
        /// </summary>
        public string Id { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the agent ID
        /// </summary>
        public string AgentId { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the timestamp
        /// </summary>
        public DateTime Timestamp { get; set; }
        
        /// <summary>
        /// Gets or sets the source
        /// </summary>
        public string Source { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the severity
        /// </summary>
        public string Severity { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the message
        /// </summary>
        public string Message { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the category
        /// </summary>
        public string Category { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the hostname
        /// </summary>
        public string Hostname { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the IP address
        /// </summary>
        public string IpAddress { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the username
        /// </summary>
        public string Username { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the process name
        /// </summary>
        public string ProcessName { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the process ID
        /// </summary>
        public int? ProcessId { get; set; }
        
        /// <summary>
        /// Gets or sets the thread ID
        /// </summary>
        public int? ThreadId { get; set; }
        
        /// <summary>
        /// Gets or sets the event ID
        /// </summary>
        public int? EventId { get; set; }
        
        /// <summary>
        /// Gets or sets the raw log data
        /// </summary>
        public string RawData { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the log properties
        /// </summary>
        public Dictionary<string, string> Properties { get; set; } = new Dictionary<string, string>();
        
        /// <summary>
        /// Gets or sets the correlation ID
        /// </summary>
        public string CorrelationId { get; set; } = string.Empty;
    }
    
    /// <summary>
    /// Log query data transfer object
    /// </summary>
    public class LogQueryDto
    {
        /// <summary>
        /// Gets or sets the search term
        /// </summary>
        public string SearchTerm { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the agent ID
        /// </summary>
        public string AgentId { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the severity
        /// </summary>
        public string Severity { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the start time
        /// </summary>
        public DateTime? StartTime { get; set; }
        
        /// <summary>
        /// Gets or sets the end time
        /// </summary>
        public DateTime? EndTime { get; set; }
        
        /// <summary>
        /// Gets or sets the category
        /// </summary>
        public string Category { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the source
        /// </summary>
        public string Source { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the event ID
        /// </summary>
        public int? EventId { get; set; }
        
        /// <summary>
        /// Gets or sets the hostname
        /// </summary>
        public string Hostname { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the username
        /// </summary>
        public string Username { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the limit
        /// </summary>
        public int Limit { get; set; } = 100;
        
        /// <summary>
        /// Gets or sets the offset
        /// </summary>
        public int Offset { get; set; } = 0;
        
        /// <summary>
        /// Gets or sets the sort field
        /// </summary>
        public string SortField { get; set; } = "Timestamp";
        
        /// <summary>
        /// Gets or sets the sort direction
        /// </summary>
        public string SortDirection { get; set; } = "desc";
    }
    
    /// <summary>
    /// Log summary data transfer object
    /// </summary>
    public class LogSummaryDto
    {
        /// <summary>
        /// Gets or sets the total logs
        /// </summary>
        public int TotalLogs { get; set; }
        
        /// <summary>
        /// Gets or sets the start time
        /// </summary>
        public DateTime StartTime { get; set; }
        
        /// <summary>
        /// Gets or sets the end time
        /// </summary>
        public DateTime EndTime { get; set; }
        
        /// <summary>
        /// Gets or sets the severity counts
        /// </summary>
        public Dictionary<string, int> SeverityCounts { get; set; } = new Dictionary<string, int>();
        
        /// <summary>
        /// Gets or sets the source counts
        /// </summary>
        public Dictionary<string, int> SourceCounts { get; set; } = new Dictionary<string, int>();
        
        /// <summary>
        /// Gets or sets the category counts
        /// </summary>
        public Dictionary<string, int> CategoryCounts { get; set; } = new Dictionary<string, int>();
        
        /// <summary>
        /// Gets or sets the agent counts
        /// </summary>
        public Dictionary<string, int> AgentCounts { get; set; } = new Dictionary<string, int>();
        
        /// <summary>
        /// Gets or sets the hourly distribution
        /// </summary>
        public Dictionary<string, int> HourlyDistribution { get; set; } = new Dictionary<string, int>();
    }
    
    /// <summary>
    /// Log trends data transfer object
    /// </summary>
    public class LogTrendsDto
    {
        /// <summary>
        /// Gets or sets the time points
        /// </summary>
        public List<DateTime> TimePoints { get; set; } = new List<DateTime>();
        
        /// <summary>
        /// Gets or sets the total counts
        /// </summary>
        public List<int> TotalCounts { get; set; } = new List<int>();
        
        /// <summary>
        /// Gets or sets the severity counts
        /// </summary>
        public Dictionary<string, List<int>> SeverityCounts { get; set; } = new Dictionary<string, List<int>>();
        
        /// <summary>
        /// Gets or sets the source counts
        /// </summary>
        public Dictionary<string, List<int>> SourceCounts { get; set; } = new Dictionary<string, List<int>>();
        
        /// <summary>
        /// Gets or sets the time interval
        /// </summary>
        public string TimeInterval { get; set; } = "day";
        
        /// <summary>
        /// Gets or sets the start time
        /// </summary>
        public DateTime StartTime { get; set; }
        
        /// <summary>
        /// Gets or sets the end time
        /// </summary>
        public DateTime EndTime { get; set; }
    }
    
    /// <summary>
    /// Log anomaly data transfer object
    /// </summary>
    public class LogAnomalyDto
    {
        /// <summary>
        /// Gets or sets the ID
        /// </summary>
        public string Id { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the timestamp
        /// </summary>
        public DateTime Timestamp { get; set; }
        
        /// <summary>
        /// Gets or sets the anomaly type
        /// </summary>
        public string AnomalyType { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the description
        /// </summary>
        public string Description { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the severity
        /// </summary>
        public string Severity { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the confidence score
        /// </summary>
        public double ConfidenceScore { get; set; }
        
        /// <summary>
        /// Gets or sets the related log IDs
        /// </summary>
        public List<string> RelatedLogIds { get; set; } = new List<string>();
        
        /// <summary>
        /// Gets or sets the affected agents
        /// </summary>
        public List<string> AffectedAgents { get; set; } = new List<string>();
        
        /// <summary>
        /// Gets or sets the details
        /// </summary>
        public Dictionary<string, string> Details { get; set; } = new Dictionary<string, string>();
    }
    
    /// <summary>
    /// Log pattern data transfer object
    /// </summary>
    public class LogPatternDto
    {
        /// <summary>
        /// Gets or sets the ID
        /// </summary>
        public string Id { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the signature
        /// </summary>
        public string Signature { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the description
        /// </summary>
        public string Description { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the occurrence count
        /// </summary>
        public int OccurrenceCount { get; set; }
        
        /// <summary>
        /// Gets or sets when the pattern was first seen
        /// </summary>
        public DateTime FirstSeen { get; set; }
        
        /// <summary>
        /// Gets or sets when the pattern was last seen
        /// </summary>
        public DateTime LastSeen { get; set; }
        
        /// <summary>
        /// Gets or sets the severity
        /// </summary>
        public string Severity { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the sources
        /// </summary>
        public List<string> Sources { get; set; } = new List<string>();
        
        /// <summary>
        /// Gets or sets the sample logs
        /// </summary>
        public List<LogEntryDto> SampleLogs { get; set; } = new List<LogEntryDto>();
    }
    
    /// <summary>
    /// Log correlation data transfer object
    /// </summary>
    public class LogCorrelationDto
    {
        /// <summary>
        /// Gets or sets the base log
        /// </summary>
        public LogEntryDto BaseLog { get; set; } = new LogEntryDto();
        
        /// <summary>
        /// Gets or sets the correlated logs
        /// </summary>
        public List<LogEntryDto> CorrelatedLogs { get; set; } = new List<LogEntryDto>();
        
        /// <summary>
        /// Gets or sets the correlation type
        /// </summary>
        public string CorrelationType { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the time window in minutes
        /// </summary>
        public int TimeWindowMinutes { get; set; }
        
        /// <summary>
        /// Gets or sets the correlation score
        /// </summary>
        public double CorrelationScore { get; set; }
        
        /// <summary>
        /// Gets or sets the related alerts
        /// </summary>
        public List<AlertDto> RelatedAlerts { get; set; } = new List<AlertDto>();
    }
} 