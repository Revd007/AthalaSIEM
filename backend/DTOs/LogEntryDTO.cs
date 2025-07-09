using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using Backend.Models;

namespace Backend.DTOs
{
    /// <summary>
    /// Data transfer object for log entries
    /// </summary>
    public class LogEntryDto
    {
        /// <summary>
        /// Gets or sets the log entry ID
        /// </summary>
        public string Id { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the agent ID that generated the log
        /// </summary>
        public string AgentId { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the timestamp of the log entry
        /// </summary>
        public DateTime Timestamp { get; set; } = DateTime.UtcNow;
        
        /// <summary>
        /// Gets or sets the log level (Information, Warning, Error, Critical)
        /// </summary>
        public string Level { get; set; } = "Information";
        
        /// <summary>
        /// Gets or sets the log source (Application, System, Security, etc.)
        /// </summary>
        public string Source { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the log category
        /// </summary>
        public string? Category { get; set; }
        
        /// <summary>
        /// Gets or sets the log message
        /// </summary>
        public string Message { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the exception details if applicable
        /// </summary>
        public string? Exception { get; set; }
        
        /// <summary>
        /// Gets or sets the stack trace if applicable
        /// </summary>
        public string? StackTrace { get; set; }
        
        /// <summary>
        /// Gets or sets the process ID that generated the log
        /// </summary>
        public int ProcessId { get; set; }
        
        /// <summary>
        /// Gets or sets the process name that generated the log
        /// </summary>
        public string? ProcessName { get; set; }
        
        /// <summary>
        /// Gets or sets the thread ID that generated the log
        /// </summary>
        public int ThreadId { get; set; }
        
        /// <summary>
        /// Gets or sets the user that was associated with the log
        /// </summary>
        public string? Username { get; set; }
        
        /// <summary>
        /// Gets or sets additional properties associated with the log
        /// </summary>
        public Dictionary<string, object>? Properties { get; set; }
        
        /// <summary>
        /// Gets or sets the date and time when the log entry was received by the server
        /// </summary>
        public DateTime ReceivedAt { get; set; } = DateTime.UtcNow;
        
        // Additional fields for comprehensive SIEM processing
        public long EventId { get; set; }
        public string IPAddress { get; set; } = string.Empty;
        public string MachineName { get; set; } = string.Empty;
        public string? UserId { get; set; }
        public string? RequestPath { get; set; }
        public string? RequestId { get; set; }
        public string? ClientIp { get; set; }
        public bool Processed { get; set; }
        public DateTime? ProcessedAt { get; set; }
        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
        public string? Details { get; set; }
        public string? SecurityRelevance { get; set; }
        public string? CollectorType { get; set; }
        public string? ComputerName { get; set; }
        public string? LogHash { get; set; }
        public string? SearchIndex { get; set; }
        public string? Severity { get; set; }
    }

    /// <summary>
    /// DTO for submitting a batch of logs from agents
    /// </summary>
    public class LogBatchDto
    {
        [Required]
        public string AgentId { get; set; } = string.Empty;
        
        [Required]
        public List<LogEntryDto> Logs { get; set; } = new();
        
        public DateTime BatchTimestamp { get; set; } = DateTime.UtcNow;
        public string? BatchId { get; set; }
        public string? CollectorType { get; set; }
        public Dictionary<string, object>? Metadata { get; set; }
    }

    /// <summary>
    /// DTO for log batch processing response
    /// </summary>
    public class LogBatchResponseDto
    {
        public bool Success { get; set; }
        public int ProcessedCount { get; set; }
        public int FailedCount { get; set; }
        public string Message { get; set; } = string.Empty;
        public string? BatchId { get; set; }
        public long ProcessingTimeMs { get; set; }
        public List<string>? Errors { get; set; }
        public DateTime ProcessedAt { get; set; } = DateTime.UtcNow;
    }

    /// <summary>
    /// DTO for log batch processing result (internal)
    /// </summary>
    public class LogBatchProcessingResult
    {
        public int ProcessedCount { get; set; }
        public int FailedCount { get; set; }
        public string BatchId { get; set; } = string.Empty;
        public long ProcessingTimeMs { get; set; }
        public List<string> Errors { get; set; } = new();
        public List<LogEntryModels> ProcessedLogs { get; set; } = new();
    }

    /// <summary>
    /// DTO for log query parameters
    /// </summary>
    public class LogQueryDto
    {
        public string? SearchTerm { get; set; }
        public string? Level { get; set; }
        public string? Severity { get; set; }
        public string? Source { get; set; }
        public string? AgentId { get; set; }
        public DateTime? StartTime { get; set; }
        public DateTime? EndTime { get; set; }
        public int Limit { get; set; } = 100;
        public int Offset { get; set; } = 0;
        public string? SortBy { get; set; } = "Timestamp";
        public string? SortOrder { get; set; } = "desc";
        public string? SortField { get; set; } = "Timestamp";
        public string? SortDirection { get; set; } = "desc";
        public List<string>? Categories { get; set; }
        public List<string>? EventIds { get; set; }
    }

    /// <summary>
    /// DTO for log summary statistics
    /// </summary>
    public class LogSummaryDto
    {
        public int TotalLogs { get; set; }
        public Dictionary<string, int> LogsByLevel { get; set; } = new();
        public Dictionary<string, int> LogsBySource { get; set; } = new();
        public Dictionary<string, int> LogsByAgent { get; set; } = new();
        public DateTime StartTime { get; set; }
        public DateTime EndTime { get; set; }
        public int CriticalCount { get; set; }
        public int ErrorCount { get; set; }
        public int WarningCount { get; set; }
        public int InfoCount { get; set; }
        public double LogsPerHour { get; set; }
        public List<HourlyLogCount> HourlyBreakdown { get; set; } = new();
    }

    /// <summary>
    /// DTO for hourly log count
    /// </summary>
    public class HourlyLogCount
    {
        public DateTime Hour { get; set; }
        public int Count { get; set; }
        public Dictionary<string, int> ByLevel { get; set; } = new();
    }

    /// <summary>
    /// DTO for log trends
    /// </summary>
    public class LogTrendsDto
    {
        public DateTime StartTime { get; set; }
        public DateTime EndTime { get; set; }
        public string Interval { get; set; } = string.Empty;
        public Dictionary<string, List<TrendDataPoint>> TrendsByLevel { get; set; } = new();
        public Dictionary<string, List<TrendDataPoint>> TrendsBySource { get; set; } = new();
        public List<DateTime> TimePoints { get; set; } = new();
        public List<int> TotalCounts { get; set; } = new();
        public Dictionary<string, List<int>> SeverityCounts { get; set; } = new();
        public Dictionary<string, List<int>> SourceCounts { get; set; } = new();
        public string TimeInterval { get; set; } = string.Empty;
    }

    /// <summary>
    /// DTO for trend data point
    /// </summary>
    public class TrendDataPoint
    {
        public DateTime Timestamp { get; set; }
        public int Count { get; set; }
        public string? Label { get; set; }
        public Dictionary<string, object>? Metadata { get; set; }
    }

    /// <summary>
    /// DTO for log anomaly
    /// </summary>
    public class LogAnomalyDto
    {
        public string Id { get; set; } = string.Empty;
        public DateTime DetectedAt { get; set; }
        public string Type { get; set; } = string.Empty;
        public string Description { get; set; } = string.Empty;
        public double AnomalyScore { get; set; }
        public string Severity { get; set; } = string.Empty;
        public List<LogEntryDto> RelatedLogs { get; set; } = new();
        public Dictionary<string, object> Metadata { get; set; } = new();
        public DateTime Timestamp { get; set; }
        public string AnomalyType { get; set; } = string.Empty;
        public double ConfidenceScore { get; set; }
        public List<string> RelatedLogIds { get; set; } = new();
        public List<string> AffectedAgents { get; set; } = new();
        public Dictionary<string, string> Details { get; set; } = new();
    }

    /// <summary>
    /// DTO for log pattern
    /// </summary>
    public class LogPatternDto
    {
        public string Id { get; set; } = string.Empty;
        public string Pattern { get; set; } = string.Empty;
        public int Frequency { get; set; }
        public DateTime FirstSeen { get; set; }
        public DateTime LastSeen { get; set; }
        public List<string> Sources { get; set; } = new();
        public string Confidence { get; set; } = string.Empty;
        public Dictionary<string, object> Metadata { get; set; } = new();
        public string Signature { get; set; } = string.Empty;
        public string Description { get; set; } = string.Empty;
        public int OccurrenceCount { get; set; }
        public string Severity { get; set; } = string.Empty;
        public List<LogEntryDto> SampleLogs { get; set; } = new();
    }

    /// <summary>
    /// DTO for log correlation
    /// </summary>
    public class LogCorrelationDto
    {
        public string LogId { get; set; } = string.Empty;
        public List<LogEntryDto> CorrelatedLogs { get; set; } = new();
        public List<AlertDto> RelatedAlerts { get; set; } = new();
        public TimeSpan TimeWindow { get; set; }
        public double CorrelationScore { get; set; }
        public Dictionary<string, object> CorrelationFactors { get; set; } = new();
        public LogEntryDto BaseLog { get; set; } = new();
        public string CorrelationType { get; set; } = string.Empty;
        public int TimeWindowMinutes { get; set; }
    }

} 