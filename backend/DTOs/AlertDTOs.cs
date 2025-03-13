using System;
using System.Collections.Generic;

namespace Backend.DTOs
{
    /// <summary>
    /// Alert data transfer object
    /// </summary>
    public class AlertDto
    {
        /// <summary>
        /// Gets or sets the alert ID
        /// </summary>
        public string Id { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the title
        /// </summary>
        public string Title { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the description
        /// </summary>
        public string Description { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the timestamp
        /// </summary>
        public DateTime Timestamp { get; set; }
        
        /// <summary>
        /// Gets or sets the severity
        /// </summary>
        public string Severity { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the status
        /// </summary>
        public string Status { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the agent ID
        /// </summary>
        public string AgentId { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the agent name
        /// </summary>
        public string AgentName { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the source
        /// </summary>
        public string Source { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the rule ID
        /// </summary>
        public string RuleId { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the rule name
        /// </summary>
        public string RuleName { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets who the alert is assigned to
        /// </summary>
        public string AssignedTo { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets when the alert was assigned
        /// </summary>
        public DateTime? AssignedAt { get; set; }
        
        /// <summary>
        /// Gets or sets who generated the alert
        /// </summary>
        public string GeneratedBy { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets when the alert was last updated
        /// </summary>
        public DateTime? LastUpdated { get; set; }
        
        /// <summary>
        /// Gets or sets who last updated the alert
        /// </summary>
        public string LastUpdatedBy { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets when the alert was closed
        /// </summary>
        public DateTime? ClosedAt { get; set; }
        
        /// <summary>
        /// Gets or sets who closed the alert
        /// </summary>
        public string ClosedBy { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the close reason
        /// </summary>
        public string CloseReason { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the related log IDs
        /// </summary>
        public List<string> RelatedLogIds { get; set; } = new List<string>();
        
        /// <summary>
        /// Gets or sets the details
        /// </summary>
        public Dictionary<string, string> Details { get; set; } = new Dictionary<string, string>();
        
        /// <summary>
        /// Gets or sets the comments
        /// </summary>
        public List<AlertCommentDto> Comments { get; set; } = new List<AlertCommentDto>();
        
        /// <summary>
        /// Gets or sets the message
        /// </summary>
        public string Message { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets who the alert is assigned to by user ID
        /// </summary>
        public string AssignedToUserId { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets resolution notes
        /// </summary>
        public string ResolutionNotes { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets when the alert was resolved
        /// </summary>
        public DateTime? ResolvedAt { get; set; }
        
        /// <summary>
        /// Gets or sets who resolved the alert
        /// </summary>
        public string ResolvedBy { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets when the alert was created
        /// </summary>
        public DateTime CreatedAt { get; set; }
        
        /// <summary>
        /// Gets or sets when the alert was updated
        /// </summary>
        public DateTime UpdatedAt { get; set; }
    }
    
    /// <summary>
    /// Alert comment data transfer object
    /// </summary>
    public class AlertCommentDto
    {
        /// <summary>
        /// Gets or sets the comment ID
        /// </summary>
        public string Id { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the comment
        /// </summary>
        public string Comment { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the author
        /// </summary>
        public string Author { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets when the comment was created
        /// </summary>
        public DateTime CreatedAt { get; set; }
    }
    
    /// <summary>
    /// Alert query data transfer object
    /// </summary>
    public class AlertQueryDto
    {
        /// <summary>
        /// Gets or sets the search term
        /// </summary>
        public string SearchTerm { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the severity
        /// </summary>
        public string Severity { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the status
        /// </summary>
        public string Status { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the agent ID
        /// </summary>
        public string AgentId { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the source
        /// </summary>
        public string Source { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the rule ID
        /// </summary>
        public string RuleId { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets who the alert is assigned to
        /// </summary>
        public string AssignedTo { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the start time
        /// </summary>
        public DateTime? StartTime { get; set; }
        
        /// <summary>
        /// Gets or sets the end time
        /// </summary>
        public DateTime? EndTime { get; set; }
        
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
    /// Create alert data transfer object
    /// </summary>
    public class CreateAlertDto
    {
        /// <summary>
        /// Gets or sets the title
        /// </summary>
        public string Title { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the description
        /// </summary>
        public string Description { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the severity
        /// </summary>
        public string Severity { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the agent ID
        /// </summary>
        public string AgentId { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the source
        /// </summary>
        public string Source { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the rule ID
        /// </summary>
        public string RuleId { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets who the alert is assigned to
        /// </summary>
        public string AssignedTo { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets who generated the alert
        /// </summary>
        public string GeneratedBy { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the related log IDs
        /// </summary>
        public List<string> RelatedLogIds { get; set; } = new List<string>();
        
        /// <summary>
        /// Gets or sets the details
        /// </summary>
        public Dictionary<string, string> Details { get; set; } = new Dictionary<string, string>();
    }
    
    /// <summary>
    /// Update alert status data transfer object
    /// </summary>
    public class UpdateAlertStatusDto
    {
        /// <summary>
        /// Gets or sets the status
        /// </summary>
        public string Status { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets who the alert is assigned to
        /// </summary>
        public string AssignedTo { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the comment
        /// </summary>
        public string Comment { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the close reason
        /// </summary>
        public string CloseReason { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets who updated the alert
        /// </summary>
        public string UpdatedBy { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets when the alert was updated
        /// </summary>
        public DateTime UpdatedAt { get; set; }
    }
    
    /// <summary>
    /// Add alert comment data transfer object
    /// </summary>
    public class AddAlertCommentDto
    {
        /// <summary>
        /// Gets or sets the comment
        /// </summary>
        public string Comment { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the author
        /// </summary>
        public string Author { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets when the comment was created
        /// </summary>
        public DateTime CreatedAt { get; set; }
    }
    
    /// <summary>
    /// Bulk update alerts data transfer object
    /// </summary>
    public class BulkUpdateAlertsDto
    {
        /// <summary>
        /// Gets or sets the alert IDs
        /// </summary>
        public List<string> AlertIds { get; set; } = new List<string>();
        
        /// <summary>
        /// Gets or sets the status
        /// </summary>
        public string Status { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets who the alert is assigned to
        /// </summary>
        public string AssignedTo { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the comment
        /// </summary>
        public string Comment { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the close reason
        /// </summary>
        public string CloseReason { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets who updated the alert
        /// </summary>
        public string UpdatedBy { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets when the alert was updated
        /// </summary>
        public DateTime UpdatedAt { get; set; }
    }
    
    /// <summary>
    /// Bulk update result data transfer object
    /// </summary>
    public class BulkUpdateResultDto
    {
        /// <summary>
        /// Gets or sets the number of alerts updated
        /// </summary>
        public int UpdatedCount { get; set; }
        
        /// <summary>
        /// Gets or sets the number of alerts that failed to update
        /// </summary>
        public int FailedCount { get; set; }
        
        /// <summary>
        /// Gets or sets the failed alert IDs
        /// </summary>
        public List<string> FailedAlertIds { get; set; } = new List<string>();
        
        /// <summary>
        /// Gets or sets the error messages
        /// </summary>
        public Dictionary<string, string> ErrorMessages { get; set; } = new Dictionary<string, string>();
    }
    
    /// <summary>
    /// Alert summary data transfer object
    /// </summary>
    public class AlertSummaryDto
    {
        /// <summary>
        /// Gets or sets the total alerts
        /// </summary>
        public int TotalAlerts { get; set; }
        
        /// <summary>
        /// Gets or sets the open alerts
        /// </summary>
        public int OpenAlerts { get; set; }
        
        /// <summary>
        /// Gets or sets the closed alerts
        /// </summary>
        public int ClosedAlerts { get; set; }
        
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
        /// Gets or sets the status counts
        /// </summary>
        public Dictionary<string, int> StatusCounts { get; set; } = new Dictionary<string, int>();
        
        /// <summary>
        /// Gets or sets the source counts
        /// </summary>
        public Dictionary<string, int> SourceCounts { get; set; } = new Dictionary<string, int>();
        
        /// <summary>
        /// Gets or sets the agent counts
        /// </summary>
        public Dictionary<string, int> AgentCounts { get; set; } = new Dictionary<string, int>();
        
        /// <summary>
        /// Gets or sets the rule counts
        /// </summary>
        public Dictionary<string, int> RuleCounts { get; set; } = new Dictionary<string, int>();
        
        /// <summary>
        /// Gets or sets the hourly distribution
        /// </summary>
        public Dictionary<string, int> HourlyDistribution { get; set; } = new Dictionary<string, int>();
    }
    
    /// <summary>
    /// Alert trends data transfer object
    /// </summary>
    public class AlertTrendsDto
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
        /// Gets or sets the status counts
        /// </summary>
        public Dictionary<string, List<int>> StatusCounts { get; set; } = new Dictionary<string, List<int>>();
        
        /// <summary>
        /// Gets or sets the source counts
        /// </summary>
        public Dictionary<string, List<int>> SourceCounts { get; set; } = new Dictionary<string, List<int>>();
        
        /// <summary>
        /// Gets or sets the time interval
        /// </summary>
        public string TimeInterval { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the start time
        /// </summary>
        public DateTime StartTime { get; set; }
        
        /// <summary>
        /// Gets or sets the end time
        /// </summary>
        public DateTime EndTime { get; set; }
    }
} 