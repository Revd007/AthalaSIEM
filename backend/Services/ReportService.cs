using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;
using Backend.Data.Repositories;
using Backend.Models;
using Microsoft.Extensions.Logging;

namespace Backend.Services
{
    /// <summary>
    /// Service for report operations
    /// </summary>
    public class ReportService : IReportService
    {
        private readonly IReportRepository _reportRepository;
        private readonly IAgentRepository _agentRepository;
        private readonly ILogEntryRepository _logEntryRepository;
        private readonly IAlertRepository _alertRepository;
        private readonly ILogger<ReportService> _logger;
        
        /// <summary>
        /// Initializes a new instance of the <see cref="ReportService"/> class
        /// </summary>
        /// <param name="reportRepository">The report repository</param>
        /// <param name="agentRepository">The agent repository</param>
        /// <param name="logEntryRepository">The log entry repository</param>
        /// <param name="alertRepository">The alert repository</param>
        /// <param name="logger">The logger</param>
        public ReportService(
            IReportRepository reportRepository,
            IAgentRepository agentRepository,
            ILogEntryRepository logEntryRepository,
            IAlertRepository alertRepository,
            ILogger<ReportService> logger)
        {
            _reportRepository = reportRepository ?? throw new ArgumentNullException(nameof(reportRepository));
            _agentRepository = agentRepository ?? throw new ArgumentNullException(nameof(agentRepository));
            _logEntryRepository = logEntryRepository ?? throw new ArgumentNullException(nameof(logEntryRepository));
            _alertRepository = alertRepository ?? throw new ArgumentNullException(nameof(alertRepository));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }
        
        /// <inheritdoc/>
        public async Task<ReportModels> CreateReportAsync(ReportModels report)
        {
            if (report == null)
            {
                throw new ArgumentNullException(nameof(report));
            }
            
            // Set default values if not provided
            report.Id = string.IsNullOrEmpty(report.Id) ? Guid.NewGuid().ToString() : report.Id;
            report.CreatedAt = DateTime.UtcNow;
            report.UpdatedAt = DateTime.UtcNow;
            
            // Add report to database
            await _reportRepository.AddAsync(report);
            
            _logger.LogInformation("Report created: {ReportId} ({Name})", report.Id, report.Name);
            
            return report;
        }
        
        /// <inheritdoc/>
        public async Task<ReportModels> UpdateReportAsync(ReportModels report)
        {
            if (report == null)
            {
                throw new ArgumentNullException(nameof(report));
            }
            
            var existingReport = await _reportRepository.GetByIdAsync(report.Id);
            if (existingReport == null)
            {
                throw new KeyNotFoundException($"Report with ID {report.Id} not found");
            }
            
            // Update report properties
            existingReport.Name = report.Name;
            existingReport.Description = report.Description;
            existingReport.Type = report.Type;
            existingReport.Parameters = report.Parameters;
            existingReport.Schedule = report.Schedule;
            existingReport.UpdatedAt = DateTime.UtcNow;
            
            // Update report in database
            await _reportRepository.UpdateAsync(existingReport);
            
            _logger.LogInformation("Report updated: {ReportId} ({Name})", existingReport.Id, existingReport.Name);
            
            return existingReport;
        }
        
        /// <inheritdoc/>
        public async Task<ReportModels> UpdateReportScheduleAsync(string id, string schedule)
        {
            return await _reportRepository.UpdateScheduleAsync(id, schedule);
        }
        
        /// <inheritdoc/>
        public async Task<string> GenerateReportAsync(string id)
        {
            var report = await _reportRepository.GetByIdAsync(id);
            if (report == null)
            {
                throw new KeyNotFoundException($"Report with ID {id} not found");
            }
            
            // Generate report content based on type
            var content = new StringBuilder();
            
            switch (report.Type.ToLower())
            {
                case "agent_status":
                    content.Append(await GenerateAgentStatusReportAsync(report));
                    break;
                case "log_summary":
                    content.Append(await GenerateLogSummaryReportAsync(report));
                    break;
                case "alert_summary":
                    content.Append(await GenerateAlertSummaryReportAsync(report));
                    break;
                default:
                    throw new NotSupportedException($"Report type '{report.Type}' is not supported");
            }
            
            // Update report's last generated timestamp
            report.LastGeneratedAt = DateTime.UtcNow;
            await _reportRepository.UpdateAsync(report);
            
            _logger.LogInformation("Report generated: {ReportId} ({Name})", report.Id, report.Name);
            
            return content.ToString();
        }
        
        /// <inheritdoc/>
        public async Task<bool> DeleteReportAsync(string id)
        {
            try
            {
                await _reportRepository.DeleteByIdAsync(id);
                _logger.LogInformation("Report deleted: {ReportId}", id);
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error deleting report: {ReportId}", id);
                return false;
            }
        }
        
        /// <summary>
        /// Generates an agent status report
        /// </summary>
        /// <param name="report">The report configuration</param>
        /// <returns>The generated report content</returns>
        private async Task<string> GenerateAgentStatusReportAsync(ReportModels report)
        {
            var content = new StringBuilder();
            
            content.AppendLine($"# Agent Status Report: {report.Name}");
            content.AppendLine($"Generated at: {DateTime.UtcNow:yyyy-MM-dd HH:mm:ss} UTC");
            content.AppendLine();
            
            // Get all agents
            var agents = await _agentRepository.GetAllAsync();
            
            content.AppendLine("## Agent Status Summary");
            content.AppendLine();
            content.AppendLine("| Agent ID | Hostname | Status | Last Heartbeat | CPU Usage | Memory Usage | Disk Usage |");
            content.AppendLine("|----------|----------|--------|----------------|-----------|--------------|------------|");
            
            foreach (var agent in agents)
            {
                content.AppendLine($"| {agent.Id} | {agent.Hostname} | {agent.Status} | {agent.LastHeartbeat?.ToString("yyyy-MM-dd HH:mm:ss") ?? "Never"} | {agent.CpuUsage:F1}% | {agent.MemoryUsage:F1}% | {agent.DiskUsage:F1}% |");
            }
            
            return content.ToString();
        }
        
        /// <summary>
        /// Generates a log summary report
        /// </summary>
        /// <param name="report">The report configuration</param>
        /// <returns>The generated report content</returns>
        private async Task<string> GenerateLogSummaryReportAsync(ReportModels report)
        {
            var content = new StringBuilder();
            
            content.AppendLine($"# Log Summary Report: {report.Name}");
            content.AppendLine($"Generated at: {DateTime.UtcNow:yyyy-MM-dd HH:mm:ss} UTC");
            content.AppendLine();
            
            // Parse parameters
            var startTime = DateTime.UtcNow.AddDays(-1);
            var endTime = DateTime.UtcNow;
            
            if (report.Parameters != null)
            {
                if (report.Parameters.TryGetValue("startTime", out var startTimeStr) && DateTime.TryParse(startTimeStr, out var parsedStartTime))
                {
                    startTime = parsedStartTime;
                }
                
                if (report.Parameters.TryGetValue("endTime", out var endTimeStr) && DateTime.TryParse(endTimeStr, out var parsedEndTime))
                {
                    endTime = parsedEndTime;
                }
            }
            
            // Get logs in the time range
            var logs = await _logEntryRepository.GetByTimeRangeAsync(startTime, endTime, 1000, 0);
            
            // Group logs by level
            var logsByLevel = new Dictionary<string, int>();
            foreach (var log in logs)
            {
                if (!logsByLevel.ContainsKey(log.Level))
                {
                    logsByLevel[log.Level] = 0;
                }
                
                logsByLevel[log.Level]++;
            }
            
            content.AppendLine("## Log Level Summary");
            content.AppendLine();
            content.AppendLine("| Level | Count |");
            content.AppendLine("|-------|-------|");
            
            foreach (var kvp in logsByLevel)
            {
                content.AppendLine($"| {kvp.Key} | {kvp.Value} |");
            }
            
            content.AppendLine();
            content.AppendLine("## Recent Error Logs");
            content.AppendLine();
            
            // Get recent error logs
            var errorLogs = await _logEntryRepository.GetByLevelAsync("Error", 10, 0);
            
            foreach (var log in errorLogs)
            {
                content.AppendLine($"### {log.Timestamp:yyyy-MM-dd HH:mm:ss} - {log.Source}");
                content.AppendLine();
                content.AppendLine($"**Message:** {log.Message}");
                
                if (!string.IsNullOrEmpty(log.StackTrace))
                {
                    content.AppendLine();
                    content.AppendLine("```");
                    content.AppendLine(log.StackTrace);
                    content.AppendLine("```");
                }
                
                content.AppendLine();
            }
            
            return content.ToString();
        }
        
        /// <summary>
        /// Generates an alert summary report
        /// </summary>
        /// <param name="report">The report configuration</param>
        /// <returns>The generated report content</returns>
        private async Task<string> GenerateAlertSummaryReportAsync(ReportModels report)
        {
            var content = new StringBuilder();
            
            content.AppendLine($"# Alert Summary Report: {report.Name}");
            content.AppendLine($"Generated at: {DateTime.UtcNow:yyyy-MM-dd HH:mm:ss} UTC");
            content.AppendLine();
            
            // Parse parameters
            var startTime = DateTime.UtcNow.AddDays(-1);
            var endTime = DateTime.UtcNow;
            
            if (report.Parameters != null)
            {
                if (report.Parameters.TryGetValue("startTime", out var startTimeStr) && DateTime.TryParse(startTimeStr, out var parsedStartTime))
                {
                    startTime = parsedStartTime;
                }
                
                if (report.Parameters.TryGetValue("endTime", out var endTimeStr) && DateTime.TryParse(endTimeStr, out var parsedEndTime))
                {
                    endTime = parsedEndTime;
                }
            }
            
            // Get alerts in the time range
            var alerts = await _alertRepository.GetByTimeRangeAsync(startTime, endTime);
            
            // Group alerts by severity
            var alertsBySeverity = new Dictionary<string, int>();
            foreach (var alert in alerts)
            {
                string severityKey = alert.Severity.ToString();
                if (!alertsBySeverity.ContainsKey(severityKey))
                {
                    alertsBySeverity[severityKey] = 0;
                }
                
                alertsBySeverity[severityKey]++;
            }
            
            content.AppendLine("## Alert Severity Summary");
            content.AppendLine();
            content.AppendLine("| Severity | Count |");
            content.AppendLine("|----------|-------|");
            
            foreach (var kvp in alertsBySeverity)
            {
                content.AppendLine($"| {kvp.Key} | {kvp.Value} |");
            }
            
            content.AppendLine();
            content.AppendLine("## Recent Critical Alerts");
            content.AppendLine();
            
            // Get recent critical alerts
            var criticalAlerts = await _alertRepository.GetBySeverityAsync(AlertSeverityModels.Critical);
            
            foreach (var alert in criticalAlerts)
            {
                content.AppendLine($"### {alert.Timestamp:yyyy-MM-dd HH:mm:ss} - {alert.Title}");
                content.AppendLine();
                content.AppendLine($"**Agent:** {alert.AgentId}");
                content.AppendLine($"**Status:** {alert.Status}");
                content.AppendLine($"**Message:** {alert.Message}");
                
                if (!string.IsNullOrEmpty(alert.ResolutionNotes))
                {
                    content.AppendLine();
                    content.AppendLine($"**Resolution Notes:** {alert.ResolutionNotes}");
                }
                
                content.AppendLine();
            }
            
            return content.ToString();
        }

        public async Task<IEnumerable<ReportModels>> GetAllReportsAsync()
        {
            return await _reportRepository.GetAllAsync();
        }

        public async Task<ReportModels?> GetReportByIdAsync(string id)
        {
            return await _reportRepository.GetByIdAsync(id);
        }

        public async Task<IEnumerable<ReportModels>> GetReportsByUserAsync(string userId)
        {
            return await _reportRepository.GetByUserIdAsync(userId);
        }

        public async Task<IEnumerable<ReportModels>> GetReportsByNameAsync(string name)
        {
            return await _reportRepository.GetByNameAsync(name);
        }

        public async Task<IEnumerable<ReportModels>> GetReportsByTypeAsync(string type)
        {
            return await _reportRepository.GetByTypeAsync(type);
        }

        public async Task<IEnumerable<ReportModels>> GetReportsByTimeRangeAsync(DateTime startTime, DateTime endTime)
        {
            return await _reportRepository.GetByTimeRangeAsync(startTime, endTime);
        }

        public async Task<IEnumerable<ReportModels>> GetScheduledReportsAsync()
        {
            return await _reportRepository.GetScheduledReportsAsync();
        }
    }
} 