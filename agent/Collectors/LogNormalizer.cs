using AthalaSIEM.Agent.Models;
using AthalaSIEM.Agent.Security;
using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace AthalaSIEM.Agent.Collectors
{
    /// <summary>
    /// Normalizes logs from different sources into a standard format
    /// </summary>
    public class LogNormalizer : ILogNormalizer
    {
        private readonly ILogger<LogNormalizer> _logger;
        private readonly IEncryptionService _encryptionService;
        private readonly string _hostname;
        private readonly string _agentId;
        
        // Common patterns for normalized fields
        private static readonly Regex IpAddressRegex = new Regex(@"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", RegexOptions.Compiled);
        private static readonly Regex EmailRegex = new Regex(@"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", RegexOptions.Compiled);
        private static readonly Regex UrlRegex = new Regex(@"https?://[^\s/$.?#].[^\s]*", RegexOptions.Compiled);
        
        /// <summary>
        /// Initializes a new instance of the <see cref="LogNormalizer"/> class
        /// </summary>
        /// <param name="logger">Logger instance</param>
        /// <param name="encryptionService">Encryption service for computing hashes</param>
        public LogNormalizer(ILogger<LogNormalizer> logger, IEncryptionService encryptionService)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _encryptionService = encryptionService ?? throw new ArgumentNullException(nameof(encryptionService));
            
            _hostname = Dns.GetHostName();
            _agentId = Environment.GetEnvironmentVariable("ATHALA_AGENT_ID") ?? string.Empty;
        }
        
        /// <summary>
        /// Normalizes raw log data into a standardized format
        /// </summary>
        /// <param name="rawLog">Raw log data to normalize</param>
        /// <returns>Normalized log entry</returns>
        public NormalizedLogEntry Normalize(RawLogData rawLog)
        {
            if (rawLog == null)
                throw new ArgumentNullException(nameof(rawLog));
                
            try
            {
                var normalizedLog = new NormalizedLogEntry
                {
                    Id = Guid.NewGuid().ToString("N"),
                    SourceHost = rawLog.SourceHost,
                    SourceType = rawLog.SourceType,
                    Timestamp = rawLog.Timestamp,
                    RawContent = rawLog.Content,
                    Severity = MapSeverityLevel(rawLog.Severity),
                    Category = DetermineCategory(rawLog),
                    CollectorType = rawLog.CollectorType,
                    SourceIdentifier = rawLog.SourceIdentifier
                };
                
                // Extract additional fields
                var additionalFields = new Dictionary<string, string>();
                ExtractAdditionalFields(rawLog.Content, additionalFields);
                if (additionalFields.Count > 0)
                {
                    normalizedLog.AdditionalFields = additionalFields;
                }
                
                // Compute hash for integrity checking
                normalizedLog.ContentHash = _encryptionService.ComputeHash(Encoding.UTF8.GetBytes(rawLog.Content));
                
                return normalizedLog;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error normalizing log: {ErrorMessage}", ex.Message);
                
                // Return a fallback entry
                return new NormalizedLogEntry
                {
                    Id = Guid.NewGuid().ToString("N"),
                    SourceHost = rawLog.SourceHost,
                    SourceType = rawLog.SourceType,
                    Timestamp = rawLog.Timestamp,
                    RawContent = rawLog.Content,
                    Severity = "Error",
                    Category = "NormalizationError",
                    CollectorType = rawLog.CollectorType,
                    SourceIdentifier = rawLog.SourceIdentifier,
                    AdditionalFields = new Dictionary<string, string>
                    {
                        { "NormalizationError", ex.Message }
                    }
                };
            }
        }
        
        /// <summary>
        /// Asynchronously normalizes raw log data into a standardized format
        /// </summary>
        /// <param name="rawLog">Raw log data to normalize</param>
        /// <returns>Normalized log entry</returns>
        public Task<NormalizedLogEntry> NormalizeAsync(RawLogData rawLog)
        {
            return Task.FromResult(Normalize(rawLog));
        }
        
        private string MapSeverityLevel(string rawSeverity)
        {
            if (string.IsNullOrEmpty(rawSeverity))
            {
                return "Information";
            }

            // Normalize various severity levels
            string severity = rawSeverity.ToLowerInvariant();

            if (severity.Contains("crit") || severity.Contains("fatal") || severity.Contains("emerg"))
            {
                return "Critical";
            }
            else if (severity.Contains("err"))
            {
                return "Error";
            }
            else if (severity.Contains("warn"))
            {
                return "Warning";
            }
            else if (severity.Contains("info"))
            {
                return "Information";
            }
            else if (severity.Contains("debug") || severity.Contains("trace") || severity.Contains("verbose"))
            {
                return "Debug";
            }
            else
            {
                return "Information"; // Default
            }
        }
        
        /// <summary>
        /// Determines the category of the log based on the raw log data
        /// </summary>
        /// <param name="rawLog">Raw log data</param>
        /// <returns>Log category</returns>
        public string DetermineCategory(RawLogData rawLog)
        {
            string content = rawLog.Content?.ToLowerInvariant() ?? string.Empty;
            string sourceType = rawLog.SourceType?.ToLowerInvariant() ?? string.Empty;

            // Authentication related
            if (content.Contains("login") || content.Contains("logon") || 
                content.Contains("authentication") || content.Contains("auth") ||
                content.Contains("password"))
            {
                if (content.Contains("fail") || content.Contains("invalid") || 
                    content.Contains("denied") || content.Contains("error"))
                {
                    return "FailedAuthentication";
                }
                return "Authentication";
            }

            // Security related
            if (content.Contains("firewall") || content.Contains("blocked") ||
                content.Contains("denied") || content.Contains("allow") ||
                content.Contains("permit") || content.Contains("security") ||
                content.Contains("access denied"))
            {
                return "Security";
            }

            // System related
            if (content.Contains("start") || content.Contains("stop") ||
                content.Contains("shutdown") || content.Contains("boot") ||
                content.Contains("crash") || content.Contains("system") ||
                content.Contains("service") || content.Contains("daemon"))
            {
                return "System";
            }

            // Network related
            if (content.Contains("network") || content.Contains("interface") ||
                content.Contains("tcp") || content.Contains("udp") ||
                content.Contains("ip") || content.Contains("connection") ||
                content.Contains("connected") || content.Contains("disconnected"))
            {
                return "Network";
            }

            // Application specific
            if (content.Contains("exception") || content.Contains("error") ||
                content.Contains("failure") || content.Contains("failed"))
            {
                return "Application";
            }

            // Based on source type
            if (sourceType.Contains("app") || sourceType.Contains("application"))
            {
                return "Application";
            }
            else if (sourceType.Contains("sec") || sourceType.Contains("audit"))
            {
                return "Security";
            }
            else if (sourceType.Contains("sys"))
            {
                return "System";
            }

            // Default
            return "General";
        }
        
        private void ExtractAdditionalFields(string content, Dictionary<string, string> additionalFields)
        {
            if (string.IsNullOrWhiteSpace(content))
            {
                return;
            }

            // Extract IP addresses
            var ipAddresses = IpAddressRegex.Matches(content)
                .OfType<Match>()
                .Select(m => m.Value)
                .Distinct()
                .ToList();
            
            if (ipAddresses.Count > 0)
            {
                additionalFields["IpAddresses"] = string.Join(",", ipAddresses);
            }

            // Extract email addresses
            var emails = EmailRegex.Matches(content)
                .OfType<Match>()
                .Select(m => m.Value)
                .Distinct()
                .ToList();
            
            if (emails.Count > 0)
            {
                additionalFields["EmailAddresses"] = string.Join(",", emails);
            }

            // Extract URLs
            var urls = UrlRegex.Matches(content)
                .OfType<Match>()
                .Select(m => m.Value)
                .Distinct()
                .ToList();
            
            if (urls.Count > 0)
            {
                additionalFields["Urls"] = string.Join(",", urls);
            }

            // Look for user identifiers
            if (content.Contains("user:", StringComparison.OrdinalIgnoreCase) || 
                content.Contains("username:", StringComparison.OrdinalIgnoreCase))
            {
                // Simple regex to extract username patterns
                var userMatches = Regex.Matches(content, @"(?:user|username)[=:]\s*(\S+)", RegexOptions.IgnoreCase);
                if (userMatches.Count > 0)
                {
                    additionalFields["Username"] = userMatches[0].Groups[1].Value;
                }
            }
        }

        /// <summary>
        /// Maps a severity level from a source to a standardized severity level
        /// </summary>
        /// <param name="source">Source of the log</param>
        /// <param name="severity">Severity level from the source</param>
        /// <returns>Standardized severity level</returns>
        public string MapSeverity(string source, string severity)
        {
            // Implement severity mapping logic here
            return "Info";
        }
    }
} 