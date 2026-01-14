using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using System.Diagnostics.Eventing.Reader;
using System.Runtime.Versioning;
using AthalaSIEM.UniversalAgent.Models;

namespace AthalaSIEM.UniversalAgent.Core.Parser
{
    /// <summary>
    /// Windows Event Log Parser
    /// Parses Windows Event Log entries into structured format
    /// 
    /// HARD RULES:
    /// - Parser decodes and structures - does NOT normalize schema
    /// - Parser does NOT detect threats
    /// - Parser does NOT enrich data
    /// </summary>
    [SupportedOSPlatform("windows")]
    public class WindowsEventLogParser : IParser
    {
        private readonly ILogger<WindowsEventLogParser> _logger;

        // Metrics
        private long _eventsParsed = 0;
        private long _parseErrors = 0;
        private readonly DateTime _startTime = DateTime.UtcNow;

        public string Name => "WindowsEventLogParser";
        public string SourceType => "WindowsEventLog";

        public WindowsEventLogParser(ILogger<WindowsEventLogParser> logger)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }

        public bool CanParse(object rawEvent)
        {
            return rawEvent is EventRecord || rawEvent is WindowsLogEntry;
        }

        public Task<Normalizer.ParsedEvent> ParseAsync(object rawEvent)
        {
            if (rawEvent == null)
            {
                throw new ArgumentNullException(nameof(rawEvent));
            }

            try
            {
                Normalizer.ParsedEvent parsed;

                if (rawEvent is EventRecord eventRecord)
                {
                    parsed = ParseEventRecord(eventRecord);
                }
                else if (rawEvent is WindowsLogEntry logEntry)
                {
                    parsed = ParseLogEntry(logEntry);
                }
                else
                {
                    throw new ArgumentException($"Unsupported raw event type: {rawEvent.GetType().Name}");
                }

                _eventsParsed++;
                return Task.FromResult(parsed);
            }
            catch (Exception ex)
            {
                _parseErrors++;
                _logger.LogError(ex, "Error parsing Windows event: {Message}", ex.Message);
                throw;
            }
        }

        private Normalizer.ParsedEvent ParseEventRecord(EventRecord eventRecord)
        {
            var parsed = new Normalizer.ParsedEvent
            {
                RawEvent = eventRecord, // Preserve original
                Timestamp = eventRecord.TimeCreated ?? DateTime.UtcNow,
                EventId = eventRecord.Id.ToString(),
                Level = MapEventLevel(ConvertToStandardEventLevel(eventRecord.Level)),
                Message = eventRecord.FormatDescription() ?? "No description",
                SourceHost = eventRecord.MachineName,
                SourceApplication = eventRecord.ProviderName,
                SourceType = eventRecord.LogName,
                Collector = "WindowsEventLogCollector",
                Properties = ExtractProperties(eventRecord)
            };

            // Extract user information
            parsed.User = ExtractUserInfo(eventRecord);

            // Extract process information
            parsed.Process = ExtractProcessInfo(eventRecord);

            // Extract network information
            parsed.Network = ExtractNetworkInfo(eventRecord);

            // Determine event category and action
            var (categories, action, outcome) = CategorizeEvent(eventRecord);
            parsed.Categories = categories;
            parsed.Action = action;
            parsed.Outcome = outcome;

            // Extract security relevance
            parsed.SecurityRelevance = DetermineSecurityRelevance(eventRecord);

            return parsed;
        }

        private Normalizer.ParsedEvent ParseLogEntry(WindowsLogEntry logEntry)
        {
            var parsed = new Normalizer.ParsedEvent
            {
                RawEvent = logEntry, // Preserve original
                Timestamp = logEntry.Timestamp,
                EventId = logEntry.EventId,
                Level = logEntry.Level,
                Message = logEntry.Message,
                SourceHost = logEntry.ComputerName ?? logEntry.WorkstationName,
                SourceApplication = logEntry.ProviderName,
                SourceType = logEntry.LogName,
                Collector = logEntry.CollectorType,
                Properties = logEntry.Properties ?? new Dictionary<string, object>()
            };

            // Extract user information
            if (!string.IsNullOrEmpty(logEntry.Username) || !string.IsNullOrEmpty(logEntry.TargetUserName))
            {
                parsed.User = new Normalizer.ParsedUserInfo
                {
                    Name = logEntry.TargetUserName ?? logEntry.Username,
                    Domain = logEntry.TargetDomainName
                };
            }

            // Extract process information
            if (!string.IsNullOrEmpty(logEntry.ProcessName) || logEntry.ProcessId.HasValue)
            {
                parsed.Process = new Normalizer.ParsedProcessInfo
                {
                    Name = logEntry.ProcessName,
                    Pid = logEntry.ProcessId
                };
            }

            // Extract network information
            if (!string.IsNullOrEmpty(logEntry.IpAddress))
            {
                parsed.Network = new Normalizer.ParsedNetworkInfo
                {
                    SourceIp = logEntry.IpAddress
                };
            }

            // Categorize based on log entry properties
            var (categories, action, outcome) = CategorizeLogEntry(logEntry);
            parsed.Categories = categories;
            parsed.Action = action;
            parsed.Outcome = outcome;
            parsed.SecurityRelevance = logEntry.SecurityRelevance;

            return parsed;
        }

        private Dictionary<string, object> ExtractProperties(EventRecord eventRecord)
        {
            var properties = new Dictionary<string, object>();

            if (eventRecord.Properties != null)
            {
                for (int i = 0; i < eventRecord.Properties.Count; i++)
                {
                    var property = eventRecord.Properties[i];
                    var key = $"Property_{i}";
                    properties[key] = property?.Value?.ToString() ?? "";
                }
            }

            // Add standard Windows Event properties
            properties["LogName"] = eventRecord.LogName ?? "";
            properties["ProviderName"] = eventRecord.ProviderName ?? "";
            properties["RecordId"] = eventRecord.RecordId ?? 0;
            properties["Task"] = eventRecord.Task ?? 0;
            properties["Opcode"] = eventRecord.Opcode ?? 0;
            properties["Keywords"] = eventRecord.KeywordsDisplayNames?.FirstOrDefault() ?? "";

            return properties;
        }

        private Normalizer.ParsedUserInfo? ExtractUserInfo(EventRecord eventRecord)
        {
            var userName = GetPropertyValue(eventRecord, "TargetUserName") ??
                          GetPropertyValue(eventRecord, "SubjectUserName") ??
                          GetPropertyValue(eventRecord, "AccountName");

            if (string.IsNullOrEmpty(userName))
                return null;

            return new Normalizer.ParsedUserInfo
            {
                Name = userName,
                Domain = GetPropertyValue(eventRecord, "TargetDomainName") ??
                        GetPropertyValue(eventRecord, "SubjectDomainName") ??
                        GetPropertyValue(eventRecord, "DomainName") ?? null,
                Id = GetPropertyValue(eventRecord, "TargetUserSid") ??
                     GetPropertyValue(eventRecord, "SubjectUserSid") ?? null
            };
        }

        private Normalizer.ParsedProcessInfo? ExtractProcessInfo(EventRecord eventRecord)
        {
            var processName = GetPropertyValue(eventRecord, "ProcessName") ??
                             GetPropertyValue(eventRecord, "Image");

            if (string.IsNullOrEmpty(processName))
                return null;

            var pidStr = GetPropertyValue(eventRecord, "ProcessId");
            int? pid = null;
            if (!string.IsNullOrEmpty(pidStr) && int.TryParse(pidStr, out var pidValue))
            {
                pid = pidValue;
            }

            return new Normalizer.ParsedProcessInfo
            {
                Name = processName,
                Pid = pid,
                CommandLine = GetPropertyValue(eventRecord, "CommandLine")
            };
        }

        private Normalizer.ParsedNetworkInfo? ExtractNetworkInfo(EventRecord eventRecord)
        {
            var sourceIp = GetPropertyValue(eventRecord, "IpAddress") ??
                          GetPropertyValue(eventRecord, "SourceIp") ??
                          GetPropertyValue(eventRecord, "ClientAddress");

            var destIp = GetPropertyValue(eventRecord, "DestinationIp") ??
                        GetPropertyValue(eventRecord, "ServerAddress");

            if (string.IsNullOrEmpty(sourceIp) && string.IsNullOrEmpty(destIp))
                return null;

            var sourcePortStr = GetPropertyValue(eventRecord, "SourcePort");
            var destPortStr = GetPropertyValue(eventRecord, "DestinationPort") ??
                             GetPropertyValue(eventRecord, "Port");

            int? sourcePort = null;
            int? destPort = null;

            if (!string.IsNullOrEmpty(sourcePortStr) && int.TryParse(sourcePortStr, out var sp))
                sourcePort = sp;

            if (!string.IsNullOrEmpty(destPortStr) && int.TryParse(destPortStr, out var dp))
                destPort = dp;

            return new Normalizer.ParsedNetworkInfo
            {
                SourceIp = sourceIp,
                SourcePort = sourcePort,
                DestinationIp = destIp,
                DestinationPort = destPort,
                Protocol = GetPropertyValue(eventRecord, "Protocol")
            };
        }

        /// <summary>
        /// Categorize event based on log source - NO hardcoded event ID mapping
        /// Detection is done by backend ML/analytics, not by parser
        /// </summary>
        private (List<string> categories, string? action, string? outcome) CategorizeEvent(EventRecord eventRecord)
        {
            var categories = new List<string>();
            string? action = null;
            string? outcome = null;

            // Add log name as category - this is structural, not detection
            var logName = eventRecord.LogName;
            if (!string.IsNullOrEmpty(logName))
            {
                categories.Add(logName.ToLowerInvariant());
                
                // Add parent category based on log name (structural mapping only)
                categories.Add("windows");
            }

            // Add provider name as additional context (structural)
            var providerName = eventRecord.ProviderName;
            if (!string.IsNullOrEmpty(providerName))
            {
                categories.Add($"provider:{providerName.ToLowerInvariant().Replace(" ", "_")}");
            }

            // Extract outcome from keywords if available (structural extraction, not detection)
            var keywords = eventRecord.KeywordsDisplayNames;
            if (keywords != null)
            {
                foreach (var keyword in keywords)
                {
                    if (keyword != null && keyword.Contains("Success", StringComparison.OrdinalIgnoreCase))
                        outcome = "success";
                    else if (keyword != null && keyword.Contains("Failure", StringComparison.OrdinalIgnoreCase))
                        outcome = "failure";
                }
            }

            return (categories, action, outcome);
        }

        /// <summary>
        /// Categorize log entry based on source - NO hardcoded event ID mapping
        /// Detection is done by backend ML/analytics, not by parser
        /// </summary>
        private (List<string> categories, string? action, string? outcome) CategorizeLogEntry(WindowsLogEntry logEntry)
        {
            var categories = new List<string> { "windows" };
            string? action = null;
            string? outcome = null;

            // Add category from log entry if available (structural)
            if (!string.IsNullOrEmpty(logEntry.Category))
            {
                categories.Add(logEntry.Category.ToLowerInvariant());
            }

            // Add log name (structural)
            if (!string.IsNullOrEmpty(logEntry.LogName))
            {
                categories.Add(logEntry.LogName.ToLowerInvariant());
            }

            // Add provider name (structural)
            if (!string.IsNullOrEmpty(logEntry.ProviderName))
            {
                categories.Add($"provider:{logEntry.ProviderName.ToLowerInvariant().Replace(" ", "_")}");
            }

            return (categories.Distinct().ToList(), action, outcome);
        }

        /// <summary>
        /// Determine security relevance based on log level ONLY - NO hardcoded event IDs
        /// Specific event relevance is determined by backend configuration
        /// </summary>
        private string DetermineSecurityRelevance(EventRecord eventRecord)
        {
            // Determine relevance based on event level ONLY
            // Backend configuration determines which specific events are high priority
            var level = ConvertToStandardEventLevel(eventRecord.Level);
            
            return level switch
            {
                StandardEventLevel.Critical => "Critical",
                StandardEventLevel.Error => "High",
                StandardEventLevel.Warning => "Medium",
                _ => "Low" // Information, Debug, etc.
            };
        }

        private StandardEventLevel ConvertToStandardEventLevel(byte? levelByte)
        {
            if (!levelByte.HasValue)
                return StandardEventLevel.Informational;

            // Map byte values to StandardEventLevel enum
            return levelByte.Value switch
            {
                0 => StandardEventLevel.LogAlways,
                1 => StandardEventLevel.Critical,
                2 => StandardEventLevel.Error,
                3 => StandardEventLevel.Warning,
                4 => StandardEventLevel.Informational,
                5 => StandardEventLevel.Verbose,
                _ => StandardEventLevel.Informational
            };
        }

        private string MapEventLevel(StandardEventLevel level)
        {
            return level switch
            {
                StandardEventLevel.LogAlways => "Information",
                StandardEventLevel.Critical => "Critical",
                StandardEventLevel.Error => "Error",
                StandardEventLevel.Warning => "Warning",
                StandardEventLevel.Informational => "Information",
                StandardEventLevel.Verbose => "Debug",
                _ => "Information"
            };
        }

        private string? GetPropertyValue(EventRecord eventRecord, string propertyName)
        {
            if (eventRecord.Properties == null)
                return null;

            // Try to find property by name (Windows Event properties are indexed)
            // This is a simplified approach - in production, you'd use XML parsing
            try
            {
                var xml = eventRecord.ToXml();
                if (xml.Contains($"<{propertyName}>"))
                {
                    var startTag = $"<{propertyName}>";
                    var endTag = $"</{propertyName}>";
                    var startIndex = xml.IndexOf(startTag) + startTag.Length;
                    var endIndex = xml.IndexOf(endTag);
                    if (endIndex > startIndex)
                    {
                        return xml.Substring(startIndex, endIndex - startIndex).Trim();
                    }
                }
            }
            catch
            {
                // Fall through to return null
            }

            return null;
        }

        public Dictionary<string, object> GetMetrics()
        {
            var uptime = DateTime.UtcNow - _startTime;
            return new Dictionary<string, object>
            {
                ["Name"] = Name,
                ["SourceType"] = SourceType,
                ["EventsParsed"] = _eventsParsed,
                ["ParseErrors"] = _parseErrors,
                ["SuccessRate"] = _eventsParsed > 0
                    ? (double)(_eventsParsed - _parseErrors) / _eventsParsed * 100
                    : 100.0,
                ["UptimeSeconds"] = uptime.TotalSeconds,
                ["EventsPerSecond"] = uptime.TotalSeconds > 0
                    ? _eventsParsed / uptime.TotalSeconds
                    : 0.0
            };
        }
    }
}
