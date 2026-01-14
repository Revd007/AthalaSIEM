using System;
using System.Collections.Generic;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using AthalaSIEM.UniversalAgent.Models;
using AthalaSIEM.UniversalAgent.Core.Normalizer;

namespace AthalaSIEM.UniversalAgent.Core.Parser
{
    /// <summary>
    /// Syslog Parser - RFC 3164 and RFC 5424 compliant
    /// Parses syslog messages from any source (Linux, network devices, firewalls, etc.)
    /// 
    /// HARD RULES:
    /// - Parser decodes and structures - does NOT detect threats
    /// - Parser does NOT filter or enrich
    /// - Parser does NOT use hardcoded event IDs
    /// - All detection is done by backend ML/analytics
    /// </summary>
    public class SyslogParser : IParser
    {
        private readonly ILogger<SyslogParser> _logger;

        // RFC 3164 pattern: <PRI>TIMESTAMP HOSTNAME TAG: MESSAGE
        private static readonly Regex Rfc3164Pattern = new(
            @"^<(?<priority>\d{1,3})>(?<timestamp>\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+(?<hostname>\S+)\s+(?<tag>[^\s\[:]+)(?:\[(?<pid>\d+)\])?:\s*(?<message>.*)$",
            RegexOptions.Compiled);

        // RFC 5424 pattern: <PRI>VERSION TIMESTAMP HOSTNAME APP-NAME PROCID MSGID [SD] MESSAGE
        private static readonly Regex Rfc5424Pattern = new(
            @"^<(?<priority>\d{1,3})>(?<version>\d+)\s+(?<timestamp>\S+)\s+(?<hostname>\S+)\s+(?<appname>\S+)\s+(?<procid>\S+)\s+(?<msgid>\S+)\s+(?<structureddata>(?:\[.*?\])+|-)\s*(?<message>.*)$",
            RegexOptions.Compiled);

        // Structured data pattern for RFC 5424
        private static readonly Regex StructuredDataPattern = new(
            @"\[(?<sdid>[^\s\]]+)(?:\s+(?<params>[^\]]+))?\]",
            RegexOptions.Compiled);

        // Metrics
        private long _eventsParsed = 0;
        private long _parseErrors = 0;
        private long _rfc3164Count = 0;
        private long _rfc5424Count = 0;
        private readonly DateTime _startTime = DateTime.UtcNow;

        public string Name => "SyslogParser";
        public string SourceType => "Syslog";

        public SyslogParser(ILogger<SyslogParser> logger)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }

        public bool CanParse(object rawEvent)
        {
            if (rawEvent is string message)
            {
                // Check if it looks like syslog (starts with priority)
                return message.StartsWith("<") && message.Length > 3;
            }
            return rawEvent is SyslogEntry;
        }

        public Task<ParsedEvent> ParseAsync(object rawEvent)
        {
            if (rawEvent == null)
            {
                throw new ArgumentNullException(nameof(rawEvent));
            }

            try
            {
                ParsedEvent parsed;

                if (rawEvent is string message)
                {
                    parsed = ParseSyslogMessage(message);
                }
                else if (rawEvent is SyslogEntry entry)
                {
                    parsed = ParseSyslogEntry(entry);
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
                _logger.LogError(ex, "Error parsing syslog message: {Message}", ex.Message);
                throw;
            }
        }

        /// <summary>
        /// Parse raw syslog message string (RFC 3164 or RFC 5424)
        /// </summary>
        private ParsedEvent ParseSyslogMessage(string message)
        {
            // Try RFC 5424 first (more specific)
            var rfc5424Match = Rfc5424Pattern.Match(message);
            if (rfc5424Match.Success)
            {
                _rfc5424Count++;
                return ParseRfc5424(rfc5424Match, message);
            }

            // Fall back to RFC 3164
            var rfc3164Match = Rfc3164Pattern.Match(message);
            if (rfc3164Match.Success)
            {
                _rfc3164Count++;
                return ParseRfc3164(rfc3164Match, message);
            }

            // Unknown format - parse as raw message
            return ParseRawSyslog(message);
        }

        /// <summary>
        /// Parse RFC 3164 syslog message
        /// </summary>
        private ParsedEvent ParseRfc3164(Match match, string rawMessage)
        {
            var priority = int.Parse(match.Groups["priority"].Value);
            var (facility, severity) = DecodePriority(priority);

            var parsed = new ParsedEvent
            {
                RawEvent = rawMessage,
                Timestamp = ParseRfc3164Timestamp(match.Groups["timestamp"].Value),
                Level = MapSeverityToLevel(severity),
                Message = match.Groups["message"].Value,
                SourceHost = match.Groups["hostname"].Value,
                SourceApplication = match.Groups["tag"].Value,
                SourceType = "Syslog-RFC3164",
                Collector = "SyslogCollector",
                Properties = new Dictionary<string, object>
                {
                    ["syslog.priority"] = priority,
                    ["syslog.facility"] = facility,
                    ["syslog.facility_name"] = GetFacilityName(facility),
                    ["syslog.severity"] = severity,
                    ["syslog.severity_name"] = GetSeverityName(severity),
                    ["syslog.rfc"] = "3164"
                }
            };

            // Extract PID if present
            if (match.Groups["pid"].Success)
            {
                var pidStr = match.Groups["pid"].Value;
                if (int.TryParse(pidStr, out var pid))
                {
                    parsed.Process = new ParsedProcessInfo
                    {
                        Name = match.Groups["tag"].Value,
                        Pid = pid
                    };
                }
            }

            return parsed;
        }

        /// <summary>
        /// Parse RFC 5424 syslog message
        /// </summary>
        private ParsedEvent ParseRfc5424(Match match, string rawMessage)
        {
            var priority = int.Parse(match.Groups["priority"].Value);
            var (facility, severity) = DecodePriority(priority);

            var parsed = new ParsedEvent
            {
                RawEvent = rawMessage,
                Timestamp = ParseRfc5424Timestamp(match.Groups["timestamp"].Value),
                Level = MapSeverityToLevel(severity),
                Message = match.Groups["message"].Value,
                SourceHost = NilToNull(match.Groups["hostname"].Value),
                SourceApplication = NilToNull(match.Groups["appname"].Value),
                SourceType = "Syslog-RFC5424",
                Collector = "SyslogCollector",
                Properties = new Dictionary<string, object>
                {
                    ["syslog.priority"] = priority,
                    ["syslog.facility"] = facility,
                    ["syslog.facility_name"] = GetFacilityName(facility),
                    ["syslog.severity"] = severity,
                    ["syslog.severity_name"] = GetSeverityName(severity),
                    ["syslog.version"] = int.Parse(match.Groups["version"].Value),
                    ["syslog.msgid"] = NilToNull(match.Groups["msgid"].Value) ?? "",
                    ["syslog.rfc"] = "5424"
                }
            };

            // Extract ProcID
            var procId = NilToNull(match.Groups["procid"].Value);
            if (!string.IsNullOrEmpty(procId) && int.TryParse(procId, out var pid))
            {
                parsed.Process = new ParsedProcessInfo
                {
                    Name = parsed.SourceApplication,
                    Pid = pid
                };
            }

            // Parse structured data
            var structuredData = match.Groups["structureddata"].Value;
            if (structuredData != "-")
            {
                parsed.Properties["syslog.structured_data"] = ParseStructuredData(structuredData);
            }

            return parsed;
        }

        /// <summary>
        /// Parse raw syslog message (unknown format)
        /// </summary>
        private ParsedEvent ParseRawSyslog(string message)
        {
            // Try to extract priority if present
            int priority = 13; // Default: user.notice
            int facility = 1;
            int severity = 5;

            if (message.StartsWith("<"))
            {
                var endIndex = message.IndexOf('>');
                if (endIndex > 0 && int.TryParse(message.Substring(1, endIndex - 1), out priority))
                {
                    (facility, severity) = DecodePriority(priority);
                    message = message.Substring(endIndex + 1).TrimStart();
                }
            }

            return new ParsedEvent
            {
                RawEvent = message,
                Timestamp = DateTime.UtcNow,
                Level = MapSeverityToLevel(severity),
                Message = message,
                SourceType = "Syslog-Raw",
                Collector = "SyslogCollector",
                Properties = new Dictionary<string, object>
                {
                    ["syslog.priority"] = priority,
                    ["syslog.facility"] = facility,
                    ["syslog.facility_name"] = GetFacilityName(facility),
                    ["syslog.severity"] = severity,
                    ["syslog.severity_name"] = GetSeverityName(severity),
                    ["syslog.rfc"] = "unknown"
                }
            };
        }

        /// <summary>
        /// Parse SyslogEntry from existing model
        /// </summary>
        private ParsedEvent ParseSyslogEntry(SyslogEntry entry)
        {
            var (facility, severity) = (entry.Facility, entry.Severity);

            return new ParsedEvent
            {
                RawEvent = entry,
                Timestamp = entry.Timestamp,
                Level = entry.Level ?? MapSeverityToLevel(severity),
                Message = entry.Message,
                SourceHost = entry.Hostname,
                SourceApplication = entry.AppName,
                SourceType = entry.Source,
                Collector = entry.CollectorType,
                Process = !string.IsNullOrEmpty(entry.ProcId) ? new ParsedProcessInfo
                {
                    Name = entry.AppName,
                    Pid = int.TryParse(entry.ProcId, out var pid) ? pid : null
                } : null,
                Properties = new Dictionary<string, object>
                {
                    ["syslog.facility"] = facility,
                    ["syslog.facility_name"] = GetFacilityName(facility),
                    ["syslog.severity"] = severity,
                    ["syslog.severity_name"] = GetSeverityName(severity),
                    ["syslog.msgid"] = entry.MsgId ?? ""
                }
            };
        }

        /// <summary>
        /// Decode syslog priority to facility and severity
        /// </summary>
        private (int facility, int severity) DecodePriority(int priority)
        {
            // Priority = Facility * 8 + Severity
            var facility = priority / 8;
            var severity = priority % 8;
            return (facility, severity);
        }

        /// <summary>
        /// Parse RFC 3164 timestamp (e.g., "Jan  5 14:30:00")
        /// </summary>
        private DateTime ParseRfc3164Timestamp(string timestamp)
        {
            try
            {
                // RFC 3164 doesn't include year, assume current year
                var year = DateTime.UtcNow.Year;
                var fullTimestamp = $"{timestamp} {year}";

                if (DateTime.TryParseExact(fullTimestamp, 
                    new[] { "MMM  d HH:mm:ss yyyy", "MMM dd HH:mm:ss yyyy" },
                    System.Globalization.CultureInfo.InvariantCulture,
                    System.Globalization.DateTimeStyles.AssumeUniversal,
                    out var result))
                {
                    return DateTime.SpecifyKind(result, DateTimeKind.Utc);
                }
            }
            catch
            {
                // Fall through to default
            }

            return DateTime.UtcNow;
        }

        /// <summary>
        /// Parse RFC 5424 timestamp (ISO 8601)
        /// </summary>
        private DateTime ParseRfc5424Timestamp(string timestamp)
        {
            if (timestamp == "-")
                return DateTime.UtcNow;

            try
            {
                if (DateTime.TryParse(timestamp, out var result))
                {
                    return result.Kind == DateTimeKind.Utc ? result : result.ToUniversalTime();
                }
            }
            catch
            {
                // Fall through to default
            }

            return DateTime.UtcNow;
        }

        /// <summary>
        /// Parse RFC 5424 structured data
        /// </summary>
        private Dictionary<string, Dictionary<string, string>> ParseStructuredData(string structuredData)
        {
            var result = new Dictionary<string, Dictionary<string, string>>();

            var matches = StructuredDataPattern.Matches(structuredData);
            foreach (Match match in matches)
            {
                var sdId = match.Groups["sdid"].Value;
                var paramsStr = match.Groups["params"].Value;
                
                var parameters = new Dictionary<string, string>();
                if (!string.IsNullOrEmpty(paramsStr))
                {
                    // Parse key="value" pairs
                    var paramPattern = new Regex(@"(\S+)=""([^""]*)""");
                    var paramMatches = paramPattern.Matches(paramsStr);
                    foreach (Match paramMatch in paramMatches)
                    {
                        parameters[paramMatch.Groups[1].Value] = paramMatch.Groups[2].Value;
                    }
                }

                result[sdId] = parameters;
            }

            return result;
        }

        /// <summary>
        /// Convert NIL value ("-") to null
        /// </summary>
        private string? NilToNull(string value)
        {
            return value == "-" ? null : value;
        }

        /// <summary>
        /// Map syslog severity to log level
        /// </summary>
        private string MapSeverityToLevel(int severity)
        {
            return severity switch
            {
                0 => "Critical",  // Emergency
                1 => "Critical",  // Alert
                2 => "Critical",  // Critical
                3 => "Error",     // Error
                4 => "Warning",   // Warning
                5 => "Information", // Notice
                6 => "Information", // Informational
                7 => "Debug",     // Debug
                _ => "Information"
            };
        }

        /// <summary>
        /// Get facility name from facility code
        /// NO HARDCODING - just standard syslog facility mapping
        /// </summary>
        private string GetFacilityName(int facility)
        {
            return facility switch
            {
                0 => "kern",
                1 => "user",
                2 => "mail",
                3 => "daemon",
                4 => "auth",
                5 => "syslog",
                6 => "lpr",
                7 => "news",
                8 => "uucp",
                9 => "cron",
                10 => "authpriv",
                11 => "ftp",
                12 => "ntp",
                13 => "security",
                14 => "console",
                15 => "solaris-cron",
                16 => "local0",
                17 => "local1",
                18 => "local2",
                19 => "local3",
                20 => "local4",
                21 => "local5",
                22 => "local6",
                23 => "local7",
                _ => $"facility{facility}"
            };
        }

        /// <summary>
        /// Get severity name from severity code
        /// NO HARDCODING - just standard syslog severity mapping
        /// </summary>
        private string GetSeverityName(int severity)
        {
            return severity switch
            {
                0 => "emerg",
                1 => "alert",
                2 => "crit",
                3 => "err",
                4 => "warning",
                5 => "notice",
                6 => "info",
                7 => "debug",
                _ => $"severity{severity}"
            };
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
                ["Rfc3164Count"] = _rfc3164Count,
                ["Rfc5424Count"] = _rfc5424Count,
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
