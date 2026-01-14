using System;
using System.Collections.Generic;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using AthalaSIEM.UniversalAgent.Core.Normalizer;

namespace AthalaSIEM.UniversalAgent.Core.Parser
{
    /// <summary>
    /// Journalctl Parser for Linux systemd journal
    /// Parses JSON output from journalctl --output=json
    /// 
    /// HARD RULES:
    /// - Parser decodes and structures - does NOT detect threats
    /// - Parser does NOT filter or enrich
    /// - Parser does NOT use hardcoded event IDs or patterns
    /// - All detection is done by backend ML/analytics
    /// </summary>
    public class JournalctlParser : IParser
    {
        private readonly ILogger<JournalctlParser> _logger;

        // Metrics
        private long _eventsParsed = 0;
        private long _parseErrors = 0;
        private readonly DateTime _startTime = DateTime.UtcNow;

        public string Name => "JournalctlParser";
        public string SourceType => "Journalctl";

        public JournalctlParser(ILogger<JournalctlParser> logger)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }

        public bool CanParse(object rawEvent)
        {
            if (rawEvent is string jsonStr)
            {
                // Check if it looks like journalctl JSON
                return jsonStr.TrimStart().StartsWith("{") && 
                       (jsonStr.Contains("__REALTIME_TIMESTAMP") || 
                        jsonStr.Contains("_HOSTNAME") ||
                        jsonStr.Contains("SYSLOG_IDENTIFIER"));
            }
            if (rawEvent is JsonElement)
            {
                return true;
            }
            if (rawEvent is Dictionary<string, object> dict)
            {
                return dict.ContainsKey("__REALTIME_TIMESTAMP") ||
                       dict.ContainsKey("_HOSTNAME") ||
                       dict.ContainsKey("MESSAGE");
            }
            return false;
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

                if (rawEvent is string jsonStr)
                {
                    parsed = ParseJsonString(jsonStr);
                }
                else if (rawEvent is JsonElement jsonElement)
                {
                    parsed = ParseJsonElement(jsonElement);
                }
                else if (rawEvent is Dictionary<string, object> dict)
                {
                    parsed = ParseDictionary(dict);
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
                _logger.LogError(ex, "Error parsing journalctl entry: {Message}", ex.Message);
                throw;
            }
        }

        /// <summary>
        /// Parse JSON string from journalctl output
        /// </summary>
        private ParsedEvent ParseJsonString(string json)
        {
            var doc = JsonDocument.Parse(json);
            return ParseJsonElement(doc.RootElement, json);
        }

        /// <summary>
        /// Parse JsonElement
        /// </summary>
        private ParsedEvent ParseJsonElement(JsonElement element, string? rawJson = null)
        {
            var properties = new Dictionary<string, object>();

            // Extract all properties - NO filtering, backend decides what's relevant
            foreach (var prop in element.EnumerateObject())
            {
                properties[prop.Name] = GetJsonValue(prop.Value);
            }

            return CreateParsedEvent(properties, rawJson ?? element.GetRawText());
        }

        /// <summary>
        /// Parse dictionary
        /// </summary>
        private ParsedEvent ParseDictionary(Dictionary<string, object> dict)
        {
            return CreateParsedEvent(dict, dict);
        }

        /// <summary>
        /// Create ParsedEvent from properties
        /// </summary>
        private ParsedEvent CreateParsedEvent(Dictionary<string, object> props, object rawEvent)
        {
            var parsed = new ParsedEvent
            {
                RawEvent = rawEvent,
                Timestamp = ExtractTimestamp(props),
                Level = ExtractLevel(props),
                Message = ExtractStringProperty(props, "MESSAGE") ?? "",
                SourceHost = ExtractStringProperty(props, "_HOSTNAME"),
                SourceApplication = ExtractStringProperty(props, "SYSLOG_IDENTIFIER") ??
                                   ExtractStringProperty(props, "_COMM"),
                SourceType = ExtractStringProperty(props, "_TRANSPORT") ?? "journal",
                Collector = "JournalctlCollector",
                Properties = props
            };

            // Extract user information
            var uid = ExtractStringProperty(props, "_UID");
            var user = ExtractStringProperty(props, "_AUDIT_LOGINUID") ?? uid;
            if (!string.IsNullOrEmpty(user))
            {
                parsed.User = new ParsedUserInfo
                {
                    Id = uid,
                    Name = user
                };
            }

            // Extract process information
            var pid = ExtractIntProperty(props, "_PID");
            var comm = ExtractStringProperty(props, "_COMM");
            var exe = ExtractStringProperty(props, "_EXE");
            var cmdline = ExtractStringProperty(props, "_CMDLINE");

            if (pid.HasValue || !string.IsNullOrEmpty(comm))
            {
                parsed.Process = new ParsedProcessInfo
                {
                    Name = comm,
                    Pid = pid,
                    Executable = exe,
                    CommandLine = cmdline
                };
            }

            // Extract categories from systemd unit
            var unit = ExtractStringProperty(props, "_SYSTEMD_UNIT");
            if (!string.IsNullOrEmpty(unit))
            {
                parsed.Categories = new List<string> { "systemd", unit.Replace(".service", "") };
            }

            return parsed;
        }

        /// <summary>
        /// Extract timestamp from journalctl properties
        /// </summary>
        private DateTime ExtractTimestamp(Dictionary<string, object> props)
        {
            // Try __REALTIME_TIMESTAMP (microseconds since epoch)
            if (props.TryGetValue("__REALTIME_TIMESTAMP", out var realtimeObj))
            {
                if (long.TryParse(realtimeObj?.ToString(), out var microseconds))
                {
                    var epoch = new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc);
                    return epoch.AddTicks(microseconds * 10); // Convert microseconds to ticks
                }
            }

            // Try _SOURCE_REALTIME_TIMESTAMP
            if (props.TryGetValue("_SOURCE_REALTIME_TIMESTAMP", out var sourceObj))
            {
                if (long.TryParse(sourceObj?.ToString(), out var microseconds))
                {
                    var epoch = new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc);
                    return epoch.AddTicks(microseconds * 10);
                }
            }

            return DateTime.UtcNow;
        }

        /// <summary>
        /// Extract log level from journalctl properties
        /// </summary>
        private string ExtractLevel(Dictionary<string, object> props)
        {
            // Try PRIORITY (syslog priority 0-7)
            if (props.TryGetValue("PRIORITY", out var priorityObj))
            {
                if (int.TryParse(priorityObj?.ToString(), out var priority))
                {
                    return MapPriorityToLevel(priority);
                }
            }

            // Try SYSLOG_FACILITY to infer level
            if (props.TryGetValue("SYSLOG_FACILITY", out var facilityObj))
            {
                // Just use default for facility
            }

            return "Information";
        }

        /// <summary>
        /// Map syslog priority to level
        /// NO detection - just standard mapping
        /// </summary>
        private string MapPriorityToLevel(int priority)
        {
            return priority switch
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
        /// Extract string property
        /// </summary>
        private string? ExtractStringProperty(Dictionary<string, object> props, string key)
        {
            if (props.TryGetValue(key, out var value))
            {
                return value?.ToString();
            }
            return null;
        }

        /// <summary>
        /// Extract int property
        /// </summary>
        private int? ExtractIntProperty(Dictionary<string, object> props, string key)
        {
            if (props.TryGetValue(key, out var value))
            {
                if (int.TryParse(value?.ToString(), out var result))
                {
                    return result;
                }
            }
            return null;
        }

        /// <summary>
        /// Get value from JsonElement
        /// </summary>
        private object GetJsonValue(JsonElement element)
        {
            return element.ValueKind switch
            {
                JsonValueKind.String => element.GetString() ?? "",
                JsonValueKind.Number => element.TryGetInt64(out var l) ? l : element.GetDouble(),
                JsonValueKind.True => true,
                JsonValueKind.False => false,
                JsonValueKind.Null => "",
                JsonValueKind.Array => element.GetRawText(),
                JsonValueKind.Object => element.GetRawText(),
                _ => element.GetRawText()
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
