using System;
using System.Collections.Generic;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using AthalaSIEM.UniversalAgent.Core.Normalizer;

namespace AthalaSIEM.UniversalAgent.Core.Parser
{
    /// <summary>
    /// Generic Log Parser
    /// Fallback parser for any log format not handled by specific parsers
    /// Uses heuristics to extract structure from unstructured logs
    /// 
    /// HARD RULES:
    /// - Parser decodes and structures - does NOT detect threats
    /// - Parser does NOT filter or enrich
    /// - Parser does NOT use hardcoded patterns for detection
    /// - All detection is done by backend ML/analytics
    /// </summary>
    public class GenericLogParser : IParser
    {
        private readonly ILogger<GenericLogParser> _logger;

        // Generic timestamp patterns (common formats)
        private static readonly Regex[] TimestampPatterns = new[]
        {
            // ISO 8601
            new Regex(@"(?<ts>\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)", RegexOptions.Compiled),
            // Common log format
            new Regex(@"(?<ts>\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2}\s+[+-]\d{4})", RegexOptions.Compiled),
            // Syslog-like
            new Regex(@"(?<ts>\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})", RegexOptions.Compiled),
            // Generic date time
            new Regex(@"(?<ts>\d{4}[-/]\d{2}[-/]\d{2}\s+\d{2}:\d{2}:\d{2})", RegexOptions.Compiled)
        };

        // Generic log level patterns
        private static readonly Regex LevelPattern = new(
            @"\b(?<level>TRACE|DEBUG|INFO(?:RMATION)?|WARN(?:ING)?|ERROR|CRIT(?:ICAL)?|FATAL|EMERG(?:ENCY)?|ALERT|NOTICE)\b",
            RegexOptions.Compiled | RegexOptions.IgnoreCase);

        // Generic key-value pattern
        private static readonly Regex KeyValuePattern = new(
            @"(?<key>[a-zA-Z_][a-zA-Z0-9_]*)[=:](?<value>""[^""]*""|'[^']*'|\[[^\]]*\]|\{[^}]*\}|\S+)",
            RegexOptions.Compiled);

        // IP address pattern
        private static readonly Regex IpPattern = new(
            @"\b(?<ip>(?:\d{1,3}\.){3}\d{1,3})\b",
            RegexOptions.Compiled);

        // Metrics
        private long _eventsParsed = 0;
        private long _parseErrors = 0;
        private readonly DateTime _startTime = DateTime.UtcNow;

        public string Name => "GenericLogParser";
        public string SourceType => "Generic";

        public GenericLogParser(ILogger<GenericLogParser> logger)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }

        public bool CanParse(object rawEvent)
        {
            // Generic parser can attempt to parse anything
            return rawEvent != null;
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

                if (rawEvent is string strEvent)
                {
                    // Try JSON first
                    if (strEvent.TrimStart().StartsWith("{") || strEvent.TrimStart().StartsWith("["))
                    {
                        try
                        {
                            parsed = ParseJson(strEvent);
                        }
                        catch
                        {
                            // Fall back to text parsing
                            parsed = ParseText(strEvent);
                        }
                    }
                    else
                    {
                        parsed = ParseText(strEvent);
                    }
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
                    // Convert to string and parse
                    parsed = ParseText(rawEvent.ToString() ?? "");
                }

                _eventsParsed++;
                return Task.FromResult(parsed);
            }
            catch (Exception ex)
            {
                _parseErrors++;
                _logger.LogError(ex, "Error parsing log: {Message}", ex.Message);
                throw;
            }
        }

        /// <summary>
        /// Parse JSON string
        /// </summary>
        private ParsedEvent ParseJson(string json)
        {
            var doc = JsonDocument.Parse(json);
            return ParseJsonElement(doc.RootElement, json);
        }

        /// <summary>
        /// Parse JsonElement
        /// </summary>
        private ParsedEvent ParseJsonElement(JsonElement element, string? rawJson = null)
        {
            var props = new Dictionary<string, object>();
            ExtractJsonProperties(element, props, "");

            var parsed = new ParsedEvent
            {
                RawEvent = rawJson ?? element.GetRawText(),
                Timestamp = ExtractTimestampFromProps(props),
                Level = ExtractLevelFromProps(props),
                Message = ExtractMessageFromProps(props),
                SourceType = "JSON",
                Collector = "GenericCollector",
                Properties = props
            };

            ExtractCommonFieldsFromProps(props, parsed);
            return parsed;
        }

        /// <summary>
        /// Parse dictionary
        /// </summary>
        private ParsedEvent ParseDictionary(Dictionary<string, object> dict)
        {
            var parsed = new ParsedEvent
            {
                RawEvent = dict,
                Timestamp = ExtractTimestampFromProps(dict),
                Level = ExtractLevelFromProps(dict),
                Message = ExtractMessageFromProps(dict),
                SourceType = "Dictionary",
                Collector = "GenericCollector",
                Properties = dict
            };

            ExtractCommonFieldsFromProps(dict, parsed);
            return parsed;
        }

        /// <summary>
        /// Parse text log
        /// </summary>
        private ParsedEvent ParseText(string text)
        {
            var parsed = new ParsedEvent
            {
                RawEvent = text,
                Timestamp = ExtractTimestampFromText(text),
                Level = ExtractLevelFromText(text),
                Message = text,
                SourceType = "Text",
                Collector = "GenericCollector",
                Properties = new Dictionary<string, object>()
            };

            // Extract key-value pairs
            ExtractKeyValuePairs(text, parsed);

            // Extract IP addresses
            ExtractIpAddresses(text, parsed);

            return parsed;
        }

        /// <summary>
        /// Extract JSON properties recursively
        /// </summary>
        private void ExtractJsonProperties(JsonElement element, Dictionary<string, object> props, string prefix)
        {
            switch (element.ValueKind)
            {
                case JsonValueKind.Object:
                    foreach (var prop in element.EnumerateObject())
                    {
                        var key = string.IsNullOrEmpty(prefix) ? prop.Name : $"{prefix}.{prop.Name}";
                        if (prop.Value.ValueKind == JsonValueKind.Object || prop.Value.ValueKind == JsonValueKind.Array)
                        {
                            ExtractJsonProperties(prop.Value, props, key);
                        }
                        else
                        {
                            props[key] = GetJsonValue(prop.Value);
                        }
                    }
                    break;
                case JsonValueKind.Array:
                    props[prefix] = element.GetRawText();
                    break;
                default:
                    props[prefix] = GetJsonValue(element);
                    break;
            }
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
                _ => element.GetRawText()
            };
        }

        /// <summary>
        /// Extract timestamp from properties
        /// </summary>
        private DateTime ExtractTimestampFromProps(Dictionary<string, object> props)
        {
            // Common timestamp field names
            var timestampKeys = new[] { "timestamp", "@timestamp", "time", "datetime", "date", "created_at", "ts" };

            foreach (var key in timestampKeys)
            {
                if (props.TryGetValue(key, out var value))
                {
                    if (DateTime.TryParse(value?.ToString(), out var dt))
                    {
                        return dt.Kind == DateTimeKind.Utc ? dt : dt.ToUniversalTime();
                    }
                    // Try Unix timestamp
                    if (long.TryParse(value?.ToString(), out var unixTs))
                    {
                        if (unixTs > 1000000000000) // Milliseconds
                            return DateTimeOffset.FromUnixTimeMilliseconds(unixTs).UtcDateTime;
                        else
                            return DateTimeOffset.FromUnixTimeSeconds(unixTs).UtcDateTime;
                    }
                }
            }

            return DateTime.UtcNow;
        }

        /// <summary>
        /// Extract timestamp from text
        /// </summary>
        private DateTime ExtractTimestampFromText(string text)
        {
            foreach (var pattern in TimestampPatterns)
            {
                var match = pattern.Match(text);
                if (match.Success)
                {
                    if (DateTime.TryParse(match.Groups["ts"].Value, out var dt))
                    {
                        return dt.Kind == DateTimeKind.Utc ? dt : dt.ToUniversalTime();
                    }
                }
            }
            return DateTime.UtcNow;
        }

        /// <summary>
        /// Extract log level from properties
        /// </summary>
        private string ExtractLevelFromProps(Dictionary<string, object> props)
        {
            var levelKeys = new[] { "level", "severity", "log_level", "loglevel", "priority" };

            foreach (var key in levelKeys)
            {
                if (props.TryGetValue(key, out var value))
                {
                    return NormalizeLevel(value?.ToString() ?? "");
                }
            }

            return "Information";
        }

        /// <summary>
        /// Extract log level from text
        /// </summary>
        private string ExtractLevelFromText(string text)
        {
            var match = LevelPattern.Match(text);
            if (match.Success)
            {
                return NormalizeLevel(match.Groups["level"].Value);
            }
            return "Information";
        }

        /// <summary>
        /// Normalize log level
        /// </summary>
        private string NormalizeLevel(string level)
        {
            return level.ToUpperInvariant() switch
            {
                "TRACE" or "DEBUG" => "Debug",
                "INFO" or "INFORMATION" or "NOTICE" => "Information",
                "WARN" or "WARNING" => "Warning",
                "ERROR" or "ERR" => "Error",
                "CRIT" or "CRITICAL" or "FATAL" or "EMERG" or "EMERGENCY" or "ALERT" => "Critical",
                _ => "Information"
            };
        }

        /// <summary>
        /// Extract message from properties
        /// </summary>
        private string ExtractMessageFromProps(Dictionary<string, object> props)
        {
            var messageKeys = new[] { "message", "msg", "text", "log", "content", "body" };

            foreach (var key in messageKeys)
            {
                if (props.TryGetValue(key, out var value) && value != null)
                {
                    return value.ToString() ?? "";
                }
            }

            return "";
        }

        /// <summary>
        /// Extract common fields from properties
        /// </summary>
        private void ExtractCommonFieldsFromProps(Dictionary<string, object> props, ParsedEvent parsed)
        {
            // Host
            var hostKeys = new[] { "host", "hostname", "host.name", "source_host" };
            foreach (var key in hostKeys)
            {
                if (props.TryGetValue(key, out var value))
                {
                    parsed.SourceHost = value?.ToString();
                    break;
                }
            }

            // Application
            var appKeys = new[] { "application", "app", "service", "service.name" };
            foreach (var key in appKeys)
            {
                if (props.TryGetValue(key, out var value))
                {
                    parsed.SourceApplication = value?.ToString();
                    break;
                }
            }

            // User
            var userKeys = new[] { "user", "username", "user.name" };
            foreach (var key in userKeys)
            {
                if (props.TryGetValue(key, out var value))
                {
                    parsed.User = new ParsedUserInfo { Name = value?.ToString() };
                    break;
                }
            }
        }

        /// <summary>
        /// Extract key-value pairs from text
        /// </summary>
        private void ExtractKeyValuePairs(string text, ParsedEvent parsed)
        {
            var matches = KeyValuePattern.Matches(text);
            foreach (Match match in matches)
            {
                var key = match.Groups["key"].Value;
                var value = match.Groups["value"].Value.Trim('"', '\'');
                parsed.Properties[key] = value;
            }
        }

        /// <summary>
        /// Extract IP addresses from text
        /// </summary>
        private void ExtractIpAddresses(string text, ParsedEvent parsed)
        {
            var matches = IpPattern.Matches(text);
            if (matches.Count > 0)
            {
                parsed.Network = new ParsedNetworkInfo
                {
                    SourceIp = matches[0].Groups["ip"].Value
                };
                if (matches.Count > 1)
                {
                    parsed.Network.DestinationIp = matches[1].Groups["ip"].Value;
                }
            }
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
