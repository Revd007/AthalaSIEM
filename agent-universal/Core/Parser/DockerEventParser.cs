using System;
using System.Collections.Generic;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using AthalaSIEM.UniversalAgent.Core.Normalizer;

namespace AthalaSIEM.UniversalAgent.Core.Parser
{
    /// <summary>
    /// Docker Event Parser
    /// Parses Docker container events, logs, and metrics
    /// 
    /// HARD RULES:
    /// - Parser decodes and structures - does NOT detect threats
    /// - Parser does NOT filter or enrich
    /// - Parser does NOT use hardcoded patterns
    /// - All detection is done by backend ML/analytics
    /// </summary>
    public class DockerEventParser : IParser
    {
        private readonly ILogger<DockerEventParser> _logger;

        // Metrics
        private long _eventsParsed = 0;
        private long _parseErrors = 0;
        private readonly DateTime _startTime = DateTime.UtcNow;

        public string Name => "DockerEventParser";
        public string SourceType => "Docker";

        public DockerEventParser(ILogger<DockerEventParser> logger)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }

        public bool CanParse(object rawEvent)
        {
            if (rawEvent is string jsonStr)
            {
                // Check if it looks like Docker event JSON
                return jsonStr.TrimStart().StartsWith("{") &&
                       (jsonStr.Contains("\"Type\":") ||
                        jsonStr.Contains("\"Action\":") ||
                        jsonStr.Contains("\"Actor\":") ||
                        jsonStr.Contains("\"container_id\":") ||
                        jsonStr.Contains("\"image\":"));
            }
            if (rawEvent is JsonElement)
            {
                return true;
            }
            if (rawEvent is Dictionary<string, object> dict)
            {
                return dict.ContainsKey("Type") ||
                       dict.ContainsKey("Action") ||
                       dict.ContainsKey("container_id") ||
                       dict.ContainsKey("image");
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
                _logger.LogError(ex, "Error parsing Docker event: {Message}", ex.Message);
                throw;
            }
        }

        /// <summary>
        /// Parse JSON string from Docker
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

            // Extract all properties - NO filtering
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
                Level = "Information",
                Message = BuildMessage(props),
                SourceHost = ExtractStringProperty(props, "node") ?? Environment.MachineName,
                SourceApplication = "docker",
                SourceType = ExtractStringProperty(props, "Type") ?? "container",
                Collector = "DockerEventCollector",
                Properties = props
            };

            // Extract action as event action
            parsed.Action = ExtractStringProperty(props, "Action");

            // Build categories from event type
            var eventType = ExtractStringProperty(props, "Type");
            if (!string.IsNullOrEmpty(eventType))
            {
                parsed.Categories = new List<string> { "container", eventType.ToLowerInvariant() };
            }

            // Extract container/image information
            var containerId = ExtractContainerId(props);
            var containerName = ExtractContainerName(props);
            var imageName = ExtractStringProperty(props, "image") ?? 
                           ExtractStringProperty(props, "from");

            if (!string.IsNullOrEmpty(containerId) || !string.IsNullOrEmpty(containerName))
            {
                // Use process fields to store container info (semantic mapping)
                parsed.Process = new ParsedProcessInfo
                {
                    Name = containerName ?? containerId,
                    Executable = imageName
                };
            }

            // Extract network information if present
            var networkProps = ExtractNetworkInfo(props);
            if (networkProps != null)
            {
                parsed.Network = networkProps;
            }

            return parsed;
        }

        /// <summary>
        /// Extract timestamp from Docker event
        /// </summary>
        private DateTime ExtractTimestamp(Dictionary<string, object> props)
        {
            // Try "time" (Unix timestamp)
            if (props.TryGetValue("time", out var timeObj))
            {
                if (long.TryParse(timeObj?.ToString(), out var unixTime))
                {
                    return DateTimeOffset.FromUnixTimeSeconds(unixTime).UtcDateTime;
                }
            }

            // Try "timeNano" (nanoseconds)
            if (props.TryGetValue("timeNano", out var timeNanoObj))
            {
                if (long.TryParse(timeNanoObj?.ToString(), out var nanos))
                {
                    return DateTimeOffset.FromUnixTimeMilliseconds(nanos / 1000000).UtcDateTime;
                }
            }

            // Try "Time" (ISO format)
            if (props.TryGetValue("Time", out var timeStrObj))
            {
                if (DateTime.TryParse(timeStrObj?.ToString(), out var dt))
                {
                    return dt.Kind == DateTimeKind.Utc ? dt : dt.ToUniversalTime();
                }
            }

            return DateTime.UtcNow;
        }

        /// <summary>
        /// Build message from Docker event properties
        /// </summary>
        private string BuildMessage(Dictionary<string, object> props)
        {
            var type = ExtractStringProperty(props, "Type") ?? "event";
            var action = ExtractStringProperty(props, "Action") ?? "unknown";
            var actorId = ExtractContainerId(props) ?? ExtractStringProperty(props, "id");

            return $"Docker {type}: {action}" + 
                   (string.IsNullOrEmpty(actorId) ? "" : $" ({actorId})");
        }

        /// <summary>
        /// Extract container ID from various possible fields
        /// </summary>
        private string? ExtractContainerId(Dictionary<string, object> props)
        {
            // Try Actor.ID
            if (props.TryGetValue("Actor", out var actorObj) && actorObj is string actorJson)
            {
                try
                {
                    var actor = JsonDocument.Parse(actorJson);
                    if (actor.RootElement.TryGetProperty("ID", out var idProp))
                    {
                        return idProp.GetString();
                    }
                }
                catch { }
            }

            // Try direct fields
            return ExtractStringProperty(props, "container_id") ??
                   ExtractStringProperty(props, "id") ??
                   ExtractStringProperty(props, "ID");
        }

        /// <summary>
        /// Extract container name
        /// </summary>
        private string? ExtractContainerName(Dictionary<string, object> props)
        {
            // Try Actor.Attributes.name
            if (props.TryGetValue("Actor", out var actorObj) && actorObj is string actorJson)
            {
                try
                {
                    var actor = JsonDocument.Parse(actorJson);
                    if (actor.RootElement.TryGetProperty("Attributes", out var attrs) &&
                        attrs.TryGetProperty("name", out var nameProp))
                    {
                        return nameProp.GetString();
                    }
                }
                catch { }
            }

            return ExtractStringProperty(props, "container_name") ??
                   ExtractStringProperty(props, "name");
        }

        /// <summary>
        /// Extract network information if present
        /// </summary>
        private ParsedNetworkInfo? ExtractNetworkInfo(Dictionary<string, object> props)
        {
            var hasNetwork = false;
            var network = new ParsedNetworkInfo();

            // Check for network-related properties
            if (props.TryGetValue("Actor", out var actorObj) && actorObj is string actorJson)
            {
                try
                {
                    var actor = JsonDocument.Parse(actorJson);
                    if (actor.RootElement.TryGetProperty("Attributes", out var attrs))
                    {
                        if (attrs.TryGetProperty("ip", out var ip))
                        {
                            network.SourceIp = ip.GetString();
                            hasNetwork = true;
                        }
                        if (attrs.TryGetProperty("gateway", out var gw))
                        {
                            network.DestinationIp = gw.GetString();
                            hasNetwork = true;
                        }
                    }
                }
                catch { }
            }

            return hasNetwork ? network : null;
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
