using System;
using System.Collections.Generic;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using AthalaSIEM.UniversalAgent.Core.Normalizer;

namespace AthalaSIEM.UniversalAgent.Core.Parser
{
    /// <summary>
    /// Network Device Parser
    /// Parses logs from firewalls, routers, switches, IDS/IPS, and other network devices
    /// Supports various vendor formats via generic parsing (NOT hardcoded)
    /// 
    /// HARD RULES:
    /// - Parser decodes and structures - does NOT detect threats
    /// - Parser does NOT filter or enrich
    /// - Parser does NOT use hardcoded vendor patterns for detection
    /// - Parser extracts structure - backend ML does detection
    /// - All detection is done by backend analytics
    /// </summary>
    public class NetworkDeviceParser : IParser
    {
        private readonly ILogger<NetworkDeviceParser> _logger;

        // Generic patterns for extracting common network log fields
        // These are structural patterns, NOT detection patterns
        private static readonly Regex IpAddressPattern = new(
            @"\b(?<ip>(?:\d{1,3}\.){3}\d{1,3})\b",
            RegexOptions.Compiled);

        private static readonly Regex PortPattern = new(
            @"\b(?:port|dst_port|src_port|dport|sport)[=:\s]*(?<port>\d{1,5})\b",
            RegexOptions.Compiled | RegexOptions.IgnoreCase);

        private static readonly Regex ProtocolPattern = new(
            @"\b(?:proto|protocol)[=:\s]*(?<protocol>\w+)\b",
            RegexOptions.Compiled | RegexOptions.IgnoreCase);

        private static readonly Regex KeyValuePattern = new(
            @"(?<key>[a-zA-Z_][a-zA-Z0-9_]*)[=:](?<value>""[^""]*""|'[^']*'|\S+)",
            RegexOptions.Compiled);

        // Metrics
        private long _eventsParsed = 0;
        private long _parseErrors = 0;
        private readonly DateTime _startTime = DateTime.UtcNow;

        public string Name => "NetworkDeviceParser";
        public string SourceType => "NetworkDevice";

        public NetworkDeviceParser(ILogger<NetworkDeviceParser> logger)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }

        public bool CanParse(object rawEvent)
        {
            // Network device logs typically come as strings via syslog
            // This is a fallback parser when no specific parser matches
            return rawEvent is string;
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
                    parsed = ParseNetworkDeviceLog(message);
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
                _logger.LogError(ex, "Error parsing network device log: {Message}", ex.Message);
                throw;
            }
        }

        /// <summary>
        /// Parse network device log message
        /// Uses generic extraction - NO vendor-specific hardcoding
        /// </summary>
        private ParsedEvent ParseNetworkDeviceLog(string message)
        {
            var parsed = new ParsedEvent
            {
                RawEvent = message,
                Timestamp = DateTime.UtcNow,
                Level = "Information",
                Message = message,
                SourceType = "NetworkDevice",
                Collector = "NetworkDeviceCollector",
                Properties = new Dictionary<string, object>()
            };

            // Extract key-value pairs (common in many network device logs)
            ExtractKeyValuePairs(message, parsed);

            // Extract network information using generic patterns
            ExtractNetworkInfo(message, parsed);

            // Extract hostname/device from common patterns
            ExtractDeviceInfo(message, parsed);

            // Build categories based on extracted information
            BuildCategories(parsed);

            return parsed;
        }

        /// <summary>
        /// Extract key-value pairs from log message
        /// Many network devices use key=value or key:value format
        /// </summary>
        private void ExtractKeyValuePairs(string message, ParsedEvent parsed)
        {
            var matches = KeyValuePattern.Matches(message);
            foreach (Match match in matches)
            {
                var key = match.Groups["key"].Value.ToLowerInvariant();
                var value = match.Groups["value"].Value.Trim('"', '\'');

                // Store in properties
                parsed.Properties[$"device.{key}"] = value;

                // Map common keys to parsed fields (generic mapping)
                MapCommonField(key, value, parsed);
            }
        }

        /// <summary>
        /// Map common network device fields to parsed event
        /// This is generic mapping, NOT detection
        /// </summary>
        private void MapCommonField(string key, string value, ParsedEvent parsed)
        {
            switch (key)
            {
                // Action/status fields (many vendors use these)
                case "action":
                case "act":
                case "disposition":
                case "result":
                    parsed.Action = value;
                    break;

                // User fields
                case "user":
                case "username":
                case "usr":
                case "srcuser":
                case "src_user":
                    parsed.User ??= new ParsedUserInfo();
                    parsed.User.Name = value;
                    break;

                // Source network fields
                case "src":
                case "srcip":
                case "src_ip":
                case "source":
                case "sourceip":
                case "source_ip":
                    parsed.Network ??= new ParsedNetworkInfo();
                    parsed.Network.SourceIp = value;
                    break;

                case "srcport":
                case "src_port":
                case "sport":
                case "sourceport":
                case "source_port":
                    if (int.TryParse(value, out var srcPort))
                    {
                        parsed.Network ??= new ParsedNetworkInfo();
                        parsed.Network.SourcePort = srcPort;
                    }
                    break;

                // Destination network fields
                case "dst":
                case "dstip":
                case "dst_ip":
                case "dest":
                case "destination":
                case "destinationip":
                case "destination_ip":
                    parsed.Network ??= new ParsedNetworkInfo();
                    parsed.Network.DestinationIp = value;
                    break;

                case "dstport":
                case "dst_port":
                case "dport":
                case "destport":
                case "destination_port":
                    if (int.TryParse(value, out var dstPort))
                    {
                        parsed.Network ??= new ParsedNetworkInfo();
                        parsed.Network.DestinationPort = dstPort;
                    }
                    break;

                // Protocol
                case "proto":
                case "protocol":
                case "service":
                    parsed.Network ??= new ParsedNetworkInfo();
                    parsed.Network.Protocol = value;
                    break;

                // Bytes/packets
                case "bytes":
                case "bytesin":
                case "bytes_in":
                case "rcvdbytes":
                case "received_bytes":
                    if (long.TryParse(value, out var bytes))
                    {
                        parsed.Network ??= new ParsedNetworkInfo();
                        parsed.Network.Bytes = bytes;
                    }
                    break;

                // Host/device
                case "hostname":
                case "device":
                case "devicename":
                case "device_name":
                case "devname":
                    parsed.SourceHost = value;
                    break;

                // Application
                case "app":
                case "application":
                case "appname":
                case "app_name":
                    parsed.SourceApplication = value;
                    break;
            }
        }

        /// <summary>
        /// Extract network information using generic patterns
        /// </summary>
        private void ExtractNetworkInfo(string message, ParsedEvent parsed)
        {
            // Extract IP addresses (if not already found via key-value)
            if (parsed.Network?.SourceIp == null || parsed.Network?.DestinationIp == null)
            {
                var ipMatches = IpAddressPattern.Matches(message);
                var ips = new List<string>();
                foreach (Match match in ipMatches)
                {
                    ips.Add(match.Groups["ip"].Value);
                }

                if (ips.Count >= 1 && parsed.Network?.SourceIp == null)
                {
                    parsed.Network ??= new ParsedNetworkInfo();
                    parsed.Network.SourceIp = ips[0];
                }
                if (ips.Count >= 2 && parsed.Network?.DestinationIp == null)
                {
                    parsed.Network!.DestinationIp = ips[1];
                }
            }

            // Extract ports (if not already found)
            if (parsed.Network != null)
            {
                var portMatches = PortPattern.Matches(message);
                var portIndex = 0;
                foreach (Match match in portMatches)
                {
                    if (int.TryParse(match.Groups["port"].Value, out var port))
                    {
                        if (portIndex == 0 && !parsed.Network.SourcePort.HasValue)
                        {
                            parsed.Network.SourcePort = port;
                        }
                        else if (portIndex == 1 && !parsed.Network.DestinationPort.HasValue)
                        {
                            parsed.Network.DestinationPort = port;
                        }
                        portIndex++;
                    }
                }
            }

            // Extract protocol (if not already found)
            if (parsed.Network != null && string.IsNullOrEmpty(parsed.Network.Protocol))
            {
                var protoMatch = ProtocolPattern.Match(message);
                if (protoMatch.Success)
                {
                    parsed.Network.Protocol = protoMatch.Groups["protocol"].Value.ToUpperInvariant();
                }
            }
        }

        /// <summary>
        /// Extract device information from log
        /// </summary>
        private void ExtractDeviceInfo(string message, ParsedEvent parsed)
        {
            // If we don't have a hostname yet, try to find one
            if (string.IsNullOrEmpty(parsed.SourceHost))
            {
                // Look for common hostname patterns
                var hostnamePatterns = new[]
                {
                    @"from\s+(?<host>\S+)",
                    @"device[=:]\s*(?<host>\S+)",
                    @"host[=:]\s*(?<host>\S+)"
                };

                foreach (var pattern in hostnamePatterns)
                {
                    var match = Regex.Match(message, pattern, RegexOptions.IgnoreCase);
                    if (match.Success)
                    {
                        parsed.SourceHost = match.Groups["host"].Value;
                        break;
                    }
                }
            }
        }

        /// <summary>
        /// Build categories based on extracted information
        /// Categories are for organization, NOT detection
        /// </summary>
        private void BuildCategories(ParsedEvent parsed)
        {
            parsed.Categories = new List<string> { "network" };

            // Add category based on presence of network data
            if (parsed.Network != null)
            {
                if (!string.IsNullOrEmpty(parsed.Network.SourceIp) ||
                    !string.IsNullOrEmpty(parsed.Network.DestinationIp))
                {
                    parsed.Categories.Add("network_traffic");
                }
            }

            // Add category based on action if present
            if (!string.IsNullOrEmpty(parsed.Action))
            {
                // Normalize action to lowercase for consistent categorization
                var normalizedAction = parsed.Action.ToLowerInvariant();
                parsed.Categories.Add($"action_{normalizedAction}");
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
