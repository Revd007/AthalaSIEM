using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using AthalaSIEM.UniversalAgent.Models;

namespace AthalaSIEM.UniversalAgent.Core.Normalizer
{
    /// <summary>
    /// Production-grade Athala ECS-lite Normalizer
    /// Maps parsed events to Athala ECS-lite schema following specification
    /// 
    /// HARD RULES (from specification):
    /// - Normalizer MUST NOT detect
    /// - Normalizer MUST NOT parse (that's Parser's job)
    /// - Normalizer MUST NOT enrich (that's Enricher's job)
    /// - Normalizer MUST preserve raw_event
    /// - Normalizer MUST output consistent schema across all platforms
    /// </summary>
    public class AthalaEcsLiteNormalizer : INormalizer
    {
        private readonly ILogger<AthalaEcsLiteNormalizer> _logger;
        private readonly string _agentId;
        private readonly string _agentName;
        private readonly string _agentVersion;
        private readonly string _hostName;
        private readonly OsInfo _hostOs;

        // Metrics
        private long _eventsNormalized = 0;
        private long _normalizationErrors = 0;
        private readonly DateTime _startTime = DateTime.UtcNow;

        public string Name => "AthalaEcsLiteNormalizer";

        public AthalaEcsLiteNormalizer(
            ILogger<AthalaEcsLiteNormalizer> logger,
            string agentId,
            string agentName,
            string agentVersion,
            string hostName,
            OsInfo hostOs)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _agentId = agentId ?? throw new ArgumentNullException(nameof(agentId));
            _agentName = agentName ?? throw new ArgumentNullException(nameof(agentName));
            _agentVersion = agentVersion ?? "1.0.0";
            _hostName = hostName ?? Environment.MachineName;
            _hostOs = hostOs ?? throw new ArgumentNullException(nameof(hostOs));
        }

        /// <summary>
        /// Normalizes a parsed event to Athala ECS-lite schema
        /// </summary>
        public Task<AthalaEcsLiteEvent> NormalizeAsync(ParsedEvent parsedEvent)
        {
            if (parsedEvent == null)
            {
                throw new ArgumentNullException(nameof(parsedEvent));
            }

            try
            {
                var normalized = new AthalaEcsLiteEvent
                {
                    // Core timestamp - REQUIRED
                    Timestamp = parsedEvent.Timestamp.Kind == DateTimeKind.Utc
                        ? parsedEvent.Timestamp
                        : parsedEvent.Timestamp.ToUniversalTime(),

                    // Agent information - REQUIRED
                    Agent = new AgentInfo
                    {
                        Id = _agentId,
                        Name = _agentName,
                        Version = _agentVersion,
                        Type = DetermineAgentType()
                    },

                    // Host information - REQUIRED
                    Host = new HostInfo
                    {
                        Name = _hostName,
                        Os = _hostOs
                    },

                    // Event categorization - REQUIRED
                    Event = new EventInfo
                    {
                        Category = parsedEvent.Categories ?? new List<string>(),
                        Action = parsedEvent.Action,
                        Outcome = parsedEvent.Outcome,
                        Code = parsedEvent.EventId,
                        Severity = MapSeverityToNumeric(parsedEvent.Level)
                    },

                    // Log level - REQUIRED
                    Log = new LogInfo
                    {
                        Level = NormalizeLogLevel(parsedEvent.Level),
                        Logger = parsedEvent.SourceApplication,
                        Original = parsedEvent.Message
                    },

                    // User information (when available)
                    User = parsedEvent.User != null ? new UserInfo
                    {
                        Name = parsedEvent.User.Name,
                        Id = parsedEvent.User.Id,
                        Domain = parsedEvent.User.Domain,
                        FullName = parsedEvent.User.FullName,
                        Email = parsedEvent.User.Email
                    } : null,

                    // Process information (when available)
                    Process = parsedEvent.Process != null ? new ProcessInfo
                    {
                        Name = parsedEvent.Process.Name,
                        Pid = parsedEvent.Process.Pid,
                        Ppid = parsedEvent.Process.Ppid,
                        CommandLine = parsedEvent.Process.CommandLine,
                        Executable = parsedEvent.Process.Executable,
                        WorkingDirectory = parsedEvent.Process.WorkingDirectory,
                        Parent = parsedEvent.Process.ParentName != null ? new ProcessParentInfo
                        {
                            Name = parsedEvent.Process.ParentName,
                            Pid = parsedEvent.Process.ParentPid,
                            CommandLine = parsedEvent.Process.ParentCommandLine
                        } : null
                    } : null,

                    // Network information (when available)
                    Network = parsedEvent.Network != null ? new NetworkInfo
                    {
                        Protocol = parsedEvent.Network.Protocol,
                        Transport = parsedEvent.Network.Transport,
                        Bytes = parsedEvent.Network.Bytes,
                        Packets = parsedEvent.Network.Packets
                    } : null,

                    // Source information (when available)
                    Source = parsedEvent.Network?.SourceIp != null ? new SourceInfo
                    {
                        Ip = parsedEvent.Network.SourceIp,
                        Port = parsedEvent.Network.SourcePort,
                        Address = parsedEvent.Network.SourceIp
                    } : null,

                    // Destination information (when available)
                    Destination = parsedEvent.Network?.DestinationIp != null ? new DestinationInfo
                    {
                        Ip = parsedEvent.Network.DestinationIp,
                        Port = parsedEvent.Network.DestinationPort,
                        Address = parsedEvent.Network.DestinationIp
                    } : null,

                    // File information (when available)
                    File = parsedEvent.File != null ? new FileInfo
                    {
                        Name = parsedEvent.File.Name,
                        Path = parsedEvent.File.Path,
                        Extension = parsedEvent.File.Extension,
                        Size = parsedEvent.File.Size,
                        MimeType = parsedEvent.File.MimeType,
                        Hash = CreateFileHashInfo(parsedEvent.File),
                        Owner = parsedEvent.File.Owner,
                        Group = parsedEvent.File.Group
                    } : null,

                    // Athala extensions - REQUIRED (MUST preserve raw_event)
                    Athala = new AthalaExtensions
                    {
                        RawEvent = parsedEvent.RawEvent, // CRITICAL: Preserve original
                        Collector = parsedEvent.Collector,
                        SourceType = parsedEvent.SourceType ?? "",
                        PipelineStage = "normalized",
                        OriginalEventId = parsedEvent.EventId,
                        SecurityRelevance = parsedEvent.SecurityRelevance,
                        Metadata = parsedEvent.Properties.Any() ? parsedEvent.Properties : null
                    }
                };

                _eventsNormalized++;
                return Task.FromResult(normalized);
            }
            catch (Exception ex)
            {
                _normalizationErrors++;
                _logger.LogError(ex, "Error normalizing event: {Message}", ex.Message);
                throw;
            }
        }

        /// <summary>
        /// Determines agent type based on OS
        /// </summary>
        private string DetermineAgentType()
        {
            return _hostOs.Platform?.ToLowerInvariant() switch
            {
                "windows" => "Windows",
                "linux" => "Linux",
                "darwin" => "macOS",
                _ => "Custom"
            };
        }

        /// <summary>
        /// Normalizes log level to standard values
        /// </summary>
        private string NormalizeLogLevel(string? level)
        {
            if (string.IsNullOrEmpty(level))
                return "Information";

            return level.ToUpperInvariant() switch
            {
                "VERBOSE" or "TRACE" or "DEBUG" => "Debug",
                "INFO" or "INFORMATION" => "Information",
                "WARN" or "WARNING" => "Warning",
                "ERR" or "ERROR" => "Error",
                "CRIT" or "CRITICAL" or "FATAL" => "Critical",
                _ => "Information"
            };
        }

        /// <summary>
        /// Maps log level to numeric severity (0-7, RFC 5424)
        /// </summary>
        private long? MapSeverityToNumeric(string? level)
        {
            if (string.IsNullOrEmpty(level))
                return 6; // Informational

            return level.ToUpperInvariant() switch
            {
                "DEBUG" or "VERBOSE" or "TRACE" => 7, // Debug
                "INFO" or "INFORMATION" => 6,         // Informational
                "WARN" or "WARNING" => 4,             // Warning
                "ERR" or "ERROR" => 3,                // Error
                "CRIT" or "CRITICAL" or "FATAL" => 2, // Critical
                _ => 6                                 // Default: Informational
            };
        }

        /// <summary>
        /// Creates file hash information object
        /// </summary>
        private FileHashInfo? CreateFileHashInfo(ParsedFileInfo file)
        {
            if (string.IsNullOrEmpty(file.HashMd5) &&
                string.IsNullOrEmpty(file.HashSha1) &&
                string.IsNullOrEmpty(file.HashSha256))
            {
                return null;
            }

            return new FileHashInfo
            {
                Md5 = file.HashMd5,
                Sha1 = file.HashSha1,
                Sha256 = file.HashSha256
            };
        }

        /// <summary>
        /// Gets normalization metrics
        /// </summary>
        public Dictionary<string, object> GetMetrics()
        {
            var uptime = DateTime.UtcNow - _startTime;
            return new Dictionary<string, object>
            {
                ["Name"] = Name,
                ["EventsNormalized"] = _eventsNormalized,
                ["NormalizationErrors"] = _normalizationErrors,
                ["SuccessRate"] = _eventsNormalized > 0
                    ? (double)(_eventsNormalized - _normalizationErrors) / _eventsNormalized * 100
                    : 100.0,
                ["UptimeSeconds"] = uptime.TotalSeconds,
                ["EventsPerSecond"] = uptime.TotalSeconds > 0
                    ? _eventsNormalized / uptime.TotalSeconds
                    : 0.0
            };
        }
    }
}
