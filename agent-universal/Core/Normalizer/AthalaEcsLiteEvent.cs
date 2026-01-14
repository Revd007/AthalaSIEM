using System;
using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace AthalaSIEM.UniversalAgent.Core.Normalizer
{
    /// <summary>
    /// Athala ECS-lite Normalized Event Schema
    /// Production-grade event normalization following the specification
    /// All events MUST be normalized to this schema before export
    /// </summary>
    public class AthalaEcsLiteEvent
    {
        // Core timestamp - REQUIRED
        [JsonPropertyName("@timestamp")]
        public DateTime Timestamp { get; set; }

        // Agent information - REQUIRED
        [JsonPropertyName("agent")]
        public AgentInfo Agent { get; set; } = new();

        // Host information - REQUIRED
        [JsonPropertyName("host")]
        public HostInfo Host { get; set; } = new();

        // Event categorization - REQUIRED
        [JsonPropertyName("event")]
        public EventInfo Event { get; set; } = new();

        // Log level - REQUIRED
        [JsonPropertyName("log")]
        public LogInfo Log { get; set; } = new();

        // User identity (when available)
        [JsonPropertyName("user")]
        public UserInfo? User { get; set; }

        // Process information (when available)
        [JsonPropertyName("process")]
        public ProcessInfo? Process { get; set; }

        // Network information (when available)
        [JsonPropertyName("network")]
        public NetworkInfo? Network { get; set; }

        // Source information (when available)
        [JsonPropertyName("source")]
        public SourceInfo? Source { get; set; }

        // Destination information (when available)
        [JsonPropertyName("destination")]
        public DestinationInfo? Destination { get; set; }

        // File information (when available)
        [JsonPropertyName("file")]
        public FileInfo? File { get; set; }

        // Athala-specific extensions - REQUIRED
        [JsonPropertyName("athala")]
        public AthalaExtensions Athala { get; set; } = new();
    }

    /// <summary>
    /// Agent information block
    /// </summary>
    public class AgentInfo
    {
        [JsonPropertyName("id")]
        public string Id { get; set; } = "";

        [JsonPropertyName("name")]
        public string Name { get; set; } = "";

        [JsonPropertyName("version")]
        public string? Version { get; set; }

        [JsonPropertyName("type")]
        public string? Type { get; set; }
    }

    /// <summary>
    /// Host information block
    /// </summary>
    public class HostInfo
    {
        [JsonPropertyName("name")]
        public string Name { get; set; } = "";

        [JsonPropertyName("os")]
        public OsInfo? Os { get; set; }

        [JsonPropertyName("ip")]
        public List<string>? Ip { get; set; }

        [JsonPropertyName("mac")]
        public List<string>? Mac { get; set; }
    }

    /// <summary>
    /// Operating system information
    /// </summary>
    public class OsInfo
    {
        [JsonPropertyName("name")]
        public string Name { get; set; } = "";

        [JsonPropertyName("version")]
        public string? Version { get; set; }

        [JsonPropertyName("platform")]
        public string? Platform { get; set; }

        [JsonPropertyName("family")]
        public string? Family { get; set; }
    }

    /// <summary>
    /// Event categorization block
    /// </summary>
    public class EventInfo
    {
        [JsonPropertyName("category")]
        public List<string> Category { get; set; } = new();

        [JsonPropertyName("action")]
        public string? Action { get; set; }

        [JsonPropertyName("outcome")]
        public string? Outcome { get; set; }

        [JsonPropertyName("type")]
        public List<string>? Type { get; set; }

        [JsonPropertyName("kind")]
        public string? Kind { get; set; }

        [JsonPropertyName("severity")]
        public long? Severity { get; set; }

        [JsonPropertyName("code")]
        public string? Code { get; set; }
    }

    /// <summary>
    /// Log information block
    /// </summary>
    public class LogInfo
    {
        [JsonPropertyName("level")]
        public string Level { get; set; } = "";

        [JsonPropertyName("logger")]
        public string? Logger { get; set; }

        [JsonPropertyName("original")]
        public string? Original { get; set; }
    }

    /// <summary>
    /// User identity information
    /// </summary>
    public class UserInfo
    {
        [JsonPropertyName("name")]
        public string? Name { get; set; }

        [JsonPropertyName("id")]
        public string? Id { get; set; }

        [JsonPropertyName("domain")]
        public string? Domain { get; set; }

        [JsonPropertyName("full_name")]
        public string? FullName { get; set; }

        [JsonPropertyName("email")]
        public string? Email { get; set; }
    }

    /// <summary>
    /// Process information
    /// </summary>
    public class ProcessInfo
    {
        [JsonPropertyName("name")]
        public string? Name { get; set; }

        [JsonPropertyName("pid")]
        public int? Pid { get; set; }

        [JsonPropertyName("ppid")]
        public int? Ppid { get; set; }

        [JsonPropertyName("command_line")]
        public string? CommandLine { get; set; }

        [JsonPropertyName("executable")]
        public string? Executable { get; set; }

        [JsonPropertyName("working_directory")]
        public string? WorkingDirectory { get; set; }

        [JsonPropertyName("parent")]
        public ProcessParentInfo? Parent { get; set; }
    }

    /// <summary>
    /// Parent process information
    /// </summary>
    public class ProcessParentInfo
    {
        [JsonPropertyName("name")]
        public string? Name { get; set; }

        [JsonPropertyName("pid")]
        public int? Pid { get; set; }

        [JsonPropertyName("command_line")]
        public string? CommandLine { get; set; }
    }

    /// <summary>
    /// Network information
    /// </summary>
    public class NetworkInfo
    {
        [JsonPropertyName("protocol")]
        public string? Protocol { get; set; }

        [JsonPropertyName("transport")]
        public string? Transport { get; set; }

        [JsonPropertyName("direction")]
        public string? Direction { get; set; }

        [JsonPropertyName("bytes")]
        public long? Bytes { get; set; }

        [JsonPropertyName("packets")]
        public long? Packets { get; set; }
    }

    /// <summary>
    /// Source information
    /// </summary>
    public class SourceInfo
    {
        [JsonPropertyName("ip")]
        public string? Ip { get; set; }

        [JsonPropertyName("port")]
        public int? Port { get; set; }

        [JsonPropertyName("address")]
        public string? Address { get; set; }

        [JsonPropertyName("mac")]
        public string? Mac { get; set; }

        [JsonPropertyName("domain")]
        public string? Domain { get; set; }

        [JsonPropertyName("geo")]
        public GeoInfo? Geo { get; set; }
    }

    /// <summary>
    /// Destination information
    /// </summary>
    public class DestinationInfo
    {
        [JsonPropertyName("ip")]
        public string? Ip { get; set; }

        [JsonPropertyName("port")]
        public int? Port { get; set; }

        [JsonPropertyName("address")]
        public string? Address { get; set; }

        [JsonPropertyName("mac")]
        public string? Mac { get; set; }

        [JsonPropertyName("domain")]
        public string? Domain { get; set; }

        [JsonPropertyName("geo")]
        public GeoInfo? Geo { get; set; }
    }

    /// <summary>
    /// Geographic information
    /// </summary>
    public class GeoInfo
    {
        [JsonPropertyName("country_iso_code")]
        public string? CountryIsoCode { get; set; }

        [JsonPropertyName("city_name")]
        public string? CityName { get; set; }

        [JsonPropertyName("region_name")]
        public string? RegionName { get; set; }

        [JsonPropertyName("location")]
        public GeoLocation? Location { get; set; }
    }

    /// <summary>
    /// Geographic location coordinates
    /// </summary>
    public class GeoLocation
    {
        [JsonPropertyName("lat")]
        public double? Lat { get; set; }

        [JsonPropertyName("lon")]
        public double? Lon { get; set; }
    }

    /// <summary>
    /// File information
    /// </summary>
    public class FileInfo
    {
        [JsonPropertyName("name")]
        public string? Name { get; set; }

        [JsonPropertyName("path")]
        public string? Path { get; set; }

        [JsonPropertyName("extension")]
        public string? Extension { get; set; }

        [JsonPropertyName("size")]
        public long? Size { get; set; }

        [JsonPropertyName("mime_type")]
        public string? MimeType { get; set; }

        [JsonPropertyName("hash")]
        public FileHashInfo? Hash { get; set; }

        [JsonPropertyName("owner")]
        public string? Owner { get; set; }

        [JsonPropertyName("group")]
        public string? Group { get; set; }
    }

    /// <summary>
    /// File hash information
    /// </summary>
    public class FileHashInfo
    {
        [JsonPropertyName("md5")]
        public string? Md5 { get; set; }

        [JsonPropertyName("sha1")]
        public string? Sha1 { get; set; }

        [JsonPropertyName("sha256")]
        public string? Sha256 { get; set; }
    }

    /// <summary>
    /// Athala-specific extensions
    /// MUST preserve raw_event for forensics
    /// </summary>
    public class AthalaExtensions
    {
        /// <summary>
        /// Original raw event - MUST be preserved for forensics
        /// </summary>
        [JsonPropertyName("raw_event")]
        public object RawEvent { get; set; } = new();

        /// <summary>
        /// Collector that generated this event
        /// </summary>
        [JsonPropertyName("collector")]
        public string Collector { get; set; } = "";

        /// <summary>
        /// Source type (e.g., Security, System, Application)
        /// </summary>
        [JsonPropertyName("source_type")]
        public string SourceType { get; set; } = "";

        /// <summary>
        /// Pipeline stage where this event was normalized
        /// </summary>
        [JsonPropertyName("pipeline_stage")]
        public string PipelineStage { get; set; } = "normalized";

        /// <summary>
        /// Original event ID (Windows Event ID, Syslog Facility, etc.)
        /// </summary>
        [JsonPropertyName("original_event_id")]
        public string? OriginalEventId { get; set; }

        /// <summary>
        /// Security relevance level
        /// </summary>
        [JsonPropertyName("security_relevance")]
        public string? SecurityRelevance { get; set; }

        /// <summary>
        /// Additional metadata
        /// </summary>
        [JsonPropertyName("metadata")]
        public Dictionary<string, object>? Metadata { get; set; }
    }
}
