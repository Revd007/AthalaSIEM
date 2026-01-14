using System;
using System.Collections.Generic;

namespace AthalaSIEM.UniversalAgent.Core.Normalizer
{
    /// <summary>
    /// Parsed event structure (output from Parser stage)
    /// Parser decodes and structures raw logs but does NOT normalize schema
    /// </summary>
    public class ParsedEvent
    {
        /// <summary>
        /// Original raw event (preserved for forensics)
        /// </summary>
        public object RawEvent { get; set; } = new();

        /// <summary>
        /// Event timestamp
        /// </summary>
        public DateTime Timestamp { get; set; }

        /// <summary>
        /// Event ID (Windows Event ID, Syslog Facility/Severity, etc.)
        /// </summary>
        public string? EventId { get; set; }

        /// <summary>
        /// Event level/severity
        /// </summary>
        public string? Level { get; set; }

        /// <summary>
        /// Event message
        /// </summary>
        public string Message { get; set; } = "";

        /// <summary>
        /// Source system/hostname
        /// </summary>
        public string? SourceHost { get; set; }

        /// <summary>
        /// Source application/service
        /// </summary>
        public string? SourceApplication { get; set; }

        /// <summary>
        /// Source type (Security, System, Application, etc.)
        /// </summary>
        public string? SourceType { get; set; }

        /// <summary>
        /// Collector that generated this event
        /// </summary>
        public string Collector { get; set; } = "";

        /// <summary>
        /// User information (if available)
        /// </summary>
        public ParsedUserInfo? User { get; set; }

        /// <summary>
        /// Process information (if available)
        /// </summary>
        public ParsedProcessInfo? Process { get; set; }

        /// <summary>
        /// Network information (if available)
        /// </summary>
        public ParsedNetworkInfo? Network { get; set; }

        /// <summary>
        /// File information (if available)
        /// </summary>
        public ParsedFileInfo? File { get; set; }

        /// <summary>
        /// Event category (authentication, authorization, etc.)
        /// </summary>
        public List<string>? Categories { get; set; }

        /// <summary>
        /// Event action (login, logout, file_access, etc.)
        /// </summary>
        public string? Action { get; set; }

        /// <summary>
        /// Event outcome (success, failure, unknown)
        /// </summary>
        public string? Outcome { get; set; }

        /// <summary>
        /// Additional structured properties
        /// </summary>
        public Dictionary<string, object> Properties { get; set; } = new();

        /// <summary>
        /// Security relevance level
        /// </summary>
        public string? SecurityRelevance { get; set; }
    }

    /// <summary>
    /// Parsed user information
    /// </summary>
    public class ParsedUserInfo
    {
        public string? Name { get; set; }
        public string? Id { get; set; }
        public string? Domain { get; set; }
        public string? FullName { get; set; }
        public string? Email { get; set; }
    }

    /// <summary>
    /// Parsed process information
    /// </summary>
    public class ParsedProcessInfo
    {
        public string? Name { get; set; }
        public int? Pid { get; set; }
        public int? Ppid { get; set; }
        public string? CommandLine { get; set; }
        public string? Executable { get; set; }
        public string? WorkingDirectory { get; set; }
        public string? ParentName { get; set; }
        public int? ParentPid { get; set; }
        public string? ParentCommandLine { get; set; }
    }

    /// <summary>
    /// Parsed network information
    /// </summary>
    public class ParsedNetworkInfo
    {
        public string? SourceIp { get; set; }
        public int? SourcePort { get; set; }
        public string? DestinationIp { get; set; }
        public int? DestinationPort { get; set; }
        public string? Protocol { get; set; }
        public string? Transport { get; set; }
        public long? Bytes { get; set; }
        public long? Packets { get; set; }
    }

    /// <summary>
    /// Parsed file information
    /// </summary>
    public class ParsedFileInfo
    {
        public string? Name { get; set; }
        public string? Path { get; set; }
        public string? Extension { get; set; }
        public long? Size { get; set; }
        public string? MimeType { get; set; }
        public string? HashMd5 { get; set; }
        public string? HashSha1 { get; set; }
        public string? HashSha256 { get; set; }
        public string? Owner { get; set; }
        public string? Group { get; set; }
    }
}
