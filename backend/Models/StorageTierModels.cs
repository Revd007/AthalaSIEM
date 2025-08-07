using System;
using System.Collections.Generic;

namespace Backend.Models
{
    /// <summary>
    /// Storage tier enumeration for log lifecycle management
    /// </summary>
    public enum StorageTier
    {
        Hot,    // 0-90 days - Database (PostgreSQL)
        Warm,   // 90 days - 7 years - Archive files (.json.gz)
        Cold    // 7+ years - External storage (S3, Azure Blob, etc.)
    }

    /// <summary>
    /// Storage tier configuration
    /// </summary>
    public class StorageTierConfiguration
    {
        public StorageTier Tier { get; set; }
        public int RetentionDays { get; set; }
        public string StorageLocation { get; set; } = string.Empty;
        public bool EnableCompression { get; set; } = true;
        public long MaxFileSize { get; set; } = 100 * 1024 * 1024; // 100MB default
        public string CompressionFormat { get; set; } = "gzip";
        public Dictionary<string, object> ExternalStorageConfig { get; set; } = new();
    }

    // ArchiveFile is already defined in ArchiveModels.cs - removed duplicate

    // StorageUsageInfo is already defined in ArchiveModels.cs - removed duplicate

    /// <summary>
    /// Log restoration request for moving logs from archive back to hot storage
    /// </summary>
    public class LogRestorationRequest
    {
        public string ArchiveFileName { get; set; } = string.Empty;
        public DateTime? StartDate { get; set; }
        public DateTime? EndDate { get; set; }
        public string? CollectorType { get; set; }
        public bool RestoreToDatabase { get; set; } = true;
        public bool KeepArchiveFile { get; set; } = true;
        public Dictionary<string, object> Filters { get; set; } = new();
    }

    /// <summary>
    /// Archive query request for searching archived logs
    /// </summary>
    public class ArchiveQueryRequest
    {
        public DateTime StartDate { get; set; }
        public DateTime EndDate { get; set; }
        public string? CollectorType { get; set; }
        public string? SearchQuery { get; set; }
        public List<string> EventIds { get; set; } = new();
        public string? LogLevel { get; set; }
        public int MaxResults { get; set; } = 10000;
        public bool IncludeMetadata { get; set; } = false;
        public string ExportFormat { get; set; } = "json"; // json, csv, xml
    }

    /// <summary>
    /// Archive query result
    /// </summary>
    public class ArchiveQueryResult
    {
        public bool Success { get; set; }
        public string Message { get; set; } = string.Empty;
        public List<LogEntryModels> Logs { get; set; } = new();
        public int TotalFound { get; set; }
        public List<string> SourceFiles { get; set; } = new();
        public TimeSpan QueryDuration { get; set; }
        public Dictionary<string, object> Statistics { get; set; } = new();
    }
}
