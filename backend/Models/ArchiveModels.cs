using System;
using System.Collections.Generic;

namespace Backend.Models
{
    public class ArchiveResult
    {
        public bool Success { get; set; }
        public int ArchivedCount { get; set; }
        public string? ArchiveFileName { get; set; }
        public long CompressedSize { get; set; }
        public string? Error { get; set; }
    }

    public class ArchiveFileInfo
    {
        public string FileName { get; set; } = string.Empty;
        public string FilePath { get; set; } = string.Empty;
        public string CollectorType { get; set; } = string.Empty;
        public DateTime Date { get; set; }
        public long Size { get; set; }
        public bool IsCompressed { get; set; }
        public DateTime CreatedAt { get; set; }
        public int LogCount { get; set; }
    }

    public class LogArchiveQuery
    {
        public string? AgentId { get; set; }
        public string? Level { get; set; }
        public string? SearchTerm { get; set; }
        public DateTime? FromDate { get; set; }
        public DateTime? ToDate { get; set; }
        public int? Limit { get; set; }
    }

    public class ArchiveStatisticsSummary
    {
        public int TotalArchives { get; set; }
        public long TotalArchivedLogs { get; set; }
        public long TotalCompressedSize { get; set; }
        public double CompressionRatio { get; set; }
        public DateTime LastArchiveDate { get; set; }
        public Dictionary<string, ArchiveStatistics> CollectorStatistics { get; set; } = new();
    }

    public class ArchiveStatistics
    {
        public int ArchiveCount { get; set; }
        public long TotalLogs { get; set; }
        public long TotalCompressedSize { get; set; }
        public long TotalUncompressedSize { get; set; }
        public DateTime LastArchiveDate { get; set; }
    }

    public class CollectorArchiveSettings
    {
        public bool EnableCompression { get; set; } = true;
        public bool IncludeDetails { get; set; } = false;
        public bool IncludeCategory { get; set; } = true;
        public bool CompressMessages { get; set; } = false;
        public int RetentionDays { get; set; } = 90;
    }

    public class LogArchiveData
    {
        public ArchiveMetadata? Metadata { get; set; }
        public List<ArchivedLogEntry> Logs { get; set; } = new();
    }

    public class ArchiveMetadata
    {
        public DateTime CreatedAt { get; set; }
        public int LogCount { get; set; }
        public string CollectorType { get; set; } = string.Empty;
        public DateRange? DateRange { get; set; }
        public string Version { get; set; } = string.Empty;
    }

    public class DateRange
    {
        public DateTime From { get; set; }
        public DateTime To { get; set; }
    }

    public class ArchivedLogEntry
    {
        public string Id { get; set; } = string.Empty;
        public DateTime Timestamp { get; set; }
        public string Level { get; set; } = string.Empty;
        public string Source { get; set; } = string.Empty;
        public string Message { get; set; } = string.Empty;
        public string? AgentId { get; set; }
        public string? Details { get; set; }
        public string? Category { get; set; }
        public bool IsMessageCompressed { get; set; }
    }

    // Additional models from ILogArchivingService.cs
    public class ArchiveFile
    {
        public string FileName { get; set; } = string.Empty;
        public long Size { get; set; }
        public DateTime CreatedAt { get; set; }
        public string CollectorType { get; set; } = string.Empty;
        public DateTime StartDate { get; set; }
        public DateTime EndDate { get; set; }
        public int RecordCount { get; set; }
        public string CompressionType { get; set; } = string.Empty;
        public string CheckSum { get; set; } = string.Empty;
        
        // Additional properties used in controller
        public DateTime Date { get; set; }
        public bool IsCompressed { get; set; }
        public int LogCount { get; set; }
    }

    public class StorageUsageInfo
    {
        public long TotalSize { get; set; }
        public long AvailableSpace { get; set; }
        public int TotalFiles { get; set; }
        public Dictionary<string, long> SizeByCollector { get; set; } = new();
        public Dictionary<string, int> FilesByCollector { get; set; } = new();
        public DateTime OldestArchive { get; set; }
        public DateTime NewestArchive { get; set; }
        public double CompressionRatio { get; set; }
    }

    public class ArchiveSearchResult
    {
        public List<ArchiveLogEntry> Results { get; set; } = new();
        public int TotalCount { get; set; }
        public string Query { get; set; } = string.Empty;
        public DateTime SearchDate { get; set; }
        public List<string> SearchedFiles { get; set; } = new();
        public TimeSpan SearchDuration { get; set; }
    }

    public class ArchiveLogEntry
    {
        public string Id { get; set; } = string.Empty;
        public DateTime Timestamp { get; set; }
        public string Level { get; set; } = string.Empty;
        public string Source { get; set; } = string.Empty;
        public string Message { get; set; } = string.Empty;
        public string CollectorType { get; set; } = string.Empty;
        public string AgentId { get; set; } = string.Empty;
        public string ArchiveFile { get; set; } = string.Empty;
    }
} 