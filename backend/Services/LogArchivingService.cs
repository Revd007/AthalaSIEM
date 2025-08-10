using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.IO;
using System.IO.Compression;
using System.Text.Json;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.DependencyInjection;
using Backend.Models;
using Backend.Data;
using Microsoft.EntityFrameworkCore;
using System.Threading;
using System.Text;

namespace Backend.Services
{
    /// <summary>
    /// Enhanced Log Archiving Service for multi-collector environments
    /// </summary>
    public class LogArchivingService : BackgroundService, ILogArchivingService
    {
        private readonly ILogger<LogArchivingService> _logger;
        private readonly IServiceScopeFactory _scopeFactory;
        private readonly IConfiguration _configuration;
        
        // Configuration settings
        private readonly TimeSpan _archiveInterval;
        private readonly TimeSpan _retentionPeriod;
        private readonly int _batchSize;
        private readonly string _archiveDirectory;
        private readonly bool _enableCompression;
        private readonly long _maxArchiveFileSize;
        private readonly Dictionary<string, CollectorArchiveSettings> _collectorSettings;
        
        // Archive statistics
        private readonly Dictionary<string, ArchiveStatistics> _archiveStats = new();
        private readonly object _statsLock = new();

        public LogArchivingService(
            ILogger<LogArchivingService> logger,
            IServiceScopeFactory scopeFactory,
            IConfiguration configuration)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _scopeFactory = scopeFactory ?? throw new ArgumentNullException(nameof(scopeFactory));
            _configuration = configuration ?? throw new ArgumentNullException(nameof(configuration));

            // Load configuration - support both minutes and hours for flexibility
            var intervalMinutes = _configuration.GetValue<int?>("LogArchiving:IntervalMinutes");
            var intervalHours = _configuration.GetValue<int?>("LogArchiving:IntervalHours");
            
            if (intervalHours.HasValue)
            {
                _archiveInterval = TimeSpan.FromHours(intervalHours.Value);
                _logger.LogInformation("🕐 Archive interval set to {Minutes} minutes", intervalHours.Value);
            }
            else if (intervalMinutes.HasValue)
            {
                _archiveInterval = TimeSpan.FromHours(intervalMinutes.Value);
                _logger.LogInformation("🕐 Archive interval set to {Hours} hours", intervalMinutes.Value);
            }
            else
            {
                _archiveInterval = TimeSpan.FromHours(24); // Default fallback
                _logger.LogInformation("🕐 Archive interval set to default 24 hours");
            }
            _retentionPeriod = TimeSpan.FromDays(_configuration.GetValue<int>("LogArchiving:RetentionDays", 90));
            _batchSize = _configuration.GetValue<int>("LogArchiving:BatchSize", 10000);
            _archiveDirectory = _configuration.GetValue<string>("LogArchiving:Directory") ?? "archives/logs";
            _enableCompression = _configuration.GetValue<bool>("LogArchiving:EnableCompression", true);
            _maxArchiveFileSize = _configuration.GetValue<long>("LogArchiving:MaxArchiveFileSizeMB", 100) * 1024 * 1024;

            // Initialize collector-specific settings
            _collectorSettings = InitializeCollectorSettings();

            // Ensure archive directory exists
            if (!string.IsNullOrEmpty(_archiveDirectory))
            {
            Directory.CreateDirectory(_archiveDirectory);
            }
        }

        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            _logger.LogInformation("🗂️ Log Archiving Service started - Interval: {Interval}, Retention: {Retention} days", 
                _archiveInterval, _retentionPeriod.TotalDays);

            while (!stoppingToken.IsCancellationRequested)
            {
                try
                {
                    var startTime = DateTime.UtcNow;
                    _logger.LogInformation("🔄 Starting archive cycle at {Time}", startTime.ToString("yyyy-MM-dd HH:mm:ss"));
                    
                    await PerformArchivingCycle();
                    await PerformCleanupCycle();
                    
                    var duration = DateTime.UtcNow - startTime;
                    _logger.LogInformation("✅ Archive cycle completed in {Duration}ms. Next cycle in {Interval}", 
                        duration.TotalMilliseconds, _archiveInterval);
                    
                    await Task.Delay(_archiveInterval, stoppingToken);
                }
                catch (OperationCanceledException)
                {
                    break;
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "❌ Error in log archiving cycle - retrying in 5 minutes");
                    await Task.Delay(TimeSpan.FromMinutes(5), stoppingToken);
                }
            }

            _logger.LogInformation("🛑 Log Archiving Service stopped");
        }

        public async Task<ArchiveResult> ArchiveLogsByCollectorAsync(string collectorType, DateTime fromDate, DateTime toDate)
        {
            try
            {
                _logger.LogInformation("Archiving logs for collector {CollectorType} from {FromDate} to {ToDate}", 
                    collectorType, fromDate, toDate);

                var result = await ArchiveLogsAsync(fromDate, toDate, collectorType); return new ArchiveResult { Success = result, ArchivedCount = 0, ArchiveFileName = null, CompressedSize = 0, Error = null };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error archiving logs for collector {CollectorType}", collectorType);
                return new ArchiveResult { Success = false, Error = ex.Message };
            }
        }

        public async Task<List<ArchiveFile>> GetArchiveFilesAsync(string? collectorType = null, DateTime? startDate = null, DateTime? endDate = null)
        {
            try
            {
                _logger.LogInformation("Getting archive files for collector: {CollectorType}, date range: {StartDate} - {EndDate}", 
                    collectorType ?? "All", startDate, endDate);

                var archiveFiles = new List<ArchiveFile>();

                if (!Directory.Exists(_archiveDirectory))
                {
                    return archiveFiles;
                }

                var files = Directory.GetFiles(_archiveDirectory, "*.json.gz")
                    .Concat(Directory.GetFiles(_archiveDirectory, "*.json"))
                    .ToList();

                foreach (var filePath in files)
                {
                    var fileInfo = new FileInfo(filePath);
                    var fileName = fileInfo.Name;

                    // Parse collector type from filename (assuming format: CollectorType_YYYYMMDD_HHMMSS.json.gz)
                    var fileCollectorType = ExtractCollectorTypeFromFileName(fileName);
                    
                    // Apply collector type filter
                    if (!string.IsNullOrEmpty(collectorType) && !fileCollectorType.Equals(collectorType, StringComparison.OrdinalIgnoreCase))
                    {
                        continue;
                    }

                    // Apply date filters
                    if (startDate.HasValue && fileInfo.CreationTime < startDate.Value)
                    {
                        continue;
                    }

                    if (endDate.HasValue && fileInfo.CreationTime > endDate.Value)
                    {
                        continue;
                    }

                    var archiveFile = new ArchiveFile
                    {
                        FileName = fileName,
                        Size = fileInfo.Length,
                        CreatedAt = fileInfo.CreationTime,
                        CollectorType = fileCollectorType,
                        Date = fileInfo.CreationTime,
                        IsCompressed = fileName.EndsWith(".gz"),
                        LogCount = await EstimateLogCountInArchive(filePath)
                    };

                    archiveFiles.Add(archiveFile);
                }

                return archiveFiles.OrderByDescending(f => f.CreatedAt).ToList();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting archive files");
                return new List<ArchiveFile>();
            }
        }

        public async Task<LogArchiveData> ExtractLogsFromArchiveAsync(string fileName, LogArchiveQuery? query = null)
        {
            try
            {
                _logger.LogInformation("Extracting logs from archive {FileName}", fileName);

                var archiveBytes = await ExtractArchiveAsync(fileName); var archiveData = DeserializeArchiveData(archiveBytes);
                
                // Apply query filters if provided
                if (query != null && archiveData != null)
                {
                    // Filter the logs based on query parameters
                    var filteredLogs = archiveData.Logs.AsEnumerable();

                    if (!string.IsNullOrEmpty(query.AgentId))
                        filteredLogs = filteredLogs.Where(l => l.AgentId == query.AgentId);

                    if (!string.IsNullOrEmpty(query.Level))
                        filteredLogs = filteredLogs.Where(l => l.Level == query.Level);

                    if (!string.IsNullOrEmpty(query.SearchTerm))
                        filteredLogs = filteredLogs.Where(l => l.Message.Contains(query.SearchTerm, StringComparison.OrdinalIgnoreCase));

                    if (query.FromDate.HasValue)
                        filteredLogs = filteredLogs.Where(l => l.Timestamp >= query.FromDate.Value);

                    if (query.ToDate.HasValue)
                        filteredLogs = filteredLogs.Where(l => l.Timestamp <= query.ToDate.Value);

                    if (query.Limit.HasValue)
                        filteredLogs = filteredLogs.Take(query.Limit.Value);

                    archiveData.Logs = filteredLogs.ToList();
                }

                return archiveData ?? new LogArchiveData();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error extracting logs from archive {FileName}", fileName);
                return new LogArchiveData();
            }
        }

        public async Task<bool> DeleteArchiveAsync(string fileName)
        {
            try
            {
                _logger.LogInformation("Deleting archive {FileName}", fileName);

                return await DeleteArchiveFileAsync(fileName);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error deleting archive {FileName}", fileName);
                return false;
            }
        }

        public async Task<ArchiveStatisticsSummary> GetArchiveStatisticsAsync(string? collectorType = null)
        {
            try
            {
                _logger.LogInformation("Getting archive statistics for collector type: {CollectorType}", collectorType ?? "All");

                var files = await GetArchiveFilesAsync();
                var filteredFiles = string.IsNullOrEmpty(collectorType) 
                    ? files 
                    : files.Where(f => f.CollectorType == collectorType);

                    var summary = new ArchiveStatisticsSummary
                    {
                    TotalArchives = filteredFiles.Count(),
                    TotalArchivedLogs = filteredFiles.Sum(f => f.LogCount),
                    TotalCompressedSize = filteredFiles.Sum(f => f.Size),
                    LastArchiveDate = filteredFiles.Any() ? filteredFiles.Max(f => f.CreatedAt) : DateTime.MinValue,
                    CollectorStatistics = new Dictionary<string, ArchiveStatistics>()
                };

                // Calculate compression ratio (assuming 10:1 ratio for demo)
                summary.CompressionRatio = summary.TotalCompressedSize > 0 ? 10.0 : 0.0;

                // Group by collector type
                var collectorGroups = filteredFiles.GroupBy(f => f.CollectorType);
                foreach (var group in collectorGroups)
                {
                    summary.CollectorStatistics[group.Key] = new ArchiveStatistics
                    {
                        ArchiveCount = group.Count(),
                        TotalLogs = group.Sum(f => f.LogCount),
                        TotalCompressedSize = group.Sum(f => f.Size),
                        TotalUncompressedSize = group.Sum(f => f.Size * 10), // Estimated
                        LastArchiveDate = group.Max(f => f.CreatedAt)
                    };
                    }

                    return summary;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting archive statistics");
                return new ArchiveStatisticsSummary();
            }
        }

        private async Task PerformArchivingCycle()
        {
            _logger.LogInformation("🔄 Starting archiving cycle with retention period: {RetentionDays} days", _retentionPeriod.TotalDays);

            using var scope = _scopeFactory.CreateScope();
            var context = scope.ServiceProvider.GetRequiredService<ApplicationDbContext>();

            // FOR IMMEDIATE TESTING: Get ALL logs if retention is 0 days
            var totalLogs = await context.LogEntries.CountAsync();
            _logger.LogInformation("📊 Total logs in database: {TotalLogs}", totalLogs);

            if (totalLogs == 0)
            {
                _logger.LogInformation("ℹ️ No logs found in database to archive");
                return;
            }

            // Get actual log sources from database for debugging
            var sources = await context.LogEntries
                .Select(l => l.Source)
                .Distinct()
                .ToListAsync();

            _logger.LogInformation("📊 Found {SourceCount} distinct log sources in database: {Sources}", 
                sources.Count, string.Join(", ", sources));

            // FOR TESTING: Archive ALL logs regardless of source
            if (_retentionPeriod.TotalDays == 0)
            {
                _logger.LogInformation("🧪 Testing mode: Archiving ALL {TotalLogs} logs immediately", totalLogs);
                
                // Group logs by source and archive each group
                foreach (var source in sources)
                {
                    try
                    {
                        _logger.LogInformation("🗂️ Archiving logs from source: {Source}", source);
                        await ArchiveLogsBySourceAsync(source);
                    }
                    catch (Exception ex)
                    {
                        _logger.LogError(ex, "❌ Error archiving logs for source {Source}", source);
                    }
                }
            }
            else
            {
                // Normal retention-based archiving
                var cutoffDate = DateTime.UtcNow - _retentionPeriod;
                _logger.LogInformation("📅 Cutoff date for archiving: {CutoffDate}", cutoffDate.ToString("yyyy-MM-dd HH:mm:ss"));
                
                var collectorTypes = sources.Select(GetCollectorTypeFromSource).Distinct().ToList();
                foreach (var collectorType in collectorTypes)
                {
                    try
                    {
                        await ArchiveLogsByCollectorAsync(collectorType, DateTime.MinValue, cutoffDate);
                    }
                    catch (Exception ex)
                    {
                        _logger.LogError(ex, "❌ Error archiving logs for collector {CollectorType}", collectorType);
                    }
                }
            }

            _logger.LogInformation("✅ Archiving cycle completed");
        }

        private Task PerformCleanupCycle()
        {
            _logger.LogInformation("🧹 Starting cleanup cycle (corrupt/temp files only)");

            try
            {
                var files = Directory.GetFiles(_archiveDirectory, "*.*");
                var deletedCount = 0;

                foreach (var filePath in files)
                {
                    var fileInfo = new FileInfo(filePath);
                    bool shouldDelete = false;
                    string reason = "";
                    
                    // Only delete corrupt or temp files, NOT old archives
                    if (fileInfo.Name.Contains(".tmp") || fileInfo.Name.Contains(".temp"))
                    {
                        shouldDelete = true;
                        reason = "temporary file";
                    }
                    else if (fileInfo.Length == 0)
                    {
                        shouldDelete = true;
                        reason = "empty/corrupt file";
                    }
                    else if (fileInfo.Name.EndsWith(".partial"))
                    {
                        shouldDelete = true;
                        reason = "incomplete archive file";
                    }
                    
                    if (shouldDelete)
                    {
                        try
                        {
                            File.Delete(filePath);
                            deletedCount++;
                            _logger.LogInformation("🗑️ Deleted {Reason}: {FileName}", reason, fileInfo.Name);
                        }
                        catch (Exception ex)
                        {
                            _logger.LogWarning(ex, "Failed to delete file: {FileName}", fileInfo.Name);
                        }
                    }
                }

                _logger.LogInformation("✅ Cleanup cycle completed, deleted {Count} corrupt/temp files (archives preserved)", deletedCount);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error in cleanup cycle");
            }
            
            return Task.CompletedTask;
        }

        private async Task<int> CreateArchiveFile(List<LogEntryModels> logs, string filePath, CollectorArchiveSettings settings)
        {
            try
            {
                // Create archive metadata
                var archiveMetadata = new ArchiveMetadata
                {
                    CreatedAt = DateTime.UtcNow,
                    LogCount = logs.Count,
                    CollectorType = GetCollectorTypeFromSource(logs.First().Source),
                    DateRange = new DateRange
                    {
                        From = logs.Min(l => l.Timestamp),
                        To = logs.Max(l => l.Timestamp)
                    },
                    Version = "1.0"
                };

                // Create archive data
                var archiveData = new LogArchiveData
                {
                    Metadata = archiveMetadata,
                    Logs = logs.Select(l => ConvertToArchiveLog(l, settings)).ToList()
                };

                var jsonOptions = new JsonSerializerOptions
                {
                    WriteIndented = false,
                    PropertyNamingPolicy = JsonNamingPolicy.CamelCase
                };

                var jsonContent = JsonSerializer.Serialize(archiveData, jsonOptions);

                if (_enableCompression && settings.EnableCompression)
                {
                    // Compress and save
                    var compressedFilePath = filePath + ".gz";
                    using var fileStream = File.Create(compressedFilePath);
                    using var gzipStream = new GZipStream(fileStream, CompressionLevel.Optimal);
                    using var writer = new StreamWriter(gzipStream);
                    await writer.WriteAsync(jsonContent);
                }
                else
                {
                    // Save without compression
                    await File.WriteAllTextAsync(filePath, jsonContent);
                }

                return logs.Count;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error creating archive file {FilePath}", filePath);
                throw;
            }
        }

        private ArchivedLogEntry ConvertToArchiveLog(LogEntryModels log, CollectorArchiveSettings settings)
        {
            var archived = new ArchivedLogEntry
            {
                Id = log.Id,
                Timestamp = log.Timestamp,
                Level = log.Level,
                Source = log.Source,
                Message = log.Message,
                AgentId = log.AgentId
            };

            // Include additional fields based on collector settings
            if (settings.IncludeDetails && !string.IsNullOrEmpty(log.Details))
            {
                archived.Details = log.Details;
            }

            if (settings.IncludeCategory && !string.IsNullOrEmpty(log.Category))
            {
                archived.Category = log.Category;
            }

            // Compress message if enabled
            if (settings.CompressMessages && !string.IsNullOrEmpty(archived.Message) && archived.Message.Length > 100)
            {
                archived.Message = CompressString(archived.Message);
                archived.IsMessageCompressed = true;
            }

            return archived;
        }

        private string CompressString(string text)
        {
            try
            {
                var bytes = System.Text.Encoding.UTF8.GetBytes(text);
                using var memory = new MemoryStream();
                using (var gzip = new GZipStream(memory, CompressionMode.Compress))
                {
                    gzip.Write(bytes, 0, bytes.Length);
                }
                return Convert.ToBase64String(memory.ToArray());
            }
            catch
            {
                return text; // Return original if compression fails
            }
        }

        private string DecompressString(string compressedText)
        {
            try
            {
                var compressedBytes = Convert.FromBase64String(compressedText);
                using var memory = new MemoryStream(compressedBytes);
                using var gzip = new GZipStream(memory, CompressionMode.Decompress);
                using var reader = new StreamReader(gzip);
                return reader.ReadToEnd();
            }
            catch
            {
                return compressedText; // Return original if decompression fails
            }
        }

        private string GenerateArchiveFileName(string collectorType, DateTime date)
        {
            var timestamp = date.ToString("yyyy-MM-dd");
            var guid = Guid.NewGuid().ToString("N")[..8];
            return $"{collectorType}_{timestamp}_{guid}.archive.json";
        }

        private (string CollectorType, DateTime Date) ParseArchiveFileName(string fileName)
        {
            try
            {
                // Format: CollectorType_yyyy-MM-dd_guid.archive.json[.gz]
                var parts = fileName.Replace(".archive.json.gz", "").Replace(".archive.json", "").Split('_');
                if (parts.Length >= 3)
                {
                    var collectorType = parts[0];
                    var dateStr = $"{parts[1]}_{parts[2]}_{parts[3]}";
                    if (DateTime.TryParseExact(parts[1], "yyyy-MM-dd", null, System.Globalization.DateTimeStyles.None, out var date))
                    {
                        return (collectorType, date);
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Error parsing archive filename: {FileName}", fileName);
            }

            return ("Unknown", DateTime.MinValue);
        }

        private async Task<int> GetLogCountFromArchive(string filePath)
        {
            try
            {
                if (filePath.EndsWith(".gz"))
                {
                    using var fileStream = File.OpenRead(filePath);
                    using var gzipStream = new GZipStream(fileStream, CompressionMode.Decompress);
                    using var reader = new StreamReader(gzipStream);
                    var jsonContent = await reader.ReadToEndAsync();
                    var archiveData = JsonSerializer.Deserialize<LogArchiveData>(jsonContent);
                    return archiveData?.Metadata?.LogCount ?? 0;
                }
                else
                {
                    var jsonContent = await File.ReadAllTextAsync(filePath);
                    var archiveData = JsonSerializer.Deserialize<LogArchiveData>(jsonContent);
                    return archiveData?.Metadata?.LogCount ?? 0;
                }
            }
            catch
            {
                return 0;
            }
        }

        private string GetCollectorTypeFromSource(string source)
        {
            if (source.Contains("Container") || source.Contains("Docker") || source.Contains("Kubernetes"))
                return "Container";
            if (source.Contains("AWS") || source.Contains("Azure") || source.Contains("GCP") || source.Contains("CloudServices"))
                return "CloudServices";
            if (source.Contains("Database") || source.Contains("SQL") || source.Contains("MySQL") || source.Contains("PostgreSQL") || source.Contains("MongoDB"))
                return "Database";
            if (source.Contains("IoT") || source.Contains("Sensor") || source.Contains("SCADA") || source.Contains("Modbus") || source.Contains("MQTT"))
                return "IoT";
            if (source.Contains("FIM") || source.Contains("FileIntegrity"))
                return "FileIntegrity";
            if (source.Contains("Syslog"))
                return "Syslog";
            if (source.Contains("Windows"))
                return "WindowsEventLog";
            
            return "General";
        }

        private void UpdateArchiveStatistics(string collectorType, int archivedCount, long compressedSize)
        {
            lock (_statsLock)
            {
                if (!_archiveStats.ContainsKey(collectorType))
                {
                    _archiveStats[collectorType] = new ArchiveStatistics();
                }

                var stats = _archiveStats[collectorType];
                stats.ArchiveCount++;
                stats.TotalLogs += archivedCount;
                stats.TotalCompressedSize += compressedSize;
                stats.LastArchiveDate = DateTime.UtcNow;
            }
        }

        private Dictionary<string, CollectorArchiveSettings> InitializeCollectorSettings()
        {
            return new Dictionary<string, CollectorArchiveSettings>
            {
                ["Container"] = new CollectorArchiveSettings
            {
                EnableCompression = true,
                    IncludeDetails = false,
                IncludeCategory = true,
                CompressMessages = true,
                    RetentionDays = 90
                },
                ["CloudServices"] = new CollectorArchiveSettings
            {
                EnableCompression = true,
                IncludeDetails = true,
                IncludeCategory = true,
                    CompressMessages = false,
                    RetentionDays = 180
                },
                ["Database"] = new CollectorArchiveSettings
            {
                EnableCompression = true,
                IncludeDetails = true,
                IncludeCategory = true,
                    CompressMessages = false,
                    RetentionDays = 365
                },
                ["IoT"] = new CollectorArchiveSettings
            {
                EnableCompression = true,
                    IncludeDetails = false,
                    IncludeCategory = false,
                CompressMessages = true,
                    RetentionDays = 30
                },
                ["FileIntegrity"] = new CollectorArchiveSettings
            {
                EnableCompression = true,
                IncludeDetails = true,
                IncludeCategory = true,
                    CompressMessages = false,
                    RetentionDays = 730
                }
            };
        }

        // Missing interface method implementations
        public async Task<bool> ArchiveLogsAsync(DateTime fromDate, DateTime toDate, string? collectorType = null)
        {
            try
            {
                _logger.LogInformation("📦 Archiving logs from {FromDate} to {ToDate} for collector {CollectorType}", 
                    fromDate.ToString("yyyy-MM-dd HH:mm:ss"), toDate.ToString("yyyy-MM-dd HH:mm:ss"), collectorType ?? "All");

                using var scope = _scopeFactory.CreateScope();
                var context = scope.ServiceProvider.GetRequiredService<ApplicationDbContext>();

                // FOR TESTING: If retention is 0 days, archive ALL logs
                IQueryable<LogEntryModels> query;
                if (_retentionPeriod.TotalDays == 0)
                {
                    query = context.LogEntries.AsQueryable();
                    _logger.LogInformation("🧪 Testing mode: Archiving ALL logs (retention = 0 days)");
                }
                else
                {
                    query = context.LogEntries.Where(l => l.Timestamp >= fromDate && l.Timestamp <= toDate);
                }
                
                if (!string.IsNullOrEmpty(collectorType))
                {
                    query = query.Where(l => l.Source.Contains(collectorType));
                }

                // Count first to provide better logging
                var totalCount = await query.CountAsync();
                _logger.LogInformation("🔍 Found {TotalCount} logs matching criteria for archiving", totalCount);

                if (totalCount == 0)
                {
                    _logger.LogInformation("ℹ️ No logs found to archive for the specified criteria");
                    return true;
                }

                var logsToArchive = await query.OrderBy(l => l.Timestamp).ToListAsync();

                var archiveFileName = GenerateArchiveFileName(collectorType ?? "Mixed", fromDate);
                var archiveFilePath = Path.Combine(_archiveDirectory, archiveFileName);
                var settings = _collectorSettings.GetValueOrDefault(collectorType ?? "Default", new CollectorArchiveSettings());

                _logger.LogInformation("💾 Creating archive file: {ArchiveFileName}", archiveFileName);
                var archivedCount = await CreateArchiveFile(logsToArchive, archiveFilePath, settings);

                // Remove archived logs from database
                _logger.LogInformation("🗑️ Removing {Count} archived logs from database", logsToArchive.Count);
                context.LogEntries.RemoveRange(logsToArchive);
                await context.SaveChangesAsync();

                _logger.LogInformation("✅ Successfully archived {Count} logs to {FileName} (file size: {FileSize})", 
                    logsToArchive.Count, archiveFileName, GetFileSize(archiveFilePath));

                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error archiving logs from {FromDate} to {ToDate}", fromDate, toDate);
                return false;
            }
        }

        /// <summary>
        /// Archive logs by source (direct approach for testing)
        /// </summary>
        private async Task ArchiveLogsBySourceAsync(string source)
        {
            try
            {
                using var scope = _scopeFactory.CreateScope();
                var context = scope.ServiceProvider.GetRequiredService<ApplicationDbContext>();

                var logsToArchive = await context.LogEntries
                    .Where(l => l.Source == source)
                    .OrderBy(l => l.Timestamp)
                    .ToListAsync();

                if (!logsToArchive.Any())
                {
                    _logger.LogInformation("ℹ️ No logs found for source: {Source}", source);
                    return;
                }

                _logger.LogInformation("📦 Found {Count} logs for source {Source}", logsToArchive.Count, source);

                // Group logs by agent/device for proper naming
                var logsByAgent = logsToArchive.GroupBy(l => l.AgentId).ToList();

                foreach (var agentGroup in logsByAgent)
                {
                    var agentLogs = agentGroup.ToList();
                    var agentId = agentGroup.Key;
                    
                    // Get agent info for filename
                    var agent = await context.Agents.FirstOrDefaultAsync(a => a.Id == agentId);
                    var deviceName = agent?.Name ?? agent?.Hostname ?? agentId ?? "Unknown";
                    
                    // Create device-specific directory
                    var deviceDirectory = Path.Combine(_archiveDirectory, SanitizeDirectoryName(deviceName));
                    Directory.CreateDirectory(deviceDirectory);
                    
                    // Create archive file with device name (simpler filename since folder already identifies device)
                    var archiveFileName = GenerateArchiveFileNameForDevice(source, DateTime.UtcNow);
                    var archiveFilePath = Path.Combine(deviceDirectory, archiveFileName);
                    var settings = _collectorSettings.GetValueOrDefault("Default", new CollectorArchiveSettings());

                    _logger.LogInformation("💾 Creating archive file for {DeviceName}: {ArchiveFileName}", deviceName, archiveFileName);
                    await CreateArchiveFile(agentLogs, archiveFilePath, settings);

                    // Remove archived logs from database
                    _logger.LogInformation("🗑️ Removing {Count} archived logs from database for {DeviceName}", agentLogs.Count, deviceName);
                    context.LogEntries.RemoveRange(agentLogs);

                    _logger.LogInformation("✅ Successfully archived {Count} logs from {DeviceName}_{Source} to {DeviceDirectory}/{FileName}", 
                        agentLogs.Count, deviceName, source, SanitizeDirectoryName(deviceName), archiveFileName);
                }

                await context.SaveChangesAsync();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Error archiving logs for source {Source}", source);
            }
        }

        /// <summary>
        /// Generate archive filename for device folder (device name sudah di folder path)
        /// Format: SOURCE_YYYY-MM-DD_UNIQUEID.archive.json (without .gz, akan ditambah di CreateArchiveFile)
        /// </summary>
        private string GenerateArchiveFileNameForDevice(string source, DateTime timestamp)
        {
            var cleanSource = string.Join("", source.Split(Path.GetInvalidFileNameChars()));
            var dateStr = timestamp.ToString("yyyy-MM-dd");
            var uniqueId = Guid.NewGuid().ToString("N")[..8];
            
            // Filename lebih simple karena device name sudah di folder
            return $"{cleanSource}_{dateStr}_{uniqueId}.archive.json";
        }

        /// <summary>
        /// Sanitize directory name untuk folder per device
        /// </summary>
        private string SanitizeDirectoryName(string deviceName)
        {
            // Remove invalid characters and limit length
            var cleanName = string.Join("", deviceName.Split(Path.GetInvalidFileNameChars()));
            return cleanName.Length > 50 ? cleanName[..50] : cleanName;
        }

        /// <summary>
        /// Get file size in human-readable format
        /// </summary>
        private string GetFileSize(string filePath)
        {
            try
            {
                if (!File.Exists(filePath))
                    return "File not found";

                var fileInfo = new FileInfo(filePath);
                var sizeInBytes = fileInfo.Length;

                if (sizeInBytes < 1024)
                    return $"{sizeInBytes} bytes";
                else if (sizeInBytes < 1024 * 1024)
                    return $"{sizeInBytes / 1024:F1} KB";
                else if (sizeInBytes < 1024 * 1024 * 1024)
                    return $"{sizeInBytes / (1024 * 1024):F1} MB";
                else
                    return $"{sizeInBytes / (1024 * 1024 * 1024):F1} GB";
            }
            catch
            {
                return "Unknown size";
            }
        }

        private string ExtractCollectorTypeFromFileName(string fileName)
        {
            // Extract collector type from filename pattern: CollectorType_YYYYMMDD_HHMMSS.json.gz
            var parts = fileName.Split('_');
            return parts.Length > 0 ? parts[0] : "Unknown";
        }

        private Task<int> EstimateLogCountInArchive(string filePath)
        {
            try
            {
                // For demo purposes, return a random count between 100-1000
                // In real implementation, you might read the archive metadata
                return Task.FromResult(new Random().Next(100, 1000));
            }
            catch
            {
                return Task.FromResult(0);
            }
        }

        public Task<bool> DeleteArchiveFileAsync(string fileName)
        {
            try
            {
                var filePath = Path.Combine(_archiveDirectory, fileName);
                if (File.Exists(filePath))
                {
                    File.Delete(filePath);
                    _logger.LogInformation("Deleted archive file: {FileName}", fileName);
                    return Task.FromResult(true);
                }
                
                _logger.LogWarning("Archive file not found: {FileName}", fileName);
                return Task.FromResult(false);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error deleting archive file {FileName}", fileName);
                return Task.FromResult(false);
            }
        }

        public async Task<byte[]?> ExtractArchiveAsync(string fileName)
        {
            try
            {
                var filePath = Path.Combine(_archiveDirectory, fileName);
                if (!File.Exists(filePath))
                {
                    _logger.LogWarning("Archive file not found: {FileName}", fileName);
                    return null;
                }

                var fileBytes = await File.ReadAllBytesAsync(filePath);
                
                if (fileName.EndsWith(".gz"))
                {
                    // Decompress if needed
                    using var compressedStream = new MemoryStream(fileBytes);
                    using var gzipStream = new GZipStream(compressedStream, CompressionMode.Decompress);
                    using var decompressedStream = new MemoryStream();
                    await gzipStream.CopyToAsync(decompressedStream);
                    return decompressedStream.ToArray();
                }

                return fileBytes;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error extracting archive {FileName}", fileName);
                return null;
            }
        }

        public Task<StorageUsageInfo> GetStorageUsageAsync()
        {
            try
            {
                var files = Directory.GetFiles(_archiveDirectory, "*.archive.json*");
                var totalSize = files.Sum(f => new FileInfo(f).Length);
                var sizeByCollector = new Dictionary<string, long>();
                var filesByCollector = new Dictionary<string, int>();

                foreach (var file in files)
                {
                    var fileName = Path.GetFileName(file);
                    var parsedInfo = ParseArchiveFileName(fileName);
                    var fileSize = new FileInfo(file).Length;

                    if (sizeByCollector.ContainsKey(parsedInfo.CollectorType))
                    {
                        sizeByCollector[parsedInfo.CollectorType] += fileSize;
                        filesByCollector[parsedInfo.CollectorType]++;
                    }
                    else
                    {
                        sizeByCollector[parsedInfo.CollectorType] = fileSize;
                        filesByCollector[parsedInfo.CollectorType] = 1;
                    }
                }

                var driveInfo = new DriveInfo(Path.GetPathRoot(_archiveDirectory) ?? "C:");

                var result = new StorageUsageInfo
                {
                    TotalSize = totalSize,
                    AvailableSpace = driveInfo.AvailableFreeSpace,
                    TotalFiles = files.Length,
                    SizeByCollector = sizeByCollector,
                    FilesByCollector = filesByCollector,
                    OldestArchive = files.Length > 0 ? files.Min(f => new FileInfo(f).CreationTimeUtc) : DateTime.UtcNow,
                    NewestArchive = files.Length > 0 ? files.Max(f => new FileInfo(f).CreationTimeUtc) : DateTime.UtcNow,
                    CompressionRatio = 0.7 // TODO: Calculate actual compression ratio
                };
                
                return Task.FromResult(result);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting storage usage information");
                throw;
            }
        }

        public async Task<bool> ValidateArchiveAsync(string fileName)
        {
            try
            {
                var filePath = Path.Combine(_archiveDirectory, fileName);
                if (!File.Exists(filePath))
                {
                    return false;
                }

                // Basic validation - check if file can be read and parsed
                var extractedData = await ExtractArchiveAsync(fileName);
                if (extractedData == null)
                {
                    return false;
                }

                // Try to parse as JSON
                var jsonString = Encoding.UTF8.GetString(extractedData);
                var archiveData = JsonSerializer.Deserialize<LogArchiveData>(jsonString);
                
                return archiveData?.Logs != null && archiveData.Metadata != null;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error validating archive {FileName}", fileName);
                return false;
            }
        }

        public async Task<ArchiveSearchResult> SearchArchiveAsync(string searchTerm, DateTime? startDate = null, DateTime? endDate = null)
        {
            try
            {
                var searchStart = DateTime.UtcNow;
                var results = new List<ArchiveLogEntry>();
                var searchedFiles = new List<string>();

                var files = Directory.GetFiles(_archiveDirectory, "*.archive.json*");

                foreach (var filePath in files)
                {
                    var fileName = Path.GetFileName(filePath);
                    var parsedInfo = ParseArchiveFileName(fileName);

                    // Filter by date range if specified
                    if (startDate.HasValue && parsedInfo.Date < startDate.Value)
                        continue;
                    if (endDate.HasValue && parsedInfo.Date > endDate.Value)
                        continue;

                    searchedFiles.Add(fileName);

                    var extractedData = await ExtractArchiveAsync(fileName);
                    if (extractedData == null) continue;

                    var jsonString = Encoding.UTF8.GetString(extractedData);
                    var archiveData = JsonSerializer.Deserialize<LogArchiveData>(jsonString);

                    if (archiveData?.Logs == null) continue;

                    foreach (var log in archiveData.Logs)
                    {
                        if (log.Message.Contains(searchTerm, StringComparison.OrdinalIgnoreCase) ||
                            log.Source.Contains(searchTerm, StringComparison.OrdinalIgnoreCase))
                        {
                            results.Add(new ArchiveLogEntry
                            {
                                Id = log.Id,
                                Timestamp = log.Timestamp,
                                Level = log.Level,
                                Source = log.Source,
                                Message = log.Message,
                                CollectorType = parsedInfo.CollectorType,
                                AgentId = log.AgentId ?? "",
                                ArchiveFile = fileName
                            });
                        }
                    }
                }

                return new ArchiveSearchResult
                {
                    Results = results,
                    TotalCount = results.Count,
                    Query = searchTerm,
                    SearchDate = DateTime.UtcNow,
                    SearchedFiles = searchedFiles,
                    SearchDuration = DateTime.UtcNow - searchStart
                };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error searching archives for term {SearchTerm}", searchTerm);
                throw;
            }
        }

        public Task CleanupOldArchivesAsync()
        {
            try
            {
                _logger.LogInformation("Starting cleanup of old archives");

                var files = Directory.GetFiles(_archiveDirectory, "*.archive.json*");
                var deletedCount = 0;

                foreach (var filePath in files)
                {
                    var fileName = Path.GetFileName(filePath);
                    var parsedInfo = ParseArchiveFileName(fileName);
                    var settings = _collectorSettings.GetValueOrDefault(parsedInfo.CollectorType, new CollectorArchiveSettings());
                    
                    var retentionCutoff = DateTime.UtcNow.AddDays(-settings.RetentionDays);
                    
                    if (parsedInfo.Date < retentionCutoff)
                    {
                        File.Delete(filePath);
                        deletedCount++;
                        _logger.LogInformation("Deleted old archive: {FileName}", fileName);
                    }
                }

                _logger.LogInformation("Cleanup completed. Deleted {Count} old archive files", deletedCount);
                return Task.CompletedTask;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error during archive cleanup");
                throw;
            }
        }

        public async Task<bool> IsHealthyAsync()
        {
            try
            {
                // Check if archive directory exists and is writable
                if (!Directory.Exists(_archiveDirectory))
                {
                    return false;
                }

                // Try to create a test file
                var testFile = Path.Combine(_archiveDirectory, "health_check.tmp");
                await File.WriteAllTextAsync(testFile, "health check");
                File.Delete(testFile);

                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Health check failed for log archiving service");
                return false;
            }
        }

        /// <summary>
        /// Loads archived logs based on query criteria
        /// </summary>
        public async Task<ArchiveQueryResult> LoadArchivedLogsAsync(ArchiveQueryRequest request)
        {
            try
            {
                _logger.LogInformation("Loading archived logs for query: {Query}", JsonSerializer.Serialize(request));
                
                var result = new ArchiveQueryResult
                {
                    Success = true,
                    Logs = new List<LogEntryModels>(),
                    TotalFound = 0
                };

                var archiveFiles = await GetArchiveFilesAsync(request.CollectorType, request.StartDate, request.EndDate);
                
                foreach (var archiveFile in archiveFiles)
                {
                    try
                    {
                        var logs = await LoadLogsFromArchiveFileAsync(archiveFile.FileName, request);
                        result.Logs.AddRange(logs);
                    }
                    catch (Exception ex)
                    {
                        _logger.LogError(ex, "Error loading logs from archive file: {File}", archiveFile.FileName);
                    }
                }

                result.TotalFound = result.Logs.Count;
                return result;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error loading archived logs");
                return new ArchiveQueryResult { Success = false, Message = ex.Message };
            }
        }

        /// <summary>
        /// Restores logs from archive back to hot storage
        /// </summary>
        public async Task<bool> RestoreLogsFromArchiveAsync(LogRestorationRequest request)
        {
            try
            {
                _logger.LogInformation("Restoring logs from archive: {Request}", JsonSerializer.Serialize(request));
                
                using var scope = _scopeFactory.CreateScope();
                var context = scope.ServiceProvider.GetRequiredService<ApplicationDbContext>();
                
                var archiveQuery = new ArchiveQueryRequest
                {
                    StartDate = request.StartDate ?? DateTime.MinValue,
                    EndDate = request.EndDate ?? DateTime.MaxValue,
                    CollectorType = request.CollectorType
                };
                
                var archivedLogs = await LoadArchivedLogsAsync(archiveQuery);
                if (!archivedLogs.Success || !archivedLogs.Logs.Any())
                {
                    return false;
                }

                // Restore logs to database
                context.LogEntries.AddRange(archivedLogs.Logs);
                await context.SaveChangesAsync();
                
                _logger.LogInformation("Restored {Count} logs to hot storage", archivedLogs.Logs.Count);
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error restoring logs from archive");
                return false;
            }
        }

        /// <summary>
        /// Moves archive file to cold storage
        /// </summary>
        public async Task<bool> MoveToColdStorageAsync(ArchiveFile archiveFile)
        {
            try
            {
                _logger.LogInformation("Moving archive to cold storage: {File}", archiveFile.FileName);
                
                var coldStorageConfig = _configuration.GetSection("Storage:ColdStorage");
                var coldStoragePath = coldStorageConfig.GetValue<string>("Path");
                
                if (string.IsNullOrEmpty(coldStoragePath))
                {
                    _logger.LogWarning("Cold storage path not configured");
                    return false;
                }

                var sourceFile = Path.Combine(_archiveDirectory, archiveFile.FileName);
                var destinationFile = Path.Combine(coldStoragePath, archiveFile.FileName);
                
                Directory.CreateDirectory(Path.GetDirectoryName(destinationFile)!);
                File.Move(sourceFile, destinationFile);
                
                _logger.LogInformation("Successfully moved {File} to cold storage", archiveFile.FileName);
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error moving archive to cold storage: {File}", archiveFile.FileName);
                return false;
            }
        }

        /// <summary>
        /// Exports archived logs to specified format
        /// </summary>
        public async Task<string?> ExportArchivedLogsAsync(ArchiveQueryRequest request)
        {
            try
            {
                _logger.LogInformation("Exporting archived logs: {Query}", JsonSerializer.Serialize(request));
                
                var archivedLogs = await LoadArchivedLogsAsync(request);
                if (!archivedLogs.Success || !archivedLogs.Logs.Any())
                {
                    return null;
                }

                var exportFormat = request.ExportFormat ?? "json";
                var timestamp = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss");
                var fileName = $"athala_siem_export_{timestamp}.{exportFormat.ToLower()}";
                var exportPath = Path.Combine(_archiveDirectory, "exports", fileName);
                
                Directory.CreateDirectory(Path.GetDirectoryName(exportPath)!);
                
                switch (exportFormat.ToLower())
                {
                    case "json":
                        await File.WriteAllTextAsync(exportPath, JsonSerializer.Serialize(archivedLogs.Logs, new JsonSerializerOptions { WriteIndented = true }));
                        break;
                        
                    case "csv":
                        await ExportToCsvAsync(exportPath, archivedLogs.Logs);
                        break;
                        
                    default:
                        throw new NotSupportedException($"Export format {exportFormat} not supported");
                }
                
                _logger.LogInformation("Exported {Count} logs to {File}", archivedLogs.Logs.Count, exportPath);
                return exportPath;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error exporting archived logs");
                return null;
            }
        }

        private async Task<List<LogEntryModels>> LoadLogsFromArchiveFileAsync(string filePath, ArchiveQueryRequest request)
        {
            var logs = new List<LogEntryModels>();
            
            try
            {
                byte[] compressedData = await File.ReadAllBytesAsync(filePath);
                string jsonContent;
                
                using (var memoryStream = new MemoryStream(compressedData))
                using (var gzipStream = new GZipStream(memoryStream, CompressionMode.Decompress))
                using (var reader = new StreamReader(gzipStream))
                {
                    jsonContent = await reader.ReadToEndAsync();
                }
                
                var archiveData = JsonSerializer.Deserialize<LogArchiveData>(jsonContent);
                if (archiveData?.Logs != null)
                {
                    var convertedLogs = archiveData.Logs
                        .Where(archivedLog => MatchesArchivedQuery(archivedLog, request))
                        .Select(ConvertArchivedLogToLogEntry)
                        .ToList();
                    logs.AddRange(convertedLogs);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error loading logs from archive file: {File}", filePath);
            }
            
            return logs;
        }

        private bool MatchesArchivedQuery(ArchivedLogEntry log, ArchiveQueryRequest request)
        {
            if (!string.IsNullOrEmpty(request.CollectorType) && log.Source != request.CollectorType)
                return false;
                
            if (!string.IsNullOrEmpty(request.LogLevel) && log.Level != request.LogLevel)
                return false;
                
            if (!string.IsNullOrEmpty(request.SearchQuery) && 
                !log.Message.Contains(request.SearchQuery, StringComparison.OrdinalIgnoreCase))
                return false;
                
            return true;
        }

        private LogEntryModels ConvertArchivedLogToLogEntry(ArchivedLogEntry archivedLog)
        {
            return new LogEntryModels
            {
                Id = archivedLog.Id,
                Timestamp = archivedLog.Timestamp,
                Level = archivedLog.Level,
                Source = archivedLog.Source,
                Message = archivedLog.Message,
                AgentId = archivedLog.AgentId,
                Details = archivedLog.Details,
                Category = archivedLog.Category,
                ProcessId = 0 // Not available in archived format
            };
        }

        private async Task ExportToCsvAsync(string filePath, List<LogEntryModels> logs)
        {
            var csv = new StringBuilder();
            csv.AppendLine("Timestamp,Level,Source,Message,AgentId,ProcessId");
            
            foreach (var log in logs)
            {
                csv.AppendLine($"{log.Timestamp:yyyy-MM-dd HH:mm:ss},{log.Level},{log.Source},{EscapeCsv(log.Message)},{log.AgentId},{log.ProcessId}");
            }
            
            await File.WriteAllTextAsync(filePath, csv.ToString());
        }

        private string EscapeCsv(string value)
        {
            if (string.IsNullOrEmpty(value)) return "";
            if (value.Contains(",") || value.Contains("\"") || value.Contains("\n"))
            {
                return $"\"{value.Replace("\"", "\"\"")}\"";
            }
            return value;
        }

        private LogArchiveData DeserializeArchiveData(byte[]? archiveBytes)
        {
            if (archiveBytes == null || archiveBytes.Length == 0)
            {
                return new LogArchiveData();
            }

            try
            {
                var json = System.Text.Encoding.UTF8.GetString(archiveBytes);
                var archiveData = JsonSerializer.Deserialize<LogArchiveData>(json);
                return archiveData ?? new LogArchiveData();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error deserializing archive data");
                return new LogArchiveData();
            }
        }
    }
}



