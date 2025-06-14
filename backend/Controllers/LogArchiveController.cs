using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Authorization;
using Backend.Services;
using Backend.Models;
using System;
using System.Threading.Tasks;
using System.Collections.Generic;

namespace Backend.Controllers
{
    /// <summary>
    /// API controller for log archiving operations with multi-collector support
    /// </summary>
    [Authorize]
    [ApiController]
    [Route("api/[controller]")]
    public class LogArchiveController : ControllerBase
    {
        private readonly ILogArchivingService _logArchivingService;
        private readonly ILogger<LogArchiveController> _logger;

        public LogArchiveController(
            ILogArchivingService logArchivingService,
            ILogger<LogArchiveController> logger)
        {
            _logArchivingService = logArchivingService ?? throw new ArgumentNullException(nameof(logArchivingService));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }

        /// <summary>
        /// Archives logs for a specific collector type
        /// </summary>
        /// <param name="collectorType">The collector type (Container, CloudServices, Database, IoT, FileIntegrity)</param>
        /// <param name="olderThan">Optional cutoff date - logs older than this date will be archived</param>
        /// <returns>Archive operation result</returns>
        [HttpPost("archive/{collectorType}")]
        public async Task<ActionResult<ArchiveResult>> ArchiveLogsByCollector(
            string collectorType,
            [FromQuery] DateTime? olderThan = null)
        {
            try
            {
                if (string.IsNullOrEmpty(collectorType))
                {
                    return BadRequest("Collector type is required");
                }

                var validCollectorTypes = new[] { "Container", "CloudServices", "Database", "IoT", "FileIntegrity", "General" };
                if (!validCollectorTypes.Contains(collectorType, StringComparer.OrdinalIgnoreCase))
                {
                    return BadRequest($"Invalid collector type. Valid types: {string.Join(", ", validCollectorTypes)}");
                }

                var result = await _logArchivingService.ArchiveLogsByCollectorAsync(collectorType, olderThan ?? DateTime.UtcNow.AddDays(-30), DateTime.UtcNow);
                
                if (result.Success)
                {
                    return Ok(result);
                }
                else
                {
                    return BadRequest(result.Error);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error archiving logs for collector {CollectorType}", collectorType);
                return StatusCode(500, "Internal server error during log archiving");
            }
        }

        /// <summary>
        /// Gets list of archive files with optional filtering
        /// </summary>
        /// <param name="collectorType">Filter by collector type</param>
        /// <param name="from">Filter archives from this date</param>
        /// <param name="to">Filter archives to this date</param>
        /// <returns>List of archive files</returns>
        [HttpGet("files")]
        public async Task<ActionResult<List<ArchiveFileInfo>>> GetArchiveFiles(
            [FromQuery] string? collectorType = null,
            [FromQuery] DateTime? from = null,
            [FromQuery] DateTime? to = null)
        {
            try
            {
                if (!string.IsNullOrEmpty(collectorType))
                {
                    var validCollectorTypes = new[] { "Container", "CloudServices", "Database", "IoT", "FileIntegrity", "General" };
                    if (!validCollectorTypes.Contains(collectorType, StringComparer.OrdinalIgnoreCase))
                    {
                        return BadRequest($"Invalid collector type. Valid types: {string.Join(", ", validCollectorTypes)}");
                    }
                }

                var archiveFiles = await _logArchivingService.GetArchiveFilesAsync(collectorType, from, to);
                return Ok(archiveFiles);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting archive files");
                return StatusCode(500, "Internal server error while retrieving archive files");
            }
        }

        /// <summary>
        /// Extracts logs from an archive file
        /// </summary>
        /// <param name="archiveFileName">The archive file name</param>
        /// <param name="query">Optional query parameters for filtering extracted logs</param>
        /// <returns>List of log entries from the archive</returns>
        [HttpPost("extract")]
        public async Task<ActionResult<List<LogEntryModels>>> ExtractLogsFromArchive(
            [FromQuery] string archiveFileName,
            [FromBody] LogArchiveQuery? query = null)
        {
            try
            {
                if (string.IsNullOrEmpty(archiveFileName))
                {
                    return BadRequest("Archive file name is required");
                }

                var logs = await _logArchivingService.ExtractLogsFromArchiveAsync(archiveFileName, query);
                return Ok(logs);
            }
            catch (FileNotFoundException)
            {
                return NotFound($"Archive file '{archiveFileName}' not found");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error extracting logs from archive {ArchiveFileName}", archiveFileName);
                return StatusCode(500, "Internal server error during log extraction");
            }
        }

        /// <summary>
        /// Deletes an archive file
        /// </summary>
        /// <param name="archiveFileName">The archive file name to delete</param>
        /// <returns>Success status</returns>
        [HttpDelete("files/{archiveFileName}")]
        public async Task<ActionResult> DeleteArchive(string archiveFileName)
        {
            try
            {
                if (string.IsNullOrEmpty(archiveFileName))
                {
                    return BadRequest("Archive file name is required");
                }

                var deleted = await _logArchivingService.DeleteArchiveAsync(archiveFileName);
                
                if (deleted)
                {
                    return Ok(new { message = "Archive deleted successfully" });
                }
                else
                {
                    return NotFound($"Archive file '{archiveFileName}' not found");
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error deleting archive {ArchiveFileName}", archiveFileName);
                return StatusCode(500, "Internal server error during archive deletion");
            }
        }

        /// <summary>
        /// Gets archive statistics summary
        /// </summary>
        /// <returns>Archive statistics including collector breakdowns</returns>
        [HttpGet("statistics")]
        public async Task<ActionResult<ArchiveStatisticsSummary>> GetArchiveStatistics()
        {
            try
            {
                var statistics = await _logArchivingService.GetArchiveStatisticsAsync();
                return Ok(statistics);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting archive statistics");
                return StatusCode(500, "Internal server error while retrieving archive statistics");
            }
        }

        /// <summary>
        /// Gets archive statistics for a specific collector type
        /// </summary>
        /// <param name="collectorType">The collector type</param>
        /// <returns>Collector-specific archive statistics</returns>
        [HttpGet("statistics/{collectorType}")]
        public async Task<ActionResult<object>> GetCollectorArchiveStatistics(string collectorType)
        {
            try
            {
                if (string.IsNullOrEmpty(collectorType))
                {
                    return BadRequest("Collector type is required");
                }

                var validCollectorTypes = new[] { "Container", "CloudServices", "Database", "IoT", "FileIntegrity", "General" };
                if (!validCollectorTypes.Contains(collectorType, StringComparer.OrdinalIgnoreCase))
                {
                    return BadRequest($"Invalid collector type. Valid types: {string.Join(", ", validCollectorTypes)}");
                }

                var allStatistics = await _logArchivingService.GetArchiveStatisticsAsync();
                
                if (allStatistics.CollectorStatistics.TryGetValue(collectorType, out var collectorStats))
                {
                    var result = new
                    {
                        CollectorType = collectorType,
                        Statistics = collectorStats,
                        Archives = await _logArchivingService.GetArchiveFilesAsync(collectorType),
                        RetentionInfo = GetCollectorRetentionInfo(collectorType)
                    };

                    return Ok(result);
                }
                else
                {
                    return Ok(new 
                    {
                        CollectorType = collectorType,
                        Statistics = new ArchiveStatistics(),
                        Archives = new List<ArchiveFileInfo>(),
                        RetentionInfo = GetCollectorRetentionInfo(collectorType)
                    });
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting collector archive statistics for {CollectorType}", collectorType);
                return StatusCode(500, "Internal server error while retrieving collector archive statistics");
            }
        }

        /// <summary>
        /// Archives all logs older than specified date across all collectors
        /// </summary>
        /// <param name="olderThan">Cutoff date for archiving</param>
        /// <returns>Summary of archive operations</returns>
        [HttpPost("archive-all")]
        public async Task<ActionResult<object>> ArchiveAllLogs([FromQuery] DateTime? olderThan = null)
        {
            try
            {
                var cutoffDate = olderThan ?? DateTime.UtcNow.AddDays(-30); // Default 30 days
                var collectorTypes = new[] { "Container", "CloudServices", "Database", "IoT", "FileIntegrity", "General" };
                var results = new List<object>();

                foreach (var collectorType in collectorTypes)
                {
                    try
                    {
                        var result = await _logArchivingService.ArchiveLogsByCollectorAsync(collectorType, cutoffDate, DateTime.UtcNow);
                        results.Add(new
                        {
                            CollectorType = collectorType,
                            Success = result.Success,
                            ArchivedCount = result.ArchivedCount,
                            ArchiveFileName = result.ArchiveFileName,
                            CompressedSize = result.CompressedSize,
                            Error = result.Error
                        });
                    }
                    catch (Exception ex)
                    {
                        _logger.LogError(ex, "Error archiving logs for collector {CollectorType}", collectorType);
                        results.Add(new
                        {
                            CollectorType = collectorType,
                            Success = false,
                            ArchivedCount = 0,
                            ArchiveFileName = (string?)null,
                            CompressedSize = 0L,
                            Error = ex.Message
                        });
                    }
                }

                var summary = new
                {
                    CutoffDate = cutoffDate,
                    TotalArchivedLogs = results.Where(r => (bool)r.GetType().GetProperty("Success")!.GetValue(r)!)
                                              .Sum(r => (int)r.GetType().GetProperty("ArchivedCount")!.GetValue(r)!),
                    SuccessfulCollectors = results.Count(r => (bool)r.GetType().GetProperty("Success")!.GetValue(r)!),
                    FailedCollectors = results.Count(r => !(bool)r.GetType().GetProperty("Success")!.GetValue(r)!),
                    Results = results
                };

                return Ok(summary);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error performing bulk archive operation");
                return StatusCode(500, "Internal server error during bulk archive operation");
            }
        }

        /// <summary>
        /// Validates archive file integrity
        /// </summary>
        /// <param name="archiveFileName">The archive file name to validate</param>
        /// <returns>Validation result</returns>
        [HttpPost("validate/{archiveFileName}")]
        public async Task<ActionResult<object>> ValidateArchive(string archiveFileName)
        {
            try
            {
                if (string.IsNullOrEmpty(archiveFileName))
                {
                    return BadRequest("Archive file name is required");
                }

                // Basic validation by attempting to extract logs
                var archiveData = await _logArchivingService.ExtractLogsFromArchiveAsync(archiveFileName);
                
                var validation = new
                {
                    ArchiveFileName = archiveFileName,
                    IsValid = true,
                    LogCount = archiveData.Logs.Count,
                    ValidatedAt = DateTime.UtcNow,
                    Issues = new List<string>()
                };

                return Ok(validation);
            }
            catch (FileNotFoundException)
            {
                return Ok(new
                {
                    ArchiveFileName = archiveFileName,
                    IsValid = false,
                    LogCount = 0,
                    ValidatedAt = DateTime.UtcNow,
                    Issues = new List<string> { "Archive file not found" }
                });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error validating archive {ArchiveFileName}", archiveFileName);
                return Ok(new
                {
                    ArchiveFileName = archiveFileName,
                    IsValid = false,
                    LogCount = 0,
                    ValidatedAt = DateTime.UtcNow,
                    Issues = new List<string> { $"Validation error: {ex.Message}" }
                });
            }
        }

        /// <summary>
        /// Gets storage usage summary for archives
        /// </summary>
        /// <returns>Storage usage breakdown by collector</returns>
        [HttpGet("storage-usage")]
        public async Task<ActionResult<object>> GetStorageUsage()
        {
            try
            {
                var archiveFiles = await _logArchivingService.GetArchiveFilesAsync();
                
                var storageUsage = archiveFiles
                    .GroupBy(f => f.CollectorType)
                    .Select(g => new
                    {
                        CollectorType = g.Key,
                        TotalFiles = g.Count(),
                        TotalSize = g.Sum(f => f.Size),
                        TotalLogs = g.Sum(f => f.LogCount),
                        CompressedFiles = g.Count(f => f.IsCompressed),
                        OldestArchive = g.Min(f => f.Date),
                        NewestArchive = g.Max(f => f.Date),
                        AverageFileSize = g.Average(f => f.Size)
                    })
                    .OrderByDescending(u => u.TotalSize)
                    .ToList();

                var totalUsage = new
                {
                    TotalFiles = archiveFiles.Count,
                    TotalSize = archiveFiles.Sum(f => f.Size),
                    TotalLogs = archiveFiles.Sum(f => f.LogCount),
                    CompressedFiles = archiveFiles.Count(f => f.IsCompressed),
                    CompressionRatio = archiveFiles.Count(f => f.IsCompressed) / (double)Math.Max(archiveFiles.Count, 1),
                    CollectorBreakdown = storageUsage
                };

                return Ok(totalUsage);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting storage usage");
                return StatusCode(500, "Internal server error while retrieving storage usage");
            }
        }

        /// <summary>
        /// Health check endpoint for log archiving service
        /// </summary>
        /// <returns>Health status</returns>
        [HttpGet("health")]
        public async Task<ActionResult<object>> GetHealthStatus()
        {
            try
            {
                var statistics = await _logArchivingService.GetArchiveStatisticsAsync();
                
                var healthStatus = new
                {
                    Status = "Healthy",
                    Timestamp = DateTime.UtcNow,
                    Services = new
                    {
                        ArchivingService = "Operational",
                        CompressionEngine = "Active",
                        StorageAccess = "Available"
                    },
                    Statistics = new
                    {
                        TotalArchives = statistics.TotalArchives,
                        TotalArchivedLogs = statistics.TotalArchivedLogs,
                        LastArchiveDate = statistics.LastArchiveDate
                    },
                    Version = "1.0.0"
                };

                return Ok(healthStatus);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error checking log archiving health");
                return StatusCode(500, "Internal server error during health check");
            }
        }

        private object GetCollectorRetentionInfo(string collectorType)
        {
            // This would typically come from configuration
            var retentionPolicies = new Dictionary<string, object>
            {
                ["Container"] = new { RetentionDays = 180, Reason = "Container logs need extended retention for troubleshooting" },
                ["CloudServices"] = new { RetentionDays = 365, Reason = "Compliance requirements for cloud service logs" },
                ["Database"] = new { RetentionDays = 270, Reason = "Database audit requirements" },
                ["IoT"] = new { RetentionDays = 90, Reason = "IoT sensor data has shorter retention needs" },
                ["FileIntegrity"] = new { RetentionDays = 365, Reason = "Security compliance for file integrity monitoring" },
                ["General"] = new { RetentionDays = 90, Reason = "Standard log retention policy" }
            };

            return retentionPolicies.GetValueOrDefault(collectorType, retentionPolicies["General"]);
        }
    }
} 

