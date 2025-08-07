using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Configuration;
using Backend.Services;
using Backend.Models;
using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.IO;

namespace Backend.Controllers
{
    /// <summary>
    /// Controller for 3-tier log archive management and user access
    /// Provides enterprise-grade archive access like Splunk, Wazuh, ManageEngine
    /// </summary>
    [ApiController]
    [Route("api/[controller]")]
    public class LogArchiveController : ControllerBase
    {
        private readonly ILogger<LogArchiveController> _logger;
        private readonly ILogArchivingService _archivingService;
        private readonly IConfiguration _configuration;

        public LogArchiveController(
            ILogger<LogArchiveController> logger,
            ILogArchivingService archivingService,
            IConfiguration configuration)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _archivingService = archivingService ?? throw new ArgumentNullException(nameof(archivingService));
            _configuration = configuration ?? throw new ArgumentNullException(nameof(configuration));
        }

        /// <summary>
        /// Get storage usage across all tiers (HOT/WARM/COLD)
        /// </summary>
        [HttpGet("storage-usage")]
        public async Task<ActionResult<StorageUsageInfo>> GetStorageUsage()
        {
            try
            {
                var usage = await _archivingService.GetStorageUsageAsync();
                return Ok(usage);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting storage usage");
                return StatusCode(500, "Internal server error");
            }
        }

        /// <summary>
        /// List all archive files for user browsing
        /// </summary>
        [HttpGet("files")]
        public async Task<ActionResult<List<ArchiveFile>>> GetArchiveFiles(
            [FromQuery] string? collectorType = null,
            [FromQuery] DateTime? startDate = null,
            [FromQuery] DateTime? endDate = null)
        {
            try
            {
                var files = await _archivingService.GetArchiveFilesAsync(collectorType, startDate, endDate);
                return Ok(files);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting archive files");
                return StatusCode(500, "Internal server error");
            }
        }

        /// <summary>
        /// Query archived logs (WARM tier access)
        /// </summary>
        [HttpPost("query")]
        public async Task<ActionResult<ArchiveQueryResult>> QueryArchivedLogs([FromBody] ArchiveQueryRequest request)
        {
            try
            {
                if (request == null)
                {
                    return BadRequest("Query request is required");
                }

                var result = await _archivingService.LoadArchivedLogsAsync(request);
                return Ok(result);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error querying archived logs");
                return StatusCode(500, "Internal server error");
            }
        }

        /// <summary>
        /// Export archived logs to user-specified format (JSON, CSV, XML)
        /// </summary>
        [HttpPost("export")]
        public async Task<ActionResult> ExportArchivedLogs([FromBody] ArchiveQueryRequest request)
        {
            try
            {
                if (request == null)
                {
                    return BadRequest("Export request is required");
                }

                var filePath = await _archivingService.ExportArchivedLogsAsync(request);
                
                if (string.IsNullOrEmpty(filePath) || !System.IO.File.Exists(filePath))
                {
                    return NotFound("Export file not found");
                }

                var fileBytes = await System.IO.File.ReadAllBytesAsync(filePath);
                var fileName = Path.GetFileName(filePath);
                
                var contentType = request.ExportFormat.ToLower() switch
                {
                    "csv" => "text/csv",
                    "xml" => "application/xml",
                    _ => "application/json"
                };

                return File(fileBytes, contentType, fileName);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error exporting archived logs");
                return StatusCode(500, "Internal server error");
            }
        }

        /// <summary>
        /// Restore logs from archive back to HOT storage (database)
        /// </summary>
        [HttpPost("restore")]
        public async Task<ActionResult> RestoreLogsFromArchive([FromBody] LogRestorationRequest request)
        {
            try
            {
                if (request == null)
                {
                    return BadRequest("Restoration request is required");
                }

                var success = await _archivingService.RestoreLogsFromArchiveAsync(request);
                
                if (success)
                {
                    return Ok(new { message = "Logs restored successfully", archiveFile = request.ArchiveFileName });
                }
                else
                {
                    return BadRequest("Failed to restore logs from archive");
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error restoring logs from archive");
                return StatusCode(500, "Internal server error");
            }
        }

        /// <summary>
        /// Download archive file directly (for manual backup)
        /// </summary>
        [HttpGet("download/{fileName}")]
        public async Task<ActionResult> DownloadArchiveFile(string fileName)
        {
            try
            {
                if (string.IsNullOrEmpty(fileName))
                {
                    return BadRequest("File name is required");
                }

                var archiveFiles = await _archivingService.GetArchiveFilesAsync();
                var archiveFile = archiveFiles.FirstOrDefault(f => f.FileName == fileName);
                
                var filePath = Path.Combine(_configuration.GetValue<string>("Storage:ArchiveDirectory") ?? "archives", fileName);
                
                if (archiveFile == null || !System.IO.File.Exists(filePath))
                {
                    return NotFound("Archive file not found");
                }

                var fileBytes = await System.IO.File.ReadAllBytesAsync(filePath);
                var contentType = archiveFile.IsCompressed ? "application/gzip" : "application/json";
                
                return File(fileBytes, contentType, fileName);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error downloading archive file: {FileName}", fileName);
                return StatusCode(500, "Internal server error");
            }
        }

        /// <summary>
        /// Move archive to COLD storage (external)
        /// </summary>
        [HttpPost("move-to-cold/{fileName}")]
        public async Task<ActionResult> MoveToColdStorage(string fileName)
        {
            try
            {
                if (string.IsNullOrEmpty(fileName))
                {
                    return BadRequest("File name is required");
                }

                var archiveFiles = await _archivingService.GetArchiveFilesAsync();
                var archiveFile = archiveFiles.FirstOrDefault(f => f.FileName == fileName);
                
                if (archiveFile == null)
                {
                    return NotFound("Archive file not found");
                }

                var success = await _archivingService.MoveToColdStorageAsync(archiveFile);
                
                if (success)
                {
                    return Ok(new { message = "Archive moved to cold storage successfully", fileName });
                }
                else
                {
                    return BadRequest("Failed to move archive to cold storage");
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error moving archive to cold storage: {FileName}", fileName);
                return StatusCode(500, "Internal server error");
            }
        }

        /// <summary>
        /// Trigger manual archiving from HOT to WARM storage
        /// </summary>
        [HttpPost("archive")]
        public async Task<ActionResult> TriggerArchiving(
            [FromQuery] DateTime fromDate,
            [FromQuery] DateTime toDate,
            [FromQuery] string? collectorType = null)
        {
            try
            {
                var success = await _archivingService.ArchiveLogsAsync(fromDate, toDate, collectorType);
                
                if (success)
                {
                    return Ok(new 
                    { 
                        message = "Archiving completed successfully", 
                        fromDate, 
                        toDate, 
                        collectorType = collectorType ?? "All" 
                    });
                }
                else
                {
                    return BadRequest("Failed to archive logs");
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error triggering manual archiving");
                return StatusCode(500, "Internal server error");
            }
        }
    }
}