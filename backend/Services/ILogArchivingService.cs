using Backend.Models;
using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace Backend.Services
{
    /// <summary>
    /// Interface for log archiving service with multi-collector support
    /// </summary>
    public interface ILogArchivingService
    {
        /// <summary>
        /// Archives logs for the specified date range and collector type
        /// </summary>
        /// <param name="fromDate">Start date</param>
        /// <param name="toDate">End date</param>
        /// <param name="collectorType">Optional collector type filter</param>
        /// <returns>True if successful</returns>
        Task<bool> ArchiveLogsAsync(DateTime fromDate, DateTime toDate, string? collectorType = null);
        
        /// <summary>
        /// Gets list of archive files
        /// </summary>
        /// <param name="collectorType">Optional collector type filter</param>
        /// <param name="startDate">Optional start date filter</param>
        /// <param name="endDate">Optional end date filter</param>
        /// <returns>List of archive files</returns>
        Task<List<ArchiveFile>> GetArchiveFilesAsync(string? collectorType = null, DateTime? startDate = null, DateTime? endDate = null);
        
        /// <summary>
        /// Deletes an archive file
        /// </summary>
        /// <param name="fileName">File name to delete</param>
        /// <returns>True if successful</returns>
        Task<bool> DeleteArchiveFileAsync(string fileName);
        
        /// <summary>
        /// Extracts data from an archive file
        /// </summary>
        /// <param name="fileName">Archive file name</param>
        /// <returns>Extracted data or null</returns>
        Task<byte[]?> ExtractArchiveAsync(string fileName);
        
        /// <summary>
        /// Gets storage usage information
        /// </summary>
        /// <returns>Storage usage info</returns>
        Task<StorageUsageInfo> GetStorageUsageAsync();
        
        /// <summary>
        /// Validates an archive file
        /// </summary>
        /// <param name="fileName">Archive file name</param>
        /// <returns>True if valid</returns>
        Task<bool> ValidateArchiveAsync(string fileName);
        
        /// <summary>
        /// Searches within archive files
        /// </summary>
        /// <param name="searchTerm">Search term</param>
        /// <param name="startDate">Optional start date</param>
        /// <param name="endDate">Optional end date</param>
        /// <returns>Search results</returns>
        Task<ArchiveSearchResult> SearchArchiveAsync(string searchTerm, DateTime? startDate = null, DateTime? endDate = null);
        
        /// <summary>
        /// Cleans up old archives based on retention policies
        /// </summary>
        /// <returns>Task</returns>
        Task CleanupOldArchivesAsync();
        
        /// <summary>
        /// Checks if the service is healthy
        /// </summary>
        /// <returns>True if healthy</returns>
        Task<bool> IsHealthyAsync();

        /// <summary>
        /// Archives logs by collector type
        /// </summary>
        /// <param name="collectorType">Collector type</param>
        /// <param name="fromDate">Start date</param>
        /// <param name="toDate">End date</param>
        /// <returns>Archive result</returns>
        Task<ArchiveResult> ArchiveLogsByCollectorAsync(string collectorType, DateTime fromDate, DateTime toDate);

        /// <summary>
        /// Extracts logs from archive
        /// </summary>
        /// <param name="fileName">Archive file name</param>
        /// <param name="query">Search query</param>
        /// <returns>Extracted logs</returns>
        Task<LogArchiveData> ExtractLogsFromArchiveAsync(string fileName, LogArchiveQuery? query = null);

        /// <summary>
        /// Deletes an archive file
        /// </summary>
        /// <param name="fileName">Archive file name</param>
        /// <returns>True if deleted successfully</returns>
        Task<bool> DeleteArchiveAsync(string fileName);

        /// <summary>
        /// Gets archive statistics
        /// </summary>
        /// <param name="collectorType">Optional collector type filter</param>
        /// <returns>Archive statistics</returns>
        Task<ArchiveStatisticsSummary> GetArchiveStatisticsAsync(string? collectorType = null);
    }
} 