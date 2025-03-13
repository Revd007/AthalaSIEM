using System;
using System.IO;
using System.IO.Compression;
using System.Text;
using System.Text.Json;
using System.Collections.Generic;
using Microsoft.Extensions.Logging;
using AthalaSIEM.Agent.Models;

namespace AthalaSIEM.Agent.Communication
{
    /// <summary>
    /// Utility for compressing and decompressing log data
    /// </summary>
    public class LogCompressor : ILogCompressor
    {
        private readonly ILogger<LogCompressor> _logger;
        private readonly LogCompressionOptions _options;
        
        /// <summary>
        /// Initializes a new instance of the <see cref="LogCompressor"/> class
        /// </summary>
        /// <param name="logger">The logger</param>
        /// <param name="options">Compression options</param>
        public LogCompressor(ILogger<LogCompressor> logger, LogCompressionOptions options)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _options = options ?? new LogCompressionOptions();
        }
        
        /// <summary>
        /// Compresses log data
        /// </summary>
        /// <param name="data">The data to compress</param>
        /// <returns>Compressed data with metadata</returns>
        public CompressedData Compress(byte[] data)
        {
            if (data == null || data.Length == 0)
                throw new ArgumentException("Data cannot be null or empty", nameof(data));
            
            // If compression is disabled or data is too small, return uncompressed
            if (!_options.UseCompression || data.Length < _options.CompressionThreshold)
            {
                return new CompressedData
                {
                    Data = data,
                    IsCompressed = false,
                    OriginalSize = data.Length,
                    CompressedSize = data.Length
                };
            }
            
            try
            {
                using (var outputStream = new MemoryStream())
                {
                    // Map our compression level to .NET compression level
                    var compressionLevel = MapCompressionLevel(_options.Level);
                    
                    using (var gzipStream = new GZipStream(outputStream, compressionLevel, true))
                    {
                        gzipStream.Write(data, 0, data.Length);
                    }
                    
                    var compressedData = outputStream.ToArray();
                    
                    // Only use compressed data if it's actually smaller
                    if (compressedData.Length < data.Length)
                    {
                        _logger.LogTrace("Compressed data from {OriginalSize} to {CompressedSize} bytes ({CompressionRatio:P2})",
                            data.Length, compressedData.Length, 1 - ((double)compressedData.Length / data.Length));
                        
                        return new CompressedData
                        {
                            Data = compressedData,
                            IsCompressed = true,
                            OriginalSize = data.Length,
                            CompressedSize = compressedData.Length
                        };
                    }
                    else
                    {
                        _logger.LogTrace("Compression not beneficial: {OriginalSize} vs {CompressedSize} bytes", 
                            data.Length, compressedData.Length);
                        
                        return new CompressedData
                        {
                            Data = data,
                            IsCompressed = false,
                            OriginalSize = data.Length,
                            CompressedSize = data.Length
                        };
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error compressing data");
                
                // Return uncompressed data if compression fails
                return new CompressedData
                {
                    Data = data,
                    IsCompressed = false,
                    OriginalSize = data.Length,
                    CompressedSize = data.Length
                };
            }
        }
        
        /// <summary>
        /// Decompresses log data
        /// </summary>
        /// <param name="compressedData">The compressed data</param>
        /// <returns>Decompressed data</returns>
        public byte[] Decompress(CompressedData compressedData)
        {
            if (compressedData == null)
                throw new ArgumentNullException(nameof(compressedData));
                
            if (compressedData.Data == null || compressedData.Data.Length == 0)
                throw new ArgumentException("Compressed data cannot be null or empty", nameof(compressedData));
                
            // If not compressed, return the original data
            if (!compressedData.IsCompressed)
                return compressedData.Data;
                
            try
            {
                using (var inputStream = new MemoryStream(compressedData.Data))
                using (var gzipStream = new GZipStream(inputStream, CompressionMode.Decompress))
                using (var outputStream = new MemoryStream())
                {
                    gzipStream.CopyTo(outputStream);
                    return outputStream.ToArray();
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error decompressing data");
                throw new InvalidOperationException("Failed to decompress data", ex);
            }
        }
        
        /// <summary>
        /// Compresses a log batch
        /// </summary>
        /// <param name="batch">The batch to compress</param>
        /// <returns>Compressed batch</returns>
        public CompressedBatch CompressBatch(LogBatch batch)
        {
            if (batch == null)
                throw new ArgumentNullException(nameof(batch));
                
            if (batch.Logs == null || batch.Logs.Count == 0)
                return new CompressedBatch { 
                    BatchId = batch.BatchId, 
                    AgentId = batch.AgentId, 
                    IsCompressed = false,
                    Data = Array.Empty<byte>()
                };
                
            try
            {
                // Serialize the batch
                string json = JsonSerializer.Serialize(batch);
                byte[] data = Encoding.UTF8.GetBytes(json);
                
                // Compress the data
                var compressedData = Compress(data);
                
                return new CompressedBatch
                {
                    BatchId = batch.BatchId,
                    AgentId = batch.AgentId,
                    CreatedAt = batch.CreatedAt,
                    BatchSize = batch.BatchSize,
                    Data = compressedData.Data,
                    IsCompressed = compressedData.IsCompressed,
                    OriginalSize = compressedData.OriginalSize,
                    CompressedSize = compressedData.CompressedSize
                };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error compressing batch {BatchId}", batch.BatchId);
                
                // Return uncompressed batch on error
                return new CompressedBatch
                {
                    BatchId = batch.BatchId,
                    AgentId = batch.AgentId,
                    CreatedAt = batch.CreatedAt,
                    BatchSize = batch.BatchSize,
                    IsCompressed = false,
                    Data = Array.Empty<byte>()
                };
            }
        }
        
        /// <summary>
        /// Decompresses a log batch
        /// </summary>
        /// <param name="compressedBatch">The compressed batch</param>
        /// <returns>Decompressed batch</returns>
        public LogBatch DecompressBatch(CompressedBatch compressedBatch)
        {
            if (compressedBatch == null)
                throw new ArgumentNullException(nameof(compressedBatch));
                
            if (compressedBatch.Data == null || compressedBatch.Data.Length == 0)
                throw new ArgumentException("Compressed batch data cannot be null or empty", nameof(compressedBatch));
                
            try
            {
                byte[] data;
                
                if (compressedBatch.IsCompressed)
                {
                    // Create appropriate compressed data object for decompression
                    var compressedData = new CompressedData
                    {
                        Data = compressedBatch.Data,
                        IsCompressed = true,
                        OriginalSize = compressedBatch.OriginalSize,
                        CompressedSize = compressedBatch.CompressedSize
                    };
                    
                    // Decompress the data
                    data = Decompress(compressedData);
                }
                else
                {
                    data = compressedBatch.Data;
                }
                
                // Deserialize the batch
                string json = Encoding.UTF8.GetString(data);
                var batch = JsonSerializer.Deserialize<LogBatch>(json);
                
                if (batch == null)
                    throw new InvalidOperationException("Failed to deserialize batch");
                    
                return batch;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error decompressing batch {BatchId}", compressedBatch.BatchId);
                throw new InvalidOperationException($"Failed to decompress batch {compressedBatch.BatchId}", ex);
            }
        }
        
        /// <summary>
        /// Maps compression level to .NET compression level
        /// </summary>
        /// <param name="level">The compression level</param>
        /// <returns>The .NET compression level</returns>
        private System.IO.Compression.CompressionLevel MapCompressionLevel(Models.CompressionLevel level)
        {
            return level switch
            {
                Models.CompressionLevel.NoCompression => System.IO.Compression.CompressionLevel.NoCompression,
                Models.CompressionLevel.Fastest => System.IO.Compression.CompressionLevel.Fastest,
                Models.CompressionLevel.Optimal => System.IO.Compression.CompressionLevel.Optimal,
                Models.CompressionLevel.Maximum => System.IO.Compression.CompressionLevel.Optimal, // .NET doesn't have a "Maximum" level
                _ => System.IO.Compression.CompressionLevel.Optimal
            };
        }
    }
    
    /// <summary>
    /// Interface for log compression operations
    /// </summary>
    public interface ILogCompressor
    {
        /// <summary>
        /// Compresses log data
        /// </summary>
        /// <param name="data">The data to compress</param>
        /// <returns>Compressed data with metadata</returns>
        CompressedData Compress(byte[] data);
        
        /// <summary>
        /// Decompresses log data
        /// </summary>
        /// <param name="compressedData">The compressed data</param>
        /// <returns>Decompressed data</returns>
        byte[] Decompress(CompressedData compressedData);
        
        /// <summary>
        /// Compresses a log batch
        /// </summary>
        /// <param name="batch">The batch to compress</param>
        /// <returns>Compressed batch</returns>
        CompressedBatch CompressBatch(LogBatch batch);
        
        /// <summary>
        /// Decompresses a log batch
        /// </summary>
        /// <param name="compressedBatch">The compressed batch</param>
        /// <returns>Decompressed batch</returns>
        LogBatch DecompressBatch(CompressedBatch compressedBatch);
    }
    
    /// <summary>
    /// Represents compressed data with metadata
    /// </summary>
    public class CompressedData
    {
        /// <summary>
        /// Gets or sets the compressed data
        /// </summary>
        public required byte[] Data { get; set; }
        
        /// <summary>
        /// Gets or sets a value indicating whether the data is compressed
        /// </summary>
        public bool IsCompressed { get; set; }
        
        /// <summary>
        /// Gets or sets the original size of the data in bytes
        /// </summary>
        public int OriginalSize { get; set; }
        
        /// <summary>
        /// Gets or sets the compressed size of the data in bytes
        /// </summary>
        public int CompressedSize { get; set; }
    }
    
    /// <summary>
    /// Represents a compressed log batch
    /// </summary>
    public class CompressedBatch
    {
        /// <summary>
        /// Gets or sets the batch ID
        /// </summary>
        public string BatchId { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the agent ID
        /// </summary>
        public string AgentId { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the time the batch was created
        /// </summary>
        public DateTime CreatedAt { get; set; }
        
        /// <summary>
        /// Gets or sets the number of logs in the batch
        /// </summary>
        public int BatchSize { get; set; }
        
        /// <summary>
        /// Gets or sets the compressed data
        /// </summary>
        public required byte[] Data { get; set; }
        
        /// <summary>
        /// Gets or sets a value indicating whether the data is compressed
        /// </summary>
        public bool IsCompressed { get; set; }
        
        /// <summary>
        /// Gets or sets the original size of the data in bytes
        /// </summary>
        public int OriginalSize { get; set; }
        
        /// <summary>
        /// Gets or sets the compressed size of the data in bytes
        /// </summary>
        public int CompressedSize { get; set; }
    }
} 