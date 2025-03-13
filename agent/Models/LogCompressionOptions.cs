using System;

namespace AthalaSIEM.Agent.Models
{
    /// <summary>
    /// Compression options for log entries
    /// </summary>
    public class LogCompressionOptions
    {
        /// <summary>
        /// Gets or sets a value indicating whether to use compression
        /// </summary>
        public bool UseCompression { get; set; } = true;
        
        /// <summary>
        /// Gets or sets the compression level
        /// </summary>
        public CompressionLevel Level { get; set; } = CompressionLevel.Optimal;
        
        /// <summary>
        /// Gets or sets the compression threshold in bytes
        /// </summary>
        public int CompressionThreshold { get; set; } = 1024; // Only compress if payload is larger than 1KB
        
        /// <summary>
        /// Gets or sets a value indicating whether to compress individual logs
        /// </summary>
        public bool CompressIndividualLogs { get; set; } = false;
        
        /// <summary>
        /// Gets or sets a value indicating whether to use fast compression
        /// </summary>
        public bool UseFastCompression { get; set; } = true;
    }
    
    /// <summary>
    /// Compression level for log entries
    /// </summary>
    public enum CompressionLevel
    {
        /// <summary>
        /// No compression
        /// </summary>
        NoCompression = 0,
        
        /// <summary>
        /// Fastest compression
        /// </summary>
        Fastest = 1,
        
        /// <summary>
        /// Optimal balance between speed and compression
        /// </summary>
        Optimal = 2,
        
        /// <summary>
        /// Maximum compression
        /// </summary>
        Maximum = 3
    }
} 