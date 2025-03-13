// This file is being kept as a reference but the interface is defined elsewhere
// to avoid duplicate definitions.

/*
using System.Threading.Tasks;

namespace Backend.Services
{
    /// <summary>
    /// Service interface for installer operations
    /// </summary>
    public interface IInstallerService
    {
        /// <summary>
        /// Generates an installer package
        /// </summary>
        /// <param name="type">The installer type</param>
        /// <returns>The installer package</returns>
        Task<InstallerPackage?> GenerateInstallerPackage(string type);
        
        /// <summary>
        /// Gets information about an installer
        /// </summary>
        /// <param name="type">The installer type</param>
        /// <param name="baseUrl">The base URL</param>
        /// <returns>The installer information</returns>
        Task<InstallerInfo> GetInstallerInfo(string type, string baseUrl);
        
        /// <summary>
        /// Builds the agent
        /// </summary>
        /// <returns>True if successful, false otherwise</returns>
        bool BuildAgent();
        
        /// <summary>
        /// Generates an installer
        /// </summary>
        /// <param name="type">The installer type</param>
        /// <returns>The installer package</returns>
        Task<InstallerPackage> GenerateInstaller(string type);
    }
    
    /// <summary>
    /// Represents an installer package
    /// </summary>
    public class InstallerPackage
    {
        /// <summary>
        /// Gets or sets the file name
        /// </summary>
        public string? FileName { get; set; }
        
        /// <summary>
        /// Gets or sets the content type
        /// </summary>
        public string? ContentType { get; set; }
        
        /// <summary>
        /// Gets or sets the content
        /// </summary>
        public byte[] Content { get; set; } = Array.Empty<byte>();
    }
    
    /// <summary>
    /// Represents installer information
    /// </summary>
    public class InstallerInfo
    {
        /// <summary>
        /// Gets or sets the installer type
        /// </summary>
        public string Type { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the installer version
        /// </summary>
        public string Version { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the installer file name
        /// </summary>
        public string FileName { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the installer size in bytes
        /// </summary>
        public long Size { get; set; }
        
        /// <summary>
        /// Gets or sets the installer download URL
        /// </summary>
        public string DownloadUrl { get; set; } = string.Empty;
        
        /// <summary>
        /// Gets or sets the installer last modified time
        /// </summary>
        public DateTime LastModified { get; set; }
        
        /// <summary>
        /// Gets or sets the installer SHA256 hash
        /// </summary>
        public string? Sha256Hash { get; set; }
        
        /// <summary>
        /// Gets or sets the installer requirements
        /// </summary>
        public string? Requirements { get; set; }
        
        /// <summary>
        /// Gets or sets the installer description
        /// </summary>
        public string? Description { get; set; }
    }
}
*/ 