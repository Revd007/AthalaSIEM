#nullable enable

namespace Backend.Models
{
    /// <summary>
    /// Represents a downloadable installer package
    /// </summary>
    public class InstallerPackage
    {
        /// <summary>
        /// The binary content of the installer
        /// </summary>
        public byte[] Content { get; set; } = Array.Empty<byte>();

        /// <summary>
        /// The MIME type of the installer file
        /// </summary>
        public string ContentType { get; set; } = "application/octet-stream";

        /// <summary>
        /// The filename of the installer
        /// </summary>
        public string FileName { get; set; } = string.Empty;
    }

    /// <summary>
    /// Represents information about an installer package
    /// </summary>
    public class InstallerInfo
    {
        /// <summary>
        /// The type of the installer (e.g., windows, linux)
        /// </summary>
        public string Type { get; set; } = string.Empty;

        /// <summary>
        /// The version of the installer
        /// </summary>
        public string Version { get; set; } = string.Empty;

        /// <summary>
        /// The content type of the installer
        /// </summary>
        public string? ContentType { get; set; }

        /// <summary>
        /// The filename of the installer
        /// </summary>
        public string FileName { get; set; } = string.Empty;

        /// <summary>
        /// The size of the installer in bytes
        /// </summary>
        public long Size { get; set; }

        /// <summary>
        /// The download URL for the installer
        /// </summary>
        public string DownloadUrl { get; set; } = string.Empty;

        /// <summary>
        /// The last modified timestamp of the installer
        /// </summary>
        public DateTime LastModified { get; set; }

        /// <summary>
        /// The SHA256 hash of the installer
        /// </summary>
        public string? Sha256Hash { get; set; }

        /// <summary>
        /// The system requirements for the installer
        /// </summary>
        public string? Requirements { get; set; }

        /// <summary>
        /// The description of the installer
        /// </summary>
        public string? Description { get; set; }

        /// <summary>
        /// The server URL for agent configuration
        /// </summary>
        public string? ServerUrl { get; set; }
    }
}

#nullable restore 