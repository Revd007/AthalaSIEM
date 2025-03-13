namespace Backend.Models
{
    /// <summary>
    /// Alert severity levels
    /// </summary>
    public enum SeverityModels
    {
        /// <summary>
        /// Informational alert, no action required
        /// </summary>
        Info = 0,
        
        /// <summary>
        /// Low severity alert, action may be required
        /// </summary>
        Low = 1,
        
        /// <summary>
        /// Medium severity alert, action recommended
        /// </summary>
        Medium = 2,
        
        /// <summary>
        /// High severity alert, action required
        /// </summary>
        High = 3,
        
        /// <summary>
        /// Critical severity alert, immediate action required
        /// </summary>
        Critical = 4
    }
    
    /// <summary>
    /// Alias for SeverityModels to maintain backward compatibility
    /// </summary>
    public enum AlertSeverityModels
    {
        /// <summary>
        /// Informational alert, no action required
        /// </summary>
        Info = 0,
        
        /// <summary>
        /// Low severity alert, action may be required
        /// </summary>
        Low = 1,
        
        /// <summary>
        /// Medium severity alert, action recommended
        /// </summary>
        Medium = 2,
        
        /// <summary>
        /// High severity alert, action required
        /// </summary>
        High = 3,
        
        /// <summary>
        /// Critical severity alert, immediate action required
        /// </summary>
        Critical = 4
    }
} 