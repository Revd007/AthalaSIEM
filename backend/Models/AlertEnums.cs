using System;

namespace Backend.Models
{
    /// <summary>
    /// Alert status values
    /// </summary>
    public enum AlertStatusModels
    {
        /// <summary>
        /// New alert that has not been acknowledged
        /// </summary>
        New = 0,
        
        /// <summary>
        /// Alert has been acknowledged but not resolved
        /// </summary>
        Acknowledged = 1,
        
        /// <summary>
        /// Alert is being investigated
        /// </summary>
        InProgress = 2,
        
        /// <summary>
        /// Alert has been resolved
        /// </summary>
        Resolved = 3,
        
        /// <summary>
        /// Alert was a false positive
        /// </summary>
        FalsePositive = 4,
        
        /// <summary>
        /// Alert was closed without resolution
        /// </summary>
        Closed = 5
    }
} 