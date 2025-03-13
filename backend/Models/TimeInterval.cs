namespace Backend.Models
{
    /// <summary>
    /// Represents a time interval for data aggregation
    /// </summary>
    public enum TimeInterval
    {
        /// <summary>
        /// Minute interval
        /// </summary>
        Minute,
        
        /// <summary>
        /// Hour interval
        /// </summary>
        Hour,
        
        /// <summary>
        /// Day interval
        /// </summary>
        Day,
        
        /// <summary>
        /// Week interval
        /// </summary>
        Week,
        
        /// <summary>
        /// Month interval
        /// </summary>
        Month
    }
} 