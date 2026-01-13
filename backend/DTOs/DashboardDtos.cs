namespace Backend.DTOs
{
    /// <summary>
    /// Update dashboard layout request
    /// </summary>
    public class UpdateDashboardLayoutRequest
    {
        /// <summary>
        /// Gets or sets the layout
        /// </summary>
        public string Layout { get; set; } = string.Empty;
    }
}
