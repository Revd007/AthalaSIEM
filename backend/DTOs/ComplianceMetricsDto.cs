namespace Backend.DTOs;

/// <summary>
/// DTO for compliance metrics
/// </summary>
public class ComplianceMetricsDto
{
    public int OverallCompliance { get; set; }
    public int ControlsAtRisk { get; set; }
    public int PendingReviews { get; set; }
    public string? NextAuditDate { get; set; }
    public int TotalControls { get; set; }
    public int CompliantControls { get; set; }
    public int NonCompliantControls { get; set; }
}
