namespace Backend.DTOs;

/// <summary>
/// DTO for compliance control
/// </summary>
public class ComplianceControlDto
{
    public string Id { get; set; } = string.Empty;
    public string Title { get; set; } = string.Empty;
    public string Status { get; set; } = string.Empty; // compliant, non-compliant, in-progress
    public string LastAssessed { get; set; } = string.Empty;
    public string NextAssessment { get; set; } = string.Empty;
    public List<string> Evidence { get; set; } = new();
    public string Assignee { get; set; } = string.Empty;
    public string Framework { get; set; } = string.Empty;
    public string Section { get; set; } = string.Empty;
}
