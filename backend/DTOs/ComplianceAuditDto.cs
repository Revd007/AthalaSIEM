namespace Backend.DTOs;

/// <summary>
/// DTO for compliance audit
/// </summary>
public class ComplianceAuditDto
{
    public string Id { get; set; } = string.Empty;
    public string Title { get; set; } = string.Empty;
    public string Status { get; set; } = string.Empty; // completed, in-progress, scheduled
    public string StartDate { get; set; } = string.Empty;
    public string EndDate { get; set; } = string.Empty;
    public string Auditor { get; set; } = string.Empty;
    public int? Score { get; set; }
    public int Findings { get; set; }
    public string Framework { get; set; } = string.Empty;
}
