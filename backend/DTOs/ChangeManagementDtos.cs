using System;
using System.Collections.Generic;

namespace Backend.DTOs
{
    public class ChangeRequestDto
    {
        public string Id { get; set; } = string.Empty;
        public string Title { get; set; } = string.Empty;
        public string Type { get; set; } = "normal";
        public string Status { get; set; } = "pending";
        public string Requester { get; set; } = string.Empty;
        public DateTime DateSubmitted { get; set; }
        public DateTime? Implementation { get; set; }
        public string Risk { get; set; } = "low";
        public List<string> Approvers { get; set; } = new();
        public string Description { get; set; } = string.Empty;
    }

    public class CreateChangeRequestDto
    {
        public string Title { get; set; } = string.Empty;
        public string? Type { get; set; }
        public string? Requester { get; set; }
        public DateTime? Implementation { get; set; }
        public string? Risk { get; set; }
        public List<string>? Approvers { get; set; }
        public string Description { get; set; } = string.Empty;
    }

    public class UpdateChangeStatusRequest
    {
        public string Status { get; set; } = string.Empty;
    }

    public class ChangeManagementStats
    {
        public int TotalRequests { get; set; }
        public int PendingRequests { get; set; }
        public int ApprovedRequests { get; set; }
        public int ImplementedRequests { get; set; }
        public int RejectedRequests { get; set; }
        public int EmergencyRequests { get; set; }
        public int HighRiskRequests { get; set; }
    }
}
