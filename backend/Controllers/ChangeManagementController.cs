using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Authorization;
using Microsoft.Extensions.Logging;
using Backend.DTOs;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Backend.Controllers
{
    [Authorize]
    [ApiController]
    [Route("api/change-management")]
    public class ChangeManagementController : ControllerBase
    {
        private readonly ILogger<ChangeManagementController> _logger;
        private static readonly List<ChangeRequestDto> _changeRequests = new();

        public ChangeManagementController(ILogger<ChangeManagementController> logger)
        {
            _logger = logger;
            SeedDefaultData();
        }

        private void SeedDefaultData()
        {
            if (!_changeRequests.Any())
            {
                _changeRequests.AddRange(new[]
                {
                    new ChangeRequestDto
                    {
                        Id = "CR-001",
                        Title = "Firewall Rule Update",
                        Type = "standard",
                        Status = "pending",
                        Requester = "John Doe",
                        DateSubmitted = DateTime.UtcNow.AddDays(-2),
                        Implementation = DateTime.UtcNow.AddDays(3),
                        Risk = "medium",
                        Approvers = new List<string> { "Security Team", "Network Team" },
                        Description = "Update firewall rules to accommodate new application servers"
                    },
                    new ChangeRequestDto
                    {
                        Id = "CR-002",
                        Title = "Emergency Patch Deployment",
                        Type = "emergency",
                        Status = "approved",
                        Requester = "Jane Smith",
                        DateSubmitted = DateTime.UtcNow.AddDays(-1),
                        Implementation = DateTime.UtcNow,
                        Risk = "high",
                        Approvers = new List<string> { "Security Team", "System Admin" },
                        Description = "Deploy critical security patch to address zero-day vulnerability"
                    },
                    new ChangeRequestDto
                    {
                        Id = "CR-003",
                        Title = "Database Migration",
                        Type = "normal",
                        Status = "implemented",
                        Requester = "Mike Johnson",
                        DateSubmitted = DateTime.UtcNow.AddDays(-5),
                        Implementation = DateTime.UtcNow.AddDays(-1),
                        Risk = "low",
                        Approvers = new List<string> { "DBA Team" },
                        Description = "Migrate production database to new server cluster"
                    }
                });
            }
        }

        [HttpGet]
        public ActionResult<List<ChangeRequestDto>> GetChangeRequests(
            [FromQuery] string? status = null,
            [FromQuery] string? type = null,
            [FromQuery] string? risk = null)
        {
            var requests = _changeRequests.Where(r => r != null).ToList();
            
            if (!string.IsNullOrEmpty(status))
                requests = requests.Where(r => r != null && !string.IsNullOrEmpty(r.Status) && r.Status.Equals(status, StringComparison.OrdinalIgnoreCase)).ToList();
            
            if (!string.IsNullOrEmpty(type))
                requests = requests.Where(r => r != null && !string.IsNullOrEmpty(r.Type) && r.Type.Equals(type, StringComparison.OrdinalIgnoreCase)).ToList();

            if (!string.IsNullOrEmpty(risk))
                requests = requests.Where(r => r != null && !string.IsNullOrEmpty(r.Risk) && r.Risk.Equals(risk, StringComparison.OrdinalIgnoreCase)).ToList();

            return Ok(requests.OrderByDescending(r => r.DateSubmitted).ToList());
        }

        [HttpGet("{id}")]
        public ActionResult<ChangeRequestDto> GetChangeRequest(string id)
        {
            var request = _changeRequests.FirstOrDefault(r => r != null && r.Id == id);
            if (request == null) return NotFound();
            return Ok(request);
        }

        [HttpPost]
        public ActionResult<ChangeRequestDto> CreateChangeRequest([FromBody] CreateChangeRequestDto request)
        {
            var nextId = $"CR-{(_changeRequests.Count + 1):D3}";
            var changeRequest = new ChangeRequestDto
            {
                Id = nextId,
                Title = request.Title,
                Type = request.Type ?? "normal",
                Status = "pending",
                Requester = request.Requester ?? "Unknown",
                DateSubmitted = DateTime.UtcNow,
                Implementation = request.Implementation,
                Risk = request.Risk ?? "low",
                Approvers = request.Approvers ?? new List<string>(),
                Description = request.Description
            };

            _changeRequests.Add(changeRequest);
            _logger.LogInformation("Created change request: {RequestId}", changeRequest.Id);
            return CreatedAtAction(nameof(GetChangeRequest), new { id = changeRequest.Id }, changeRequest);
        }

        [HttpPut("{id}")]
        public ActionResult<ChangeRequestDto> UpdateChangeRequest(string id, [FromBody] CreateChangeRequestDto request)
        {
            var changeRequest = _changeRequests.FirstOrDefault(r => r != null && r.Id == id);
            if (changeRequest == null) return NotFound();

            changeRequest.Title = request.Title;
            changeRequest.Type = request.Type ?? changeRequest.Type;
            changeRequest.Implementation = request.Implementation ?? changeRequest.Implementation;
            changeRequest.Risk = request.Risk ?? changeRequest.Risk;
            changeRequest.Approvers = request.Approvers ?? changeRequest.Approvers;
            changeRequest.Description = request.Description;

            return Ok(changeRequest);
        }

        [HttpPatch("{id}/status")]
        public ActionResult<ChangeRequestDto> UpdateStatus(string id, [FromBody] UpdateChangeStatusRequest request)
        {
            var changeRequest = _changeRequests.FirstOrDefault(r => r != null && r.Id == id);
            if (changeRequest == null) return NotFound();

            changeRequest.Status = request.Status;
            _logger.LogInformation("Updated change request {RequestId} status to {Status}", id, request.Status);
            return Ok(changeRequest);
        }

        [HttpDelete("{id}")]
        public ActionResult DeleteChangeRequest(string id)
        {
            var request = _changeRequests.FirstOrDefault(r => r != null && r.Id == id);
            if (request == null) return NotFound();
            _changeRequests.Remove(request);
            return NoContent();
        }

        [HttpGet("statistics")]
        public ActionResult<ChangeManagementStats> GetStatistics()
        {
            return Ok(new ChangeManagementStats
            {
                TotalRequests = _changeRequests.Count,
                PendingRequests = _changeRequests.Count(r => r != null && r.Status == "pending"),
                ApprovedRequests = _changeRequests.Count(r => r != null && r.Status == "approved"),
                ImplementedRequests = _changeRequests.Count(r => r != null && r.Status == "implemented"),
                RejectedRequests = _changeRequests.Count(r => r != null && r.Status == "rejected"),
                EmergencyRequests = _changeRequests.Count(r => r != null && r.Type == "emergency"),
                HighRiskRequests = _changeRequests.Count(r => r != null && r.Risk == "high")
            });
        }
    }
}
