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
    [Route("api/automated-response")]
    public class AutomatedResponseController : ControllerBase
    {
        private readonly ILogger<AutomatedResponseController> _logger;
        private static readonly List<AutomatedActionDto> _actions = new();
        private static readonly List<AutomatedRuleDto> _rules = new();

        public AutomatedResponseController(ILogger<AutomatedResponseController> logger)
        {
            _logger = logger;
            SeedDefaultData();
        }

        private void SeedDefaultData()
        {
            if (!_rules.Any())
            {
                _rules.AddRange(new[]
                {
                    new AutomatedRuleDto
                    {
                        Id = Guid.NewGuid().ToString(),
                        Name = "Suspicious Login Block",
                        Description = "Block IPs after multiple failed logins",
                        Status = "active",
                        Triggers = 5,
                        LastTriggered = DateTime.UtcNow.AddHours(-1),
                        ActionType = "block",
                        Conditions = new { failedLogins = 5, timeWindow = "5m" }
                    },
                    new AutomatedRuleDto
                    {
                        Id = Guid.NewGuid().ToString(),
                        Name = "Endpoint Isolation",
                        Description = "Isolate endpoints showing malware indicators",
                        Status = "active",
                        Triggers = 2,
                        LastTriggered = DateTime.UtcNow.AddHours(-3),
                        ActionType = "isolate",
                        Conditions = new { malwareScore = 75 }
                    },
                    new AutomatedRuleDto
                    {
                        Id = Guid.NewGuid().ToString(),
                        Name = "Malware Scan Trigger",
                        Description = "Trigger malware scan on suspicious file activity",
                        Status = "active",
                        Triggers = 8,
                        LastTriggered = DateTime.UtcNow.AddMinutes(-30),
                        ActionType = "scan",
                        Conditions = new { fileTypes = new[] { ".exe", ".dll", ".ps1" } }
                    }
                });
            }

            if (!_actions.Any())
            {
                _actions.AddRange(new[]
                {
                    new AutomatedActionDto
                    {
                        Id = Guid.NewGuid().ToString(),
                        Type = "block",
                        Trigger = "Malicious IP Detection",
                        Status = "success",
                        Timestamp = DateTime.UtcNow,
                        Target = "192.168.1.100",
                        Details = "Blocked malicious IP after multiple failed login attempts",
                        Result = "IP blocked for 24 hours"
                    },
                    new AutomatedActionDto
                    {
                        Id = Guid.NewGuid().ToString(),
                        Type = "isolate",
                        Trigger = "Ransomware Behavior",
                        Status = "success",
                        Timestamp = DateTime.UtcNow.AddMinutes(-30),
                        Target = "WORKSTATION-01",
                        Details = "Isolated endpoint showing ransomware indicators",
                        Result = "Endpoint isolated from network"
                    },
                    new AutomatedActionDto
                    {
                        Id = Guid.NewGuid().ToString(),
                        Type = "scan",
                        Trigger = "Suspicious File Detected",
                        Status = "in-progress",
                        Timestamp = DateTime.UtcNow.AddMinutes(-5),
                        Target = "SERVER-DB-01",
                        Details = "Full malware scan initiated due to suspicious file creation",
                        Result = null
                    }
                });
            }
        }

        [HttpGet("actions")]
        public ActionResult<PaginatedResult<AutomatedActionDto>> GetActions(
            [FromQuery] string? type = null,
            [FromQuery] string? status = null,
            [FromQuery] int page = 1,
            [FromQuery] int pageSize = 20)
        {
            var actions = _actions.Where(a => a != null).ToList();
            
            if (!string.IsNullOrEmpty(type))
                actions = actions.Where(a => a != null && !string.IsNullOrEmpty(a.Type) && a.Type.Equals(type, StringComparison.OrdinalIgnoreCase)).ToList();
            
            if (!string.IsNullOrEmpty(status))
                actions = actions.Where(a => a != null && !string.IsNullOrEmpty(a.Status) && a.Status.Equals(status, StringComparison.OrdinalIgnoreCase)).ToList();

            var total = actions.Count;
            var items = actions
                .OrderByDescending(a => a.Timestamp)
                .Skip((page - 1) * pageSize)
                .Take(pageSize)
                .ToList();

            return Ok(new PaginatedResult<AutomatedActionDto>
            {
                Items = items,
                TotalCount = total,
                Page = page,
                PageSize = pageSize
            });
        }

        [HttpGet("actions/{id}")]
        public ActionResult<AutomatedActionDto> GetAction(string id)
        {
            var action = _actions.FirstOrDefault(a => a != null && a.Id == id);
            if (action == null) return NotFound();
            return Ok(action);
        }

        [HttpGet("rules")]
        public ActionResult<List<AutomatedRuleDto>> GetRules([FromQuery] string? status = null)
        {
            var rules = _rules.Where(r => r != null).ToList();
            
            if (!string.IsNullOrEmpty(status))
            {
                rules = rules.Where(r => r != null && !string.IsNullOrEmpty(r.Status) && r.Status.Equals(status, StringComparison.OrdinalIgnoreCase)).ToList();
            }

            return Ok(rules);
        }

        [HttpGet("rules/{id}")]
        public ActionResult<AutomatedRuleDto> GetRule(string id)
        {
            var rule = _rules.FirstOrDefault(r => r != null && r.Id == id);
            if (rule == null) return NotFound();
            return Ok(rule);
        }

        [HttpPost("rules")]
        public ActionResult<AutomatedRuleDto> CreateRule([FromBody] CreateAutomatedRuleRequest request)
        {
            var rule = new AutomatedRuleDto
            {
                Id = Guid.NewGuid().ToString(),
                Name = request.Name ?? string.Empty,
                Description = request.Description ?? string.Empty,
                Status = "active",
                Triggers = 0,
                ActionType = request.ActionType ?? string.Empty,
                Conditions = request.Conditions
            };

            _rules.Add(rule);
            _logger.LogInformation("Created automated response rule: {RuleId}", rule.Id);
            return CreatedAtAction(nameof(GetRule), new { id = rule.Id }, rule);
        }

        [HttpPut("rules/{id}")]
        public ActionResult<AutomatedRuleDto> UpdateRule(string id, [FromBody] CreateAutomatedRuleRequest request)
        {
            var rule = _rules.FirstOrDefault(r => r != null && r.Id == id);
            if (rule == null) return NotFound();

            rule.Name = request.Name ?? rule.Name;
            rule.Description = request.Description ?? rule.Description;
            rule.ActionType = request.ActionType ?? rule.ActionType;
            rule.Conditions = request.Conditions;

            return Ok(rule);
        }

        [HttpPatch("rules/{id}/status")]
        public ActionResult<AutomatedRuleDto> UpdateRuleStatus(string id, [FromBody] UpdateStatusRequest request)
        {
            var rule = _rules.FirstOrDefault(r => r != null && r.Id == id);
            if (rule == null) return NotFound();

            rule.Status = request.Status ?? rule.Status;
            return Ok(rule);
        }

        [HttpDelete("rules/{id}")]
        public ActionResult DeleteRule(string id)
        {
            var rule = _rules.FirstOrDefault(r => r != null && r.Id == id);
            if (rule == null) return NotFound();
            _rules.Remove(rule);
            return NoContent();
        }

        [HttpGet("statistics")]
        public ActionResult<AutomatedResponseStats> GetStatistics()
        {
            var today = DateTime.UtcNow.Date;
            var actionsToday = _actions.Count(a => a != null && a.Timestamp.Date == today);
            var successfulActions = _actions.Count(a => a != null && a.Status == "success");
            var totalActions = _actions.Count(a => a != null);

            return Ok(new AutomatedResponseStats
            {
                ActionsToday = actionsToday,
                SuccessRate = totalActions > 0 ? (double)successfulActions / totalActions * 100 : 0,
                AverageResponseTime = 1.2,
                ActiveRules = _rules.Count(r => r != null && !string.IsNullOrEmpty(r.Status) && r.Status == "active"),
                TotalActions = totalActions
            });
        }

        [HttpGet("metrics")]
        public ActionResult<List<ResponseMetricDto>> GetMetrics([FromQuery] int hours = 24)
        {
            var metrics = Enumerable.Range(0, hours).Select(i => new ResponseMetricDto
            {
                Time = DateTime.UtcNow.AddHours(-hours + i).ToString("HH:00"),
                Actions = Random.Shared.Next(5, 30),
                ResponseTime = Math.Round(Random.Shared.NextDouble() * 2, 2)
            }).ToList();

            return Ok(metrics);
        }
    }
}
