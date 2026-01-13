using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Authorization;
using Microsoft.Extensions.Logging;
using Backend.DTOs;
using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Linq;

namespace Backend.Controllers
{
    [Authorize]
    [ApiController]
    [Route("api/detection-rules")]
    public class DetectionRulesController : ControllerBase
    {
        private readonly ILogger<DetectionRulesController> _logger;
        private static readonly List<SigmaRuleDto> _sigmaRules = new();
        private static readonly List<YaraRuleDto> _yaraRules = new();

        public DetectionRulesController(ILogger<DetectionRulesController> logger)
        {
            _logger = logger;
            SeedDefaultRules();
        }

        private void SeedDefaultRules()
        {
            if (!_sigmaRules.Any())
            {
                _sigmaRules.AddRange(new[]
                {
                    new SigmaRuleDto
                    {
                        Id = Guid.NewGuid().ToString(),
                        Title = "Suspicious Service Creation",
                        Description = "Detects suspicious service creation from unusual processes",
                        Status = "active",
                        Level = "high",
                        Logsource = "windows/security",
                        Tags = new List<string> { "attack.persistence", "attack.t1543.003" },
                        LastModified = DateTime.UtcNow,
                        Matches = 3,
                        Content = "title: Suspicious Service Creation\nstatus: experimental\ndescription: Detects suspicious service creation"
                    },
                    new SigmaRuleDto
                    {
                        Id = Guid.NewGuid().ToString(),
                        Title = "PowerShell Download Cradle",
                        Description = "Detects PowerShell download cradles",
                        Status = "active",
                        Level = "medium",
                        Logsource = "windows/powershell",
                        Tags = new List<string> { "attack.execution", "attack.t1059.001" },
                        LastModified = DateTime.UtcNow,
                        Matches = 7,
                        Content = "title: PowerShell Download Cradle\nstatus: experimental"
                    }
                });
            }

            if (!_yaraRules.Any())
            {
                _yaraRules.AddRange(new[]
                {
                    new YaraRuleDto
                    {
                        Id = Guid.NewGuid().ToString(),
                        Name = "Detect_Cobalt_Strike",
                        Description = "Detects Cobalt Strike beacon patterns",
                        Category = "Malware",
                        Severity = "critical",
                        Status = "active",
                        LastModified = DateTime.UtcNow,
                        Matches = 5,
                        Content = "rule Detect_Cobalt_Strike { meta: description = \"Detects Cobalt Strike\" strings: $a = \"beacon\" condition: $a }"
                    },
                    new YaraRuleDto
                    {
                        Id = Guid.NewGuid().ToString(),
                        Name = "Suspicious_PowerShell",
                        Description = "Detects suspicious PowerShell execution patterns",
                        Category = "Behavior",
                        Severity = "high",
                        Status = "active",
                        LastModified = DateTime.UtcNow,
                        Matches = 12,
                        Content = "rule Suspicious_PowerShell { condition: true }"
                    }
                });
            }
        }

        [HttpGet("sigma")]
        public ActionResult<List<SigmaRuleDto>> GetSigmaRules([FromQuery] string? status = null, [FromQuery] string? level = null)
        {
            var rules = _sigmaRules.Where(r => r != null).ToList();
            
            if (!string.IsNullOrEmpty(status))
                rules = rules.Where(r => r != null && !string.IsNullOrEmpty(r.Status) && r.Status.Equals(status, StringComparison.OrdinalIgnoreCase)).ToList();
            
            if (!string.IsNullOrEmpty(level))
                rules = rules.Where(r => r != null && !string.IsNullOrEmpty(r.Level) && r.Level.Equals(level, StringComparison.OrdinalIgnoreCase)).ToList();

            return Ok(rules);
        }

        [HttpGet("sigma/{id}")]
        public ActionResult<SigmaRuleDto> GetSigmaRule(string id)
        {
            var rule = _sigmaRules.FirstOrDefault(r => r != null && r.Id == id);
            if (rule == null) return NotFound();
            return Ok(rule);
        }

        [HttpPost("sigma")]
        public ActionResult<SigmaRuleDto> CreateSigmaRule([FromBody] CreateSigmaRuleRequest request)
        {
            var rule = new SigmaRuleDto
            {
                Id = Guid.NewGuid().ToString(),
                Title = request.Title,
                Description = request.Description,
                Status = request.Status ?? "testing",
                Level = request.Level ?? "medium",
                Logsource = request.Logsource ?? string.Empty,
                Tags = request.Tags ?? new List<string>(),
                LastModified = DateTime.UtcNow,
                Matches = 0,
                Content = request.Content
            };

            _sigmaRules.Add(rule);
            _logger.LogInformation("Created SIGMA rule: {RuleId}", rule.Id);
            return CreatedAtAction(nameof(GetSigmaRule), new { id = rule.Id }, rule);
        }

        [HttpPut("sigma/{id}")]
        public ActionResult<SigmaRuleDto> UpdateSigmaRule(string id, [FromBody] CreateSigmaRuleRequest request)
        {
            var rule = _sigmaRules.FirstOrDefault(r => r != null && r.Id == id);
            if (rule == null) return NotFound();

            rule.Title = request.Title;
            rule.Description = request.Description;
            rule.Status = request.Status ?? rule.Status;
            rule.Level = request.Level ?? rule.Level;
            rule.Logsource = request.Logsource ?? rule.Logsource;
            rule.Tags = request.Tags ?? rule.Tags;
            rule.Content = request.Content;
            rule.LastModified = DateTime.UtcNow;

            return Ok(rule);
        }

        [HttpDelete("sigma/{id}")]
        public ActionResult DeleteSigmaRule(string id)
        {
            var rule = _sigmaRules.FirstOrDefault(r => r != null && r.Id == id);
            if (rule == null) return NotFound();
            _sigmaRules.Remove(rule);
            return NoContent();
        }

        [HttpPost("sigma/{id}/test")]
        public async Task<ActionResult<RuleTestResult>> TestSigmaRule(string id)
        {
            var rule = _sigmaRules.FirstOrDefault(r => r != null && r.Id == id);
            if (rule == null) return NotFound();

            await Task.Delay(500);
            
            return Ok(new RuleTestResult
            {
                RuleId = id,
                Success = true,
                Matches = Random.Shared.Next(0, 10),
                ExecutionTime = Random.Shared.NextDouble() * 2,
                TestedAt = DateTime.UtcNow
            });
        }

        [HttpGet("yara")]
        public ActionResult<List<YaraRuleDto>> GetYaraRules([FromQuery] string? status = null, [FromQuery] string? severity = null)
        {
            var rules = _yaraRules.Where(r => r != null).ToList();
            
            if (!string.IsNullOrEmpty(status))
                rules = rules.Where(r => r != null && !string.IsNullOrEmpty(r.Status) && r.Status.Equals(status, StringComparison.OrdinalIgnoreCase)).ToList();
            
            if (!string.IsNullOrEmpty(severity))
                rules = rules.Where(r => r != null && !string.IsNullOrEmpty(r.Severity) && r.Severity.Equals(severity, StringComparison.OrdinalIgnoreCase)).ToList();

            return Ok(rules);
        }

        [HttpGet("yara/{id}")]
        public ActionResult<YaraRuleDto> GetYaraRule(string id)
        {
            var rule = _yaraRules.FirstOrDefault(r => r != null && r.Id == id);
            if (rule == null) return NotFound();
            return Ok(rule);
        }

        [HttpPost("yara")]
        public ActionResult<YaraRuleDto> CreateYaraRule([FromBody] CreateYaraRuleRequest request)
        {
            var rule = new YaraRuleDto
            {
                Id = Guid.NewGuid().ToString(),
                Name = request.Name,
                Description = request.Description,
                Category = request.Category ?? "General",
                Severity = request.Severity ?? "medium",
                Status = request.Status ?? "testing",
                LastModified = DateTime.UtcNow,
                Matches = 0,
                Content = request.Content
            };

            _yaraRules.Add(rule);
            _logger.LogInformation("Created YARA rule: {RuleId}", rule.Id);
            return CreatedAtAction(nameof(GetYaraRule), new { id = rule.Id }, rule);
        }

        [HttpPut("yara/{id}")]
        public ActionResult<YaraRuleDto> UpdateYaraRule(string id, [FromBody] CreateYaraRuleRequest request)
        {
            var rule = _yaraRules.FirstOrDefault(r => r != null && r.Id == id);
            if (rule == null) return NotFound();

            rule.Name = request.Name;
            rule.Description = request.Description;
            rule.Category = request.Category ?? rule.Category;
            rule.Severity = request.Severity ?? rule.Severity;
            rule.Status = request.Status ?? rule.Status;
            rule.Content = request.Content;
            rule.LastModified = DateTime.UtcNow;

            return Ok(rule);
        }

        [HttpDelete("yara/{id}")]
        public ActionResult DeleteYaraRule(string id)
        {
            var rule = _yaraRules.FirstOrDefault(r => r != null && r.Id == id);
            if (rule == null) return NotFound();
            _yaraRules.Remove(rule);
            return NoContent();
        }

        [HttpPost("yara/{id}/test")]
        public async Task<ActionResult<RuleTestResult>> TestYaraRule(string id)
        {
            var rule = _yaraRules.FirstOrDefault(r => r != null && r.Id == id);
            if (rule == null) return NotFound();

            await Task.Delay(500);
            
            return Ok(new RuleTestResult
            {
                RuleId = id,
                Success = true,
                Matches = Random.Shared.Next(0, 10),
                ExecutionTime = Random.Shared.NextDouble() * 2,
                TestedAt = DateTime.UtcNow
            });
        }
    }
}
