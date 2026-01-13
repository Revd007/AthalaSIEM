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
    [Route("api/playbooks")]
    public class PlaybooksController : ControllerBase
    {
        private readonly ILogger<PlaybooksController> _logger;
        private static readonly List<PlaybookDto> _playbooks = new();
        private static readonly List<PlaybookRunDto> _runs = new();

        public PlaybooksController(ILogger<PlaybooksController> logger)
        {
            _logger = logger;
            SeedDefaultPlaybooks();
        }

        private void SeedDefaultPlaybooks()
        {
            if (!_playbooks.Any())
            {
                _playbooks.AddRange(new[]
                {
                    new PlaybookDto
                    {
                        Id = Guid.NewGuid().ToString(),
                        Name = "Ransomware Hunt",
                        Description = "Detect potential ransomware activity",
                        Author = "Security Team",
                        Category = "Malware",
                        Status = "active",
                        LastRun = DateTime.UtcNow.AddHours(-2),
                        LastModified = DateTime.UtcNow,
                        Steps = new List<PlaybookStepDto>
                        {
                            new() { Id = "1", Type = "query", Name = "File Operations", Description = "Search for suspicious file operations", Config = new { query = "source=\"windows\" EventID=4663" } },
                            new() { Id = "2", Type = "enrichment", Name = "Enrich IOCs", Description = "Enrich found indicators", Config = new { sources = new[] { "virustotal", "alienvault" } } }
                        }
                    },
                    new PlaybookDto
                    {
                        Id = Guid.NewGuid().ToString(),
                        Name = "Lateral Movement Detection",
                        Description = "Detect lateral movement patterns",
                        Author = "SOC Team",
                        Category = "Threat Hunting",
                        Status = "active",
                        LastModified = DateTime.UtcNow,
                        Steps = new List<PlaybookStepDto>
                        {
                            new() { Id = "1", Type = "query", Name = "Remote Sessions", Description = "Find remote session events", Config = new { query = "EventID=4624 LogonType=10" } }
                        }
                    }
                });
            }
        }

        [HttpGet]
        public ActionResult<List<PlaybookDto>> GetPlaybooks([FromQuery] string? status = null, [FromQuery] string? category = null)
        {
            var playbooks = _playbooks.Where(p => p != null).ToList();
            
            if (!string.IsNullOrEmpty(status))
                playbooks = playbooks.Where(p => p != null && !string.IsNullOrEmpty(p.Status) && p.Status.Equals(status, StringComparison.OrdinalIgnoreCase)).ToList();
            
            if (!string.IsNullOrEmpty(category))
                playbooks = playbooks.Where(p => p != null && !string.IsNullOrEmpty(p.Category) && p.Category.Equals(category, StringComparison.OrdinalIgnoreCase)).ToList();

            return Ok(playbooks);
        }

        [HttpGet("{id}")]
        public ActionResult<PlaybookDto> GetPlaybook(string id)
        {
            var playbook = _playbooks.FirstOrDefault(p => p != null && p.Id == id);
            if (playbook == null) return NotFound();
            return Ok(playbook);
        }

        [HttpPost]
        public ActionResult<PlaybookDto> CreatePlaybook([FromBody] CreatePlaybookRequest request)
        {
            var playbook = new PlaybookDto
            {
                Id = Guid.NewGuid().ToString(),
                Name = request.Name,
                Description = request.Description,
                Author = request.Author ?? "Unknown",
                Category = request.Category ?? "General",
                Status = "draft",
                LastModified = DateTime.UtcNow,
                Steps = request.Steps ?? new List<PlaybookStepDto>()
            };

            _playbooks.Add(playbook);
            _logger.LogInformation("Created playbook: {PlaybookId}", playbook.Id);
            return CreatedAtAction(nameof(GetPlaybook), new { id = playbook.Id }, playbook);
        }

        [HttpPut("{id}")]
        public ActionResult<PlaybookDto> UpdatePlaybook(string id, [FromBody] CreatePlaybookRequest request)
        {
            var playbook = _playbooks.FirstOrDefault(p => p != null && p.Id == id);
            if (playbook == null) return NotFound();

            playbook.Name = request.Name;
            playbook.Description = request.Description;
            playbook.Author = request.Author ?? playbook.Author;
            playbook.Category = request.Category ?? playbook.Category;
            playbook.Steps = request.Steps ?? playbook.Steps;
            playbook.LastModified = DateTime.UtcNow;

            return Ok(playbook);
        }

        [HttpDelete("{id}")]
        public ActionResult DeletePlaybook(string id)
        {
            var playbook = _playbooks.FirstOrDefault(p => p != null && p.Id == id);
            if (playbook == null) return NotFound();
            _playbooks.Remove(playbook);
            return NoContent();
        }

        [HttpPost("{id}/run")]
        public Task<ActionResult<PlaybookRunDto>> RunPlaybook(string id)
        {
            var playbook = _playbooks.FirstOrDefault(p => p != null && p.Id == id);
            if (playbook == null) return Task.FromResult<ActionResult<PlaybookRunDto>>(NotFound());

            var run = new PlaybookRunDto
            {
                Id = Guid.NewGuid().ToString(),
                PlaybookId = id,
                PlaybookName = playbook.Name,
                Status = "running",
                StartTime = DateTime.UtcNow,
                Results = new List<PlaybookStepResultDto>()
            };

            _runs.Add(run);
            playbook.LastRun = DateTime.UtcNow;

            _ = Task.Run(async () =>
            {
                await Task.Delay(2000);
                run.Status = "completed";
                run.EndTime = DateTime.UtcNow;
                run.Results = playbook.Steps.Select(s => new PlaybookStepResultDto
                {
                    StepId = s.Id,
                    StepName = s.Name,
                    Status = "success",
                    Output = $"Step {s.Name} completed successfully",
                    ExecutedAt = DateTime.UtcNow
                }).ToList();
            });

            return Task.FromResult<ActionResult<PlaybookRunDto>>(Ok(run));
        }

        [HttpGet("runs")]
        public ActionResult<List<PlaybookRunDto>> GetPlaybookRuns([FromQuery] string? playbookId = null)
        {
            var runs = _runs.AsEnumerable();
            
            if (!string.IsNullOrEmpty(playbookId))
                runs = runs.Where(r => r.PlaybookId == playbookId);

            return Ok(runs.OrderByDescending(r => r.StartTime).Take(50).ToList());
        }

        [HttpGet("runs/{runId}")]
        public ActionResult<PlaybookRunDto> GetPlaybookRun(string runId)
        {
            var run = _runs.FirstOrDefault(r => r != null && r.Id == runId);
            if (run == null) return NotFound();
            return Ok(run);
        }
    }
}
