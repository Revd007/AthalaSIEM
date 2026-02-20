using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Claims;
using System.Text.Json;
using System.Threading.Tasks;
using Backend.Data;
using Backend.Models;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging;

namespace Backend.Controllers
{
    /// <summary>
    /// Generic system settings API: GET/PUT per category with database persistence.
    /// Categories: security, agents, monitoring, notifications, network, compliance, backup, integrations.
    /// </summary>
    [ApiController]
    [Route("api/[controller]")]
    [Authorize]
    public class SettingsController : ControllerBase
    {
        private static readonly HashSet<string> AllowedCategories = new(StringComparer.OrdinalIgnoreCase)
        {
            "security", "agents", "monitoring", "notifications", "network", "compliance", "backup", "integrations"
        };

        private readonly ApplicationDbContext _context;
        private readonly ILogger<SettingsController> _logger;

        public SettingsController(ApplicationDbContext context, ILogger<SettingsController> logger)
        {
            _context = context ?? throw new ArgumentNullException(nameof(context));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }

        /// <summary>
        /// Gets all settings for a category as a single JSON object (key -> value).
        /// </summary>
        [HttpGet("{category}")]
        public async Task<ActionResult<Dictionary<string, JsonElement>>> GetSettings(string category)
        {
            if (!AllowedCategories.Contains(category))
                return BadRequest(new { message = "Invalid settings category." });

            var rows = await _context.SystemSettings
                .Where(s => s.Category == category)
                .ToListAsync();

            var result = new Dictionary<string, JsonElement>(StringComparer.OrdinalIgnoreCase);
            foreach (var row in rows)
            {
                if (string.IsNullOrEmpty(row.ValueJson))
                    continue;
                try
                {
                    var el = JsonSerializer.Deserialize<JsonElement>(row.ValueJson);
                    result[row.Key] = el;
                }
                catch
                {
                    // store as string value if not valid JSON
                    result[row.Key] = JsonSerializer.SerializeToElement(row.ValueJson);
                }
            }

            return Ok(result);
        }

        /// <summary>
        /// Updates settings for a category. Body is a JSON object; each key is upserted.
        /// </summary>
        [HttpPut("{category}")]
        [Authorize(Roles = "Admin")]
        public async Task<ActionResult> PutSettings(string category, [FromBody] Dictionary<string, JsonElement> payload)
        {
            if (!AllowedCategories.Contains(category))
                return BadRequest(new { message = "Invalid settings category." });

            if (payload == null || payload.Count == 0)
                return Ok();

            var userId = User.FindFirstValue(ClaimTypes.NameIdentifier) ?? User.Identity?.Name ?? "system";
            var now = DateTime.UtcNow;

            foreach (var kv in payload)
            {
                var key = kv.Key;
                if (string.IsNullOrWhiteSpace(key) || key.Length > 100)
                    continue;

                var valueJson = kv.Value.GetRawText();

                var existing = await _context.SystemSettings
                    .FirstOrDefaultAsync(s => s.Category == category && s.Key == key);

                if (existing != null)
                {
                    existing.ValueJson = valueJson;
                    existing.UpdatedAt = now;
                    existing.UpdatedBy = userId;
                }
                else
                {
                    _context.SystemSettings.Add(new SystemSetting
                    {
                        Category = category,
                        Key = key,
                        ValueJson = valueJson,
                        UpdatedAt = now,
                        UpdatedBy = userId
                    });
                }
            }

            await _context.SaveChangesAsync();
            _logger.LogInformation("Settings category {Category} updated by {User}", category, userId);
            return Ok();
        }
    }
}
