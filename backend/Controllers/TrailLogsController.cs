using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using AthalaSIEM.Data;
using AthalaSIEM.Models;

namespace AthalaSIEM.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    [Authorize]
    public class TrailLogsController : ControllerBase
    {
        private readonly ApplicationDbContext _context;

        public TrailLogsController(ApplicationDbContext context)
        {
            _context = context;
        }

        // GET: api/trail-logs
        [HttpGet]
        public async Task<ActionResult<IEnumerable<TrailLog>>> GetTrailLogs(
            [FromQuery] string userId = null,
            [FromQuery] string action = null,
            [FromQuery] DateTime? startDate = null,
            [FromQuery] DateTime? endDate = null)
        {
            var query = _context.TrailLogs.AsQueryable();

            if (!string.IsNullOrEmpty(userId))
                query = query.Where(t => t.UserId == userId);

            if (!string.IsNullOrEmpty(action))
                query = query.Where(t => t.Action == action);

            if (startDate.HasValue)
                query = query.Where(t => t.Timestamp >= startDate.Value);

            if (endDate.HasValue)
                query = query.Where(t => t.Timestamp <= endDate.Value);

            return await query.OrderByDescending(t => t.Timestamp).ToListAsync();
        }

        // POST: api/trail-logs
        [HttpPost]
        public async Task<ActionResult<TrailLog>> CreateTrailLog(TrailLog trailLog)
        {
            trailLog.Timestamp = DateTime.UtcNow;
            _context.TrailLogs.Add(trailLog);
            await _context.SaveChangesAsync();

            return CreatedAtAction(nameof(GetTrailLogs), new { id = trailLog.Id }, trailLog);
        }

        // GET: api/trail-logs/stats
        [HttpGet("stats")]
        public async Task<ActionResult<object>> GetTrailLogStats(
            [FromQuery] DateTime? startDate = null,
            [FromQuery] DateTime? endDate = null)
        {
            var query = _context.TrailLogs.AsQueryable();

            if (startDate.HasValue)
                query = query.Where(t => t.Timestamp >= startDate.Value);

            if (endDate.HasValue)
                query = query.Where(t => t.Timestamp <= endDate.Value);

            var stats = new
            {
                TotalLogs = await query.CountAsync(),
                ActionsByUser = await query
                    .GroupBy(t => t.UserId)
                    .Select(g => new
                    {
                        UserId = g.Key,
                        ActionCount = g.Count()
                    })
                    .ToListAsync(),
                ActionsByType = await query
                    .GroupBy(t => t.Action)
                    .Select(g => new
                    {
                        Action = g.Key,
                        Count = g.Count()
                    })
                    .ToListAsync()
            };

            return stats;
        }
    }
} 