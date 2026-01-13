using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Backend.Data;
using Backend.Models;
using Backend.DTOs;
using System.Text.Json;
using Microsoft.AspNetCore.Authorization;

namespace Backend.Controllers
{
    /// <summary>
    /// FIM Configuration Controller - Web interface for FIM management
    /// Following ManageEngine EventLog Analyzer and Splunk FIM patterns
    /// </summary>
    [ApiController]
    [Route("api/[controller]")]
    [Authorize]
    public class FIMController : ControllerBase
    {
        private readonly ApplicationDbContext _context;
        private readonly ILogger<FIMController> _logger;

        public FIMController(ApplicationDbContext context, ILogger<FIMController> logger)
        {
            _context = context;
            _logger = logger;
        }

        #region FIM Configuration Management

        /// <summary>
        /// Get all FIM configurations
        /// </summary>
        [HttpGet("configurations")]
        public async Task<ActionResult<IEnumerable<FIMConfigurationDto>>> GetFIMConfigurations()
        {
            try
            {
                var configurations = await _context.FIMConfigurations
                    .OrderByDescending(c => c.CreatedAt)
                    .ToListAsync();

                var dtos = configurations.Select(MapToDto).ToList();
                return Ok(dtos);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error retrieving FIM configurations");
                return StatusCode(500, "Internal server error");
            }
        }

        /// <summary>
        /// Get FIM configuration by ID
        /// </summary>
        [HttpGet("configurations/{id}")]
        public async Task<ActionResult<FIMConfigurationDto>> GetFIMConfiguration(string id)
        {
            try
            {
                var configuration = await _context.FIMConfigurations
                    .FirstOrDefaultAsync(c => c.Id == id);

                if (configuration == null)
                {
                    return NotFound($"FIM configuration with ID {id} not found");
                }

                return Ok(MapToDto(configuration));
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error retrieving FIM configuration {ConfigId}", id);
                return StatusCode(500, "Internal server error");
            }
        }

        /// <summary>
        /// Create new FIM configuration
        /// </summary>
        [HttpPost("configurations")]
        public async Task<ActionResult<FIMConfigurationDto>> CreateFIMConfiguration([FromBody] FIMConfigurationRequestDto request)
        {
            try
            {
                if (!ModelState.IsValid)
                {
                    return BadRequest(ModelState);
                }

                var configuration = new FIMConfiguration
                {
                    Id = Guid.NewGuid().ToString(),
                    Name = request.Name,
                    Description = request.Description,
                    CreatedBy = User.Identity?.Name ?? "system",
                    RulesJson = JsonSerializer.Serialize(request.Rules.Select(MapRuleRequestToDto)),
                    GlobalSettingsJson = JsonSerializer.Serialize(request.GlobalSettings ?? new FIMGlobalSettingsDto()),
                    TargetAgentsJson = JsonSerializer.Serialize(request.TargetAgents),
                    SupportedOSJson = JsonSerializer.Serialize(new[] { "Windows", "Linux", "macOS" })
                };

                _context.FIMConfigurations.Add(configuration);
                await _context.SaveChangesAsync();

                _logger.LogInformation("Created FIM configuration {ConfigName} with ID {ConfigId}", 
                    configuration.Name, configuration.Id);

                return CreatedAtAction(nameof(GetFIMConfiguration), 
                    new { id = configuration.Id }, MapToDto(configuration));
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error creating FIM configuration");
                return StatusCode(500, "Internal server error");
            }
        }

        /// <summary>
        /// Update FIM configuration
        /// </summary>
        [HttpPut("configurations/{id}")]
        public async Task<ActionResult<FIMConfigurationDto>> UpdateFIMConfiguration(string id, [FromBody] FIMConfigurationRequestDto request)
        {
            try
            {
                if (!ModelState.IsValid)
                {
                    return BadRequest(ModelState);
                }

                var configuration = await _context.FIMConfigurations
                    .FirstOrDefaultAsync(c => c.Id == id);

                if (configuration == null)
                {
                    return NotFound($"FIM configuration with ID {id} not found");
                }

                // Update configuration
                configuration.Name = request.Name;
                configuration.Description = request.Description;
                configuration.UpdatedAt = DateTime.UtcNow;
                configuration.RulesJson = JsonSerializer.Serialize(request.Rules.Select(MapRuleRequestToDto));
                configuration.GlobalSettingsJson = JsonSerializer.Serialize(request.GlobalSettings ?? new FIMGlobalSettingsDto());
                configuration.TargetAgentsJson = JsonSerializer.Serialize(request.TargetAgents);

                await _context.SaveChangesAsync();

                _logger.LogInformation("Updated FIM configuration {ConfigName} with ID {ConfigId}", 
                    configuration.Name, configuration.Id);

                return Ok(MapToDto(configuration));
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error updating FIM configuration {ConfigId}", id);
                return StatusCode(500, "Internal server error");
            }
        }

        /// <summary>
        /// Delete FIM configuration
        /// </summary>
        [HttpDelete("configurations/{id}")]
        public async Task<ActionResult> DeleteFIMConfiguration(string id)
        {
            try
            {
                var configuration = await _context.FIMConfigurations
                    .FirstOrDefaultAsync(c => c.Id == id);

                if (configuration == null)
                {
                    return NotFound($"FIM configuration with ID {id} not found");
                }

                _context.FIMConfigurations.Remove(configuration);
                await _context.SaveChangesAsync();

                _logger.LogInformation("Deleted FIM configuration {ConfigName} with ID {ConfigId}", 
                    configuration.Name, configuration.Id);

                return NoContent();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error deleting FIM configuration {ConfigId}", id);
                return StatusCode(500, "Internal server error");
            }
        }

        /// <summary>
        /// Get FIM configurations for specific agent
        /// </summary>
        [HttpGet("configurations/agent/{agentId}")]
        public async Task<ActionResult<IEnumerable<FIMConfigurationDto>>> GetFIMConfigurationsForAgent(string agentId)
        {
            try
            {
                var configurations = await _context.FIMConfigurations
                    .Where(c => c.Enabled)
                    .ToListAsync();

                // Filter configurations that target this agent
                var filteredConfigurations = configurations.Where(c =>
                {
                    var targetAgents = JsonSerializer.Deserialize<List<string>>(c.TargetAgentsJson) ?? new List<string>();
                    return targetAgents.Contains(agentId) || targetAgents.Contains("*");
                }).ToList();

                var dtos = filteredConfigurations.Select(MapToDto).ToList();
                return Ok(dtos);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error retrieving FIM configurations for agent {AgentId}", agentId);
                return StatusCode(500, "Internal server error");
            }
        }

        #endregion

        #region FIM Templates Management

        /// <summary>
        /// Get all FIM templates
        /// </summary>
        [HttpGet("templates")]
        public async Task<ActionResult<IEnumerable<FIMTemplateDto>>> GetFIMTemplates()
        {
            try
            {
                var templates = await _context.FIMTemplates
                    .OrderBy(t => t.IsBuiltIn ? 0 : 1)
                    .ThenBy(t => t.Name)
                    .ToListAsync();

                var dtos = templates.Select(MapTemplateToDto).ToList();

                // Add built-in templates if they don't exist in database
                await EnsureBuiltInTemplatesExist();

                return Ok(dtos);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error retrieving FIM templates");
                return StatusCode(500, "Internal server error");
            }
        }

        /// <summary>
        /// Get FIM templates for specific OS
        /// </summary>
        [HttpGet("templates/os/{operatingSystem}")]
        public async Task<ActionResult<IEnumerable<FIMTemplateDto>>> GetFIMTemplatesForOS(string operatingSystem)
        {
            try
            {
                var templates = await _context.FIMTemplates.ToListAsync();

                var filteredTemplates = templates.Where(t =>
                {
                    var supportedOS = JsonSerializer.Deserialize<List<string>>(t.SupportedOSJson) ?? new List<string>();
                    return supportedOS.Contains(operatingSystem) || supportedOS.Contains("*");
                }).ToList();

                var dtos = filteredTemplates.Select(MapTemplateToDto).ToList();
                return Ok(dtos);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error retrieving FIM templates for OS {OS}", operatingSystem);
                return StatusCode(500, "Internal server error");
            }
        }

        /// <summary>
        /// Create FIM configuration from template
        /// </summary>
        [HttpPost("configurations/from-template/{templateId}")]
        public async Task<ActionResult<FIMConfigurationDto>> CreateConfigurationFromTemplate(
            string templateId, 
            [FromBody] CreateFromTemplateRequest request)
        {
            try
            {
                var template = await _context.FIMTemplates
                    .FirstOrDefaultAsync(t => t.Id == templateId);

                if (template == null)
                {
                    return NotFound($"FIM template with ID {templateId} not found");
                }

                // Deserialize template rules
                var templateRules = JsonSerializer.Deserialize<List<FIMRuleDto>>(template.TemplateRulesJson) ?? new List<FIMRuleDto>();
                var templateVariables = JsonSerializer.Deserialize<Dictionary<string, string>>(template.VariablesJson) ?? new Dictionary<string, string>();

                // Apply variable substitution
                var rules = templateRules.Select(rule =>
                {
                    var newRule = new FIMRuleDto
                    {
                        Id = Guid.NewGuid().ToString(),
                        Name = rule.Name,
                        Description = rule.Description,
                        MonitorPath = SubstituteVariables(rule.MonitorPath, templateVariables, request.Variables),
                        MonitoringMode = rule.MonitoringMode,
                        MonitoringOptions = rule.MonitoringOptions,
                        Filters = rule.Filters,
                        SecurityLevel = rule.SecurityLevel,
                        AlertSettings = rule.AlertSettings,
                        Tags = rule.Tags
                    };
                    return newRule;
                }).ToList();

                // Create configuration
                var configuration = new FIMConfiguration
                {
                    Id = Guid.NewGuid().ToString(),
                    Name = request.ConfigurationName,
                    Description = $"Configuration created from template: {template.Name}",
                    CreatedBy = User.Identity?.Name ?? "system",
                    RulesJson = JsonSerializer.Serialize(rules),
                    GlobalSettingsJson = JsonSerializer.Serialize(new FIMGlobalSettingsDto()),
                    TargetAgentsJson = JsonSerializer.Serialize(request.TargetAgents),
                    SupportedOSJson = template.SupportedOSJson
                };

                _context.FIMConfigurations.Add(configuration);
                await _context.SaveChangesAsync();

                _logger.LogInformation("Created FIM configuration {ConfigName} from template {TemplateName}", 
                    configuration.Name, template.Name);

                return CreatedAtAction(nameof(GetFIMConfiguration), 
                    new { id = configuration.Id }, MapToDto(configuration));
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error creating configuration from template {TemplateId}", templateId);
                return StatusCode(500, "Internal server error");
            }
        }

        #endregion

        #region FIM Events

        /// <summary>
        /// Get FIM events
        /// </summary>
        [HttpGet("events")]
        public async Task<ActionResult<IEnumerable<FIMEventDto>>> GetFIMEvents(
            [FromQuery] string? agentId = null,
            [FromQuery] DateTime? startDate = null,
            [FromQuery] DateTime? endDate = null,
            [FromQuery] int page = 1,
            [FromQuery] int pageSize = 100)
        {
            try
            {
                var query = _context.FIMEvents.AsQueryable();

                // Apply filters
                if (!string.IsNullOrEmpty(agentId))
                {
                    query = query.Where(e => e.AgentId == agentId);
                }

                if (startDate.HasValue)
                {
                    query = query.Where(e => e.Timestamp >= startDate.Value);
                }

                if (endDate.HasValue)
                {
                    query = query.Where(e => e.Timestamp <= endDate.Value);
                }

                // Apply pagination
                var events = await query
                    .OrderByDescending(e => e.Timestamp)
                    .Skip((page - 1) * pageSize)
                    .Take(pageSize)
                    .ToListAsync();

                var dtos = events.Select(MapEventToDto).ToList();
                return Ok(dtos);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error retrieving FIM events");
                return StatusCode(500, "Internal server error");
            }
        }

        /// <summary>
        /// Create FIM event (called by agents)
        /// </summary>
        [HttpPost("events")]
        [AllowAnonymous] // Agents will use API key authentication
        public async Task<ActionResult> CreateFIMEvent([FromBody] FIMEventDto eventDto)
        {
            try
            {
                if (!ModelState.IsValid)
                {
                    return BadRequest(ModelState);
                }

                var fimEvent = new FIMEvent
                {
                    Id = eventDto.Id,
                    RuleId = eventDto.RuleId,
                    RuleName = eventDto.RuleName,
                    AgentId = eventDto.AgentId,
                    Timestamp = eventDto.Timestamp,
                    FilePath = eventDto.FilePath,
                    EventType = eventDto.EventType,
                    OldFilePath = eventDto.OldFilePath,
                    OldFileInfoJson = JsonSerializer.Serialize(eventDto.OldFileInfo),
                    NewFileInfoJson = JsonSerializer.Serialize(eventDto.NewFileInfo),
                    User = eventDto.User,
                    Process = eventDto.Process,
                    ProcessId = eventDto.ProcessId,
                    SecurityLevel = eventDto.SecurityLevel,
                    MetadataJson = JsonSerializer.Serialize(eventDto.Metadata),
                    AlertGenerated = eventDto.AlertGenerated,
                    TagsJson = JsonSerializer.Serialize(eventDto.Tags)
                };

                _context.FIMEvents.Add(fimEvent);
                await _context.SaveChangesAsync();

                _logger.LogInformation("Created FIM event for file {FilePath} on agent {AgentId}", 
                    eventDto.FilePath, eventDto.AgentId);

                return Ok();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error creating FIM event");
                return StatusCode(500, "Internal server error");
            }
        }

        #endregion

        #region Private Helper Methods

        private FIMConfigurationDto MapToDto(FIMConfiguration configuration)
        {
            return new FIMConfigurationDto
            {
                Id = configuration.Id,
                Name = configuration.Name,
                Description = configuration.Description,
                Enabled = configuration.Enabled,
                CreatedAt = configuration.CreatedAt,
                UpdatedAt = configuration.UpdatedAt,
                CreatedBy = configuration.CreatedBy,
                Rules = JsonSerializer.Deserialize<List<FIMRuleDto>>(configuration.RulesJson) ?? new List<FIMRuleDto>(),
                GlobalSettings = JsonSerializer.Deserialize<FIMGlobalSettingsDto>(configuration.GlobalSettingsJson) ?? new FIMGlobalSettingsDto(),
                TargetAgents = JsonSerializer.Deserialize<List<string>>(configuration.TargetAgentsJson) ?? new List<string>(),
                SupportedOS = JsonSerializer.Deserialize<List<string>>(configuration.SupportedOSJson) ?? new List<string>()
            };
        }

        private FIMTemplateDto MapTemplateToDto(FIMTemplate template)
        {
            return new FIMTemplateDto
            {
                Id = template.Id,
                Name = template.Name,
                Description = template.Description,
                Category = template.Category,
                TemplateRules = JsonSerializer.Deserialize<List<FIMRuleDto>>(template.TemplateRulesJson) ?? new List<FIMRuleDto>(),
                SupportedOS = JsonSerializer.Deserialize<List<string>>(template.SupportedOSJson) ?? new List<string>(),
                Variables = JsonSerializer.Deserialize<Dictionary<string, string>>(template.VariablesJson) ?? new Dictionary<string, string>(),
                IsBuiltIn = template.IsBuiltIn,
                CreatedAt = template.CreatedAt,
                UpdatedAt = template.UpdatedAt,
                CreatedBy = template.CreatedBy
            };
        }

        private FIMEventDto MapEventToDto(FIMEvent fimEvent)
        {
            return new FIMEventDto
            {
                Id = fimEvent.Id,
                RuleId = fimEvent.RuleId,
                RuleName = fimEvent.RuleName,
                AgentId = fimEvent.AgentId,
                Timestamp = fimEvent.Timestamp,
                FilePath = fimEvent.FilePath,
                EventType = fimEvent.EventType,
                OldFilePath = fimEvent.OldFilePath,
                OldFileInfo = JsonSerializer.Deserialize<FIMFileInfoDto>(fimEvent.OldFileInfoJson),
                NewFileInfo = JsonSerializer.Deserialize<FIMFileInfoDto>(fimEvent.NewFileInfoJson),
                User = fimEvent.User,
                Process = fimEvent.Process,
                ProcessId = fimEvent.ProcessId,
                SecurityLevel = fimEvent.SecurityLevel,
                Metadata = JsonSerializer.Deserialize<Dictionary<string, object>>(fimEvent.MetadataJson) ?? new Dictionary<string, object>(),
                AlertGenerated = fimEvent.AlertGenerated,
                Tags = JsonSerializer.Deserialize<List<string>>(fimEvent.TagsJson) ?? new List<string>()
            };
        }

        private FIMRuleDto MapRuleRequestToDto(FIMRuleRequestDto request)
        {
            return new FIMRuleDto
            {
                Id = Guid.NewGuid().ToString(),
                Name = request.Name,
                Description = request.Description,
                MonitorPath = request.MonitorPath,
                MonitoringMode = request.MonitoringMode,
                MonitoringOptions = request.MonitoringOptions ?? new FIMMonitoringOptionsDto(),
                Filters = request.Filters ?? new FIMFiltersDto(),
                SecurityLevel = request.SecurityLevel,
                AlertSettings = request.AlertSettings ?? new FIMAlertSettingsDto(),
                Tags = request.Tags
            };
        }

        private string SubstituteVariables(string text, Dictionary<string, string> templateVariables, Dictionary<string, string>? userVariables)
        {
            // Substitute template variables
            foreach (var variable in templateVariables)
            {
                text = text.Replace($"${{{variable.Key}}}", variable.Value);
            }

            // Substitute user-provided variables
            if (userVariables != null)
            {
                foreach (var variable in userVariables)
                {
                    text = text.Replace($"${{{variable.Key}}}", variable.Value);
                }
            }

            return text;
        }

        private async Task EnsureBuiltInTemplatesExist()
        {
            // Check if built-in templates exist, if not create them
            var existingBuiltInTemplates = await _context.FIMTemplates
                .Where(t => t.IsBuiltIn)
                .Select(t => t.Id)
                .ToListAsync();

            var builtInTemplateIds = new[] { "builtin-windows-system", "builtin-linux-system", "builtin-web-application" };
            
            foreach (var templateId in builtInTemplateIds)
            {
                if (!existingBuiltInTemplates.Contains(templateId))
                {
                    await CreateBuiltInTemplate(templateId);
                }
            }
        }

        private async Task CreateBuiltInTemplate(string templateId)
        {
            FIMTemplate? template = templateId switch
            {
                "builtin-windows-system" => CreateWindowsSystemTemplate(),
                "builtin-linux-system" => CreateLinuxSystemTemplate(),
                "builtin-web-application" => CreateWebApplicationTemplate(),
                _ => null
            };

            if (template != null)
            {
                _context.FIMTemplates.Add(template);
                await _context.SaveChangesAsync();
            }
        }

        private FIMTemplate CreateWindowsSystemTemplate()
        {
            var rules = new List<FIMRuleDto>
            {
                new FIMRuleDto
                {
                    Name = "Windows System32 Drivers",
                    Description = "Monitor Windows system drivers",
                    MonitorPath = @"C:\Windows\System32\drivers\*",
                    SecurityLevel = "Critical",
                    MonitoringOptions = new FIMMonitoringOptionsDto
                    {
                        MonitorCreation = true,
                        MonitorModification = true,
                        MonitorDeletion = true,
                        MonitorHashes = true,
                        HashAlgorithm = "SHA256"
                    }
                },
                new FIMRuleDto
                {
                    Name = "Windows Registry Hives",
                    Description = "Monitor Windows registry hive files",
                    MonitorPath = @"C:\Windows\System32\config\*",
                    SecurityLevel = "Critical",
                    Filters = new FIMFiltersDto
                    {
                        ExcludeExtensions = new List<string> { ".tmp", ".log" }
                    }
                }
            };

            return new FIMTemplate
            {
                Id = "builtin-windows-system",
                Name = "Windows System Files",
                Description = "Monitor critical Windows system files and directories",
                Category = "System",
                IsBuiltIn = true,
                SupportedOSJson = JsonSerializer.Serialize(new[] { "Windows" }),
                TemplateRulesJson = JsonSerializer.Serialize(rules),
                VariablesJson = JsonSerializer.Serialize(new Dictionary<string, string>()),
                CreatedBy = "system"
            };
        }

        private FIMTemplate CreateLinuxSystemTemplate()
        {
            var rules = new List<FIMRuleDto>
            {
                new FIMRuleDto
                {
                    Name = "System Binaries",
                    Description = "Monitor system binaries in /bin and /sbin",
                    MonitorPath = "/bin/*,/sbin/*,/usr/bin/*,/usr/sbin/*",
                    SecurityLevel = "Critical",
                    MonitoringOptions = new FIMMonitoringOptionsDto
                    {
                        MonitorCreation = true,
                        MonitorModification = true,
                        MonitorDeletion = true,
                        MonitorPermissions = true,
                        MonitorHashes = true
                    }
                },
                new FIMRuleDto
                {
                    Name = "Configuration Files",
                    Description = "Monitor system configuration files",
                    MonitorPath = "/etc/*",
                    SecurityLevel = "High",
                    Filters = new FIMFiltersDto
                    {
                        IncludeExtensions = new List<string> { ".conf", ".cfg", ".config", "" },
                        ExcludeDirectories = new List<string> { "/etc/systemd/system", "/etc/udev" }
                    }
                }
            };

            return new FIMTemplate
            {
                Id = "builtin-linux-system",
                Name = "Linux System Files",
                Description = "Monitor critical Linux system files and directories",
                Category = "System",
                IsBuiltIn = true,
                SupportedOSJson = JsonSerializer.Serialize(new[] { "Linux" }),
                TemplateRulesJson = JsonSerializer.Serialize(rules),
                VariablesJson = JsonSerializer.Serialize(new Dictionary<string, string>()),
                CreatedBy = "system"
            };
        }

        private FIMTemplate CreateWebApplicationTemplate()
        {
            var rules = new List<FIMRuleDto>
            {
                new FIMRuleDto
                {
                    Name = "Web Root Files",
                    Description = "Monitor web application files",
                    MonitorPath = "${WEB_ROOT}/*",
                    SecurityLevel = "High",
                    MonitoringOptions = new FIMMonitoringOptionsDto
                    {
                        RecursiveMonitoring = true,
                        MonitorHashes = true
                    },
                    Filters = new FIMFiltersDto
                    {
                        IncludeExtensions = new List<string> { ".php", ".html", ".js", ".css", ".jsp", ".asp", ".aspx" }
                    }
                }
            };

            var variables = new Dictionary<string, string>
            {
                { "WEB_ROOT", "/var/www/html" },
                { "CONFIG_PATH", "/etc/apache2" }
            };

            return new FIMTemplate
            {
                Id = "builtin-web-application",
                Name = "Web Application Files",
                Description = "Monitor web application files and configurations",
                Category = "Application",
                IsBuiltIn = true,
                SupportedOSJson = JsonSerializer.Serialize(new[] { "Windows", "Linux" }),
                TemplateRulesJson = JsonSerializer.Serialize(rules),
                VariablesJson = JsonSerializer.Serialize(variables),
                CreatedBy = "system"
            };
        }

        #endregion
    }
} 