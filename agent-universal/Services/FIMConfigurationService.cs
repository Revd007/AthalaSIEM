using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Configuration;
using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Text.Json;
using System.Threading.Tasks;
using AthalaSIEM.UniversalAgent.DTOs;
using System.Linq;

namespace AthalaSIEM.UniversalAgent.Services
{
    /// <summary>
    /// FIM Configuration Service - Manages FIM configuration from backend
    /// Agent-side service that fetches dynamic FIM configuration from SIEM backend
    /// Following clean architecture principles with proper separation of concerns
    /// </summary>
    public class FIMConfigurationService
    {
        private readonly ILogger<FIMConfigurationService> _logger;
        private readonly IConfiguration _configuration;
        private readonly HttpClient _httpClient;
        private readonly string _agentId;
        private readonly string _apiKey;
        private readonly string _backendUrl;

        public FIMConfigurationService(
            ILogger<FIMConfigurationService> logger, 
            IConfiguration configuration,
            HttpClient httpClient)
        {
            _logger = logger;
            _configuration = configuration;
            _httpClient = httpClient;
            
            _agentId = _configuration["Agent:Id"] ?? Environment.MachineName;
            _apiKey = _configuration["Agent:ApiKey"] ?? "";
            _backendUrl = _configuration["Agent:ManagerUrl"] ?? "";
            
            // Configure HTTP client
            if (!string.IsNullOrEmpty(_apiKey))
            {
                _httpClient.DefaultRequestHeaders.Add("X-API-Key", _apiKey);
            }
        }

        /// <summary>
        /// Fetch FIM configurations for this agent from backend
        /// </summary>
        public async Task<List<FIMConfigurationDto>> GetFIMConfigurationsAsync()
        {
            try
            {
                if (string.IsNullOrEmpty(_backendUrl))
                {
                    _logger.LogWarning("Backend URL not configured - cannot fetch FIM configurations");
                    return new List<FIMConfigurationDto>();
                }

                var url = $"{_backendUrl}/api/fim/configurations/agent/{_agentId}";
                _logger.LogDebug("Fetching FIM configurations from: {Url}", url);

                var response = await _httpClient.GetAsync(url);
                
                if (response.IsSuccessStatusCode)
                {
                    var json = await response.Content.ReadAsStringAsync();
                    var configurations = JsonSerializer.Deserialize<List<FIMConfigurationDto>>(json, new JsonSerializerOptions
                    {
                        PropertyNameCaseInsensitive = true
                    }) ?? new List<FIMConfigurationDto>();

                    _logger.LogInformation("Retrieved {Count} FIM configurations from backend", configurations.Count);
                    return configurations;
                }
                else
                {
                    _logger.LogWarning("Failed to fetch FIM configurations: {StatusCode} - {ReasonPhrase}", 
                        response.StatusCode, response.ReasonPhrase);
                    return new List<FIMConfigurationDto>();
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error fetching FIM configurations from backend");
                return new List<FIMConfigurationDto>();
            }
        }

        /// <summary>
        /// Send FIM event to backend
        /// </summary>
        public async Task<bool> SendFIMEventAsync(FIMEventDto fimEvent)
        {
            try
            {
                if (string.IsNullOrEmpty(_backendUrl))
                {
                    _logger.LogWarning("Backend URL not configured - cannot send FIM event");
                    return false;
                }

                var url = $"{_backendUrl}/api/fim/events";
                var json = JsonSerializer.Serialize(fimEvent);
                var content = new StringContent(json, System.Text.Encoding.UTF8, "application/json");

                var response = await _httpClient.PostAsync(url, content);
                
                if (response.IsSuccessStatusCode)
                {
                    _logger.LogDebug("Successfully sent FIM event for file {FilePath}", fimEvent.FilePath);
                    return true;
                }
                else
                {
                    _logger.LogWarning("Failed to send FIM event: {StatusCode} - {ReasonPhrase}", 
                        response.StatusCode, response.ReasonPhrase);
                    return false;
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error sending FIM event to backend");
                return false;
            }
        }

        /// <summary>
        /// Get available FIM templates from backend
        /// </summary>
        public async Task<List<FIMTemplateDto>> GetFIMTemplatesAsync(string? operatingSystem = null)
        {
            try
            {
                if (string.IsNullOrEmpty(_backendUrl))
                {
                    _logger.LogWarning("Backend URL not configured - cannot fetch FIM templates");
                    return new List<FIMTemplateDto>();
                }

                var url = string.IsNullOrEmpty(operatingSystem) 
                    ? $"{_backendUrl}/api/fim/templates"
                    : $"{_backendUrl}/api/fim/templates/os/{operatingSystem}";

                var response = await _httpClient.GetAsync(url);
                
                if (response.IsSuccessStatusCode)
                {
                    var json = await response.Content.ReadAsStringAsync();
                    var templates = JsonSerializer.Deserialize<List<FIMTemplateDto>>(json, new JsonSerializerOptions
                    {
                        PropertyNameCaseInsensitive = true
                    }) ?? new List<FIMTemplateDto>();

                    _logger.LogInformation("Retrieved {Count} FIM templates from backend", templates.Count);
                    return templates;
                }
                else
                {
                    _logger.LogWarning("Failed to fetch FIM templates: {StatusCode} - {ReasonPhrase}", 
                        response.StatusCode, response.ReasonPhrase);
                    return new List<FIMTemplateDto>();
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error fetching FIM templates from backend");
                return new List<FIMTemplateDto>();
            }
        }

        /// <summary>
        /// Convert FIM configuration from backend to agent format
        /// </summary>
        public Dictionary<string, object> ConvertToAgentConfiguration(FIMConfigurationDto configuration)
        {
            var agentConfig = new Dictionary<string, object>();

            if (configuration.Rules.Any())
            {
                var monitoredPaths = configuration.Rules
                    .Where(r => r.Enabled)
                    .Select(r => r.MonitorPath)
                    .ToArray();

                agentConfig["MonitoredPaths"] = monitoredPaths;
                agentConfig["ScanIntervalMinutes"] = configuration.GlobalSettings.DefaultScanInterval;
                agentConfig["MaxEventBuffer"] = configuration.GlobalSettings.MaxEventBuffer;
                agentConfig["EnableCompression"] = configuration.GlobalSettings.EnableCompression;
                agentConfig["EnableBaseline"] = configuration.GlobalSettings.EnableBaseline;
            }

            return agentConfig;
        }

        /// <summary>
        /// Check if FIM configuration has been updated
        /// </summary>
        public async Task<bool> HasConfigurationUpdatedAsync(string lastConfigurationVersion)
        {
            try
            {
                var configurations = await GetFIMConfigurationsAsync();
                
                // Simple version check - in production this would be more sophisticated
                var currentVersion = string.Join("|", configurations.Select(c => $"{c.Id}:{c.Name}"));
                return currentVersion != lastConfigurationVersion;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error checking FIM configuration updates");
                return false;
            }
        }
    }
} 
