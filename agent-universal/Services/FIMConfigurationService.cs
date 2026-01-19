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

        public FIMConfigurationService(
            ILogger<FIMConfigurationService> logger, 
            IConfiguration configuration,
            HttpClient httpClient)
        {
            _logger = logger;
            _configuration = configuration;
            _httpClient = httpClient;
        }

        /// <summary>
        /// Gets the current API key from configuration (updated after registration)
        /// </summary>
        private string GetApiKey()
        {
            return _configuration["Agent:ApiKey"] ?? "";
        }

        /// <summary>
        /// Gets the current agent ID from configuration
        /// </summary>
        private string GetAgentId()
        {
            return _configuration["Agent:Id"] ?? Environment.MachineName;
        }

        /// <summary>
        /// Gets the backend URL from configuration
        /// </summary>
        private string GetBackendUrl()
        {
            return _configuration["Agent:ManagerUrl"] ?? "";
        }

        /// <summary>
        /// Creates an HTTP request message with authentication headers
        /// </summary>
        private HttpRequestMessage CreateAuthenticatedRequest(HttpMethod method, string url)
        {
            var request = new HttpRequestMessage(method, url);
            var apiKey = GetApiKey();
            if (!string.IsNullOrEmpty(apiKey))
            {
                request.Headers.Add("X-API-Key", apiKey);
            }
            return request;
        }

        /// <summary>
        /// Fetch FIM configurations for this agent from backend
        /// </summary>
        public async Task<List<FIMConfigurationDto>> GetFIMConfigurationsAsync()
        {
            try
            {
                var backendUrl = GetBackendUrl();
                if (string.IsNullOrEmpty(backendUrl))
                {
                    _logger.LogWarning("Backend URL not configured - cannot fetch FIM configurations");
                    return new List<FIMConfigurationDto>();
                }

                var agentId = GetAgentId();
                var url = $"{backendUrl}/api/fim/configurations/agent/{agentId}";
                _logger.LogDebug("Fetching FIM configurations from: {Url}", url);

                var request = CreateAuthenticatedRequest(HttpMethod.Get, url);
                var response = await _httpClient.SendAsync(request);
                
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
                var backendUrl = GetBackendUrl();
                if (string.IsNullOrEmpty(backendUrl))
                {
                    _logger.LogWarning("Backend URL not configured - cannot send FIM event");
                    return false;
                }
                var url = $"{backendUrl}/api/fim/events";
                var json = JsonSerializer.Serialize(fimEvent);
                var content = new StringContent(json, System.Text.Encoding.UTF8, "application/json");

                var request = CreateAuthenticatedRequest(HttpMethod.Post, url);
                request.Content = content;
                var response = await _httpClient.SendAsync(request);
                
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
                var backendUrl = GetBackendUrl();
                if (string.IsNullOrEmpty(backendUrl))
                {
                    _logger.LogWarning("Backend URL not configured - cannot fetch FIM templates");
                    return new List<FIMTemplateDto>();
                }
                var url = string.IsNullOrEmpty(operatingSystem) 
                    ? $"{backendUrl}/api/fim/templates"
                    : $"{backendUrl}/api/fim/templates/os/{operatingSystem}";

                var request = CreateAuthenticatedRequest(HttpMethod.Get, url);
                var response = await _httpClient.SendAsync(request);
                
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
