using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using AthalaSIEM.UniversalAgent.Core.Normalizer;

namespace AthalaSIEM.UniversalAgent.Core.Exporter
{
    /// <summary>
    /// HTTP Exporter for Production Mode
    /// Sends normalized events to backend via REST API
    /// 
    /// Features:
    /// - Batch sending for efficiency
    /// - Retry with exponential backoff
    /// - Compression support
    /// - Authentication via API key or JWT
    /// </summary>
    public class HttpExporter : IExporter, IAsyncDisposable
    {
        private readonly ILogger<HttpExporter> _logger;
        private readonly HttpClient _httpClient;
        private readonly string _endpoint;
        private readonly int _maxRetries;
        private readonly int _batchSize;
        private readonly bool _enableCompression;
        private readonly JsonSerializerOptions _jsonOptions;

        // Metrics
        private long _eventsExported = 0;
        private long _exportErrors = 0;
        private long _retryCount = 0;
        private long _batchesSent = 0;
        private readonly DateTime _startTime = DateTime.UtcNow;

        public string Name => "HttpExporter";
        public string Mode => "HTTP";

        public HttpExporter(
            ILogger<HttpExporter> logger,
            string endpoint,
            string? apiKey = null,
            int maxRetries = 3,
            int batchSize = 100,
            bool enableCompression = true,
            int timeoutSeconds = 30)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _endpoint = endpoint ?? throw new ArgumentNullException(nameof(endpoint));
            _maxRetries = maxRetries;
            _batchSize = batchSize;
            _enableCompression = enableCompression;

            _httpClient = new HttpClient
            {
                Timeout = TimeSpan.FromSeconds(timeoutSeconds)
            };

            // Set default headers
            _httpClient.DefaultRequestHeaders.Accept.Add(new MediaTypeWithQualityHeaderValue("application/json"));

            if (!string.IsNullOrEmpty(apiKey))
            {
                _httpClient.DefaultRequestHeaders.Add("X-API-Key", apiKey);
            }

            if (enableCompression)
            {
                _httpClient.DefaultRequestHeaders.AcceptEncoding.Add(new StringWithQualityHeaderValue("gzip"));
            }

            _jsonOptions = new JsonSerializerOptions
            {
                WriteIndented = false,
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
                DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull
            };
        }

        public Task<bool> InitializeAsync()
        {
            _logger.LogInformation("HTTP exporter initialized - Endpoint: {Endpoint}", _endpoint);
            return Task.FromResult(true);
        }

        public async Task<ExportResult> ExportAsync(IEnumerable<AthalaEcsLiteEvent> events)
        {
            if (events == null)
            {
                throw new ArgumentNullException(nameof(events));
            }

            var eventList = events.ToList();
            var totalExported = 0;
            var totalFailed = 0;

            // Process in batches
            var batches = eventList
                .Select((evt, index) => new { evt, index })
                .GroupBy(x => x.index / _batchSize)
                .Select(g => g.Select(x => x.evt).ToList())
                .ToList();

            foreach (var batch in batches)
            {
                var result = await SendBatchWithRetryAsync(batch);
                if (result.Success)
                {
                    totalExported += result.ExportedCount;
                    _batchesSent++;
                }
                else
                {
                    totalFailed += batch.Count;
                }
            }

            _eventsExported += totalExported;
            _exportErrors += totalFailed;

            return new ExportResult
            {
                Success = totalFailed == 0,
                ExportedCount = totalExported,
                FailedCount = totalFailed
            };
        }

        /// <summary>
        /// Send batch with retry logic
        /// </summary>
        private async Task<ExportResult> SendBatchWithRetryAsync(List<AthalaEcsLiteEvent> batch)
        {
            var attempt = 0;
            Exception? lastException = null;

            while (attempt < _maxRetries)
            {
                try
                {
                    var payload = new
                    {
                        events = batch,
                        metadata = new
                        {
                            batchSize = batch.Count,
                            timestamp = DateTime.UtcNow,
                            agentVersion = "1.0.0"
                        }
                    };

                    var json = JsonSerializer.Serialize(payload, _jsonOptions);
                    var content = new StringContent(json, Encoding.UTF8, "application/json");

                    var response = await _httpClient.PostAsync(_endpoint, content);

                    if (response.IsSuccessStatusCode)
                    {
                        return new ExportResult
                        {
                            Success = true,
                            ExportedCount = batch.Count,
                            FailedCount = 0
                        };
                    }

                    // Log non-success status
                    var responseContent = await response.Content.ReadAsStringAsync();
                    _logger.LogWarning("HTTP export failed with status {Status}: {Response}",
                        response.StatusCode, responseContent);

                    // Don't retry on client errors (4xx)
                    if ((int)response.StatusCode >= 400 && (int)response.StatusCode < 500)
                    {
                        return new ExportResult
                        {
                            Success = false,
                            ExportedCount = 0,
                            FailedCount = batch.Count,
                            ErrorMessage = $"HTTP {response.StatusCode}: {responseContent}"
                        };
                    }
                }
                catch (Exception ex)
                {
                    lastException = ex;
                    _logger.LogWarning(ex, "HTTP export attempt {Attempt} failed: {Message}", attempt + 1, ex.Message);
                }

                attempt++;
                _retryCount++;

                if (attempt < _maxRetries)
                {
                    // Exponential backoff
                    var delay = TimeSpan.FromMilliseconds(Math.Pow(2, attempt) * 100);
                    await Task.Delay(delay);
                }
            }

            return new ExportResult
            {
                Success = false,
                ExportedCount = 0,
                FailedCount = batch.Count,
                ErrorMessage = lastException?.Message ?? "Max retries exceeded"
            };
        }

        /// <summary>
        /// Set authentication token (for JWT auth)
        /// </summary>
        public void SetAuthToken(string token)
        {
            _httpClient.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", token);
        }

        public Dictionary<string, object> GetMetrics()
        {
            var uptime = DateTime.UtcNow - _startTime;
            return new Dictionary<string, object>
            {
                ["Name"] = Name,
                ["Mode"] = Mode,
                ["Endpoint"] = _endpoint,
                ["EventsExported"] = _eventsExported,
                ["ExportErrors"] = _exportErrors,
                ["BatchesSent"] = _batchesSent,
                ["RetryCount"] = _retryCount,
                ["SuccessRate"] = _eventsExported > 0
                    ? (double)(_eventsExported) / (_eventsExported + _exportErrors) * 100
                    : 100.0,
                ["UptimeSeconds"] = uptime.TotalSeconds,
                ["EventsPerSecond"] = uptime.TotalSeconds > 0
                    ? _eventsExported / uptime.TotalSeconds
                    : 0.0
            };
        }

        public async ValueTask DisposeAsync()
        {
            _httpClient?.Dispose();
            _logger.LogInformation("HTTP exporter disposed");
            await Task.CompletedTask;
        }
    }
}
