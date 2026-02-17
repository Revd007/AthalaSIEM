using AthalaSIEM.Agent.Models;
using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.Data;
using Microsoft.Data.SqlClient;
using System.Threading;
using System.Threading.Tasks;
using System.Text.Json;
using System.Linq;
using System.Collections.Concurrent;
using MySql.Data.MySqlClient;
using Npgsql;
using MongoDB.Driver;
using System.Text.RegularExpressions;

namespace AthalaSIEM.Agent.Collectors
{
    /// <summary>
    /// Database collector for monitoring database activities and security events
    /// </summary>
    public class DatabaseCollector : ILogCollector
    {
        private readonly ILogger<DatabaseCollector> _logger;
        private readonly ILogNormalizer _normalizer;
        private bool _isRunning;
        private bool _isPaused;
        private string _errorMessage = string.Empty;
        private CollectorSettings _settings = new();
        
        // Configuration
        private List<DatabaseConnection> _connections = new();
        private bool _enableQueryMonitoring = true;
        private bool _enableAccessMonitoring = true;
        private bool _enableSchemaChangeMonitoring = true;
        private bool _enablePerformanceMonitoring = false;
        private int _queryThresholdMs = 5000;
        private int _collectionIntervalSeconds = 30;
        private int _maxEventsPerBatch = 100;
        private readonly Queue<NormalizedLogEntry> _eventBuffer = new();
        private Timer? _collectionTimer;
        private CancellationTokenSource? _cancellationTokenSource;
        
        /// <summary>
        /// Event raised when a log is collected
        /// </summary>
        public event EventHandler<NormalizedLogEntry>? LogCollected;

        /// <summary>
        /// Gets the type of the collector
        /// </summary>
        public string CollectorType => "Database";

        /// <summary>
        /// Gets the status of the collector
        /// </summary>
        public CollectorStatus Status => _isRunning ? (_isPaused ? CollectorStatus.Paused : CollectorStatus.Running) : 
                                        (!string.IsNullOrEmpty(_errorMessage) ? CollectorStatus.Error : CollectorStatus.Stopped);

        /// <summary>
        /// Gets the error message if the collector is in an error state
        /// </summary>
        public string ErrorMessage => _errorMessage;

        /// <summary>
        /// Gets a value indicating whether the collector is running
        /// </summary>
        public bool IsRunning => _isRunning;

        /// <summary>
        /// Gets a value indicating whether the collector is paused
        /// </summary>
        public bool IsPaused => _isPaused;

        /// <summary>
        /// Gets the collector settings
        /// </summary>
        public CollectorSettings Settings => _settings;

        public DatabaseCollector(ILogger<DatabaseCollector> logger, ILogNormalizer normalizer)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _normalizer = normalizer ?? throw new ArgumentNullException(nameof(normalizer));
        }

        /// <summary>
        /// Initializes the collector with the specified settings
        /// </summary>
        /// <param name="settings">Collector settings</param>
        /// <returns>True if initialization was successful, otherwise false</returns>
        public bool Initialize(CollectorSettings settings)
        {
            _settings = settings ?? throw new ArgumentNullException(nameof(settings));
            _logger.LogInformation("Initializing Database Collector");

            try
            {
                ParseSettings();
                ValidateConnections();
                _logger.LogInformation("Database Collector initialized - Connections: {Count}, Query monitoring: {QueryMon}, Access monitoring: {AccessMon}", 
                    _connections.Count, _enableQueryMonitoring, _enableAccessMonitoring);
                return true;
            }
            catch (Exception ex)
            {
                _errorMessage = ex.Message;
                _logger.LogError(ex, "Failed to initialize Database Collector");
                return false;
            }
        }

        /// <summary>
        /// Starts the collector
        /// </summary>
        public async Task StartAsync()
        {
            await Task.CompletedTask;
            if (_isRunning) return;

            try
            {
                _logger.LogInformation("Starting Database Collector");
                _cancellationTokenSource = new CancellationTokenSource();

                // Start collection timer
                _collectionTimer = new Timer(async _ => await CollectDatabaseEventsAsync(), 
                    null, TimeSpan.Zero, TimeSpan.FromSeconds(_collectionIntervalSeconds));

                _isRunning = true;
                _isPaused = false;
                _errorMessage = string.Empty;

                _logger.LogInformation("Database Collector started successfully");
            }
            catch (Exception ex)
            {
                _errorMessage = ex.Message;
                _logger.LogError(ex, "Failed to start Database Collector");
                throw;
            }
        }

        /// <summary>
        /// Stops the collector
        /// </summary>
        public async Task StopAsync()
        {
            await Task.CompletedTask;
            if (!_isRunning) return;

            try
            {
                _logger.LogInformation("Stopping Database Collector");
                
                _isRunning = false;
                _cancellationTokenSource?.Cancel();
                
                _collectionTimer?.Dispose();
                
                // Process remaining events
                ProcessEventBuffer();
                
                _logger.LogInformation("Database Collector stopped");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error stopping Database Collector");
            }
        }

        /// <summary>
        /// Pauses the collector
        /// </summary>
        public Task PauseAsync()
        {
            _isPaused = true;
            _logger.LogInformation("Database Collector paused");
            return Task.CompletedTask;
        }

        /// <summary>
        /// Resumes the collector
        /// </summary>
        public Task ResumeAsync()
        {
            _isPaused = false;
            _logger.LogInformation("Database Collector resumed");
            return Task.CompletedTask;
        }

        /// <summary>
        /// Collects logs on demand
        /// </summary>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>The number of logs collected</returns>
        public async Task<int> CollectLogsAsync(CancellationToken cancellationToken)
        {
            if (_isPaused || !_isRunning)
                return 0;

            int collectedCount = 0;

            try
            {
                await CollectDatabaseEventsAsync();
                collectedCount = _eventBuffer.Count;
                ProcessEventBuffer();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error collecting database logs");
                _errorMessage = ex.Message;
            }

            return collectedCount;
        }

        /// <summary>
        /// Gets collector statistics
        /// </summary>
        public CollectorStats GetStats()
        {
            return new CollectorStats
            {
                IsRunning = _isRunning,
                IsPaused = _isPaused,
                LastError = _errorMessage,
                FilesMonitored = _connections.Count,
                WatchersActive = _isRunning ? _connections.Count : 0
            };
        }

        private void ParseSettings()
        {
            if (_settings.Properties.ContainsKey("DatabaseConnections"))
            {
                var connectionsJson = _settings.Properties["DatabaseConnections"];
                _connections = JsonSerializer.Deserialize<List<DatabaseConnection>>(connectionsJson) ?? new List<DatabaseConnection>();
            }

            if (_settings.Properties.ContainsKey("EnableQueryMonitoring"))
            {
                bool.TryParse(_settings.Properties["EnableQueryMonitoring"], out _enableQueryMonitoring);
            }

            if (_settings.Properties.ContainsKey("EnableAccessMonitoring"))
            {
                bool.TryParse(_settings.Properties["EnableAccessMonitoring"], out _enableAccessMonitoring);
            }

            if (_settings.Properties.ContainsKey("EnableSchemaChangeMonitoring"))
            {
                bool.TryParse(_settings.Properties["EnableSchemaChangeMonitoring"], out _enableSchemaChangeMonitoring);
            }

            if (_settings.Properties.ContainsKey("EnablePerformanceMonitoring"))
            {
                bool.TryParse(_settings.Properties["EnablePerformanceMonitoring"], out _enablePerformanceMonitoring);
            }

            if (_settings.Properties.ContainsKey("QueryThresholdMs"))
            {
                int.TryParse(_settings.Properties["QueryThresholdMs"], out _queryThresholdMs);
            }

            if (_settings.Properties.ContainsKey("CollectionIntervalSeconds"))
            {
                int.TryParse(_settings.Properties["CollectionIntervalSeconds"], out _collectionIntervalSeconds);
            }

            if (_settings.Properties.ContainsKey("MaxEventsPerBatch"))
            {
                int.TryParse(_settings.Properties["MaxEventsPerBatch"], out _maxEventsPerBatch);
            }
        }

        private void ValidateConnections()
        {
            foreach (var connection in _connections)
            {
                try
                {
                    TestConnection(connection);
                    _logger.LogInformation("Validated database connection: {DatabaseType} - {DatabaseName}", 
                        connection.DatabaseType, connection.DatabaseName);
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Failed to validate database connection: {DatabaseType} - {DatabaseName}", 
                        connection.DatabaseType, connection.DatabaseName);
                }
            }
        }

        private void TestConnection(DatabaseConnection connection)
        {
            switch (connection.DatabaseType.ToLowerInvariant())
            {
                case "sqlserver":
                    using (var conn = new SqlConnection(connection.ConnectionString))
                    {
                        conn.Open();
                    }
                    break;
                case "mysql":
                    using (var conn = new MySqlConnection(connection.ConnectionString))
                    {
                        conn.Open();
                    }
                    break;
                case "postgresql":
                    using (var conn = new NpgsqlConnection(connection.ConnectionString))
                    {
                        conn.Open();
                    }
                    break;
                case "mongodb":
                    var client = new MongoClient(connection.ConnectionString);
                    var database = client.GetDatabase(connection.DatabaseName);
                    _ = database.RunCommandAsync((Command<MongoDB.Bson.BsonDocument>)"{ping:1}").Result;
                    break;
                default:
                    throw new NotSupportedException($"Database type {connection.DatabaseType} is not supported");
            }
        }

        private async Task CollectDatabaseEventsAsync()
        {
            if (_isPaused) return;

            foreach (var connection in _connections)
            {
                try
                {
                    await CollectFromDatabaseAsync(connection);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error collecting from database: {DatabaseName}", connection.DatabaseName);
                }
            }

            ProcessEventBuffer();
        }

        private async Task CollectFromDatabaseAsync(DatabaseConnection connection)
        {
            switch (connection.DatabaseType.ToLowerInvariant())
            {
                case "sqlserver":
                    await CollectFromSqlServerAsync(connection);
                    break;
                case "mysql":
                    await CollectFromMySqlAsync(connection);
                    break;
                case "postgresql":
                    await CollectFromPostgreSqlAsync(connection);
                    break;
                case "mongodb":
                    await CollectFromMongoDbAsync(connection);
                    break;
            }
        }

        private async Task CollectFromSqlServerAsync(DatabaseConnection connection)
        {
            using var conn = new SqlConnection(connection.ConnectionString);
            await conn.OpenAsync();

            // Collect query events
            if (_enableQueryMonitoring)
            {
                await CollectSqlServerQueryEventsAsync(conn, connection);
            }

            // Collect access events
            if (_enableAccessMonitoring)
            {
                await CollectSqlServerAccessEventsAsync(conn, connection);
            }

            // Collect schema changes
            if (_enableSchemaChangeMonitoring)
            {
                await CollectSqlServerSchemaEventsAsync(conn, connection);
            }
        }

        private async Task CollectSqlServerQueryEventsAsync(SqlConnection conn, DatabaseConnection connection)
        {
            var query = @"
                SELECT TOP 100
                    session_id,
                    start_time,
                    command,
                    database_name,
                    user_name,
                    host_name,
                    program_name,
                    text as query_text,
                    cpu_time,
                    total_elapsed_time,
                    reads,
                    writes
                FROM sys.dm_exec_requests r
                CROSS APPLY sys.dm_exec_sql_text(r.sql_handle)
                WHERE total_elapsed_time > @threshold
                ORDER BY start_time DESC";

            using var cmd = new SqlCommand(query, conn);
            cmd.Parameters.AddWithValue("@threshold", _queryThresholdMs * 1000); // Convert to microseconds

            using var reader = await cmd.ExecuteReaderAsync();
            while (await reader.ReadAsync())
            {
                var logEntry = CreateDatabaseLogEntry(
                    connection,
                    "SlowQuery",
                    $"Slow query detected: {reader["query_text"]}",
                    new Dictionary<string, object>
                    {
                        ["session_id"] = reader["session_id"],
                        ["database_name"] = reader["database_name"],
                        ["user_name"] = reader["user_name"],
                        ["host_name"] = reader["host_name"],
                        ["program_name"] = reader["program_name"],
                        ["cpu_time"] = reader["cpu_time"],
                        ["elapsed_time"] = reader["total_elapsed_time"],
                        ["reads"] = reader["reads"],
                        ["writes"] = reader["writes"],
                        ["query_text"] = reader["query_text"]
                    }
                );

                lock (_eventBuffer)
                {
                    if (_eventBuffer.Count < _maxEventsPerBatch * 10)
                    {
                        _eventBuffer.Enqueue(logEntry);
                    }
                }
            }
        }

        private async Task CollectSqlServerAccessEventsAsync(SqlConnection conn, DatabaseConnection connection)
        {
            await Task.CompletedTask;
            // Implementation for collecting access events from SQL Server audit logs
            // This would typically require SQL Server Audit to be configured
            _logger.LogDebug("Collecting SQL Server access events for {DatabaseName}", connection.DatabaseName);
        }

        private async Task CollectSqlServerSchemaEventsAsync(SqlConnection conn, DatabaseConnection connection)
        {
            await Task.CompletedTask;
            // Implementation for collecting schema change events
            // This would typically use DDL triggers or event notifications
            _logger.LogDebug("Collecting SQL Server schema events for {DatabaseName}", connection.DatabaseName);
        }

        private async Task CollectFromMySqlAsync(DatabaseConnection connection)
        {
            using var conn = new MySqlConnection(connection.ConnectionString);
            await conn.OpenAsync();

            if (_enableQueryMonitoring)
            {
                await CollectMySqlSlowQueriesAsync(conn, connection);
            }

            if (_enableAccessMonitoring)
            {
                await CollectMySqlGeneralLogAsync(conn, connection);
            }
        }

        private async Task CollectMySqlSlowQueriesAsync(MySqlConnection conn, DatabaseConnection connection)
        {
            var query = @"
                SELECT start_time, user_host, query_time, lock_time, rows_sent, rows_examined, db, sql_text
                FROM mysql.slow_log 
                WHERE start_time > DATE_SUB(NOW(), INTERVAL 1 HOUR)
                ORDER BY start_time DESC
                LIMIT 100";

            try
            {
                using var cmd = new MySqlCommand(query, conn);
                using var reader = await cmd.ExecuteReaderAsync();
                
                while (await reader.ReadAsync())
                {
                    var logEntry = CreateDatabaseLogEntry(
                        connection,
                        "SlowQuery",
                        $"MySQL slow query: {reader["sql_text"]}",
                        new Dictionary<string, object>
                        {
                            ["start_time"] = reader["start_time"],
                            ["user_host"] = reader["user_host"],
                            ["query_time"] = reader["query_time"],
                            ["lock_time"] = reader["lock_time"],
                            ["rows_sent"] = reader["rows_sent"],
                            ["rows_examined"] = reader["rows_examined"],
                            ["database"] = reader["db"],
                            ["sql_text"] = reader["sql_text"]
                        }
                    );

                    lock (_eventBuffer)
                    {
                        if (_eventBuffer.Count < _maxEventsPerBatch * 10)
                        {
                            _eventBuffer.Enqueue(logEntry);
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Could not access MySQL slow log for {DatabaseName}", connection.DatabaseName);
            }
        }

        private async Task CollectMySqlGeneralLogAsync(MySqlConnection conn, DatabaseConnection connection)
        {
            await Task.CompletedTask;
            // Implementation for MySQL general log monitoring
            _logger.LogDebug("Collecting MySQL general log for {DatabaseName}", connection.DatabaseName);
        }

        private async Task CollectFromPostgreSqlAsync(DatabaseConnection connection)
        {
            using var conn = new NpgsqlConnection(connection.ConnectionString);
            await conn.OpenAsync();

            if (_enableQueryMonitoring)
            {
                await CollectPostgreSqlSlowQueriesAsync(conn, connection);
            }

            if (_enableAccessMonitoring)
            {
                await CollectPostgreSqlActivityAsync(conn, connection);
            }
        }

        private async Task CollectPostgreSqlSlowQueriesAsync(NpgsqlConnection conn, DatabaseConnection connection)
        {
            var query = @"
                SELECT query, calls, total_time, min_time, max_time, mean_time, rows, 
                       100.0 * shared_blks_hit / nullif(shared_blks_hit + shared_blks_read, 0) AS hit_percent
                FROM pg_stat_statements 
                WHERE mean_time > @threshold
                ORDER BY total_time DESC
                LIMIT 100";

            try
            {
                using var cmd = new NpgsqlCommand(query, conn);
                cmd.Parameters.AddWithValue("@threshold", _queryThresholdMs);

                using var reader = await cmd.ExecuteReaderAsync();
                
                while (await reader.ReadAsync())
                {
                    var logEntry = CreateDatabaseLogEntry(
                        connection,
                        "SlowQuery",
                        $"PostgreSQL slow query: {reader["query"]}",
                        new Dictionary<string, object>
                        {
                            ["query"] = reader["query"],
                            ["calls"] = reader["calls"],
                            ["total_time"] = reader["total_time"],
                            ["min_time"] = reader["min_time"],
                            ["max_time"] = reader["max_time"],
                            ["mean_time"] = reader["mean_time"],
                            ["rows"] = reader["rows"],
                            ["hit_percent"] = reader["hit_percent"]
                        }
                    );

                    lock (_eventBuffer)
                    {
                        if (_eventBuffer.Count < _maxEventsPerBatch * 10)
                        {
                            _eventBuffer.Enqueue(logEntry);
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Could not access pg_stat_statements for {DatabaseName}", connection.DatabaseName);
            }
        }

        private async Task CollectPostgreSqlActivityAsync(NpgsqlConnection conn, DatabaseConnection connection)
        {
            var query = @"
                SELECT pid, usename, application_name, client_addr, backend_start, query_start, 
                       state, query
                FROM pg_stat_activity 
                WHERE state = 'active' AND query NOT LIKE '%pg_stat_activity%'
                ORDER BY query_start DESC";

            using var cmd = new NpgsqlCommand(query, conn);
            using var reader = await cmd.ExecuteReaderAsync();
            
            while (await reader.ReadAsync())
            {
                var logEntry = CreateDatabaseLogEntry(
                    connection,
                    "ActiveQuery",
                    $"PostgreSQL active query: {reader["query"]}",
                    new Dictionary<string, object>
                    {
                        ["pid"] = reader["pid"],
                        ["username"] = reader["usename"],
                        ["application_name"] = reader["application_name"],
                        ["client_addr"] = reader["client_addr"],
                        ["backend_start"] = reader["backend_start"],
                        ["query_start"] = reader["query_start"],
                        ["state"] = reader["state"],
                        ["query"] = reader["query"]
                    }
                );

                lock (_eventBuffer)
                {
                    if (_eventBuffer.Count < _maxEventsPerBatch * 10)
                    {
                        _eventBuffer.Enqueue(logEntry);
                    }
                }
            }
        }

        private async Task CollectFromMongoDbAsync(DatabaseConnection connection)
        {
            var client = new MongoClient(connection.ConnectionString);
            var database = client.GetDatabase(connection.DatabaseName);

            if (_enableQueryMonitoring)
            {
                await CollectMongoDbSlowOperationsAsync(database, connection);
            }

            if (_enableAccessMonitoring)
            {
                await CollectMongoDbOperationsAsync(database, connection);
            }
        }

        private async Task CollectMongoDbSlowOperationsAsync(IMongoDatabase database, DatabaseConnection connection)
        {
            try
            {
                // Get profiling data for slow operations
                var profilingCollection = database.GetCollection<MongoDB.Bson.BsonDocument>("system.profile");
                
                var filter = new MongoDB.Bson.BsonDocument
                {
                    ["ts"] = new MongoDB.Bson.BsonDocument("$gte", DateTime.UtcNow.AddHours(-1)),
                    ["millis"] = new MongoDB.Bson.BsonDocument("$gte", _queryThresholdMs)
                };

                var cursor = await profilingCollection.FindAsync(filter);
                var documents = await cursor.ToListAsync();

                foreach (var doc in documents)
                {
                    var logEntry = CreateDatabaseLogEntry(
                        connection,
                        "SlowOperation",
                        $"MongoDB slow operation: {doc.GetValue("command", "")}",
                        new Dictionary<string, object>
                        {
                            ["timestamp"] = doc.GetValue("ts", DateTime.UtcNow),
                            ["operation"] = doc.GetValue("op", ""),
                            ["namespace"] = doc.GetValue("ns", ""),
                            ["duration_ms"] = doc.GetValue("millis", 0),
                            ["command"] = doc.GetValue("command", "")?.ToString() ?? "",
                            ["user"] = doc.GetValue("user", ""),
                            ["client"] = doc.GetValue("client", "")
                        }
                    );

                    lock (_eventBuffer)
                    {
                        if (_eventBuffer.Count < _maxEventsPerBatch * 10)
                        {
                            _eventBuffer.Enqueue(logEntry);
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Could not access MongoDB profiling data for {DatabaseName}", connection.DatabaseName);
            }
        }

        private async Task CollectMongoDbOperationsAsync(IMongoDatabase database, DatabaseConnection connection)
        {
            await Task.CompletedTask;
            // Implementation for MongoDB current operations monitoring
            _logger.LogDebug("Collecting MongoDB operations for {DatabaseName}", connection.DatabaseName);
        }

        private NormalizedLogEntry CreateDatabaseLogEntry(DatabaseConnection connection, string eventType, string message, Dictionary<string, object> details)
        {
            var severity = DetermineSeverity(eventType, details);
            var threatIndicators = AnalyzeThreatIndicators(eventType, message, details);

            return new NormalizedLogEntry
            {
                Id = Guid.NewGuid().ToString(),
                Timestamp = DateTime.UtcNow,
                Level = severity,
                Source = $"Database/{connection.DatabaseType}",
                Category = "DatabaseSecurity",
                EventId = eventType,
                Message = message,
                Details = JsonSerializer.Serialize(new
                {
                    database_type = connection.DatabaseType,
                    database_name = connection.DatabaseName,
                    server = connection.Server,
                    event_type = eventType,
                    details = details,
                    threat_indicators = threatIndicators
                }),
                Tags = CreateTags(eventType, connection.DatabaseType, threatIndicators),
                Severity = severity
            };
        }

        private string DetermineSeverity(string eventType, Dictionary<string, object> details)
        {
            switch (eventType)
            {
                case "SlowQuery":
                    if (details.ContainsKey("elapsed_time") && long.TryParse(details["elapsed_time"].ToString(), out var elapsedTime))
                    {
                        return elapsedTime > _queryThresholdMs * 10 ? "High" : "Medium";
                    }
                    return "Medium";
                
                case "FailedLogin":
                case "UnauthorizedAccess":
                    return "High";
                
                case "SchemaChange":
                case "PrivilegeEscalation":
                    return "Critical";
                
                case "ActiveQuery":
                case "SlowOperation":
                    return "Low";
                
                default:
                    return "Information";
            }
        }

        private List<string> AnalyzeThreatIndicators(string eventType, string message, Dictionary<string, object> details)
        {
            var indicators = new List<string>();

            // SQL Injection patterns
            if (details.ContainsKey("query_text") || details.ContainsKey("sql_text") || details.ContainsKey("query"))
            {
                var query = details.GetValueOrDefault("query_text")?.ToString() ??
                           details.GetValueOrDefault("sql_text")?.ToString() ??
                           details.GetValueOrDefault("query")?.ToString() ?? "";

                var sqlInjectionPatterns = new[]
                {
                    "union select", "or 1=1", "'; drop", "exec xp_", "script>", "javascript:",
                    "waitfor delay", "benchmark(", "sleep(", "pg_sleep("
                };

                foreach (var pattern in sqlInjectionPatterns)
                {
                    if (query.ToLowerInvariant().Contains(pattern))
                    {
                        indicators.Add("sql_injection_attempt");
                        break;
                    }
                }

                // Bulk operations
                if (query.ToLowerInvariant().Contains("select") && query.Contains("*"))
                {
                    indicators.Add("bulk_data_access");
                }

                // Administrative operations
                var adminPatterns = new[] { "drop table", "drop database", "truncate", "delete from", "alter table" };
                foreach (var pattern in adminPatterns)
                {
                    if (query.ToLowerInvariant().Contains(pattern))
                    {
                        indicators.Add("administrative_operation");
                        break;
                    }
                }
            }

            // After-hours access
            var currentHour = DateTime.Now.Hour;
            if (currentHour < 6 || currentHour > 22)
            {
                indicators.Add("after_hours_access");
            }

            // Excessive resource usage
            if (eventType == "SlowQuery" && details.ContainsKey("elapsed_time"))
            {
                if (long.TryParse(details["elapsed_time"].ToString(), out var elapsedTime) && elapsedTime > _queryThresholdMs * 50)
                {
                    indicators.Add("excessive_resource_usage");
                }
            }

            return indicators;
        }

        private List<string> CreateTags(string eventType, string databaseType, List<string> threatIndicators)
        {
            var tags = new List<string> { "database", databaseType.ToLower(), eventType.ToLower() };
            
            if (threatIndicators.Any())
            {
                tags.Add("threat_detected");
                tags.AddRange(threatIndicators);
            }
            
            return tags;
        }

        private void ProcessEventBuffer()
        {
            var eventsToProcess = new List<NormalizedLogEntry>();
            
            lock (_eventBuffer)
            {
                var count = Math.Min(_maxEventsPerBatch, _eventBuffer.Count);
                for (int i = 0; i < count; i++)
                {
                    if (_eventBuffer.Count > 0)
                    {
                        eventsToProcess.Add(_eventBuffer.Dequeue());
                    }
                }
            }
            
            foreach (var eventEntry in eventsToProcess)
            {
                LogCollected?.Invoke(this, eventEntry);
            }
            
            if (eventsToProcess.Count > 0)
            {
                _logger.LogInformation("Processed {Count} database events", eventsToProcess.Count);
            }
        }

        /// <summary>
        /// Disposes resources
        /// </summary>
        public void Dispose()
        {
            StopAsync().Wait();
            _collectionTimer?.Dispose();
            _cancellationTokenSource?.Dispose();
        }
    }

    /// <summary>
    /// Database connection configuration
    /// </summary>
    public class DatabaseConnection
    {
        public string DatabaseType { get; set; } = string.Empty; // SqlServer, MySQL, PostgreSQL, MongoDB
        public string ConnectionString { get; set; } = string.Empty;
        public string DatabaseName { get; set; } = string.Empty;
        public string Server { get; set; } = string.Empty;
        public bool EnableMonitoring { get; set; } = true;
        public List<string> MonitoredTables { get; set; } = new();
        public Dictionary<string, string> CustomQueries { get; set; } = new();
    }
} 