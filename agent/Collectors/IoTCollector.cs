using AthalaSIEM.Agent.Models;
using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using System.Text.RegularExpressions;

namespace AthalaSIEM.Agent.Collectors
{
    /// <summary>
    /// IoT Collector for industrial devices, sensors, and SCADA systems
    /// </summary>
    public class IoTCollector : ILogCollector
    {
        private readonly ILogger<IoTCollector> _logger;
        private readonly ILogNormalizer _normalizer;
        private bool _isRunning;
        private bool _isPaused;
        private string _errorMessage = string.Empty;
        private CollectorSettings _settings = new();
        
        // Configuration
        private bool _enableModbusLogs = false;
        private bool _enableMqttLogs = false;
        private bool _enableOpcUaLogs = false;
        private bool _enableScadaLogs = false;
        private bool _enableSensorLogs = true;
        private int _collectionInterval = 60;
        
        // Protocol-specific settings
        private string _modbusPort = "502";
        private string _mqttBroker = "localhost:1883";
        private string _opcUaEndpoint = "opc.tcp://localhost:4840";
        private string _scadaHost = "localhost";
        private List<string> _sensorEndpoints = new();
        private List<string> _deviceTypes = new();
        
        // Network listeners
        private UdpClient? _udpListener;
        private TcpListener? _tcpListener;
        private int _udpPort = 5140; // Custom IoT UDP port
        private int _tcpPort = 5141; // Custom IoT TCP port
        
        private CancellationTokenSource? _cancellationTokenSource;

        public event EventHandler<NormalizedLogEntry>? LogCollected;
        public string CollectorType => "IoT";
        public CollectorStatus Status => _isRunning ? (_isPaused ? CollectorStatus.Paused : CollectorStatus.Running) : 
                                        (!string.IsNullOrEmpty(_errorMessage) ? CollectorStatus.Error : CollectorStatus.Stopped);
        public string ErrorMessage => _errorMessage;
        public bool IsRunning => _isRunning;
        public bool IsPaused => _isPaused;
        public CollectorSettings Settings => _settings;

        public IoTCollector(ILogger<IoTCollector> logger, ILogNormalizer normalizer)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _normalizer = normalizer ?? throw new ArgumentNullException(nameof(normalizer));
        }

        public bool Initialize(CollectorSettings settings)
        {
            _settings = settings ?? throw new ArgumentNullException(nameof(settings));
            _logger.LogInformation("Initializing IoT Collector");

            try
            {
                ParseSettings();
                _logger.LogInformation("IoT Collector initialized - Modbus: {Modbus}, MQTT: {Mqtt}, OPC-UA: {OpcUa}, SCADA: {Scada}", 
                    _enableModbusLogs, _enableMqttLogs, _enableOpcUaLogs, _enableScadaLogs);
                return true;
            }
            catch (Exception ex)
            {
                _errorMessage = ex.Message;
                _logger.LogError(ex, "Failed to initialize IoT Collector");
                return false;
            }
        }

        public async Task StartAsync()
        {
            if (_isRunning) return;

            try
            {
                _logger.LogInformation("Starting IoT Collector");
                _cancellationTokenSource = new CancellationTokenSource();

                // Start UDP/TCP listeners for IoT devices
                await StartNetworkListeners();

                if (_enableModbusLogs)
                {
                    _ = Task.Run(() => CollectModbusLogsAsync(_cancellationTokenSource.Token));
                }

                if (_enableMqttLogs)
                {
                    _ = Task.Run(() => CollectMqttLogsAsync(_cancellationTokenSource.Token));
                }

                if (_enableOpcUaLogs)
                {
                    _ = Task.Run(() => CollectOpcUaLogsAsync(_cancellationTokenSource.Token));
                }

                if (_enableScadaLogs)
                {
                    _ = Task.Run(() => CollectScadaLogsAsync(_cancellationTokenSource.Token));
                }

                if (_enableSensorLogs)
                {
                    _ = Task.Run(() => CollectSensorLogsAsync(_cancellationTokenSource.Token));
                }

                _isRunning = true;
                _isPaused = false;
                _errorMessage = string.Empty;

                _logger.LogInformation("IoT Collector started successfully");
            }
            catch (Exception ex)
            {
                _errorMessage = ex.Message;
                _logger.LogError(ex, "Failed to start IoT Collector");
                throw;
            }
        }

        public async Task StopAsync()
        {
            await Task.CompletedTask;
            if (!_isRunning) return;

            try
            {
                _logger.LogInformation("Stopping IoT Collector");
                
                _cancellationTokenSource?.Cancel();
                _udpListener?.Close();
                _udpListener?.Dispose();
                _tcpListener?.Stop();
                _isRunning = false;
                
                _logger.LogInformation("IoT Collector stopped");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error stopping IoT Collector");
            }
        }

        public Task PauseAsync()
        {
            _isPaused = true;
            _logger.LogInformation("IoT Collector paused");
            return Task.CompletedTask;
        }

        public Task ResumeAsync()
        {
            _isPaused = false;
            _logger.LogInformation("IoT Collector resumed");
            return Task.CompletedTask;
        }

        public async Task<int> CollectLogsAsync(CancellationToken cancellationToken)
        {
            if (_isPaused || !_isRunning)
                return 0;

            int collectedCount = 0;

            try
            {
                if (_enableModbusLogs)
                {
                    await CollectModbusLogs();
                    collectedCount++;
                }

                if (_enableMqttLogs)
                {
                    await CollectMqttLogs();
                    collectedCount++;
                }

                if (_enableOpcUaLogs)
                {
                    await CollectOpcUaLogs();
                    collectedCount++;
                }

                if (_enableScadaLogs)
                {
                    await CollectScadaLogs();
                    collectedCount++;
                }

                if (_enableSensorLogs)
                {
                    await CollectSensorLogs();
                    collectedCount++;
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error collecting IoT logs");
                _errorMessage = ex.Message;
            }

            return collectedCount;
        }

        private void ParseSettings()
        {
            if (_settings.Properties.ContainsKey("EnableModbusLogs"))
            {
                bool.TryParse(_settings.Properties["EnableModbusLogs"], out _enableModbusLogs);
            }

            if (_settings.Properties.ContainsKey("EnableMqttLogs"))
            {
                bool.TryParse(_settings.Properties["EnableMqttLogs"], out _enableMqttLogs);
            }

            if (_settings.Properties.ContainsKey("EnableOpcUaLogs"))
            {
                bool.TryParse(_settings.Properties["EnableOpcUaLogs"], out _enableOpcUaLogs);
            }

            if (_settings.Properties.ContainsKey("EnableScadaLogs"))
            {
                bool.TryParse(_settings.Properties["EnableScadaLogs"], out _enableScadaLogs);
            }

            if (_settings.Properties.ContainsKey("EnableSensorLogs"))
            {
                bool.TryParse(_settings.Properties["EnableSensorLogs"], out _enableSensorLogs);
            }

            if (_settings.Properties.ContainsKey("CollectionInterval"))
            {
                int.TryParse(_settings.Properties["CollectionInterval"], out _collectionInterval);
            }

            if (_settings.Properties.ContainsKey("ModbusPort"))
            {
                _modbusPort = _settings.Properties["ModbusPort"];
            }

            if (_settings.Properties.ContainsKey("MqttBroker"))
            {
                _mqttBroker = _settings.Properties["MqttBroker"];
            }

            if (_settings.Properties.ContainsKey("OpcUaEndpoint"))
            {
                _opcUaEndpoint = _settings.Properties["OpcUaEndpoint"];
            }

            if (_settings.Properties.ContainsKey("ScadaHost"))
            {
                _scadaHost = _settings.Properties["ScadaHost"];
            }

            if (_settings.Properties.ContainsKey("UdpPort"))
            {
                int.TryParse(_settings.Properties["UdpPort"], out _udpPort);
            }

            if (_settings.Properties.ContainsKey("TcpPort"))
            {
                int.TryParse(_settings.Properties["TcpPort"], out _tcpPort);
            }

            if (_settings.Properties.ContainsKey("SensorEndpoints"))
            {
                _sensorEndpoints = new List<string>(_settings.Properties["SensorEndpoints"].Split(','));
            }

            if (_settings.Properties.ContainsKey("DeviceTypes"))
            {
                _deviceTypes = new List<string>(_settings.Properties["DeviceTypes"].Split(','));
            }
        }

        private async Task StartNetworkListeners()
        {
            await Task.CompletedTask;
            try
            {
                // Start UDP listener for IoT devices
                _udpListener = new UdpClient(_udpPort);
                _logger.LogInformation("IoT UDP listener started on port {Port}", _udpPort);

                _ = Task.Run(async () =>
                {
                    while (_isRunning && _udpListener != null)
                    {
                        try
                        {
                            var result = await _udpListener.ReceiveAsync();
                            if (!_isPaused)
                            {
                                var message = Encoding.UTF8.GetString(result.Buffer);
                                await ProcessIoTMessage(message, result.RemoteEndPoint.ToString(), "UDP");
                            }
                        }
                        catch (ObjectDisposedException)
                        {
                            break;
                        }
                        catch (Exception ex)
                        {
                            _logger.LogError(ex, "Error receiving UDP IoT message");
                        }
                    }
                });

                // Start TCP listener for IoT devices
                _tcpListener = new TcpListener(IPAddress.Any, _tcpPort);
                _tcpListener.Start();
                _logger.LogInformation("IoT TCP listener started on port {Port}", _tcpPort);

                _ = Task.Run(async () =>
                {
                    while (_isRunning && _tcpListener != null)
                    {
                        try
                        {
                            var tcpClient = await _tcpListener.AcceptTcpClientAsync();
                            _ = Task.Run(() => HandleTcpClient(tcpClient));
                        }
                        catch (ObjectDisposedException)
                        {
                            break;
                        }
                        catch (Exception ex)
                        {
                            _logger.LogError(ex, "Error accepting TCP connection");
                        }
                    }
                });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error starting IoT network listeners");
                throw;
            }
        }

        private async Task HandleTcpClient(TcpClient client)
        {
            try
            {
                using (client)
                using (var stream = client.GetStream())
                using (var reader = new StreamReader(stream, Encoding.UTF8))
                {
                    var buffer = new char[4096];
                    
                    while (client.Connected && _isRunning)
                    {
                        var bytesRead = await reader.ReadAsync(buffer, 0, buffer.Length);
                        if (bytesRead == 0) break;

                        if (!_isPaused)
                        {
                            var message = new StringBuilder().Append(buffer, 0, bytesRead).ToString();
                            await ProcessIoTMessage(message, client.Client.RemoteEndPoint?.ToString() ?? "unknown", "TCP");
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error handling TCP client");
            }
        }

        private async Task CollectModbusLogsAsync(CancellationToken cancellationToken)
        {
            _logger.LogInformation("Starting Modbus logs collection");

            while (_isRunning && !cancellationToken.IsCancellationRequested)
            {
                try
                {
                    if (!_isPaused)
                    {
                        await CollectModbusLogs();
                    }

                    await Task.Delay(TimeSpan.FromSeconds(_collectionInterval), cancellationToken);
                }
                catch (OperationCanceledException)
                {
                    break;
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error collecting Modbus logs");
                    await Task.Delay(TimeSpan.FromSeconds(60), cancellationToken);
                }
            }
        }

        private async Task CollectMqttLogsAsync(CancellationToken cancellationToken)
        {
            _logger.LogInformation("Starting MQTT logs collection");

            while (_isRunning && !cancellationToken.IsCancellationRequested)
            {
                try
                {
                    if (!_isPaused)
                    {
                        await CollectMqttLogs();
                    }

                    await Task.Delay(TimeSpan.FromSeconds(_collectionInterval), cancellationToken);
                }
                catch (OperationCanceledException)
                {
                    break;
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error collecting MQTT logs");
                    await Task.Delay(TimeSpan.FromSeconds(60), cancellationToken);
                }
            }
        }

        private async Task CollectOpcUaLogsAsync(CancellationToken cancellationToken)
        {
            _logger.LogInformation("Starting OPC-UA logs collection");

            while (_isRunning && !cancellationToken.IsCancellationRequested)
            {
                try
                {
                    if (!_isPaused)
                    {
                        await CollectOpcUaLogs();
                    }

                    await Task.Delay(TimeSpan.FromSeconds(_collectionInterval), cancellationToken);
                }
                catch (OperationCanceledException)
                {
                    break;
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error collecting OPC-UA logs");
                    await Task.Delay(TimeSpan.FromSeconds(60), cancellationToken);
                }
            }
        }

        private async Task CollectScadaLogsAsync(CancellationToken cancellationToken)
        {
            _logger.LogInformation("Starting SCADA logs collection");

            while (_isRunning && !cancellationToken.IsCancellationRequested)
            {
                try
                {
                    if (!_isPaused)
                    {
                        await CollectScadaLogs();
                    }

                    await Task.Delay(TimeSpan.FromSeconds(_collectionInterval), cancellationToken);
                }
                catch (OperationCanceledException)
                {
                    break;
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error collecting SCADA logs");
                    await Task.Delay(TimeSpan.FromSeconds(60), cancellationToken);
                }
            }
        }

        private async Task CollectSensorLogsAsync(CancellationToken cancellationToken)
        {
            _logger.LogInformation("Starting sensor logs collection");

            while (_isRunning && !cancellationToken.IsCancellationRequested)
            {
                try
                {
                    if (!_isPaused)
                    {
                        await CollectSensorLogs();
                    }

                    await Task.Delay(TimeSpan.FromSeconds(_collectionInterval), cancellationToken);
                }
                catch (OperationCanceledException)
                {
                    break;
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error collecting sensor logs");
                    await Task.Delay(TimeSpan.FromSeconds(60), cancellationToken);
                }
            }
        }

        private async Task ProcessIoTMessage(string message, string sourceEndpoint, string protocol)
        {
            await Task.CompletedTask;
            try
            {
                var logEntry = ParseIoTMessage(message, sourceEndpoint, protocol);
                if (logEntry != null)
                {
                    LogCollected?.Invoke(this, logEntry);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error processing IoT message");
            }
        }

        private async Task CollectModbusLogs()
        {
            await Task.CompletedTask;
            try
            {
                // Simulate Modbus device communication
                var mockEvents = GenerateMockModbusEvents();

                foreach (var modbusEvent in mockEvents)
                {
                    var logEntry = ParseModbusLog(modbusEvent);
                    if (logEntry != null)
                    {
                        LogCollected?.Invoke(this, logEntry);
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error collecting Modbus logs");
            }
        }

        private async Task CollectMqttLogs()
        {
            await Task.CompletedTask;
            try
            {
                // Simulate MQTT broker logs
                var mockEvents = GenerateMockMqttEvents();

                foreach (var mqttEvent in mockEvents)
                {
                    var logEntry = ParseMqttLog(mqttEvent);
                    if (logEntry != null)
                    {
                        LogCollected?.Invoke(this, logEntry);
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error collecting MQTT logs");
            }
        }

        private async Task CollectOpcUaLogs()
        {
            await Task.CompletedTask;
            try
            {
                // Simulate OPC-UA server logs
                var mockEvents = GenerateMockOpcUaEvents();

                foreach (var opcEvent in mockEvents)
                {
                    var logEntry = ParseOpcUaLog(opcEvent);
                    if (logEntry != null)
                    {
                        LogCollected?.Invoke(this, logEntry);
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error collecting OPC-UA logs");
            }
        }

        private async Task CollectScadaLogs()
        {
            await Task.CompletedTask;
            try
            {
                // Simulate SCADA system logs
                var mockEvents = GenerateMockScadaEvents();

                foreach (var scadaEvent in mockEvents)
                {
                    var logEntry = ParseScadaLog(scadaEvent);
                    if (logEntry != null)
                    {
                        LogCollected?.Invoke(this, logEntry);
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error collecting SCADA logs");
            }
        }

        private async Task CollectSensorLogs()
        {
            await Task.CompletedTask;
            try
            {
                // Simulate sensor data collection
                var mockEvents = GenerateMockSensorEvents();

                foreach (var sensorEvent in mockEvents)
                {
                    var logEntry = ParseSensorLog(sensorEvent);
                    if (logEntry != null)
                    {
                        LogCollected?.Invoke(this, logEntry);
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error collecting sensor logs");
            }
        }

        private NormalizedLogEntry? ParseIoTMessage(string message, string sourceEndpoint, string protocol)
        {
            try
            {
                // Try to parse as JSON first
                try
                {
                    var jsonMessage = JsonDocument.Parse(message);
                    return ParseJsonIoTMessage(jsonMessage, sourceEndpoint, protocol);
                }
                catch
                {
                    // If not JSON, parse as plain text
                    return ParsePlainTextIoTMessage(message, sourceEndpoint, protocol);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error parsing IoT message");
                return null;
            }
        }

        private NormalizedLogEntry ParseJsonIoTMessage(JsonDocument jsonMessage, string sourceEndpoint, string protocol)
        {
            var root = jsonMessage.RootElement;
            
            var deviceId = root.TryGetProperty("device_id", out var deviceIdElement) ? deviceIdElement.GetString() : "unknown";
            var messageType = root.TryGetProperty("type", out var typeElement) ? typeElement.GetString() : "sensor_data";
            var timestamp = root.TryGetProperty("timestamp", out var tsElement) ? tsElement.GetString() : DateTime.UtcNow.ToString();

            DateTime.TryParse(timestamp, out var parsedTimestamp);

            return new NormalizedLogEntry
            {
                Id = Guid.NewGuid().ToString(),
                Timestamp = parsedTimestamp != DateTime.MinValue ? parsedTimestamp : DateTime.UtcNow,
                Level = "Information",
                Source = $"IoT/{deviceId}",
                Category = "IoTDevice",
                EventId = $"IOT_{messageType?.ToUpper()}",
                Message = $"IoT device {deviceId} sent {messageType} data",
                Details = jsonMessage.RootElement.ToString(),
                Tags = new List<string> { "iot", "device", deviceId ?? "unknown", messageType?.ToLower() ?? "unknown", protocol.ToLower() },
                Severity = DetermineIoTSeverity(messageType, root)
            };
        }

        private NormalizedLogEntry ParsePlainTextIoTMessage(string message, string sourceEndpoint, string protocol)
        {
            return new NormalizedLogEntry
            {
                Id = Guid.NewGuid().ToString(),
                Timestamp = DateTime.UtcNow,
                Level = "Information",
                Source = $"IoT/{sourceEndpoint}",
                Category = "IoTDevice",
                EventId = "IOT_MESSAGE",
                Message = $"IoT message from {sourceEndpoint}",
                Details = JsonSerializer.Serialize(new
                {
                    raw_message = message,
                    source_endpoint = sourceEndpoint,
                    protocol = protocol
                }),
                Tags = new List<string> { "iot", "device", "plaintext", protocol.ToLower() },
                Severity = "Low"
            };
        }

        private NormalizedLogEntry? ParseModbusLog(dynamic modbusEvent)
        {
            try
            {
                return new NormalizedLogEntry
                {
                    Id = Guid.NewGuid().ToString(),
                    Timestamp = DateTime.Parse(modbusEvent.timestamp.ToString()),
                    Level = modbusEvent.function_code > 127 ? "Error" : "Information",
                    Source = $"Modbus/{modbusEvent.device_id}",
                    Category = "ModbusDevice",
                    EventId = $"MODBUS_FC_{modbusEvent.function_code}",
                    Message = $"Modbus {modbusEvent.operation} on device {modbusEvent.device_id}",
                    Details = JsonSerializer.Serialize(modbusEvent),
                    Tags = new List<string> { "iot", "modbus", "industrial", modbusEvent.operation.ToString().ToLower() },
                    Severity = modbusEvent.function_code > 127 ? "Medium" : "Low"
                };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error parsing Modbus log");
                return null;
            }
        }

        private NormalizedLogEntry? ParseMqttLog(dynamic mqttEvent)
        {
            try
            {
                return new NormalizedLogEntry
                {
                    Id = Guid.NewGuid().ToString(),
                    Timestamp = DateTime.Parse(mqttEvent.timestamp.ToString()),
                    Level = "Information",
                    Source = $"MQTT/{mqttEvent.client_id}",
                    Category = "MQTTBroker",
                    EventId = $"MQTT_{mqttEvent.event_type.ToString().ToUpper()}",
                    Message = $"MQTT {mqttEvent.event_type}: {mqttEvent.topic} from {mqttEvent.client_id}",
                    Details = JsonSerializer.Serialize(mqttEvent),
                    Tags = new List<string> { "iot", "mqtt", "messaging", mqttEvent.event_type.ToString().ToLower() },
                    Severity = mqttEvent.event_type.ToString() == "disconnect" ? "Medium" : "Low"
                };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error parsing MQTT log");
                return null;
            }
        }

        private NormalizedLogEntry? ParseOpcUaLog(dynamic opcEvent)
        {
            try
            {
                return new NormalizedLogEntry
                {
                    Id = Guid.NewGuid().ToString(),
                    Timestamp = DateTime.Parse(opcEvent.timestamp.ToString()),
                    Level = opcEvent.status_code == 0 ? "Information" : "Warning",
                    Source = $"OPC-UA/{opcEvent.node_id}",
                    Category = "OPCUAServer",
                    EventId = $"OPCUA_{opcEvent.operation.ToString().ToUpper()}",
                    Message = $"OPC-UA {opcEvent.operation} on node {opcEvent.node_id}",
                    Details = JsonSerializer.Serialize(opcEvent),
                    Tags = new List<string> { "iot", "opcua", "industrial", opcEvent.operation.ToString().ToLower() },
                    Severity = opcEvent.status_code == 0 ? "Low" : "Medium"
                };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error parsing OPC-UA log");
                return null;
            }
        }

        private NormalizedLogEntry? ParseScadaLog(dynamic scadaEvent)
        {
            try
            {
                return new NormalizedLogEntry
                {
                    Id = Guid.NewGuid().ToString(),
                    Timestamp = DateTime.Parse(scadaEvent.timestamp.ToString()),
                    Level = scadaEvent.alarm_level.ToString() == "Critical" ? "Error" : "Information",
                    Source = $"SCADA/{scadaEvent.system_id}",
                    Category = "SCADASystem",
                    EventId = $"SCADA_{scadaEvent.event_type.ToString().ToUpper()}",
                    Message = $"SCADA {scadaEvent.event_type}: {scadaEvent.description}",
                    Details = JsonSerializer.Serialize(scadaEvent),
                    Tags = new List<string> { "iot", "scada", "industrial", scadaEvent.event_type.ToString().ToLower() },
                    Severity = scadaEvent.alarm_level.ToString() == "Critical" ? "High" : scadaEvent.alarm_level.ToString() == "Warning" ? "Medium" : "Low"
                };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error parsing SCADA log");
                return null;
            }
        }

        private NormalizedLogEntry? ParseSensorLog(dynamic sensorEvent)
        {
            try
            {
                return new NormalizedLogEntry
                {
                    Id = Guid.NewGuid().ToString(),
                    Timestamp = DateTime.Parse(sensorEvent.timestamp.ToString()),
                    Level = CheckSensorThresholds(sensorEvent) ? "Warning" : "Information",
                    Source = $"Sensor/{sensorEvent.sensor_id}",
                    Category = "IoTSensor",
                    EventId = "SENSOR_READING",
                    Message = $"Sensor {sensorEvent.sensor_id} reading: {sensorEvent.value} {sensorEvent.unit}",
                    Details = JsonSerializer.Serialize(sensorEvent),
                    Tags = new List<string> { "iot", "sensor", sensorEvent.sensor_type.ToString().ToLower(), "reading" },
                    Severity = CheckSensorThresholds(sensorEvent) ? "Medium" : "Low"
                };
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error parsing sensor log");
                return null;
            }
        }

        private string DetermineIoTSeverity(string? messageType, JsonElement root)
        {
            if (messageType?.ToLower().Contains("alarm") == true || messageType?.ToLower().Contains("error") == true)
                return "High";
            
            if (messageType?.ToLower().Contains("warning") == true)
                return "Medium";

            // Check for threshold violations in sensor data
            if (root.TryGetProperty("alarm", out var alarmElement) && alarmElement.GetBoolean())
                return "Medium";

            return "Low";
        }

        private bool CheckSensorThresholds(dynamic sensorEvent)
        {
            try
            {
                var value = Convert.ToDouble(sensorEvent.value);
                var sensorType = sensorEvent.sensor_type.ToString().ToLower();

                return sensorType switch
                {
                    "temperature" => value > 50 || value < -10,
                    "humidity" => value > 90 || value < 10,
                    "pressure" => value > 1000 || value < 900,
                    "vibration" => value > 10,
                    "flow" => value < 0.1,
                    _ => false
                };
            }
            catch
            {
                return false;
            }
        }

        // Mock data generators
        private List<dynamic> GenerateMockModbusEvents()
        {
            return new List<dynamic>
            {
                new {
                    timestamp = DateTime.UtcNow.ToString("o"),
                    device_id = "PLC_001",
                    function_code = 3,
                    operation = "ReadHoldingRegisters",
                    register_address = 1000,
                    register_count = 10,
                    response_data = "01 02 03 04 05 06 07 08 09 0A"
                }
            };
        }

        private List<dynamic> GenerateMockMqttEvents()
        {
            return new List<dynamic>
            {
                new {
                    timestamp = DateTime.UtcNow.ToString("o"),
                    client_id = "sensor_001",
                    event_type = "publish",
                    topic = "sensors/temperature",
                    payload = "24.5",
                    qos = 1
                }
            };
        }

        private List<dynamic> GenerateMockOpcUaEvents()
        {
            return new List<dynamic>
            {
                new {
                    timestamp = DateTime.UtcNow.ToString("o"),
                    node_id = "ns=2;i=1001",
                    operation = "read",
                    value = 75.2,
                    status_code = 0,
                    data_type = "Double"
                }
            };
        }

        private List<dynamic> GenerateMockScadaEvents()
        {
            return new List<dynamic>
            {
                new {
                    timestamp = DateTime.UtcNow.ToString("o"),
                    system_id = "HMI_001",
                    event_type = "alarm",
                    alarm_level = "Warning",
                    description = "High temperature detected in zone 3",
                    @operator = "operator1"
                }
            };
        }

        private List<dynamic> GenerateMockSensorEvents()
        {
            return new List<dynamic>
            {
                new {
                    timestamp = DateTime.UtcNow.ToString("o"),
                    sensor_id = "TEMP_001",
                    sensor_type = "temperature",
                    value = 23.5,
                    unit = "celsius",
                    location = "warehouse_zone_1"
                }
            };
        }

        public void Dispose()
        {
            StopAsync().Wait();
            _udpListener?.Dispose();
            _cancellationTokenSource?.Dispose();
        }
    }
} 