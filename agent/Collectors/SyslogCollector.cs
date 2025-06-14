using AthalaSIEM.Agent.Models;
using Microsoft.Extensions.Logging;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using System.Linq;

namespace AthalaSIEM.Agent.Collectors
{
    /// <summary>
    /// Universal Syslog Collector for Linux, Unix, FreeBSD, Network Devices, Firewalls, and other syslog-enabled devices
    /// </summary>
    public class SyslogCollector : ILogCollector
    {
        private readonly ILogger<SyslogCollector> _logger;
        private readonly ILogNormalizer _normalizer;
        private UdpClient? _udpListener;
        private TcpListener? _tcpListener;
        private bool _isRunning;
        private bool _isPaused;
        private string _errorMessage = string.Empty;
        private CollectorSettings _settings = new();
        
        // Configuration
        private int _udpPort = 514;
        private int _tcpPort = 601;
        private bool _enableUdp = true;
        private bool _enableTcp = true;
        private string _bindAddress = "0.0.0.0";
        private int _maxMessageSize = 8192;
        private int _maxConcurrentConnections = 100;
        
        // Device type detection patterns
        private readonly Dictionary<string, DeviceTypePattern> _devicePatterns = new();
        private readonly ConcurrentQueue<NormalizedLogEntry> _messageQueue = new();
        private readonly SemaphoreSlim _connectionSemaphore;
        
        // RFC3164 and RFC5424 parsing
        private static readonly Regex Rfc3164Pattern = new(@"^<(\d+)>(\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+(\S+)\s+(.+)$", RegexOptions.Compiled);
        private static readonly Regex Rfc5424Pattern = new(@"^<(\d+)>(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(.*)$", RegexOptions.Compiled);
        
        public event EventHandler<NormalizedLogEntry>? LogCollected;
        public string CollectorType => "Syslog";
        public bool IsRunning => _isRunning;
        public bool IsPaused => _isPaused;
        public string LastError => _errorMessage;
        public CollectorSettings Settings => _settings;

        public SyslogCollector(ILogger<SyslogCollector> logger, ILogNormalizer normalizer)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _normalizer = normalizer ?? throw new ArgumentNullException(nameof(normalizer));
            _connectionSemaphore = new SemaphoreSlim(_maxConcurrentConnections, _maxConcurrentConnections);
            InitializeDevicePatterns();
        }

        public void Initialize(CollectorSettings settings)
        {
            _settings = settings ?? throw new ArgumentNullException(nameof(settings));
            _logger.LogInformation("Initializing Universal Syslog Collector");

            try
            {
                ParseSettings();
                _logger.LogInformation("Universal Syslog Collector initialized - UDP: {UdpEnabled}:{UdpPort}, TCP: {TcpEnabled}:{TcpPort}", 
                    _enableUdp, _udpPort, _enableTcp, _tcpPort);
            }
            catch (Exception ex)
            {
                _errorMessage = ex.Message;
                _logger.LogError(ex, "Failed to initialize Universal Syslog Collector");
                throw;
            }
        }

        public async Task StartAsync()
        {
            if (_isRunning) return;

            try
            {
                _logger.LogInformation("Starting Universal Syslog Collector");

                if (_enableUdp)
                {
                    await StartUdpListener();
                }

                if (_enableTcp)
                {
                    await StartTcpListener();
                }

                // Start message processing task
                _ = Task.Run(ProcessMessageQueue);

                _isRunning = true;
                _isPaused = false;
                _errorMessage = string.Empty;

                _logger.LogInformation("Universal Syslog Collector started successfully");
            }
            catch (Exception ex)
            {
                _errorMessage = ex.Message;
                _logger.LogError(ex, "Failed to start Universal Syslog Collector");
                throw;
            }
        }

        public async Task StopAsync()
        {
            if (!_isRunning) return;

            try
            {
                _logger.LogInformation("Stopping Universal Syslog Collector");

                _udpListener?.Close();
                _udpListener?.Dispose();
                _udpListener = null;

                _tcpListener?.Stop();
                _tcpListener = null;

                _isRunning = false;
                _logger.LogInformation("Universal Syslog Collector stopped");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error stopping Universal Syslog Collector");
            }
        }

        public void Pause()
        {
            _isPaused = true;
            _logger.LogInformation("Universal Syslog Collector paused");
        }

        public void Resume()
        {
            _isPaused = false;
            _logger.LogInformation("Universal Syslog Collector resumed");
        }

        private void ParseSettings()
        {
            if (_settings.Properties.ContainsKey("UdpPort"))
            {
                int.TryParse(_settings.Properties["UdpPort"], out _udpPort);
            }

            if (_settings.Properties.ContainsKey("TcpPort"))
            {
                int.TryParse(_settings.Properties["TcpPort"], out _tcpPort);
            }

            if (_settings.Properties.ContainsKey("EnableUdp"))
            {
                bool.TryParse(_settings.Properties["EnableUdp"], out _enableUdp);
            }

            if (_settings.Properties.ContainsKey("EnableTcp"))
            {
                bool.TryParse(_settings.Properties["EnableTcp"], out _enableTcp);
            }

            if (_settings.Properties.ContainsKey("BindAddress"))
            {
                _bindAddress = _settings.Properties["BindAddress"];
            }

            if (_settings.Properties.ContainsKey("MaxMessageSize"))
            {
                int.TryParse(_settings.Properties["MaxMessageSize"], out _maxMessageSize);
            }

            if (_settings.Properties.ContainsKey("MaxConcurrentConnections"))
            {
                int.TryParse(_settings.Properties["MaxConcurrentConnections"], out _maxConcurrentConnections);
            }
        }

        private async Task StartUdpListener()
        {
            try
            {
                var endpoint = new IPEndPoint(IPAddress.Parse(_bindAddress), _udpPort);
                _udpListener = new UdpClient(endpoint);
                
                _logger.LogInformation("UDP Syslog listener started on {Address}:{Port}", _bindAddress, _udpPort);

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
                                await ProcessSyslogMessage(message, result.RemoteEndPoint, "UDP");
                            }
                        }
                        catch (ObjectDisposedException)
                        {
                            break;
                        }
                        catch (Exception ex)
                        {
                            _logger.LogError(ex, "Error receiving UDP syslog message");
                        }
                    }
                });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to start UDP listener on port {Port}", _udpPort);
                throw;
            }
        }

        private async Task StartTcpListener()
        {
            try
            {
                _tcpListener = new TcpListener(IPAddress.Parse(_bindAddress), _tcpPort);
                _tcpListener.Start();
                
                _logger.LogInformation("TCP Syslog listener started on {Address}:{Port}", _bindAddress, _tcpPort);

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
                _logger.LogError(ex, "Failed to start TCP listener on port {Port}", _tcpPort);
                throw;
            }
        }

        private async Task HandleTcpClient(TcpClient client)
        {
            await _connectionSemaphore.WaitAsync();
            
            try
            {
                using (client)
                using (var stream = client.GetStream())
                using (var reader = new StreamReader(stream, Encoding.UTF8))
                {
                    var buffer = new char[_maxMessageSize];
                    var messageBuilder = new StringBuilder();
                    
                    while (client.Connected && _isRunning)
                    {
                        var bytesRead = await reader.ReadAsync(buffer, 0, buffer.Length);
                        if (bytesRead == 0) break;

                        messageBuilder.Append(buffer, 0, bytesRead);
                        
                        // Process complete messages (delimited by newline)
                        string fullMessage = messageBuilder.ToString();
                        string[] messages = fullMessage.Split('\n');
                        
                        // Process all complete messages
                        for (int i = 0; i < messages.Length - 1; i++)
                        {
                            if (!string.IsNullOrWhiteSpace(messages[i]) && !_isPaused)
                            {
                                await ProcessSyslogMessage(messages[i].Trim(), client.Client.RemoteEndPoint, "TCP");
                            }
                        }
                        
                        // Keep the incomplete message for next iteration
                        messageBuilder.Clear();
                        if (!string.IsNullOrWhiteSpace(messages[messages.Length - 1]))
                        {
                            messageBuilder.Append(messages[messages.Length - 1]);
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error handling TCP client {Client}", client.Client.RemoteEndPoint);
            }
            finally
            {
                _connectionSemaphore.Release();
            }
        }

        private async Task ProcessSyslogMessage(string message, EndPoint? remoteEndPoint, string protocol)
        {
            try
            {
                var parsedMessage = ParseSyslogMessage(message, remoteEndPoint?.ToString() ?? "unknown", protocol);
                if (parsedMessage != null)
                {
                    _messageQueue.Enqueue(parsedMessage);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error processing syslog message: {Message}", message);
            }
        }

        private NormalizedLogEntry? ParseSyslogMessage(string message, string sourceIp, string protocol)
        {
            try
            {
                // Try RFC5424 format first
                var rfc5424Match = Rfc5424Pattern.Match(message);
                if (rfc5424Match.Success)
                {
                    return ParseRfc5424Message(rfc5424Match, message, sourceIp, protocol);
                }

                // Try RFC3164 format
                var rfc3164Match = Rfc3164Pattern.Match(message);
                if (rfc3164Match.Success)
                {
                    return ParseRfc3164Message(rfc3164Match, message, sourceIp, protocol);
                }

                // Fallback for non-standard formats
                return ParseNonStandardMessage(message, sourceIp, protocol);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error parsing syslog message: {Message}", message);
                return null;
            }
        }

        private NormalizedLogEntry ParseRfc5424Message(Match match, string rawMessage, string sourceIp, string protocol)
        {
            var priority = int.Parse(match.Groups[1].Value);
            var facility = priority >> 3;
            var severity = priority & 7;
            
            var version = match.Groups[2].Value;
            var timestamp = match.Groups[3].Value;
            var hostname = match.Groups[4].Value;
            var appName = match.Groups[5].Value;
            var procId = match.Groups[6].Value;
            var msgId = match.Groups[7].Value;
            var structuredData = match.Groups[8].Value;

            var deviceType = DetectDeviceType(hostname, appName, rawMessage);

            return new NormalizedLogEntry
            {
                Id = Guid.NewGuid().ToString(),
                Timestamp = ParseTimestamp(timestamp),
                Level = GetLogLevel(severity).ToString(),
                Source = $"{hostname}:{appName}",
                Category = GetFacilityName(facility),
                EventId = msgId != "-" ? msgId : $"SYSLOG_{severity}",
                Message = structuredData,
                Details = JsonSerializer.Serialize(new
                {
                    raw_message = rawMessage,
                    source_ip = sourceIp,
                    protocol = protocol,
                    rfc_format = "RFC5424",
                    priority = priority,
                    facility = facility,
                    facility_name = GetFacilityName(facility),
                    severity = severity,
                    severity_name = GetSeverityName(severity),
                    hostname = hostname,
                    app_name = appName,
                    process_id = procId,
                    message_id = msgId,
                    device_type = deviceType.Type,
                    device_vendor = deviceType.Vendor
                }),
                Tags = new List<string> { "syslog", protocol.ToLower(), deviceType.Type, GetFacilityName(facility) },
                Severity = GetSeverityLevel(severity)
            };
        }

        private NormalizedLogEntry ParseRfc3164Message(Match match, string rawMessage, string sourceIp, string protocol)
        {
            var priority = int.Parse(match.Groups[1].Value);
            var facility = priority >> 3;
            var severity = priority & 7;
            
            var timestamp = match.Groups[2].Value;
            var hostname = match.Groups[3].Value;
            var content = match.Groups[4].Value;

            var deviceType = DetectDeviceType(hostname, "", rawMessage);

            return new NormalizedLogEntry
            {
                Id = Guid.NewGuid().ToString(),
                Timestamp = ParseTimestamp(timestamp),
                Level = GetLogLevel(severity).ToString(),
                Source = hostname,
                Category = GetFacilityName(facility),
                EventId = $"SYSLOG_{severity}",
                Message = content,
                Details = JsonSerializer.Serialize(new
                {
                    raw_message = rawMessage,
                    source_ip = sourceIp,
                    protocol = protocol,
                    rfc_format = "RFC3164",
                    priority = priority,
                    facility = facility,
                    facility_name = GetFacilityName(facility),
                    severity = severity,
                    severity_name = GetSeverityName(severity),
                    hostname = hostname,
                    device_type = deviceType.Type,
                    device_vendor = deviceType.Vendor
                }),
                Tags = new List<string> { "syslog", protocol.ToLower(), deviceType.Type, GetFacilityName(facility) },
                Severity = GetSeverityLevel(severity)
            };
        }

        private NormalizedLogEntry ParseNonStandardMessage(string message, string sourceIp, string protocol)
        {
            var deviceType = DetectDeviceType("", "", message);

            return new NormalizedLogEntry
            {
                Id = Guid.NewGuid().ToString(),
                Timestamp = DateTime.UtcNow,
                Level = LogLevel.Information.ToString(),
                Source = sourceIp,
                Category = "Unknown",
                EventId = "SYSLOG_NONSTANDARD",
                Message = message,
                Details = JsonSerializer.Serialize(new
                {
                    raw_message = message,
                    source_ip = sourceIp,
                    protocol = protocol,
                    rfc_format = "NonStandard",
                    device_type = deviceType.Type,
                    device_vendor = deviceType.Vendor
                }),
                Tags = new List<string> { "syslog", protocol.ToLower(), deviceType.Type, "nonstandard" },
                Severity = "Medium"
            };
        }

        private DeviceTypeInfo DetectDeviceType(string hostname, string appName, string message)
        {
            var combined = $"{hostname} {appName} {message}".ToLowerInvariant();
            
            foreach (var pattern in _devicePatterns.Values)
            {
                foreach (var keyword in pattern.Keywords)
                {
                    if (combined.Contains(keyword))
                    {
                        return new DeviceTypeInfo { Type = pattern.DeviceType, Vendor = pattern.Vendor };
                    }
                }
            }

            return new DeviceTypeInfo { Type = "Generic", Vendor = "Unknown" };
        }

        private void InitializeDevicePatterns()
        {
            // Firewalls
            _devicePatterns["pfsense"] = new DeviceTypePattern
            {
                DeviceType = "Firewall",
                Vendor = "pfSense",
                Keywords = new[] { "pfsense", "filterlog", "openvpn", "pf:" }
            };

            _devicePatterns["checkpoint"] = new DeviceTypePattern
            {
                DeviceType = "Firewall",
                Vendor = "Check Point",
                Keywords = new[] { "checkpoint", "fwlog", "smartdefense", "cpd" }
            };

            _devicePatterns["fortinet"] = new DeviceTypePattern
            {
                DeviceType = "Firewall",
                Vendor = "Fortinet",
                Keywords = new[] { "fortigate", "fortios", "fortianalyzer", "fortinet" }
            };

            _devicePatterns["paloalto"] = new DeviceTypePattern
            {
                DeviceType = "Firewall",
                Vendor = "Palo Alto",
                Keywords = new[] { "palo alto", "pan-os", "panorama", "globalprotect" }
            };

            // Network Devices
            _devicePatterns["cisco"] = new DeviceTypePattern
            {
                DeviceType = "Network",
                Vendor = "Cisco",
                Keywords = new[] { "cisco", "ios", "nexus", "catalyst", "asa", "meraki" }
            };

            _devicePatterns["juniper"] = new DeviceTypePattern
            {
                DeviceType = "Network",
                Vendor = "Juniper",
                Keywords = new[] { "juniper", "junos", "srx", "mx", "ex", "qfx" }
            };

            _devicePatterns["aruba"] = new DeviceTypePattern
            {
                DeviceType = "Network",
                Vendor = "Aruba",
                Keywords = new[] { "aruba", "arubaos", "clearpass", "airwave" }
            };

            // Operating Systems
            _devicePatterns["freebsd"] = new DeviceTypePattern
            {
                DeviceType = "Server",
                Vendor = "FreeBSD",
                Keywords = new[] { "freebsd", "bsd", "kernel:", "su:", "sshd:", "httpd:" }
            };

            _devicePatterns["linux"] = new DeviceTypePattern
            {
                DeviceType = "Server",
                Vendor = "Linux",
                Keywords = new[] { "ubuntu", "centos", "rhel", "debian", "kernel:", "systemd", "sudo:" }
            };

            _devicePatterns["aix"] = new DeviceTypePattern
            {
                DeviceType = "Server",
                Vendor = "IBM AIX",
                Keywords = new[] { "aix", "lpar", "errpt", "oslevel" }
            };

            _devicePatterns["solaris"] = new DeviceTypePattern
            {
                DeviceType = "Server",
                Vendor = "Oracle Solaris",
                Keywords = new[] { "solaris", "sunos", "zones", "zfs" }
            };

            // Load Balancers
            _devicePatterns["f5"] = new DeviceTypePattern
            {
                DeviceType = "LoadBalancer",
                Vendor = "F5 Networks",
                Keywords = new[] { "f5", "big-ip", "ltm:", "asm:", "apm:" }
            };

            _devicePatterns["haproxy"] = new DeviceTypePattern
            {
                DeviceType = "LoadBalancer",
                Vendor = "HAProxy",
                Keywords = new[] { "haproxy", "backend", "frontend", "stats" }
            };

            // Storage
            _devicePatterns["netapp"] = new DeviceTypePattern
            {
                DeviceType = "Storage",
                Vendor = "NetApp",
                Keywords = new[] { "netapp", "ontap", "wafl", "snapmirror" }
            };

            _devicePatterns["emc"] = new DeviceTypePattern
            {
                DeviceType = "Storage",
                Vendor = "Dell EMC",
                Keywords = new[] { "emc", "vnx", "vmax", "isilon", "unity" }
            };
        }

        private async Task ProcessMessageQueue()
        {
            while (_isRunning)
            {
                try
                {
                    if (_messageQueue.TryDequeue(out var logEntry))
                    {
                        var normalizedEntry = _normalizer.NormalizeLogEntry(logEntry);
                        LogCollected?.Invoke(this, normalizedEntry);
                    }
                    else
                    {
                        await Task.Delay(100);
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error processing message queue");
                }
            }
        }

        private DateTime ParseTimestamp(string timestamp)
        {
            if (DateTime.TryParse(timestamp, out var result))
            {
                return result;
            }
            return DateTime.UtcNow;
        }

        private LogLevel GetLogLevel(int severity)
        {
            return severity switch
            {
                0 => LogLevel.Critical,  // Emergency
                1 => LogLevel.Critical,  // Alert
                2 => LogLevel.Critical,  // Critical
                3 => LogLevel.Error,     // Error
                4 => LogLevel.Warning,   // Warning
                5 => LogLevel.Warning,   // Notice
                6 => LogLevel.Information, // Info
                7 => LogLevel.Debug,     // Debug
                _ => LogLevel.Information
            };
        }

        private string GetSeverityLevel(int severity)
        {
            return severity switch
            {
                0 or 1 or 2 => "Critical",
                3 => "High",
                4 => "Medium",
                5 or 6 => "Low",
                7 => "Info",
                _ => "Medium"
            };
        }

        private string GetSeverityName(int severity)
        {
            return severity switch
            {
                0 => "Emergency",
                1 => "Alert", 
                2 => "Critical",
                3 => "Error",
                4 => "Warning",
                5 => "Notice",
                6 => "Informational",
                7 => "Debug",
                _ => "Unknown"
            };
        }

        private string GetFacilityName(int facility)
        {
            return facility switch
            {
                0 => "Kernel",
                1 => "User",
                2 => "Mail",
                3 => "Daemon",
                4 => "Security",
                5 => "Syslog",
                6 => "LinePrep",
                7 => "News",
                8 => "UUCP",
                9 => "Cron",
                10 => "Authpriv",
                11 => "FTP",
                16 => "Local0",
                17 => "Local1",
                18 => "Local2",
                19 => "Local3",
                20 => "Local4",
                21 => "Local5",
                22 => "Local6",
                23 => "Local7",
                _ => "Unknown"
            };
        }

        public void Dispose()
        {
            StopAsync().Wait();
            _udpListener?.Dispose();
            _connectionSemaphore?.Dispose();
        }
    }

    public class DeviceTypePattern
    {
        public string DeviceType { get; set; } = string.Empty;
        public string Vendor { get; set; } = string.Empty;
        public string[] Keywords { get; set; } = Array.Empty<string>();
    }

    public class DeviceTypeInfo
    {
        public string Type { get; set; } = string.Empty;
        public string Vendor { get; set; } = string.Empty;
    }
} 