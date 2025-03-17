using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Net;
using System.Net.NetworkInformation;
using System.Text.Json;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Collections.Generic;
using AthalaSIEM.Agent.Models;
using AthalaSIEM.Agent.Security;
using System.Net.Http;

namespace AthalaSIEM.Agent.Configuration
{
    /// <summary>
    /// Simple agent configuration form that appears during initial setup or manual configuration
    /// </summary>
    public partial class AgentConfigurationForm : Form
    {
        private readonly IAgentIdentityService _agentIdentityService;
        private AgentSettings _settings;
        private bool _isRegistered = false;
        private string _agentId = string.Empty;
        private string _apiKey = string.Empty;
        private string _deploymentToken = string.Empty;
        private bool _isTokenMode = false;

        // Form controls - changed to public to resolve CS9032 errors
        public required Label lblTitle;
        public required Label lblServerIp;
        public required TextBox txtServerIp;
        public required Label lblServerPort;
        public required NumericUpDown numServerPort;
        public required Label lblAgentName;
        public required TextBox txtAgentName;
        public required Button btnAddAgent;
        
        // Token mode controls
        public required Label lblTokenMode;
        public required Label lblToken;
        public required TextBox txtToken;
        
        // Connection and status controls - change from private to public
        public required Label lblConnectionStatus;
        public required PictureBox pbConnectionStatus;
        public required Label lblStatus;
        public required Label lblStatusValue;
        public required Label lblAgentId;
        public required TextBox txtAgentId;
        public required Label lblApiKey;
        public required TextBox txtApiKey;
        public required Button btnTestConnection;
        public required Button btnSave;

        public AgentConfigurationForm(IAgentIdentityService agentIdentityService, AgentSettings settings, string deploymentToken = "")
        {
            _agentIdentityService = agentIdentityService ?? throw new ArgumentNullException(nameof(agentIdentityService));
            _settings = settings ?? throw new ArgumentNullException(nameof(settings));
            _deploymentToken = deploymentToken ?? string.Empty;
            _isTokenMode = !string.IsNullOrEmpty(_deploymentToken);
            InitializeComponent();
        }

        /// <summary>
        /// Initialize form components
        /// </summary>
        private void InitializeComponent()
        {
            this.SuspendLayout();
            
            // Form settings
            this.Text = "AthalaSIEM Agent Setup";
            this.ClientSize = new Size(400, _isTokenMode ? 180 : 220); // Even smaller form in token mode
            this.MaximizeBox = false;
            this.MinimizeBox = true;
            this.StartPosition = FormStartPosition.CenterScreen;
            this.FormBorderStyle = FormBorderStyle.FixedDialog;
            this.Icon = SystemIcons.Shield;
            
            // Title
            lblTitle = new Label
            {
                Text = _isTokenMode ? "AthalaSIEM Agent Token Setup" : "AthalaSIEM Agent Setup",
                Location = new Point(20, 15),
                Width = 360,
                TextAlign = ContentAlignment.MiddleCenter,
                Font = new Font(this.Font.FontFamily, 12, FontStyle.Bold)
            };
            
            if (_isTokenMode)
            {
                // Token mode UI
                lblTokenMode = new Label
                {
                    Text = "This agent will be registered with a pre-configured deployment token.",
                Location = new Point(20, 50),
                    Width = 360,
                    TextAlign = ContentAlignment.MiddleCenter,
                    Font = new Font(this.Font.FontFamily, 9)
                };
                
                lblToken = new Label
                {
                    Text = "Token:",
                Location = new Point(20, 80),
                AutoSize = true
            };
            
                txtToken = new TextBox
                {
                    Location = new Point(80, 80),
                    Width = 260,
                    Text = _deploymentToken,
                    ReadOnly = true
                };
                
                // Add Agent button for token mode
                btnAddAgent = new Button
                {
                    Text = "Register Agent",
                    Location = new Point(140, 120),
                    Width = 120,
                    Height = 30,
                    BackColor = Color.FromArgb(0, 120, 212),
                    ForeColor = Color.White,
                    FlatStyle = FlatStyle.Flat
                };
                btnAddAgent.FlatAppearance.BorderSize = 0;
                btnAddAgent.Click += BtnAddAgent_Click;
                
                // Add only the token mode controls
                this.Controls.Add(lblTitle);
                this.Controls.Add(lblTokenMode);
                this.Controls.Add(lblToken);
                this.Controls.Add(txtToken);
                this.Controls.Add(btnAddAgent);
            }
            else
            {
                // Regular mode UI
            // Server IP field
            lblServerIp = new Label
            {
                Text = "Server IP:",
                    Location = new Point(20, 60),
                AutoSize = true
            };
            
            txtServerIp = new TextBox
            {
                    Location = new Point(140, 60),
                Width = 200,
                Text = "127.0.0.1"
            };
            
            // Server Port field
            lblServerPort = new Label
            {
                Text = "Server Port:",
                    Location = new Point(20, 90),
                AutoSize = true
            };
            
            numServerPort = new NumericUpDown
            {
                    Location = new Point(140, 90),
                Width = 80,
                Minimum = 1,
                Maximum = 65535,
                    Value = 5135
            };
            
            // Agent Name field
            lblAgentName = new Label
            {
                Text = "Agent Name:",
                    Location = new Point(20, 120),
                AutoSize = true
            };
            
            txtAgentName = new TextBox
            {
                    Location = new Point(140, 120),
                Width = 200,
                Text = Environment.MachineName
            };
            
                // Add Agent button
                btnAddAgent = new Button
                {
                    Text = "Add Agent",
                    Location = new Point(140, 160),
                Width = 120,
                Height = 30,
                    BackColor = Color.FromArgb(0, 120, 212),
                ForeColor = Color.White,
                    FlatStyle = FlatStyle.Flat
                };
                btnAddAgent.FlatAppearance.BorderSize = 0;
                btnAddAgent.Click += BtnAddAgent_Click;
                
                // Add only the needed controls to form for regular mode
            this.Controls.Add(lblTitle);
            this.Controls.Add(lblServerIp);
            this.Controls.Add(txtServerIp);
            this.Controls.Add(lblServerPort);
            this.Controls.Add(numServerPort);
            this.Controls.Add(lblAgentName);
            this.Controls.Add(txtAgentName);
                this.Controls.Add(btnAddAgent);
            }
            
            // Initialize but don't add unused controls
            lblConnectionStatus = new Label();
            pbConnectionStatus = new PictureBox();
            lblStatus = new Label();
            lblStatusValue = new Label();
            lblAgentId = new Label();
            txtAgentId = new TextBox();
            lblApiKey = new Label();
            txtApiKey = new TextBox();
            btnTestConnection = new Button();
            btnSave = new Button();
            
            this.ResumeLayout(false);
            this.PerformLayout();
            
            // Add event handler for form load
            this.Load += Form_Load;
        }

        /// <summary>
        /// Form load handler
        /// </summary>
        private async void Form_Load(object? sender, EventArgs e)
        {
            try
            {
                // Check if agent is already registered
                bool isRegistered = await _agentIdentityService.IsRegisteredAsync();
                if (isRegistered)
                {
                    // If already registered, show a message and close the form
                    MessageBox.Show("This agent is already registered and configured. The service is running in the background.",
                        "Already Registered", MessageBoxButtons.OK, MessageBoxIcon.Information);
                    this.DialogResult = DialogResult.OK;
                    this.Close();
                    return;
                }

                // If using token mode, we don't need to pre-populate fields
                if (_isTokenMode)
                {
                    return;
                }

                // Pre-populate form fields from settings if available
                if (_settings != null)
                {
                    // Extract server and port from backend URL
                    if (!string.IsNullOrEmpty(_settings.BackendApiUrl))
                    {
                        try
                        {
                            var uri = new Uri(_settings.BackendApiUrl);
                            txtServerIp.Text = uri.Host;
                            numServerPort.Value = uri.Port;
                        }
                        catch
                        {
                            // Use defaults if URL parsing fails
                            txtServerIp.Text = "localhost";
                            numServerPort.Value = 5135;
                        }
                    }

                    // Set agent name
                    txtAgentName.Text = string.IsNullOrEmpty(_settings.AgentName) ? 
                        Environment.MachineName : _settings.AgentName;
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error loading agent configuration: {ex.Message}", 
                    "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }
        
        /// <summary>
        /// Tests connection to the backend server
        /// </summary>
        private async void BtnTestConnection_Click(object sender, EventArgs e)
        {
            await TestBackendConnection();
        }
        
        /// <summary>
        /// Tests the connection to the backend
        /// </summary>
        private async Task TestBackendConnection()
        {
            // First save the current settings
            SaveFormToSettings();
            
            btnTestConnection.Enabled = false;
            btnTestConnection.Text = "Testing...";
            
            try
            {
                // If registered, try to validate the API key
                if (_isRegistered)
                {
                    bool isValid = await _agentIdentityService.ValidateApiKeyAsync();
                    UpdateConnectionStatus(isValid);
                    
                    if (isValid)
                    {
                        MessageBox.Show("Successfully connected to the backend server!", "Connection Test", 
                            MessageBoxButtons.OK, MessageBoxIcon.Information);
                    }
                    else
                    {
                        MessageBox.Show("Failed to connect to the backend server. The API key may be invalid.", 
                            "Connection Test", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                    }
                }
                else
                {
                    // Just test if we can reach the server
                    string baseUrl = $"{(numServerPort.Value == 443 ? "https" : "http")}://{txtServerIp.Text}:{numServerPort.Value}";
                    
                    try
                    {
                        using var client = new HttpClient();
                        client.Timeout = TimeSpan.FromSeconds(5); // 5 second timeout
                        
                        // Try to ping the server
                        bool canPing = await Task.Run(() => 
                        {
                            try
                            {
                                using var ping = new Ping();
                                var reply = ping.Send(txtServerIp.Text, 3000);
                                return reply.Status == IPStatus.Success;
                            }
                            catch
                            {
                                return false;
                            }
                        });
                        
                        UpdateConnectionStatus(canPing);
                        
                        if (canPing)
                        {
                            MessageBox.Show($"Backend server at {baseUrl} is reachable.", 
                                "Connection Test", MessageBoxButtons.OK, MessageBoxIcon.Information);
                        }
                        else
                        {
                            MessageBox.Show($"Could not reach the backend server at {baseUrl}. " +
                                "Please check the IP address and port.", 
                                "Connection Test", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                        }
                    }
                    catch (Exception ex)
                    {
                        UpdateConnectionStatus(false);
                        MessageBox.Show($"Error connecting to server: {ex.Message}", 
                            "Connection Test", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    }
                }
            }
            catch (Exception ex)
            {
                UpdateConnectionStatus(false);
                MessageBox.Show($"Error testing connection: {ex.Message}", 
                    "Connection Test", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            finally
            {
                btnTestConnection.Enabled = true;
                btnTestConnection.Text = "Test Connection";
            }
        }
        
        /// <summary>
        /// Updates the connection status indicator
        /// </summary>
        private void UpdateConnectionStatus(bool isConnected)
        {
            pbConnectionStatus.BackColor = isConnected ? Color.Green : Color.Red;
        }
        
        /// <summary>
        /// Updates the UI based on registration status
        /// </summary>
        private void UpdateRegistrationStatus(bool isRegistered)
        {
            if (isRegistered)
            {
                lblStatusValue.Text = "Registered";
                lblStatusValue.ForeColor = Color.Green;
                txtAgentId.Text = _agentId;
                txtApiKey.Text = _apiKey;
            }
            else
            {
                lblStatusValue.Text = "Not Registered";
                lblStatusValue.ForeColor = Color.Red;
                txtAgentId.Clear();
                txtApiKey.Clear();
            }
        }
        
        /// <summary>
        /// Loads settings into the form
        /// </summary>
        private void LoadSettingsIntoForm()
        {
            // Skip in token mode
            if (_isTokenMode)
                return;
                
            // Parse server URL into IP and port
            string serverUrl = _settings.BackendApiUrl;
            if (!string.IsNullOrEmpty(serverUrl))
            {
                try
                {
                    // Extract host and port from URL
                    Uri uri = new Uri(serverUrl);
                    txtServerIp.Text = uri.Host;
                    if (uri.Port > 0 && uri.Port != 80 && uri.Port != 443)
                    {
                        numServerPort.Value = uri.Port;
                    }
                }
                catch
                {
                    // If URL parsing fails, keep defaults
                }
            }
            
            txtAgentName.Text = _settings.AgentName;
        }

        /// <summary>
        /// Save button handler
        /// </summary>
        private void BtnSave_Click(object sender, EventArgs e)
        {
            try
            {
                SaveFormToSettings();
                SaveSettingsToFile();
                MessageBox.Show("Configuration saved successfully.", "Success", MessageBoxButtons.OK, MessageBoxIcon.Information);
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error saving configuration: {ex.Message}", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        /// <summary>
        /// Register button handler (redirects to Add Agent)
        /// </summary>
        private void BtnRegister_Click(object sender, EventArgs e)
        {
            // Simply redirect to the Add Agent handler
            BtnAddAgent_Click(sender, e);
        }

        /// <summary>
        /// Saves form values to settings object
        /// </summary>
        private void SaveFormToSettings()
        {
            // Skip in token mode as settings will come from the server
            if (_isTokenMode)
                return;
                
            // Generate API URL from IP and port
            string scheme = (int)numServerPort.Value == 443 ? "https" : "http";
            _settings.BackendApiUrl = $"{scheme}://{txtServerIp.Text}:{numServerPort.Value}";
            
            // Copy URL to gRPC as well (they should be the same)
            _settings.BackendGrpcUrl = _settings.BackendApiUrl;
            
            // Save agent name
            _settings.AgentName = txtAgentName.Text;
            
            // Set up default collectors if they don't exist
            if (_settings.Collectors == null || _settings.Collectors.Count == 0)
            {
                _settings.Collectors = new List<CollectorSettings>();
            
            // Set Windows Event Log collector by default if on Windows
                if (OperatingSystem.IsWindows())
            {
                _settings.Collectors.Add(new CollectorSettings
                {
                    Type = "WindowsEventLog",
                    Enabled = true,
                    IntervalSeconds = 30,
                    Properties = new Dictionary<string, string>
                    {
                        { "EventLogs", "Application,System,Security" },
                        { "MaxEvents", "100" }
                    }
                });
                }
                // Set Linux syslog collector by default if on Linux
                else if (OperatingSystem.IsLinux())
                {
                    _settings.Collectors.Add(new CollectorSettings
                    {
                        Type = "LinuxSyslog",
                        Enabled = true,
                        IntervalSeconds = 30,
                        Properties = new Dictionary<string, string>
                        {
                            { "LogFiles", "/var/log/syslog,/var/log/auth.log" },
                            { "MaxLinesPerRead", "1000" }
                        }
                    });
                }
            }
        }

        /// <summary>
        /// Saves settings to appsettings.json file
        /// </summary>
        private void SaveSettingsToFile()
        {
            string configPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "appsettings.json");
            
            try
            {
                // Read existing config
                string json = File.ReadAllText(configPath);
                var config = JsonSerializer.Deserialize<Dictionary<string, object>>(json, new JsonSerializerOptions { AllowTrailingCommas = true }) 
                    ?? new Dictionary<string, object>();
                
                // Update Agent section with serialized settings
                string settingsJson = JsonSerializer.Serialize(_settings, new JsonSerializerOptions { WriteIndented = true });
                var settingsObj = JsonSerializer.Deserialize<Dictionary<string, object>>(settingsJson)
                    ?? new Dictionary<string, object>();
                
                    config["Agent"] = settingsObj;
                    
                    // Write back to file
                    string updatedJson = JsonSerializer.Serialize(config, new JsonSerializerOptions { WriteIndented = true });
                    File.WriteAllText(configPath, updatedJson);
            }
            catch (Exception ex)
            {
                throw new Exception($"Failed to save settings to file: {ex.Message}", ex);
            }
        }
        
        /// <summary>
        /// HttpClient helper with timeout support
        /// </summary>
        private class HttpClientHelper : IDisposable
        {
            private readonly HttpClient _client;
            
            public HttpClientHelper()
            {
                _client = new HttpClient();
                Timeout = TimeSpan.FromSeconds(10); // Default 10 seconds
            }
            
            public TimeSpan Timeout
            {
                get => _client.Timeout;
                set => _client.Timeout = value;
            }
            
            public async Task<string> DownloadStringAsync(Uri address)
            {
                try
                {
                    return await _client.GetStringAsync(address);
                }
                catch (TaskCanceledException)
                {
                    throw new TimeoutException($"The request to {address} timed out.");
                }
            }
            
            public void Dispose()
            {
                _client.Dispose();
            }
        }

        /// <summary>
        /// Handler for the Add Agent button - registers the agent and closes the form
        /// </summary>
        private async void BtnAddAgent_Click(object? sender, EventArgs e)
        {
            if (_isTokenMode && !string.IsNullOrWhiteSpace(txtToken.Text))
            {
                _deploymentToken = txtToken.Text.Trim();
                await RegisterWithTokenAsync();
                return;
            }

            // Mode konfigurasi manual - gunakan implementasi yang ada
            if (string.IsNullOrWhiteSpace(txtServerIp.Text))
            {
                MessageBox.Show("Server URL harus diisi!", "Validasi", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                txtServerIp.Focus();
                return;
            }

            if (string.IsNullOrWhiteSpace(txtAgentName.Text))
            {
                MessageBox.Show("Nama Agent harus diisi!", "Validasi", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                txtAgentName.Focus();
                return;
            }

            btnAddAgent.Enabled = false;
            lblStatus.Text = "Mendaftarkan agent...";

            try
            {
                // Simpan pengaturan dari form
                SaveFormToSettings();

                // Daftarkan agent
                var result = await _agentIdentityService.RegisterAgentAsync(
                    txtAgentName.Text.Trim(),
                    txtServerIp.Text.Trim(),
                    (int)numServerPort.Value);

                if (result.Success)
                {
                    _isRegistered = true;
                    _agentId = result.AgentId;
                    _apiKey = result.ApiKey;

                    MessageBox.Show("Agent berhasil didaftarkan ke server SIEM!", 
                        "Registrasi Berhasil", MessageBoxButtons.OK, MessageBoxIcon.Information);

                    // Update status registrasi di UI
                    UpdateRegistrationStatus(true);

                    // Simpan pengaturan ke file
                    SaveSettingsToFile();
                }
                else
                {
                    MessageBox.Show($"Gagal mendaftarkan agent: {result.Message}", 
                        "Registrasi Gagal", MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Terjadi kesalahan saat mendaftarkan agent: {ex.Message}", 
                    "Kesalahan", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            finally
            {
                btnAddAgent.Enabled = true;
            }
        }

        private async Task RegisterWithTokenAsync()
        {
            if (string.IsNullOrWhiteSpace(_deploymentToken))
            {
                MessageBox.Show("Token deployment tidak valid. Silakan masukkan token yang valid atau gunakan konfigurasi manual.",
                    "Token Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return;
            }

            lblStatus.Text = "Mendaftarkan menggunakan token...";
            txtToken.Enabled = false;
            btnAddAgent.Enabled = false;
            
            try
            {
                // Panggil API untuk mendaftarkan dengan token
                var result = await _agentIdentityService.RegisterWithTokenAsync(_deploymentToken);
                
                if (result.Success)
                {
                    _isRegistered = true;
                    _agentId = result.AgentId;
                    _apiKey = result.ApiKey;
                    
                    // Update UI untuk menunjukkan status registrasi
                    UpdateRegistrationStatus(true);
                    
                    MessageBox.Show("Registrasi berhasil! Agent telah terdaftar menggunakan token deployment.",
                        "Registrasi Berhasil", MessageBoxButtons.OK, MessageBoxIcon.Information);
                    
                    // Simpan pengaturan baru
                    SaveFormToSettings();
                    SaveSettingsToFile();
                }
                else
                {
                    MessageBox.Show($"Gagal mendaftarkan agent: {result.Message}",
                        "Registrasi Gagal", MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Terjadi kesalahan saat mendaftarkan agent: {ex.Message}",
                    "Kesalahan", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            finally
            {
                txtToken.Enabled = true;
                btnAddAgent.Enabled = true;
            }
        }
    }
} 