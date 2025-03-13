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

        // Form controls
        private Label lblTitle;
        private Label lblConnectionStatus;
        private PictureBox pbConnectionStatus;
        private Label lblStatus;
        private Label lblStatusValue;
        private Label lblServerIp;
        private TextBox txtServerIp;
        private Label lblServerPort;
        private NumericUpDown numServerPort;
        private Label lblAgentName;
        private TextBox txtAgentName;
        private Label lblAgentId;
        private TextBox txtAgentId;
        private Label lblApiKey;
        private TextBox txtApiKey;
        private Button btnTestConnection;
        private Button btnRegister;
        private Button btnSave;

        public AgentConfigurationForm(IAgentIdentityService agentIdentityService, AgentSettings settings)
        {
            _agentIdentityService = agentIdentityService ?? throw new ArgumentNullException(nameof(agentIdentityService));
            _settings = settings ?? throw new ArgumentNullException(nameof(settings));
            InitializeComponent();
        }

        /// <summary>
        /// Initialize form components
        /// </summary>
        private void InitializeComponent()
        {
            this.SuspendLayout();
            
            // Form settings
            this.Text = "AthalaSIEM Agent Configuration";
            this.ClientSize = new Size(400, 400);
            this.MaximizeBox = false;
            this.MinimizeBox = true;
            this.StartPosition = FormStartPosition.CenterScreen;
            this.FormBorderStyle = FormBorderStyle.FixedDialog;
            this.Icon = SystemIcons.Shield;
            
            // Title
            lblTitle = new Label
            {
                Text = "AthalaSIEM Agent Configuration",
                Location = new Point(20, 15),
                Width = 360,
                TextAlign = ContentAlignment.MiddleCenter,
                Font = new Font(this.Font.FontFamily, 12, FontStyle.Bold)
            };
            
            // Connection status
            lblConnectionStatus = new Label
            {
                Text = "Backend Connection:",
                Location = new Point(20, 50),
                AutoSize = true
            };
            
            pbConnectionStatus = new PictureBox
            {
                Location = new Point(140, 48),
                Size = new Size(16, 16),
                BackColor = Color.Red // Default to red (not connected)
            };
            
            // Status section
            lblStatus = new Label
            {
                Text = "Registration Status:",
                Location = new Point(20, 80),
                AutoSize = true
            };
            
            lblStatusValue = new Label
            {
                Text = "Not Registered",
                Location = new Point(140, 80),
                AutoSize = true,
                ForeColor = Color.Red,
                Font = new Font(this.Font, FontStyle.Bold)
            };
            
            // Server IP field
            lblServerIp = new Label
            {
                Text = "Server IP:",
                Location = new Point(20, 120),
                AutoSize = true
            };
            
            txtServerIp = new TextBox
            {
                Location = new Point(140, 120),
                Width = 200,
                Text = "127.0.0.1"
            };
            
            // Server Port field
            lblServerPort = new Label
            {
                Text = "Server Port:",
                Location = new Point(20, 150),
                AutoSize = true
            };
            
            numServerPort = new NumericUpDown
            {
                Location = new Point(140, 150),
                Width = 80,
                Minimum = 1,
                Maximum = 65535,
                Value = 5001
            };
            
            // Agent Name field
            lblAgentName = new Label
            {
                Text = "Agent Name:",
                Location = new Point(20, 180),
                AutoSize = true
            };
            
            txtAgentName = new TextBox
            {
                Location = new Point(140, 180),
                Width = 200,
                Text = Environment.MachineName
            };
            
            // Agent ID field (read-only)
            lblAgentId = new Label
            {
                Text = "Agent ID:",
                Location = new Point(20, 210),
                AutoSize = true
            };
            
            txtAgentId = new TextBox
            {
                Location = new Point(140, 210),
                Width = 200,
                ReadOnly = true
            };
            
            // API Key field (read-only)
            lblApiKey = new Label
            {
                Text = "API Key:",
                Location = new Point(20, 240),
                AutoSize = true
            };
            
            txtApiKey = new TextBox
            {
                Location = new Point(140, 240),
                Width = 200,
                ReadOnly = true
            };
            
            // Test Connection button
            btnTestConnection = new Button
            {
                Text = "Test Connection",
                Location = new Point(140, 280),
                Width = 120,
                Height = 30,
                BackColor = Color.FromArgb(0, 122, 204),
                ForeColor = Color.White,
                FlatStyle = FlatStyle.Flat,
                Cursor = Cursors.Hand
            };
            btnTestConnection.Click += new EventHandler(BtnTestConnection_Click);
            
            // Register button
            btnRegister = new Button
            {
                Text = "Register",
                Location = new Point(140, 320),
                Width = 120,
                Height = 40,
                BackColor = Color.FromArgb(0, 122, 204),
                ForeColor = Color.White,
                FlatStyle = FlatStyle.Flat,
                Cursor = Cursors.Hand
            };
            btnRegister.Click += new EventHandler(BtnRegister_Click);
            
            // Save button
            btnSave = new Button
            {
                Text = "Save",
                Location = new Point(270, 320),
                Width = 80,
                Height = 40,
                BackColor = Color.FromArgb(0, 122, 204),
                ForeColor = Color.White,
                FlatStyle = FlatStyle.Flat,
                Cursor = Cursors.Hand
            };
            btnSave.Click += new EventHandler(BtnSave_Click);
            
            // Add controls to form
            this.Controls.Add(lblTitle);
            this.Controls.Add(lblConnectionStatus);
            this.Controls.Add(pbConnectionStatus);
            this.Controls.Add(lblStatus);
            this.Controls.Add(lblStatusValue);
            this.Controls.Add(lblServerIp);
            this.Controls.Add(txtServerIp);
            this.Controls.Add(lblServerPort);
            this.Controls.Add(numServerPort);
            this.Controls.Add(lblAgentName);
            this.Controls.Add(txtAgentName);
            this.Controls.Add(lblAgentId);
            this.Controls.Add(txtAgentId);
            this.Controls.Add(lblApiKey);
            this.Controls.Add(txtApiKey);
            this.Controls.Add(btnTestConnection);
            this.Controls.Add(btnRegister);
            this.Controls.Add(btnSave);
            
            // Set handlers
            this.Load += new EventHandler(Form_Load);
            
            this.ResumeLayout(false);
        }

        /// <summary>
        /// Form load handler
        /// </summary>
        private async void Form_Load(object sender, EventArgs e)
        {
            await LoadAgentStatusAsync();
            LoadSettingsIntoForm();
        }

        /// <summary>
        /// Loads the agent registration status
        /// </summary>
        private async Task LoadAgentStatusAsync()
        {
            try
            {
                _isRegistered = await _agentIdentityService.IsRegisteredAsync();
                if (_isRegistered)
                {
                    _agentId = await _agentIdentityService.GetAgentIdAsync();
                    _apiKey = await _agentIdentityService.GetApiKeyAsync();
                    UpdateRegistrationStatus(true);
                    
                    // If registered, validate connection with the backend
                    await TestBackendConnection();
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error checking agent registration status: {ex.Message}", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
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
                btnRegister.Text = "Re-Register";
                txtAgentId.Text = _agentId;
                txtApiKey.Text = _apiKey;
            }
            else
            {
                lblStatusValue.Text = "Not Registered";
                lblStatusValue.ForeColor = Color.Red;
                btnRegister.Text = "Register";
                txtAgentId.Clear();
                txtApiKey.Clear();
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
                        using var client = new WebClient();
                        client.Timeout = 5000; // 5 second timeout
                        
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
        /// Loads settings into the form
        /// </summary>
        private void LoadSettingsIntoForm()
        {
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
        /// Register button handler
        /// </summary>
        private async void BtnRegister_Click(object sender, EventArgs e)
        {
            if (string.IsNullOrWhiteSpace(txtServerIp.Text))
            {
                MessageBox.Show("Server IP is required.", "Validation Error", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }
            
            // Update settings with form values
            SaveFormToSettings();
            
            // Disable controls during registration
            btnRegister.Enabled = false;
            btnRegister.Text = "Registering...";
            
            try
            {
                // Save settings first
                SaveSettingsToFile();
                
                // Then register
                bool result = await _agentIdentityService.RegisterAgentAsync();
                if (result)
                {
                    _isRegistered = true;
                    _agentId = await _agentIdentityService.GetAgentIdAsync();
                    _apiKey = await _agentIdentityService.GetApiKeyAsync();
                    
                    UpdateRegistrationStatus(true);
                    UpdateConnectionStatus(true);
                    MessageBox.Show("Agent registered successfully with the SIEM backend!", "Success", MessageBoxButtons.OK, MessageBoxIcon.Information);
                }
                else
                {
                    UpdateConnectionStatus(false);
                    MessageBox.Show("Failed to register agent. Please check your settings and try again.", 
                        "Registration Failed", MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
            }
            catch (Exception ex)
            {
                UpdateConnectionStatus(false);
                MessageBox.Show($"Error registering agent: {ex.Message}", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            finally
            {
                btnRegister.Enabled = true;
                btnRegister.Text = _isRegistered ? "Re-Register" : "Register";
            }
        }

        /// <summary>
        /// Saves form values to settings object
        /// </summary>
        private void SaveFormToSettings()
        {
            // Generate API URL from IP and port
            string scheme = (int)numServerPort.Value == 443 ? "https" : "http";
            _settings.BackendApiUrl = $"{scheme}://{txtServerIp.Text}:{numServerPort.Value}";
            
            // Copy URL to gRPC as well
            _settings.BackendGrpcUrl = _settings.BackendApiUrl;
            
            // Save agent name
            _settings.AgentName = txtAgentName.Text;
            
            // Default values for other settings
            if (_settings.Collectors == null)
            {
                _settings.Collectors = new List<CollectorSettings>();
            }
            
            // Set Windows Event Log collector by default if on Windows
            if (OperatingSystem.IsWindows() && !_settings.Collectors.Exists(c => c.Type == "WindowsEventLog"))
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
                var config = JsonSerializer.Deserialize<Dictionary<string, object>>(json, new JsonSerializerOptions { AllowTrailingCommas = true });
                
                // Update Agent section with serialized settings
                string settingsJson = JsonSerializer.Serialize(_settings, new JsonSerializerOptions { WriteIndented = true });
                var settingsObj = JsonSerializer.Deserialize<Dictionary<string, object>>(settingsJson);
                
                if (config != null)
                {
                    config["Agent"] = settingsObj;
                    
                    // Write back to file
                    string updatedJson = JsonSerializer.Serialize(config, new JsonSerializerOptions { WriteIndented = true });
                    File.WriteAllText(configPath, updatedJson);
                }
            }
            catch (Exception ex)
            {
                throw new Exception($"Failed to save settings to file: {ex.Message}", ex);
            }
        }
        
        /// <summary>
        /// WebClient with timeout support
        /// </summary>
        private class WebClient : System.Net.WebClient
        {
            public int Timeout { get; set; } = 10000; // Default 10 seconds

            protected override WebRequest GetWebRequest(Uri address)
            {
                var request = base.GetWebRequest(address);
                if (request != null)
                {
                    request.Timeout = Timeout;
                }
                return request;
            }
        }
    }
} 