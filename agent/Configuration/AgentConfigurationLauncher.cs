using System;
using System.Diagnostics;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using AthalaSIEM.Agent.Models;
using AthalaSIEM.Agent.Security;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;

namespace AthalaSIEM.Agent.Configuration
{
    /// <summary>
    /// Launcher for the agent configuration UI
    /// </summary>
    public class AgentConfigurationLauncher
    {
        private readonly IServiceProvider _serviceProvider;
        private readonly ILogger<AgentConfigurationLauncher> _logger;
        private readonly SynchronizationContext? _syncContext;
        
        public AgentConfigurationLauncher(IServiceProvider serviceProvider, ILogger<AgentConfigurationLauncher> logger)
        {
            _serviceProvider = serviceProvider ?? throw new ArgumentNullException(nameof(serviceProvider));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            
            // Store the synchronization context from the calling thread (if running UI thread)
            // This might be null when running from a non-UI thread
            _syncContext = SynchronizationContext.Current;
        }
        
        /// <summary>
        /// Shows the agent configuration form
        /// </summary>
        /// <param name="modal">Whether to show the form modally</param>
        /// <returns>True if the agent is fully configured and registered, false otherwise</returns>
        public async Task<bool> ShowConfigurationFormAsync(bool modal = true)
        {
            return await ShowConfigurationFormAsync(string.Empty, modal);
        }

        /// <summary>
        /// Shows the agent configuration form with a deployment token
        /// </summary>
        /// <param name="token">Deployment token for pre-configuration</param>
        /// <param name="modal">Whether to show the form modally</param>
        /// <returns>True if the agent is fully configured and registered, false otherwise</returns>
        public async Task<bool> ShowConfigurationFormAsync(string token, bool modal = true)
        {
            try
            {
                _logger.LogInformation("Launching agent configuration UI{0}", 
                    !string.IsNullOrEmpty(token) ? " with deployment token" : "");
                
                // Get required services from DI container
                var agentIdentityService = _serviceProvider.GetRequiredService<IAgentIdentityService>();
                var settings = _serviceProvider.GetRequiredService<AgentSettings>();
                
                // Create a new thread for the UI
                bool isConfigured = false;
                var thread = new Thread(() =>
                {
                    try
                    {
                        Application.SetHighDpiMode(HighDpiMode.SystemAware);
                        Application.EnableVisualStyles();
                        Application.SetCompatibleTextRenderingDefault(false);
                        
                        // Create and show the form with token if provided
                        var form = new AgentConfigurationForm(agentIdentityService, settings, token)
                        {
                            lblTitle = new Label(),
                            lblServerIp = new Label(),
                            txtServerIp = new TextBox(),
                            lblServerPort = new Label(),
                            numServerPort = new NumericUpDown(),
                            lblAgentName = new Label(),
                            txtAgentName = new TextBox(),
                            btnAddAgent = new Button(),
                            lblTokenMode = new Label(),
                            lblToken = new Label(),
                            txtToken = new TextBox(),
                            lblConnectionStatus = new Label(),
                            pbConnectionStatus = new PictureBox(),
                            lblStatus = new Label(),
                            lblStatusValue = new Label(),
                            lblAgentId = new Label(),
                            txtAgentId = new TextBox(),
                            lblApiKey = new Label(),
                            txtApiKey = new TextBox(),
                            btnTestConnection = new Button(),
                            btnSave = new Button()
                        };
                        
                        if (modal)
                        {
                            form.ShowDialog();
                        }
                        else
                        {
                            form.Show();
                            Application.Run();
                        }
                        
                        // Check if the agent is registered after configuration
                        isConfigured = agentIdentityService.IsRegisteredAsync().GetAwaiter().GetResult();
                    }
                    catch (Exception ex)
                    {
                        _logger.LogError(ex, "Error in configuration UI thread");
                        
                        // Show error message on UI thread
                        MessageBox.Show($"An error occurred in the configuration UI: {ex.Message}", 
                            "Configuration Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    }
                });
                
                // Set thread as STA for WinForms
                thread.SetApartmentState(ApartmentState.STA);
                thread.Start();
                
                // Wait for thread to complete if modal
                if (modal)
                {
                    await Task.Run(() => thread.Join());
                }
                
                return isConfigured;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to launch configuration UI");
                return false;
            }
        }

        /// <summary>
        /// Registers agent without showing UI when a valid token is provided
        /// </summary>
        /// <param name="token">Deployment token</param>
        /// <returns>True if registration succeeds, false otherwise</returns>
        public async Task<bool> RegisterWithTokenSilentlyAsync(string token)
        {
            if (string.IsNullOrWhiteSpace(token))
            {
                _logger.LogError("No deployment token provided for silent registration");
                return false;
            }

            _logger.LogInformation("Attempting silent registration with token");
            
            try
            {
                // Get the identity service
                var agentIdentityService = _serviceProvider.GetRequiredService<IAgentIdentityService>();
                
                // Check if we already have an identity
                if (await agentIdentityService.HasValidIdentityAsync())
                {
                    _logger.LogInformation("Agent already has a valid identity, skipping registration");
                    return true;
                }
                
                // Register with token
                var result = await agentIdentityService.RegisterWithTokenAsync(token);
                
                if (result.Success)
                {
                    _logger.LogInformation("Silent registration successful");
                    return true;
                }
                else
                {
                    _logger.LogError($"Silent registration failed: {result.Message}");
                    return false;
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error during silent registration");
                return false;
            }
        }
        
        /// <summary>
        /// Checks if the agent is running in interactive mode (i.e., with a desktop session)
        /// </summary>
        /// <returns>True if the agent is running in interactive mode</returns>
        public static bool IsInteractiveMode()
        {
            try
            {
                // Check if running as a Windows service
                if (!OperatingSystem.IsWindows())
                {
                    // On Linux, check if running as a systemd service
                    // If DISPLAY environment variable is not set, it's likely running as a service
                    return !string.IsNullOrEmpty(Environment.GetEnvironmentVariable("DISPLAY"));
                }
                
                // On Windows, check if running as a service
                using var currentProcess = Process.GetCurrentProcess();
                return currentProcess.SessionId > 0 && Environment.UserInteractive;
            }
            catch
            {
                // If we can't determine, assume not interactive to be safe
                return false;
            }
        }
        
        /// <summary>
        /// Checks if the agent is already configured and registered
        /// </summary>
        /// <returns>True if the agent is already configured and registered</returns>
        public async Task<bool> IsAgentConfiguredAsync()
        {
            try
            {
                var agentIdentityService = _serviceProvider.GetRequiredService<IAgentIdentityService>();
                return await agentIdentityService.IsRegisteredAsync();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error checking if agent is configured");
                return false;
            }
        }
        
        /// <summary>
        /// Checks if this is a first-time installation by looking for installation markers
        /// </summary>
        /// <returns>True if this is a first-time installation</returns>
        public bool IsFirstTimeInstallation()
        {
            try
            {
                string configPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "appsettings.json");
                string installMarkerPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, ".installed");
                
                // If the install marker exists, this is not a first-time installation
                if (File.Exists(installMarkerPath))
                {
                    return false;
                }
                
                // Create the install marker
                File.WriteAllText(installMarkerPath, DateTime.UtcNow.ToString("o"));
                
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error checking if this is a first-time installation");
                return false;
            }
        }
    }
} 