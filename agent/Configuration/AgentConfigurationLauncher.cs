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
        private readonly SynchronizationContext _syncContext;
        
        public AgentConfigurationLauncher(IServiceProvider serviceProvider, ILogger<AgentConfigurationLauncher> logger)
        {
            _serviceProvider = serviceProvider ?? throw new ArgumentNullException(nameof(serviceProvider));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            
            // Store the synchronization context from the calling thread (if running UI thread)
            _syncContext = SynchronizationContext.Current;
        }
        
        /// <summary>
        /// Shows the agent configuration form
        /// </summary>
        /// <param name="modal">Whether to show the form modally</param>
        /// <returns>True if the agent is fully configured and registered, false otherwise</returns>
        public async Task<bool> ShowConfigurationFormAsync(bool modal = true)
        {
            try
            {
                _logger.LogInformation("Launching agent configuration UI");
                
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
                        
                        // Create and show the form
                        var form = new AgentConfigurationForm(agentIdentityService, settings);
                        
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