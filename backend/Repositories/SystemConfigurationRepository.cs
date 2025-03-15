using System;
using System.Threading.Tasks;
using AthalaSIEM.Backend.Models;
using Backend.Data;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging;

namespace AthalaSIEM.Backend.Repositories
{
    public class SystemConfigurationRepository : ISystemConfigurationRepository
    {
        private readonly ApplicationDbContext _context;
        private readonly ILogger<SystemConfigurationRepository> _logger;

        public SystemConfigurationRepository(
            ApplicationDbContext context,
            ILogger<SystemConfigurationRepository> logger)
        {
            _context = context;
            _logger = logger;
        }

        public async Task<SystemConfiguration> GetConfigurationAsync()
        {
            try
            {
                var config = await _context.SystemConfiguration.FirstOrDefaultAsync();
                return config ?? new SystemConfiguration();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting system configuration");
                throw;
            }
        }

        public async Task UpdateConfigurationAsync(SystemConfiguration configuration)
        {
            try
            {
                var existing = await _context.SystemConfiguration.FirstOrDefaultAsync();
                if (existing == null)
                {
                    _context.SystemConfiguration.Add(configuration);
                }
                else
                {
                    _context.Entry(existing).CurrentValues.SetValues(configuration);
                }

                await _context.SaveChangesAsync();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error updating system configuration");
                throw;
            }
        }

        public async Task<string> GetSystemSecretAsync()
        {
            try
            {
                var config = await GetConfigurationAsync();
                return config.SystemSecret ?? string.Empty;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting system secret");
                throw;
            }
        }

        public async Task SetSystemSecretAsync(string secret)
        {
            try
            {
                var config = await GetConfigurationAsync();
                config.SystemSecret = secret;
                await UpdateConfigurationAsync(config);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error setting system secret");
                throw;
            }
        }
    }
} 