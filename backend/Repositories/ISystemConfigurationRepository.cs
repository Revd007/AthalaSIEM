using System.Threading.Tasks;
using AthalaSIEM.Backend.Models;

namespace AthalaSIEM.Backend.Repositories
{
    public interface ISystemConfigurationRepository
    {
        Task<SystemConfiguration> GetConfigurationAsync();
        Task UpdateConfigurationAsync(SystemConfiguration configuration);
        Task<string> GetSystemSecretAsync();
        Task SetSystemSecretAsync(string secret);
    }
} 