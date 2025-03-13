using AthalaSIEM.Agent.Collectors;
using AthalaSIEM.Agent.Communication;
using AthalaSIEM.Agent.Models;
using AthalaSIEM.Agent.Security;
using AthalaSIEM.Agent.Services;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Options;
using System;
using System.IO;
using System.Net.Http;

namespace AthalaSIEM.Agent.Extensions
{
    public static class ServiceCollectionExtensions
    {
        public static IServiceCollection AddAgentServices(this IServiceCollection services, IConfiguration configuration)
        {
            // Register configuration
            services.Configure<AgentSettings>(configuration.GetSection("AgentSettings"));
            services.AddSingleton(sp => sp.GetRequiredService<IOptions<AgentSettings>>().Value);

            // Register HTTP client
            services.AddHttpClient("SiemBackend", (sp, client) =>
            {
                var settings = sp.GetRequiredService<AgentSettings>();
                client.BaseAddress = new Uri(settings.BackendUrl);
                client.Timeout = TimeSpan.FromMinutes(2);
            }).ConfigurePrimaryHttpMessageHandler(() =>
            {
                return new HttpClientHandler
                {
                    ServerCertificateCustomValidationCallback = (message, cert, chain, errors) => true // For development only
                };
            });

            // Register gRPC client
            services.AddGrpcClient<SiemService.SiemServiceClient>((sp, options) =>
            {
                var settings = sp.GetRequiredService<AgentSettings>();
                options.Address = new Uri(settings.BackendUrl);
            }).ConfigureChannel(options =>
            {
                options.UnsafeUseInsecureChannelCallCredentials = true; // For development only
            });

            // Register core services
            services.AddSingleton<IEncryptionService, AesEncryptionService>();
            services.AddSingleton<IAgentIdentityService, AgentIdentityService>();
            services.AddSingleton<IAgentHealthMonitor, AgentHealthMonitor>();
            services.AddSingleton<ILogForwarder, GrpcLogForwarder>();
            services.AddSingleton<ILogNormalizer, LogNormalizer>();
            services.AddSingleton<ILogCollectorFactory, LogCollectorFactory>();

            // Ensure logs directory exists
            var logsPath = Path.Combine(AppContext.BaseDirectory, "logs");
            if (!Directory.Exists(logsPath))
            {
                Directory.CreateDirectory(logsPath);
            }

            return services;
        }
    }
} 