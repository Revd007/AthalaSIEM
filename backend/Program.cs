using System;
using System.Text;
using Backend.Data;
using Backend.Data.Repositories;
using Backend.Services;
using Backend.Services.Background;
using Backend.Models;
using Microsoft.AspNetCore.Authentication.JwtBearer;
using Microsoft.AspNetCore.Builder;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.IdentityModel.Tokens;
using Microsoft.OpenApi.Models;
using Npgsql.EntityFrameworkCore.PostgreSQL;
using Grpc.Net.Client;
// using Microsoft.AspNetCore.Server.Kestrel.Https; // COMMENTED OUT FOR DEVELOPMENT
using System.Net.Security;
using System.Security.Authentication;
using System.Security.Cryptography.X509Certificates;
using Grpc.Net.Client.Configuration;
using Microsoft.AspNetCore.Server.Kestrel.Core;
using System.Net;
using System.Net.Sockets;
using System.Security.Cryptography;
using System.Text.Json.Serialization;
using Serilog;
using Serilog.Events;
using Microsoft.Extensions.Options;
using Swashbuckle.AspNetCore.Filters;
using Microsoft.AspNetCore.Mvc;

// Enable HTTP/2 over HTTP (without TLS) for gRPC
AppContext.SetSwitch("System.Net.Http.SocketsHttpHandler.Http2UnencryptedSupport", true);

var builder = WebApplication.CreateBuilder(args);

// Configure logging
builder.Logging.ClearProviders();
builder.Logging.AddConsole();
builder.Logging.AddDebug();
builder.Logging.SetMinimumLevel(LogLevel.Debug);
builder.Logging.AddFilter("Backend", LogLevel.Debug);

// Add services to the container
builder.Services.AddControllers()
    .AddJsonOptions(options =>
    {
        // Use camelCase by default, but respect JsonPropertyName attributes
        options.JsonSerializerOptions.PropertyNamingPolicy = System.Text.Json.JsonNamingPolicy.CamelCase;
        options.JsonSerializerOptions.PropertyNameCaseInsensitive = true;
        options.JsonSerializerOptions.DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull;
        options.JsonSerializerOptions.WriteIndented = false;
    })
    .ConfigureApiBehaviorOptions(options =>
    {
        options.InvalidModelStateResponseFactory = context =>
        {
            var errors = context.ModelState
                .Where(x => x.Value?.Errors.Count > 0)
                .SelectMany(x => x.Value!.Errors)
                .Select(x => x.ErrorMessage)
                .ToList();
            
            return new BadRequestObjectResult(new { 
                message = "Validation failed", 
                errors = errors 
            });
        };
    });
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen(c =>
{
    c.SwaggerDoc("v1", new OpenApiInfo { Title = "ATHALA SIEM API", Version = "v1" });
    
    // Configure Swagger to use JWT Authentication
    c.AddSecurityDefinition("Bearer", new OpenApiSecurityScheme
    {
        Description = "JWT Authorization header using the Bearer scheme. Example: \"Authorization: Bearer {token}\"",
        Name = "Authorization",
        In = ParameterLocation.Header,
        Type = SecuritySchemeType.ApiKey,
        Scheme = "Bearer"
    });
    
    c.AddSecurityRequirement(new OpenApiSecurityRequirement
    {
        {
            new OpenApiSecurityScheme
            {
                Reference = new OpenApiReference
                {
                    Type = ReferenceType.SecurityScheme,
                    Id = "Bearer"
                }
            },
            Array.Empty<string>()
        }
    });
});

// Add gRPC services
builder.Services.AddGrpc(options =>
{
    options.EnableDetailedErrors = true;
    options.MaxReceiveMessageSize = 1024 * 1024 * 100; // 100MB
    options.MaxSendMessageSize = 1024 * 1024 * 100;    // 100MB
    
    // Get gRPC URL from configuration - REQUIRED, no hardcoded defaults
    var grpcUrl = builder.Configuration["GrpcServer:Url"];
    if (string.IsNullOrEmpty(grpcUrl))
    {
        throw new InvalidOperationException("GrpcServer:Url must be configured in appsettings.json. No hardcoded defaults for security.");
    }
    Console.WriteLine($"🔧 gRPC server configured for: {grpcUrl}");
});

// Configure gRPC client
builder.Services.AddGrpcClient<AthalaSIEM.Agent.SiemService.SiemServiceClient>(options =>
{
    var grpcUrl = builder.Configuration["GrpcServer:Url"];
    if (string.IsNullOrEmpty(grpcUrl))
    {
        throw new InvalidOperationException("GrpcServer:Url configuration is required. Please configure your gRPC server URL.");
    }
    options.Address = new Uri(grpcUrl);
})
.ConfigureChannel(options =>
{
    options.HttpHandler = new SocketsHttpHandler
    {
        KeepAlivePingDelay = TimeSpan.FromSeconds(60),
        KeepAlivePingTimeout = TimeSpan.FromSeconds(30),
        PooledConnectionIdleTimeout = TimeSpan.FromMinutes(5),
        EnableMultipleHttp2Connections = true,
        // SslOptions = new SslClientAuthenticationOptions // COMMENTED OUT FOR DEVELOPMENT
        // {
        //     EnabledSslProtocols = SslProtocols.Tls12 | SslProtocols.Tls13,
        //     CertificateRevocationCheckMode = X509RevocationMode.Online,
        //     EncryptionPolicy = EncryptionPolicy.RequireEncryption
        // }
    };
});

// Configure CORS - MUST come from configuration, no hardcoded values
builder.Services.AddCors(options =>
{
    // Get allowed origins from configuration - REQUIRED, no defaults for security
    var allowedOrigins = builder.Configuration.GetSection("Cors:AllowedOrigins").Get<string[]>();
    
    if (allowedOrigins == null || allowedOrigins.Length == 0)
    {
        throw new InvalidOperationException("Cors:AllowedOrigins must be configured in appsettings.json. No hardcoded defaults for security.");
    }

    // Default policy (used by UseCors() without name)
    options.AddDefaultPolicy(corsBuilder =>
    {
        corsBuilder
            .WithOrigins(allowedOrigins)
            .AllowAnyMethod()
            .AllowAnyHeader()
            .AllowCredentials()
            .SetPreflightMaxAge(TimeSpan.FromHours(1));
        
        // Log configured origins for debugging
        var logger = LoggerFactory.Create(b => b.AddConsole()).CreateLogger("CORS");
        logger.LogInformation("🌐 CORS DefaultPolicy configured with origins: {Origins}", string.Join(", ", allowedOrigins));
    });
    
    // Named policy for controllers that explicitly use it
    options.AddPolicy("AllowFrontend", corsBuilder =>
    {
        corsBuilder
            .WithOrigins(allowedOrigins)
            .AllowAnyMethod()
            .AllowAnyHeader()
            .AllowCredentials()
            .SetPreflightMaxAge(TimeSpan.FromHours(1));
        
        var logger = LoggerFactory.Create(b => b.AddConsole()).CreateLogger("CORS");
        logger.LogInformation("🌐 CORS AllowFrontend policy configured with origins: {Origins}", string.Join(", ", allowedOrigins));
    });
});

// Configure database
builder.Services.AddDbContext<ApplicationDbContext>(options =>
{
    var connectionString = builder.Configuration.GetConnectionString("DefaultConnection");
    if (string.IsNullOrEmpty(connectionString))
    {
        throw new InvalidOperationException("Connection string 'DefaultConnection' not found");
    }
    options.UseNpgsql(connectionString);
});

// Configure JWT Authentication
var jwtSettings = builder.Configuration.GetSection("Jwt");
var key = Encoding.ASCII.GetBytes(jwtSettings["Key"] ?? throw new InvalidOperationException("JWT secret not configured"));

builder.Services.AddAuthentication(x =>
{
    x.DefaultAuthenticateScheme = JwtBearerDefaults.AuthenticationScheme;
    x.DefaultChallengeScheme = JwtBearerDefaults.AuthenticationScheme;
})
.AddJwtBearer(x =>
{
    x.RequireHttpsMetadata = false;
    x.SaveToken = true;
    x.TokenValidationParameters = new TokenValidationParameters
    {
        ValidateIssuerSigningKey = true,
        IssuerSigningKey = new SymmetricSecurityKey(key),
        ValidateIssuer = false,
        ValidateAudience = false,
        ClockSkew = TimeSpan.Zero
    };
});

// Add Authorization
builder.Services.AddAuthorization(options =>
{
    options.AddPolicy("RequireAdminRole", policy => 
        policy.RequireRole("Admin"));
        
    options.AddPolicy("RequireUserRole", policy => 
        policy.RequireRole("User", "Admin", "Operator"));
        
    options.AddPolicy("RequireAgentAccess", policy => 
        policy.RequireRole("User", "Admin", "Operator")
              .RequireClaim("permission", "agent:read"));
});

// Configure Kestrel - port MUST come from configuration
builder.WebHost.ConfigureKestrel(serverOptions =>
{
    // Get URL from configuration - REQUIRED, no hardcoded defaults
    var configuredUrl = builder.Configuration["Kestrel:Endpoints:Http:Url"];
    if (string.IsNullOrEmpty(configuredUrl))
    {
        throw new InvalidOperationException("Kestrel:Endpoints:Http:Url must be configured in appsettings.json. No hardcoded defaults for security.");
    }
    
    if (!Uri.TryCreate(configuredUrl, UriKind.Absolute, out var uri))
    {
        throw new InvalidOperationException($"Invalid Kestrel URL configuration: {configuredUrl}");
    }
    
    var httpPort = uri.Port;

    serverOptions.ListenAnyIP(httpPort, listenOptions =>
    {
        listenOptions.Protocols = HttpProtocols.Http1AndHttp2;
        listenOptions.UseConnectionLogging();
    });
    
    Console.WriteLine($"🚀 Backend server listening on port: {httpPort}");
    Console.WriteLine($"💡 Override via environment: ATHALA_Kestrel__Endpoints__Http__Url=http://0.0.0.0:YOUR_PORT");
});

// Register MediatR for CQRS
builder.Services.AddMediatR(cfg => cfg.RegisterServicesFromAssembly(typeof(Backend.Application.Commands.IngestLogCommand).Assembly));

// Register repositories (legacy - using renamed interfaces to avoid conflicts)
builder.Services.AddScoped<IUserRepository, UserRepository>();
builder.Services.AddScoped<Backend.Data.Repositories.ILegacyAgentRepository, Backend.Data.Repositories.AgentRepository>();
builder.Services.AddScoped<Backend.Data.Repositories.ILegacyLogEntryRepository, Backend.Data.Repositories.LogEntryRepository>();
builder.Services.AddScoped<Backend.Data.Repositories.IAlertRepository, Backend.Data.Repositories.AlertRepository>();
builder.Services.AddScoped<IDashboardRepository, DashboardRepository>();
builder.Services.AddScoped<IReportRepository, ReportRepository>();
builder.Services.AddScoped<AthalaSIEM.Backend.Repositories.IAgentDeploymentTokenRepository, AthalaSIEM.Backend.Repositories.AgentDeploymentTokenRepository>();

// Register new domain repositories
builder.Services.AddScoped<Backend.Domain.Interfaces.ILogRepository, Backend.Infrastructure.Data.Repositories.LogRepository>();
builder.Services.AddScoped<Backend.Domain.Interfaces.IAlertRepository, Backend.Infrastructure.Data.Repositories.AlertRepository>();
builder.Services.AddScoped<Backend.Domain.Interfaces.IDetectionRuleRepository, Backend.Infrastructure.Data.Repositories.DetectionRuleRepository>();
builder.Services.AddScoped<Backend.Domain.Interfaces.IAgentRepository, Backend.Infrastructure.Data.Repositories.AgentRepository>();
builder.Services.AddScoped<Backend.Infrastructure.Data.Repositories.INormalizedLogRepository, Backend.Infrastructure.Data.Repositories.NormalizedLogRepository>();

// Register services (legacy - keep for backward compatibility)
builder.Services.AddScoped<IAuthService, AuthService>();
builder.Services.AddScoped<IUserService, UserService>();
builder.Services.AddScoped<IAgentService, AgentService>();
builder.Services.AddScoped<IAlertService, AlertService>();
builder.Services.AddScoped<ILogService, LogService>();
builder.Services.AddScoped<ILogAnalysisService, LogAnalysisService>();
builder.Services.AddScoped<IDashboardService, DashboardService>();
builder.Services.AddScoped<IReportService, ReportService>();
builder.Services.AddScoped<IInstallerService, InstallerService>();

// Register new enhanced services for multi-collector support
builder.Services.AddScoped<IThreatIntelligenceService, ThreatIntelligenceService>();
builder.Services.AddScoped<ILogArchivingService, LogArchivingService>();

// Register new infrastructure services
builder.Services.AddScoped<Backend.Infrastructure.Normalizers.ILogNormalizer, Backend.Infrastructure.Normalizers.ECSLogNormalizer>();
builder.Services.AddScoped<Backend.Infrastructure.Detection.RuleEngine.IRuleParser, Backend.Infrastructure.Detection.RuleEngine.YamlRuleParser>();
builder.Services.AddScoped<Backend.Infrastructure.Detection.RuleEngine.IRuleExecutor, Backend.Infrastructure.Detection.RuleEngine.PatternMatchRuleExecutor>();
builder.Services.AddScoped<Backend.Infrastructure.Detection.IDetectionEngine, Backend.Infrastructure.Detection.DetectionEngine>();
builder.Services.AddScoped<Backend.Infrastructure.Correlation.ICorrelationEngine, Backend.Infrastructure.Correlation.TemporalCorrelator>();
builder.Services.AddScoped<Backend.Infrastructure.AlertProcessing.IAlertDeduplicator, Backend.Infrastructure.AlertProcessing.AlertDeduplicator>();
builder.Services.AddScoped<Backend.Infrastructure.AlertProcessing.IAlertSeverityScorer, Backend.Infrastructure.AlertProcessing.AlertSeverityScorer>();

// Register background services (legacy)
builder.Services.AddHostedService<Backend.Services.Background.AgentMonitoringService>();
builder.Services.AddHostedService<Backend.Services.Background.LogCleanupService>();
builder.Services.AddHostedService<Backend.Services.Background.AlertCleanupService>();
builder.Services.AddHostedService<LogArchivingService>();

// Register new workers
builder.Services.AddSingleton<Backend.Workers.LogNormalizationWorker>();
builder.Services.AddHostedService(provider => provider.GetRequiredService<Backend.Workers.LogNormalizationWorker>());
builder.Services.AddHostedService<Backend.Workers.DetectionWorker>();

var app = builder.Build();

// Configure the HTTP request pipeline
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

// Configure middleware in correct order - IMPORTANT: Routing must come before CORS
app.UseRouting();

// Apply CORS middleware BEFORE authentication and authorization
app.UseCors();

// Special handling for OPTIONS requests (preflight) - AFTER CORS middleware
app.Use(async (context, next) =>
{
    // If it's a preflight request, handle it directly
    if (context.Request.Method == "OPTIONS")
    {
        // Get allowed origins from configuration - REQUIRED, no hardcoded defaults
        var configuration = context.RequestServices.GetRequiredService<IConfiguration>();
        var allowedOrigins = configuration.GetSection("Cors:AllowedOrigins").Get<string[]>();
        
        if (allowedOrigins == null || allowedOrigins.Length == 0)
        {
            context.Response.StatusCode = 500;
            await context.Response.WriteAsync("CORS configuration error: Cors:AllowedOrigins must be configured in appsettings.json");
            await context.Response.CompleteAsync();
            return;
        }
        
        var origin = context.Request.Headers["Origin"].ToString();
        
        // Only set headers if origin is in allowed list
        if (!string.IsNullOrEmpty(origin) && allowedOrigins.Contains(origin))
        {
            context.Response.Headers["Access-Control-Allow-Origin"] = origin;
            context.Response.Headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, PATCH, OPTIONS";
            context.Response.Headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With, Accept";
            context.Response.Headers["Access-Control-Allow-Credentials"] = "true";
            context.Response.Headers["Access-Control-Max-Age"] = "3600";
        }
        
        context.Response.StatusCode = 200;
        await context.Response.CompleteAsync();
        return;
    }

    // For non-OPTIONS requests, continue with middleware pipeline
    await next();
});

// Only apply HTTPS redirection in production environment
// if (!app.Environment.IsDevelopment())
// {
//     app.UseHttpsRedirection(); // COMMENTED OUT FOR DEVELOPMENT
// }

app.UseAuthentication();
app.UseAuthorization();

// Map controllers
app.MapControllers();

// Map gRPC services
app.MapGrpcService<Backend.Services.SiemService>().RequireCors("AllowAll");
app.MapGet("/proto/siem.proto", async context =>
{
    await context.Response.WriteAsync(File.ReadAllText("Protos/siem.proto"));
});

// Initialize the database
using (var scope = app.Services.CreateScope())
{
    var services = scope.ServiceProvider;
    try
    {
        var context = services.GetRequiredService<ApplicationDbContext>();
        context.Database.Migrate();
        
        // Seed database with roles and admin user
        await SeedDatabase(context, services.GetRequiredService<Microsoft.Extensions.Logging.ILogger<Program>>());
    }
    catch (Exception ex)
    {
        var logger = services.GetRequiredService<Microsoft.Extensions.Logging.ILogger<Program>>();
        logger.LogError(ex, "An error occurred while migrating the database");
    }
}

app.Run();

// Database seeding
async Task SeedDatabase(ApplicationDbContext context, Microsoft.Extensions.Logging.ILogger<Program> logger)
{
    logger.LogInformation("Checking database seed data...");
    
    // Ensure roles exist
    var roles = new[] 
    { 
        RoleModels.DefaultRoles.Admin, 
        RoleModels.DefaultRoles.Operator, 
        RoleModels.DefaultRoles.Analyst, 
        RoleModels.DefaultRoles.User 
    };
    
    foreach (var roleName in roles)
    {
        if (!await context.Roles.AnyAsync(r => r.Name == roleName))
        {
            logger.LogInformation("Creating role: {Role}", roleName);
            context.Roles.Add(new RoleModels
            {
                Name = roleName,
                Description = $"Default {roleName} role",
                IsSystem = true,
                CreatedAt = DateTime.UtcNow,
                UpdatedAt = DateTime.UtcNow
            });
            await context.SaveChangesAsync();
        }
    }
    
    // Check if any admin exists, if not create a default admin
    bool adminExists = await context.Users
        .Include(u => u.UserRoles)
        .ThenInclude(ur => ur.Role)
        .AnyAsync(u => u.UserRoles.Any(ur => ur.Role.Name == RoleModels.DefaultRoles.Admin));
    
    if (!adminExists)
    {
        logger.LogInformation("No admin user found. Creating default admin user...");
        
        // Create default admin user
        var adminUser = new UserModels
        {
            Username = "admin",
            Email = "admin@athalasiem.com",
            FirstName = "System",
            LastName = "Administrator",
            IsActive = true,
            CreatedAt = DateTime.UtcNow,
            UpdatedAt = DateTime.UtcNow
        };
        
        // Generate password hash and salt (using HMACSHA512 as in AuthController.Register)
        using var hmac = new System.Security.Cryptography.HMACSHA512();
        var hashBytes = hmac.ComputeHash(Encoding.UTF8.GetBytes("Admin123!")); // Default password
        var saltBytes = hmac.Key;
        
        adminUser.PasswordHash = Convert.ToBase64String(hashBytes);
        adminUser.PasswordSalt = Convert.ToBase64String(saltBytes);
        
        context.Users.Add(adminUser);
        await context.SaveChangesAsync();
        
        // Add admin role to user
        var adminRole = await context.Roles.FirstOrDefaultAsync(r => r.Name == RoleModels.DefaultRoles.Admin);
        if (adminRole != null)
        {
            context.UserRoles.Add(new UserRoleModels
            {
                UserId = adminUser.Id,
                RoleId = adminRole.Id
            });
            await context.SaveChangesAsync();
            logger.LogInformation("Default admin user created successfully with Admin role");
        }
    }
    else
    {
        // Ensure existing admin user has Admin role
        var adminUser = await context.Users
            .Include(u => u.UserRoles)
            .ThenInclude(ur => ur.Role)
            .FirstOrDefaultAsync(u => u.Username == "admin");
            
        if (adminUser != null)
        {
            var adminRole = await context.Roles.FirstOrDefaultAsync(r => r.Name == RoleModels.DefaultRoles.Admin);
            if (adminRole != null && !adminUser.UserRoles.Any(ur => ur.Role.Name == RoleModels.DefaultRoles.Admin))
            {
                context.UserRoles.Add(new UserRoleModels
                {
                    UserId = adminUser.Id,
                    RoleId = adminRole.Id
                });
                await context.SaveChangesAsync();
                logger.LogInformation("Added Admin role to existing admin user");
            }
        }
    }
    
    logger.LogInformation("Database seeding completed");
} 