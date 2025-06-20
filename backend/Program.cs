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
builder.Services.AddControllers();
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
    options.MaxReceiveMessageSize = 16 * 1024 * 1024; // 16 MB
    options.MaxSendMessageSize = 16 * 1024 * 1024; // 16 MB
});

// Configure gRPC client
builder.Services.AddGrpcClient<AthalaSIEM.Agent.SiemService.SiemServiceClient>(options =>
{
    // options.Address = new Uri(builder.Configuration["GrpcServer:Url"] ?? "https://localhost:9596"); // COMMENTED OUT FOR DEVELOPMENT
    options.Address = new Uri("http://localhost:9595"); // CHANGED TO HTTP FOR DEVELOPMENT
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

// Configure CORS - updated to properly handle preflight requests
builder.Services.AddCors(options =>
{
    options.AddPolicy("AllowFrontend", builder =>
    {
        builder.WithOrigins(
                "http://localhost:3000",  // Development
                "http://localhost:9595",  // Backend
                "http://localhost:7654",  // Development
                "http://localhost:7655",  // Production
                // "https://localhost:9596", // Secure Production // COMMENTED OUT FOR DEVELOPMENT
                "http://localhost:7657"   // Test
            )
            .AllowAnyMethod()
            .AllowAnyHeader()
            .AllowCredentials()
            .WithExposedHeaders("Content-Disposition"); // For file downloads
    });
    
    // Add a more permissive CORS policy for gRPC clients that doesn't use credentials
    options.AddPolicy("AllowAll", builder =>
    {
        builder.WithOrigins(
                "http://localhost:3000",
                "http://localhost:9595",
                "http://localhost:7654",
                "http://localhost:7655",
                // "https://localhost:9596", // COMMENTED OUT FOR DEVELOPMENT
                "http://localhost:7657"
            )
            .AllowAnyMethod()
            .AllowAnyHeader();
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

// Configure Kestrel - Support HTTP/2 over HTTP for gRPC
builder.WebHost.ConfigureKestrel(serverOptions =>
{
    // Configure HTTP endpoint with HTTP/2 support for gRPC
    serverOptions.ListenAnyIP(9595, listenOptions =>
    {
        listenOptions.Protocols = HttpProtocols.Http1AndHttp2;
        
        // Enable HTTP/2 over HTTP (without TLS) for gRPC
        listenOptions.UseConnectionLogging();
    });
    
    // Allow HTTP/2 over HTTP (insecure) for development
    serverOptions.ConfigureEndpointDefaults(listenOptions =>
    {
        listenOptions.Protocols = HttpProtocols.Http1AndHttp2;
    });
});

// Register repositories
builder.Services.AddScoped<IUserRepository, UserRepository>();
builder.Services.AddScoped<IAgentRepository, AgentRepository>();
builder.Services.AddScoped<IAlertRepository, AlertRepository>();
builder.Services.AddScoped<ILogEntryRepository, LogEntryRepository>();
builder.Services.AddScoped<IDashboardRepository, DashboardRepository>();
builder.Services.AddScoped<IReportRepository, ReportRepository>();
builder.Services.AddScoped<AthalaSIEM.Backend.Repositories.IAgentDeploymentTokenRepository, AthalaSIEM.Backend.Repositories.AgentDeploymentTokenRepository>();

// Register services
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

// Register background services
builder.Services.AddHostedService<Backend.Services.Background.AgentMonitoringService>();
builder.Services.AddHostedService<Backend.Services.Background.LogCleanupService>();
builder.Services.AddHostedService<Backend.Services.Background.AlertCleanupService>();

// Register the new log archiving background service
builder.Services.AddHostedService<LogArchivingService>();

var app = builder.Build();

// Configure the HTTP request pipeline
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

// Configure middleware in correct order - IMPORTANT: Routing must come before CORS
app.UseRouting();

// Special handling for OPTIONS requests (preflight)
app.Use(async (context, next) =>
{
    // If it's a preflight request, handle it directly
    if (context.Request.Method == "OPTIONS")
    {
        // Apply CORS directly for OPTIONS requests
        context.Response.Headers["Access-Control-Allow-Origin"] = context.Request.Headers["Origin"];
        context.Response.Headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS";
        context.Response.Headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With";
        context.Response.Headers["Access-Control-Allow-Credentials"] = "true";
        context.Response.StatusCode = 200;
        return;
    }

    // For non-OPTIONS requests, continue with middleware pipeline
    await next();
});

// Apply CORS middleware for handling standard requests
app.UseCors("AllowFrontend");

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
app.MapGrpcService<AthalaSIEM.Backend.Services.SiemService>().RequireCors("AllowAll");
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