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

// Configure CORS - updated to properly handle preflight requests
builder.Services.AddCors(options =>
{
    options.AddPolicy("AllowFrontend", builder =>
    {
        builder.WithOrigins(
                "http://localhost:7654",  // Development
                "http://localhost:7655",  // Production
                "https://localhost:7656", // Secure Production
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
                "http://localhost:7654",
                "http://localhost:7655",
                "https://localhost:7656",
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

// Register background services
builder.Services.AddHostedService<Backend.Services.Background.AgentMonitoringService>();
builder.Services.AddHostedService<Backend.Services.Background.LogCleanupService>();
builder.Services.AddHostedService<Backend.Services.Background.AlertCleanupService>();

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
if (!app.Environment.IsDevelopment())
{
    app.UseHttpsRedirection();
}

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
        await SeedDatabase(context, services.GetRequiredService<ILogger<Program>>());
    }
    catch (Exception ex)
    {
        var logger = services.GetRequiredService<ILogger<Program>>();
        logger.LogError(ex, "An error occurred while migrating the database");
    }
}

app.Run();

// Database seeding
async Task SeedDatabase(ApplicationDbContext context, ILogger logger)
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
            Email = "admin@example.com",
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
            logger.LogInformation("Default admin user created successfully");
        }
    }
    
    logger.LogInformation("Database seeding completed");
} 