using Microsoft.AspNetCore.Connections;
using Microsoft.AspNetCore.Server.Kestrel.Core;
using Microsoft.AspNetCore.Server.Kestrel.Https;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using System;
using System.IO;
using System.Net.Security;
using System.Security.Authentication;
using System.Security.Cryptography.X509Certificates;

var builder = WebApplication.CreateBuilder(args);

// Configure Kestrel
builder.WebHost.ConfigureKestrel(options =>
{
    // Setup HTTPS
    options.ListenAnyIP(9596, listenOptions =>
    {
        listenOptions.Protocols = HttpProtocols.Http2;
        
        // Add specific HTTPS configuration
        var certificatePath = Path.Combine(builder.Environment.ContentRootPath, "certificates", "localhost.pfx");
        var certificatePassword = builder.Configuration["CertificatePassword"] ?? "password";

        if (!File.Exists(certificatePath))
        {
            // Generate a new certificate if it doesn't exist
            var certificateDirectory = Path.GetDirectoryName(certificatePath);
            if (!Directory.Exists(certificateDirectory))
            {
                Directory.CreateDirectory(certificateDirectory!);
            }
            
            // Export the development certificate
            using var process = new System.Diagnostics.Process
            {
                StartInfo = new System.Diagnostics.ProcessStartInfo
                {
                    FileName = "dotnet",
                    Arguments = $"dev-certs https -ep \"{certificatePath}\" -p \"{certificatePassword}\"",
                    RedirectStandardOutput = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                }
            };
            process.Start();
            process.WaitForExit();
        }

        listenOptions.UseHttps(new HttpsConnectionAdapterOptions
        {
            ServerCertificate = new X509Certificate2(certificatePath, certificatePassword),
            SslProtocols = SslProtocols.Tls12 | SslProtocols.Tls13,
            ClientCertificateMode = ClientCertificateMode.AllowCertificate,
            CheckCertificateRevocation = false
        });
    });

    // Setup HTTP
    options.ListenAnyIP(9595, listenOptions =>
    {
        listenOptions.Protocols = HttpProtocols.Http1AndHttp2;
    });
});

// Add gRPC
builder.Services.AddGrpc(options =>
{
    options.EnableDetailedErrors = true;
    options.MaxReceiveMessageSize = 6 * 1024 * 1024; // 6 MB
    options.MaxSendMessageSize = 6 * 1024 * 1024; // 6 MB
});

// Add services to the container
builder.Services.AddControllers();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

var app = builder.Build();

// Configure the HTTP request pipeline
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseRouting();
app.UseHttpsRedirection();
app.UseAuthorization();

app.UseEndpoints(endpoints =>
{
    endpoints.MapControllers();
    // Map your gRPC services here
    // endpoints.MapGrpcService<YourGrpcService>();
});

app.Run(); 