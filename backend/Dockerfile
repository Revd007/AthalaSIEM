FROM mcr.microsoft.com/dotnet/sdk:8.0 AS base
WORKDIR /app
EXPOSE 9595
EXPOSE 50051

# Development stage
FROM base AS development
ENV ASPNETCORE_ENVIRONMENT=Development
ENV DOTNET_USE_POLLING_FILE_WATCHER=1
WORKDIR /app
COPY backend.csproj .
RUN dotnet restore
COPY . .
RUN dotnet build backend.csproj
ENTRYPOINT ["dotnet", "run", "--project", "backend.csproj", "--urls", "http://+:9595"]

# Production build stage
FROM base AS build
WORKDIR /src
COPY ["backend.csproj", "."]
RUN dotnet restore
COPY . .
RUN dotnet build "backend.csproj" -c Release -o /app/build

# Production publish stage
FROM build AS publish
RUN dotnet publish "backend.csproj" -c Release -o /app/publish /p:UseAppHost=false

# Production final stage
FROM mcr.microsoft.com/dotnet/aspnet:8.0 AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "backend.dll"]