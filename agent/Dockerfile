FROM mcr.microsoft.com/dotnet/runtime:8.0 AS base
WORKDIR /app

FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src
COPY ["agent/agent.csproj", "agent/"]
RUN dotnet restore "agent/agent.csproj"
COPY . .
WORKDIR "/src/agent"
RUN dotnet build "agent.csproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "agent.csproj" -c Release -o /app/publish /p:UseAppHost=false

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "AthalaSIEM.Agent.dll"]
