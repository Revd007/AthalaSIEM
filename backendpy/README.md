# ATHALA SIEM Python Backend

The Python Backend component of the ATHALA SIEM (Security Information and Event Management) platform provides advanced analytics, AI-driven threat detection, and a REST API for security event processing.

## Overview

The ATHALA SIEM Python Backend offers powerful capabilities for:
- AI-driven security event analysis and threat detection
- Anomaly detection and correlation of security events
- Advanced analytics for threat intelligence
- REST API for agent and frontend communication
- Automated playbook execution for incident response
- Integration with OSINT and threat intelligence sources

## Tech Stack

- **Framework**: FastAPI
- **Database**: MS SQL Server with SQLAlchemy ORM
- **Authentication**: JWT with OAuth2
- **AI/ML**: Custom ML models for security event analysis
- **Data Processing**: Pandas, NumPy for data manipulation
- **Documentation**: OpenAPI/Swagger
- **Deployment**: Docker, Kubernetes support

## Project Structure

```
backendpy/
├── api/                # API endpoints and routes
│   └── routes/         # Route definitions for various features
├── auth/               # Authentication and authorization
├── database/           # Database models and connections
│   └── models/         # SQLAlchemy ORM models
├── ai_engine/          # AI and ML components
│   ├── models/         # ML model definitions
│   ├── training/       # Model training pipelines
│   ├── prediction/     # Inference and prediction services
│   └── evaluation/     # Model evaluation and metrics
├── analytics/          # Log and event analytics
├── automation/         # Automated response and playbooks
├── config/             # Configuration management
├── core/               # Core application functionality
├── middleware/         # FastAPI middleware components
├── monitoring/         # Application and system monitoring
├── schemas/            # Pydantic schemas for validation
├── services/           # Business logic services
├── utils/              # Utility functions and helpers
├── main.py             # Application entry point
└── config.py           # Configuration settings
```

## Core Components

### API Layer
The API module provides REST endpoints for:

- **Alerts**: Endpoint for managing security alerts
- **Events**: Endpoint for security event ingestion and querying
- **Agents**: Management of security agents
- **Users**: User management and authentication
- **Playbooks**: Automated response workflow management
- **Dashboard**: Data for dashboard visualizations
- **AI Service**: Endpoints for AI-driven analysis
- **Collectors**: Log collector configuration and management

### AI Engine
The AI engine provides advanced security analytics:

- **Event Analysis**: AI-driven analysis of security events
- **Threat Detection**: ML-based detection of security threats
- **Anomaly Detection**: Identification of unusual patterns and behaviors
- **Correlation Engine**: Correlation of related security events
- **OSINT Integration**: Open Source Intelligence integration

### Database Models
The database module defines the data structure for:

- **Users**: User accounts and permissions
- **Alerts**: Security alerts and notifications
- **Events**: Security events and logs
- **Agents**: Deployed security agents
- **Playbooks**: Automated response workflows
- **API Keys**: Authentication tokens for API access

## API Endpoints

### Authentication
- `POST /api/auth/login`: Authenticate users
- `POST /api/auth/token`: Generate JWT tokens
- `POST /api/auth/refresh`: Refresh authentication tokens

### Events
- `GET /api/events`: Query security events
- `POST /api/events`: Submit new security events
- `GET /api/events/{id}`: Get event details
- `GET /api/events/search`: Search events with advanced filtering

### Alerts
- `GET /api/alerts`: Query security alerts
- `POST /api/alerts`: Create new alerts
- `PUT /api/alerts/{id}`: Update alert information
- `GET /api/alerts/{id}`: Get alert details
- `PUT /api/alerts/{id}/status`: Update alert status

### Agents
- `GET /api/agents`: List all registered agents
- `POST /api/agents/register`: Register new agents
- `GET /api/agents/{id}`: Get agent details
- `PUT /api/agents/{id}/config`: Update agent configuration
- `GET /api/agents/{id}/health`: Get agent health status

### AI Services
- `POST /api/ai/analyze`: Analyze security events with AI
- `GET /api/ai/status`: Check AI service status
- `POST /api/ai/feedback`: Submit feedback for AI analysis
- `GET /api/ai/models`: List available AI models
- `GET /api/ai/insights`: Get AI-generated security insights

### Playbooks
- `GET /api/playbooks`: List available playbooks
- `POST /api/playbooks`: Create new playbook
- `GET /api/playbooks/{id}`: Get playbook details
- `POST /api/playbooks/{id}/execute`: Execute a playbook
- `GET /api/playbooks/{id}/history`: View playbook execution history

## AI Capabilities

### Security Event Analysis
The AI engine analyzes security events to:
- Classify events by type and severity
- Detect potential threats and attacks
- Identify false positives
- Correlate related events
- Recommend response actions

### Anomaly Detection
The system uses machine learning to detect:
- Unusual user behavior
- Network traffic anomalies
- System performance issues
- Potential data exfiltration
- Unusual login patterns

### Threat Intelligence
Integration with threat intelligence sources for:
- Known malicious IP addresses
- Malware signatures
- TTPs (Tactics, Techniques, and Procedures)
- Vulnerability information
- Emerging threat indicators

## Configuration

The application uses environment variables and `.env` files for configuration:

```python
DATABASE_URL: Connection string for the MS SQL database
SECRET_KEY: Secret key for JWT token generation
API_VERSION: API version string
LOG_LEVEL: Logging level (INFO, DEBUG, etc.)
ALLOWED_HOSTS: List of allowed hosts for the API
CORS_ORIGINS: List of allowed origins for CORS
ENVIRONMENT: Deployment environment (dev, staging, prod)
```

AI-specific configuration:

```python
TRAINING_INTERVAL: Interval for model retraining
EVALUATION_INTERVAL: Interval for model evaluation
BATCH_SIZE: Batch size for model training
LEARNING_RATE: Learning rate for model training
MAX_EPOCHS: Maximum number of training epochs
ENABLE_GPU: Whether to use GPU for ML operations
```

## Getting Started

### Prerequisites
- Python 3.9+
- MS SQL Server
- ODBC Driver for SQL Server
- Virtual environment (recommended)

### Development Setup
1. Clone the repository
2. Navigate to the backend directory
   ```bash
   cd backendpy
   ```

3. Create and activate a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

5. Create a `.env` file with required configuration
   ```
   DATABASE_URL=mssql+pyodbc://username:password@server/database?driver=ODBC+Driver+17+for+SQL+Server
   SECRET_KEY=your-secret-key
   API_VERSION=v1
   LOG_LEVEL=INFO
   ALLOWED_HOSTS=["*"]
   CORS_ORIGINS=["http://localhost:3000"]
   ENVIRONMENT=development
   DEBUG=true
   ```

6. Run the application
   ```bash
   uvicorn main:app --reload
   ```

   The API will be available at http://localhost:8000

### Docker Setup
```bash
docker build -t athala-siem-backendpy .
docker run -p 8000:8000 -e DATABASE_URL=your-db-url athala-siem-backendpy
```

## Integration Points

### Frontend Integration
The Python backend provides API endpoints for the frontend to:
- Fetch security events and alerts
- Display AI-generated insights
- Execute automated responses
- Configure and monitor agents

### Agent Integration
The backend receives data from agents including:
- Security logs and events
- System performance metrics
- Agent health status
- Configuration feedback

### External Systems Integration
The backend can integrate with:
- External threat intelligence feeds
- SIEM systems
- Ticketing systems
- Notification services
- SOAR platforms

## Development Workflow

1. Create feature branches from `develop`
2. Implement new features or fix bugs
3. Write unit and integration tests
4. Submit pull requests
5. CI pipeline validates code quality and runs tests
6. Merge to develop after approval
7. Periodic releases to main branch

## Performance Considerations

- Batch processing for high-volume event ingestion
- Database query optimization
- Caching for frequently requested data
- Asynchronous processing for AI operations
- Horizontal scaling for high availability

## License

Copyright © 2025 Athala Security Solutions

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 