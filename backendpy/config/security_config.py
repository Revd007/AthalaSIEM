from pydantic import EmailStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional

class SSLSettings(BaseSettings):
    SSL_ENABLED: bool = False
    SSL_CERT_PATH: str = "certs/cert.pem"
    SSL_KEY_PATH: str = "certs/key.pem"
    SSL_CA_PATH: Optional[str] = None
    SSL_VERIFY_CLIENT: bool = False
    SSL_CIPHERS: str = "TLS_AES_256_GCM_SHA384:TLS_CHACHA20_POLY1305_SHA256"
    SSL_PROTOCOLS: List[str] = ["TLSv1.2", "TLSv1.3"]
    SSL_PREFER_SERVER_CIPHERS: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="allow"
    )

class SecuritySettings(BaseSettings):
    # JWT Settings
    SECRET_KEY: str = "your-secret-key"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # SSL Settings
    ssl: SSLSettings = SSLSettings()
    
    # CORS Settings
    CORS_ORIGINS: List[str] = ["*"]
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["*"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    
    # Security Headers
    SECURITY_HEADERS: dict = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Content-Security-Policy": "default-src 'self'",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "geolocation=(), microphone=()"
    }

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="allow"
    )

# Create instance
settings = SecuritySettings()