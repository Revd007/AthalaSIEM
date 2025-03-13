from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os

class EncryptionService:
    def __init__(self, key: str):
        self.fernet = Fernet(self._generate_key(key))
        
    def _generate_key(self, key: str) -> bytes:
        salt = b'athala_siem_salt'  # Change this in production
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key_bytes = key.encode()
        return base64.urlsafe_b64encode(kdf.derive(key_bytes))
        
    def encrypt(self, data: str) -> str:
        return self.fernet.encrypt(data.encode()).decode()
        
    def decrypt(self, encrypted_data: str) -> str:
        return self.fernet.decrypt(encrypted_data.encode()).decode() 