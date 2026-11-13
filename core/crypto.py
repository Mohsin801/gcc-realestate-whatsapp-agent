"""Encryption at rest for the owner's Google refresh token (Fernet: AES-128-CBC + HMAC-SHA256)."""
from cryptography.fernet import Fernet, InvalidToken


class TokenCipher:
    def __init__(self, key: str):
        if not key:
            raise ValueError("TOKEN_ENCRYPTION_KEY is not set")
        try:
            self._fernet = Fernet(key.encode())
        except (ValueError, TypeError) as e:
            raise ValueError("TOKEN_ENCRYPTION_KEY must be a Fernet key (32 url-safe base64 bytes)") from e

    def encrypt(self, value: str) -> str:
        return self._fernet.encrypt(value.encode()).decode()

    def decrypt(self, value: str) -> str:
        try:
            return self._fernet.decrypt(value.encode()).decode()
        except InvalidToken as e:
            raise ValueError("stored token cannot be decrypted (TOKEN_ENCRYPTION_KEY changed?)") from e


def generate_key() -> str:
    return Fernet.generate_key().decode()
