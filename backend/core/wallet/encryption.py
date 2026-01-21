"""AES-256-GCM encryption for private keys."""

import base64
import os

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes


def _derive_key(password: str, salt: bytes) -> bytes:
    """Derive 256-bit key from password using PBKDF2."""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100_000,
    )
    return kdf.derive(password.encode())


def encrypt_private_key(private_key: str, password: str) -> tuple[str, str]:
    """
    Encrypt private key with password.

    Returns:
        Tuple of (encrypted_key_b64, salt_b64)
    """
    salt = os.urandom(16)
    key = _derive_key(password, salt)

    aesgcm = AESGCM(key)
    nonce = os.urandom(12)

    ciphertext = aesgcm.encrypt(nonce, private_key.encode(), None)
    encrypted = nonce + ciphertext

    return base64.b64encode(encrypted).decode(), base64.b64encode(salt).decode()


def decrypt_private_key(encrypted_b64: str, salt_b64: str, password: str) -> str:
    """
    Decrypt private key with password.

    Raises:
        ValueError: If password is incorrect
    """
    encrypted = base64.b64decode(encrypted_b64)
    salt = base64.b64decode(salt_b64)
    key = _derive_key(password, salt)

    nonce = encrypted[:12]
    ciphertext = encrypted[12:]

    aesgcm = AESGCM(key)
    try:
        plaintext = aesgcm.decrypt(nonce, ciphertext, None)
        return plaintext.decode()
    except InvalidTag:
        raise ValueError("Invalid password")
