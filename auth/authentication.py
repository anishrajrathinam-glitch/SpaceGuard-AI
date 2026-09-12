"""
SpaceGuard AI - Authentication & Access Control Module
Provides secure salted hashing and verification of credentials
with local SQLite storage and session management for Streamlit.
"""

import os
import sqlite3
import hashlib
import secrets
from typing import Tuple, Optional, Dict, Any


class AuthManager:
    """
    Local secure authentication manager for SpaceGuard AI prototype.
    Stores passwords using SHA-256 + 32-byte cryptographic salt.
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            auth_dir = os.path.dirname(os.path.abspath(__file__))
            self.db_path = os.path.join(auth_dir, "users.db")
        else:
            self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """
        Creates users table if it does not exist and seeds initial demo users.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                organization TEXT,
                role TEXT DEFAULT 'Researcher',
                salt TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

        # Seed default demo accounts if table is empty
        cursor.execute("SELECT COUNT(*) FROM users")
        if cursor.fetchone()[0] == 0:
            self.register_user("admin", "admin@spaceguard.ai", "SpaceGuard@2025", "SpaceGuard Mission Ops", "Administrator")
            self.register_user("scientist", "scientist@isro-imd.demo", "WeatherAI#1", "Atmospheric Research Centre", "Senior Meteorologist")

        conn.close()

    @staticmethod
    def _hash_password(password: str, salt: str) -> str:
        """
        Produces salted SHA-256 hash.
        """
        return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()

    def register_user(
        self,
        username: str,
        email: str,
        password: str,
        organization: str = "Disaster Management Authority",
        role: str = "Analyst"
    ) -> Tuple[bool, str]:
        """
        Registers a new user with validation and salted password hash.
        """
        username = username.strip().lower()
        email = email.strip().lower()

        if len(username) < 3:
            return False, "Username must be at least 3 characters long."
        if len(password) < 6:
            return False, "Password must be at least 6 characters long."
        if "@" not in email or "." not in email:
            return False, "Please provide a valid email address."

        salt = secrets.token_hex(16)
        pw_hash = self._hash_password(password, salt)

        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO users (username, email, organization, role, salt, password_hash) VALUES (?, ?, ?, ?, ?, ?)",
                (username, email, organization, role, salt, pw_hash)
            )
            conn.commit()
            return True, f"Account for '{username}' created successfully! You can now log in."
        except sqlite3.IntegrityError:
            return False, "Username or Email is already registered."
        finally:
            conn.close()

    def verify_login(self, username: str, password: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Verifies login credentials and returns user profile if authenticated.
        """
        username = username.strip().lower()
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        conn.close()

        if not user:
            return False, None

        calculated_hash = self._hash_password(password, user["salt"])
        if secrets.compare_digest(calculated_hash, user["password_hash"]):
            profile = {
                "id": user["id"],
                "username": user["username"],
                "email": user["email"],
                "organization": user["organization"],
                "role": user["role"],
            }
            return True, profile
        return False, None
