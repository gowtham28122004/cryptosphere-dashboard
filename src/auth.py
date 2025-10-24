# src/auth.py
import sqlite3
import hashlib

DB_PATH = 'data/users.db'

def hash_password(password):
    """Hashes the password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

def check_password(hashed_password, user_password):
    """Checks if the provided password matches the stored hash."""
    return hashed_password == hash_password(user_password)

def create_users_table():
    """Creates the users table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            password TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def add_user(username, password, email):
    """Adds a new user to the database."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password, email) VALUES (?, ?, ?)", (username, hash_password(password), email.lower()))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False # Email already exists
    finally:
        conn.close()

# === THIS IS THE CORRECTED FUNCTION ===
def check_user(email, password):
    """
    Checks if a user exists with the given email and if the password is correct.
    Returns the username on success.
    """
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    # Find the user by their email address (which is unique)
    c.execute("SELECT username, password FROM users WHERE email = ?", (email.lower(),))
    result = c.fetchone()
    conn.close()
    
    if result and check_password(result[1], password):
        return result[0] # Return the username on successful login
    return None