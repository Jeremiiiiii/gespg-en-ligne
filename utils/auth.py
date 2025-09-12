import json
import hashlib
import os

USERS_FILE = "data/users.json"

def hash_password(password: str) -> str:
    """Hash un mot de passe avec SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()


def load_users() -> dict:
    """Charge les utilisateurs depuis le fichier JSON"""
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r") as f:
        return json.load(f)


def save_users(users: dict):
    """Sauvegarde les utilisateurs dans le fichier JSON"""
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)


def register_user(username: str, password: str) -> bool:
    """Inscrit un nouvel utilisateur"""
    users = load_users()
    if username in users:
        return False  # utilisateur existe déjà

    users[username] = hash_password(password)
    save_users(users)
    return True


def authenticate_user(username: str, password: str) -> bool:
    """Vérifie si l'utilisateur existe et que le mot de passe est correct"""
    users = load_users()
    hashed_input = hash_password(password)
    return username in users and users[username] == hashed_input
