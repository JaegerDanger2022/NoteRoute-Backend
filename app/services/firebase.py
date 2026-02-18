import json

import firebase_admin
from firebase_admin import auth, credentials

from app.config import settings

_firebase_app: firebase_admin.App | None = None


def init_firebase() -> None:
    global _firebase_app
    if _firebase_app is not None:
        return
    service_account = json.loads(settings.FIREBASE_SERVICE_ACCOUNT_JSON)
    cred = credentials.Certificate(service_account)
    _firebase_app = firebase_admin.initialize_app(cred)


def verify_id_token(token: str) -> dict:
    """Verify a Firebase ID token and return the decoded claims."""
    return auth.verify_id_token(token)
