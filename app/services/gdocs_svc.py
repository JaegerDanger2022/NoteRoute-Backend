import asyncio
import logging

from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials

logger = logging.getLogger(__name__)

_GDOCS_MIME = "application/vnd.google-apps.document"


def _build_drive(access_token: str):
    creds = Credentials(token=access_token)
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _build_docs(access_token: str):
    creds = Credentials(token=access_token)
    return build("docs", "v1", credentials=creds, cache_discovery=False)


def _list_documents_sync(access_token: str) -> list[dict]:
    service = _build_drive(access_token)
    results = []
    page_token = None

    while True:
        kwargs: dict = {
            "q": f"mimeType='{_GDOCS_MIME}' and trashed=false",
            "fields": "nextPageToken, files(id, name, webViewLink)",
            "pageSize": 100,
        }
        if page_token:
            kwargs["pageToken"] = page_token

        resp = service.files().list(**kwargs).execute()
        for f in resp.get("files", []):
            results.append({
                "id": f["id"],
                "name": f["name"],
                "url": f.get("webViewLink"),
            })

        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    return results


def _create_document_sync(title: str, content: str, access_token: str) -> dict:
    docs_service = _build_docs(access_token)
    drive_service = _build_drive(access_token)

    # Create blank document
    doc = docs_service.documents().create(body={"title": title}).execute()
    doc_id = doc["documentId"]

    # Insert content
    docs_service.documents().batchUpdate(
        documentId=doc_id,
        body={
            "requests": [
                {
                    "insertText": {
                        "location": {"index": 1},
                        "text": content,
                    }
                }
            ]
        },
    ).execute()

    # Fetch webViewLink from Drive
    file_meta = drive_service.files().get(
        fileId=doc_id, fields="webViewLink"
    ).execute()

    return {
        "id": doc_id,
        "name": title,
        "url": file_meta.get("webViewLink"),
    }


def _append_content_sync(document_id: str, content: str, access_token: str) -> None:
    docs_service = _build_docs(access_token)
    doc = docs_service.documents().get(documentId=document_id).execute()
    end_index = doc["body"]["content"][-1]["endIndex"] - 1

    docs_service.documents().batchUpdate(
        documentId=document_id,
        body={
            "requests": [
                {
                    "insertText": {
                        "location": {"index": end_index},
                        "text": "\n" + content,
                    }
                }
            ]
        },
    ).execute()


async def list_documents(access_token: str) -> list[dict]:
    return await asyncio.to_thread(_list_documents_sync, access_token)


async def create_document(title: str, content: str, access_token: str) -> dict:
    return await asyncio.to_thread(_create_document_sync, title, content, access_token)


async def append_content(document_id: str, content: str, access_token: str) -> None:
    await asyncio.to_thread(_append_content_sync, document_id, content, access_token)
