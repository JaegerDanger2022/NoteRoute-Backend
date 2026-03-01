import asyncio
import logging
import re
from datetime import datetime, timedelta, timezone

import httpx
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials

from app.config import settings

logger = logging.getLogger(__name__)

_GDOCS_MIME = "application/vnd.google-apps.document"
_GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"


# ---------------------------------------------------------------------------
# Markdown → Google Docs batchUpdate requests
# ---------------------------------------------------------------------------

def _md_to_requests(markdown: str, start_index: int, tab_id: str | None = None) -> tuple[list[dict], int]:
    """Convert a markdown string to a list of Google Docs batchUpdate requests.

    Returns (requests, end_index) where end_index is start_index + len(plain text).

    Supported syntax:
      # H1  →  HEADING_1
      ## H2  →  HEADING_2
      ### H3  →  HEADING_3
      **bold**  →  bold text run
      _italic_ / *italic*  →  italic text run
      - item / * item  →  BULLET_LIST_ITEM (named 'glyphType': DISC)
      plain paragraph  →  NORMAL_TEXT

    Implementation note: the Google Docs API requires that text be inserted
    before formatting is applied. We therefore build two parallel lists:
      1. insertText requests (all plain text concatenated)
      2. style requests (updateParagraphStyle / updateTextStyle)
    and return them in that order so callers can batch them in a single call.
    """
    lines = markdown.splitlines()

    insert_requests: list[dict] = []
    style_requests: list[dict] = []

    idx = start_index

    def _loc(i: int) -> dict:
        loc: dict = {"index": i}
        if tab_id:
            loc["tabId"] = tab_id
        return loc

    def _range(s: int, e: int) -> dict:
        r: dict = {"startIndex": s, "endIndex": e}
        if tab_id:
            r["tabId"] = tab_id
        return r

    def _insert(text: str) -> int:
        nonlocal idx
        insert_requests.append({"insertText": {"location": _loc(idx), "text": text}})
        length = len(text)
        idx += length
        return length

    def _para_style(s: int, e: int, named_style: str) -> None:
        style_requests.append({
            "updateParagraphStyle": {
                "range": _range(s, e),
                "paragraphStyle": {"namedStyleType": named_style},
                "fields": "namedStyleType",
            }
        })

    def _bullet(s: int, e: int) -> None:
        style_requests.append({
            "createParagraphBullets": {
                "range": _range(s, e),
                "bulletPreset": "BULLET_DISC_CIRCLE_SQUARE",
            }
        })

    def _apply_inline(line: str, line_start: int) -> None:
        """Walk the plain-text line and emit updateTextStyle requests for **bold** and _italic_."""
        pos = line_start
        remaining = line
        while remaining:
            m = re.search(r'\*\*(.+?)\*\*|_(.+?)_|\*(.+?)\*', remaining)
            if not m:
                break
            pos += len(remaining[:m.start()])
            content = m.group(1) or m.group(2) or m.group(3)
            is_bold = m.group(0).startswith('**')
            content_len = len(content)
            style_requests.append({
                "updateTextStyle": {
                    "range": _range(pos, pos + content_len),
                    "textStyle": {"bold": True} if is_bold else {"italic": True},
                    "fields": "bold" if is_bold else "italic",
                }
            })
            pos += content_len
            remaining = remaining[m.end():]

    def _strip_inline(text: str) -> str:
        """Remove markdown inline markers, keeping only the inner text."""
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'_(.+?)_', r'\1', text)
        text = re.sub(r'\*(.+?)\*', r'\1', text)
        return text

    for raw_line in lines:
        # Determine paragraph type
        h1 = re.match(r'^# (.+)', raw_line)
        h2 = re.match(r'^## (.+)', raw_line)
        h3 = re.match(r'^### (.+)', raw_line)
        bullet = re.match(r'^[-*] (.+)', raw_line)

        if h3:
            plain = _strip_inline(h3.group(1))
            line_start = idx
            _insert(plain + '\n')
            _para_style(line_start, line_start + len(plain) + 1, 'HEADING_3')
            _apply_inline(plain, line_start)
        elif h2:
            plain = _strip_inline(h2.group(1))
            line_start = idx
            _insert(plain + '\n')
            _para_style(line_start, line_start + len(plain) + 1, 'HEADING_2')
            _apply_inline(plain, line_start)
        elif h1:
            plain = _strip_inline(h1.group(1))
            line_start = idx
            _insert(plain + '\n')
            _para_style(line_start, line_start + len(plain) + 1, 'HEADING_1')
            _apply_inline(plain, line_start)
        elif bullet:
            plain = _strip_inline(bullet.group(1))
            line_start = idx
            _insert(plain + '\n')
            _bullet(line_start, line_start + len(plain) + 1)
            _apply_inline(plain, line_start)
        else:
            plain = _strip_inline(raw_line)
            line_start = idx
            _insert(plain + '\n')
            if plain.strip():
                _para_style(line_start, line_start + len(plain) + 1, 'NORMAL_TEXT')
            _apply_inline(plain, line_start)

    return insert_requests + style_requests, idx


async def refresh_access_token(refresh_token: str) -> tuple[str, datetime]:
    """Exchange a refresh token for a fresh access token. Returns (access_token, expires_at)."""
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            _GOOGLE_TOKEN_URL,
            data={
                "client_id": settings.GOOGLE_CLIENT_ID,
                "client_secret": settings.GOOGLE_CLIENT_SECRET,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            },
        )
        resp.raise_for_status()
        data = resp.json()
    expires_in = data.get("expires_in", 3600)
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in)
    return data["access_token"], expires_at


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

    # Insert + format content (markdown-aware)
    requests, _ = _md_to_requests(content.strip(), start_index=1)
    if requests:
        docs_service.documents().batchUpdate(
            documentId=doc_id,
            body={"requests": requests},
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


def _list_tabs_sync(document_id: str, access_token: str) -> list[dict]:
    """Return a flat list of {tab_id, tab_title} for all tabs in a Google Doc."""
    docs_service = _build_docs(access_token)
    doc = docs_service.documents().get(documentId=document_id, includeTabsContent=True).execute()

    result: list[dict] = []

    def _walk(tab_list: list) -> None:
        for tab in tab_list:
            tab_props = tab.get("tabProperties", {})
            result.append({
                "tab_id": tab_props.get("tabId", ""),
                "tab_title": tab_props.get("title", "Untitled"),
            })
            child_tabs = tab.get("childTabs", [])
            if child_tabs:
                _walk(child_tabs)

    tabs = doc.get("tabs", [])
    logger.info("list_tabs doc_id=%s tabs_count=%d doc_keys=%s", document_id, len(tabs), list(doc.keys()))
    if tabs:
        _walk(tabs)
    else:
        # Single-body doc with no tabs structure — synthesise one entry
        result.append({"tab_id": "", "tab_title": "Document"})

    return result


def _append_content_sync(document_id: str, content: str, access_token: str, tab_id: str | None = None) -> None:
    docs_service = _build_docs(access_token)

    if tab_id:
        doc = docs_service.documents().get(documentId=document_id, includeTabsContent=True).execute()
        end_index = 1  # fallback
        for tab in doc.get("tabs", []):
            if tab.get("tabProperties", {}).get("tabId") == tab_id:
                tab_body = tab.get("documentTab", {}).get("body", {}).get("content", [])
                if tab_body:
                    end_index = tab_body[-1]["endIndex"] - 1
                break
    else:
        doc = docs_service.documents().get(documentId=document_id).execute()
        end_index = doc["body"]["content"][-1]["endIndex"] - 1

    # Insert two blank lines as a separator before the new content, then
    # convert markdown to properly-styled requests starting at end_index + 2.
    separator = "\n\n"
    sep_request = {
        "insertText": {
            "location": {"index": end_index} if not tab_id else {"index": end_index, "tabId": tab_id},
            "text": separator,
        }
    }
    content_requests, _ = _md_to_requests(content.strip(), start_index=end_index + len(separator), tab_id=tab_id)
    docs_service.documents().batchUpdate(
        documentId=document_id,
        body={"requests": [sep_request] + content_requests},
    ).execute()


def _extract_body_text(body_content: list, parts: list, max_chars: int) -> bool:
    """Append text from a body content list into parts. Returns True if max_chars reached."""
    for element in body_content:
        paragraph = element.get("paragraph")
        if not paragraph:
            continue
        for pe in paragraph.get("elements", []):
            text_run = pe.get("textRun")
            if text_run:
                parts.append(text_run.get("content", ""))
        if sum(len(p) for p in parts) >= max_chars:
            return True
    return False


def _fetch_document_text_sync(document_id: str, access_token: str, max_chars: int = 100000) -> str:
    """Extract plain text from a Google Doc (all tabs if present), capped at max_chars."""
    docs_service = _build_docs(access_token)
    doc = docs_service.documents().get(documentId=document_id, includeTabsContent=True).execute()
    parts: list[str] = []

    tabs = doc.get("tabs")
    if tabs:
        # Multi-tab document — walk every tab recursively
        def _walk_tabs(tab_list: list) -> bool:
            for tab in tab_list:
                body_content = (
                    tab.get("documentTab", {}).get("body", {}).get("content", [])
                )
                if _extract_body_text(body_content, parts, max_chars):
                    return True
                child_tabs = tab.get("childTabs", [])
                if child_tabs and _walk_tabs(child_tabs):
                    return True
            return False
        _walk_tabs(tabs)
    else:
        _extract_body_text(doc.get("body", {}).get("content", []), parts, max_chars)

    return "".join(parts)[:max_chars].strip()


async def fetch_document_text(document_id: str, access_token: str, max_chars: int = 100000) -> str:
    return await asyncio.to_thread(_fetch_document_text_sync, document_id, access_token, max_chars)


async def list_documents(access_token: str) -> list[dict]:
    return await asyncio.to_thread(_list_documents_sync, access_token)


async def get_document_metadata(document_id: str, access_token: str) -> dict:
    """Return {"id": ..., "name": ...} for a single Google Doc/Sheet/Slide by ID."""
    async with httpx.AsyncClient() as client:
        r = await client.get(
            f"https://www.googleapis.com/drive/v3/files/{document_id}",
            params={"fields": "id,name"},
            headers={"Authorization": f"Bearer {access_token}"},
        )
        if r.status_code == 404:
            raise ValueError("Document not found or not accessible")
        r.raise_for_status()
        data = r.json()
        return {"id": data["id"], "name": data.get("name", "Google Doc")}


async def create_document(title: str, content: str, access_token: str) -> dict:
    return await asyncio.to_thread(_create_document_sync, title, content, access_token)


async def list_tabs(document_id: str, access_token: str) -> list[dict]:
    return await asyncio.to_thread(_list_tabs_sync, document_id, access_token)


async def append_content(document_id: str, content: str, access_token: str, tab_id: str | None = None) -> None:
    await asyncio.to_thread(_append_content_sync, document_id, content, access_token, tab_id)
