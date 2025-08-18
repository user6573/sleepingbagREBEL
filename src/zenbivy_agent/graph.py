from __future__ import annotations
import os, base64, json
from typing import TypedDict, List, Dict, Any, Optional
import requests

# MSAL (delegated / public client + device code flow)
import msal

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic

# =========================
# ====== ENV & CONSTS =====
# =========================
TENANT_ID = os.environ["MS_TENANT_ID"]             # z.B. "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
CLIENT_ID = os.environ["MS_PUBLIC_CLIENT_ID"]      # App-Registrierung → App (client) ID, **ohne** Secret
SHARED_MAILBOX = os.environ["MS_SHARED_MAILBOX"]   # z.B. "friends@zenbivy.eu"

# Delegated Scopes (mit Shared-Mailbox Zugriff)
SCOPES = [
    "Mail.Read",
    "Mail.ReadWrite",
    "Mail.Read.Shared",
    "Mail.ReadWrite.Shared",
    "offline_access",
    "email",
    "openid",
    "profile",
]

GRAPH_BASE = "https://graph.microsoft.com/v1.0"
AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID}"

ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]

SYSTEM = (
    "Du bist sleepingbagREBEL, ein präziser, netter Mitarbeiter von Zenbivy. "
    "Zeige nicht, dass du eine KI bist. "
    "Antworte in der Sprache der Anfrage und kompakt. "
    "Gib nur die Informationen, nach denen gefragt wurde. "
    "Weise dezent auf den Newsletter hin (https://zenbivy.eu – rechts unten). "
    "Falls du die Frage nicht beantworten kannst, sage das und verweise auf friends@zenbivy.eu. "
    "Tool-Nutzung: • gear_guide • bedingungen • wieder_verfuegbar • search_web (nur wenn nötig). "
    "RAG nur als letztes Mittel. Keine Sternchen (*)."
)

# =========================
# ====== SIMPLE TOOLS =====
# =========================

_SOURCES = {
    "Größentabelle": "https://zenbivy.eu/pages/size-guide",
    "Gebrauchsanweisung": "https://zenbivy.eu/pages/owners-manual-support-document",
    "Accessory Guide": "https://zenbivy.eu/pages/accessory-guide",
    "Kontakt": "https://zenbivy.eu/pages/kontakt",
}

def _http_get(url: str, timeout: int = 20) -> str:
    resp = requests.get(url, timeout=timeout, headers={"User-Agent": "ZenbivyAgent/1.0"})
    resp.raise_for_status()
    return resp.text

@tool("bedingungen")
def bedingungen(kategorie: str) -> str:
    k = (kategorie or "").strip().lower()
    if "rabatt" in k:
        return "Rabattcode erhältst du im Newsletter (https://zenbivy.eu)."
    if "rückgabe" in k or "umtausch" in k:
        return "Rückgabe/Umtausch: 14 Tage; unbenutzt. Details auf der Website."
    if "versand" in k:
        return "Versand: EU-weit; 2–7 Werktage. Genaues auf https://zenbivy.eu."
    return "Unbekannte Kategorie. Verfügbar: Rabattcode | Rückgabe- & Umtauschbedingungen | Versandbedingungen."

@tool("gear_guide")
def gear_guide(name: str) -> dict:
    if name not in _SOURCES:
        return {"error": f"'{name}' nicht verfügbar. Options: {', '.join(_SOURCES.keys())}"}
    url = _SOURCES[name]
    try:
        html = _http_get(url)
    except Exception as e:
        return {"error": f"Fehler beim Laden: {e}", "url": url}
    text = html[:5000] + (" …" if len(html) > 5000 else "")
    return {"source": name, "url": url, "text": text}

from pathlib import Path
_DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).resolve().parents[2] / "data"))

@tool("wieder_verfuegbar")
def wieder_verfuegbar(datei: str) -> str:
    fname = f"{datei}.txt" if not datei.lower().endswith(".txt") else datei
    p = _DATA_DIR / fname
    if not p.exists():
        return f"[FEHLER] Datei nicht gefunden: {p.name}"
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return p.read_text(encoding="latin-1", errors="ignore")

@tool("search_web")
def search_web(query: str, restrict_to_zenbivy: bool = True) -> dict:
    base = "https://zenbivy.eu" if restrict_to_zenbivy else "https://duckduckgo.com/?q="
    return {"query": query, "note": "Demo-Suche", "url": base}

TOOLS = [bedingungen, gear_guide, wieder_verfuegbar, search_web]

# =========================
# ========= LLM ==========='
# =========================
llm = ChatAnthropic(model=ANTHROPIC_MODEL, api_key=ANTHROPIC_API_KEY, temperature=0.2, max_tokens=2000)
llm_with_tools = llm.bind_tools(TOOLS)
_TOOL_MAP = {t.name: t for t in TOOLS}

def run_agent_with_tools(user_text: str) -> str:
    msgs: List[Any] = [SystemMessage(content=SYSTEM), HumanMessage(content=user_text)]
    for _ in range(4):
        ai: AIMessage = llm_with_tools.invoke(msgs, config=RunnableConfig())
        msgs.append(ai)
        tool_calls = getattr(ai, "tool_calls", None) or []
        if not tool_calls:
            return ai.content or ""
        for call in tool_calls:
            name = call.get("name"); args = call.get("args") or {}; tool = _TOOL_MAP.get(name)
            if not tool:
                msgs.append(ToolMessage(content=f"[FEHLER] Tool '{name}' nicht gefunden.", name=name, tool_call_id=call.get("id")))
                continue
            try:
                res = tool.invoke(args)
            except Exception as e:
                res = {"error": str(e)}
            msgs.append(ToolMessage(content=str(res), name=name, tool_call_id=call.get("id")))
    return "Ich konnte die Anfrage nicht abschließen. Bitte schreibe an friends@zenbivy.eu."

# =========================
# === DELEGATED AUTH ====
# =========================

def _load_cache_from_state(b64: Optional[str]) -> msal.SerializableTokenCache:
    cache = msal.SerializableTokenCache()
    if b64:
        try:
            cache.deserialize(base64.b64decode(b64).decode("utf-8"))
        except Exception:
            pass
    return cache


def _dump_cache_to_b64(cache: msal.SerializableTokenCache) -> Optional[str]:
    if cache.has_state_changed:
        try:
            raw = cache.serialize()
            return base64.b64encode(raw.encode("utf-8")).decode("ascii")
        except Exception:
            return None
    return None


def _acquire_token(cache: msal.SerializableTokenCache) -> Dict[str, Any]:
    app = msal.PublicClientApplication(client_id=CLIENT_ID, authority=AUTHORITY, token_cache=cache)
    accounts = app.get_accounts()
    result = None
    if accounts:
        result = app.acquire_token_silent(SCOPES, account=accounts[0])
    if not result:
        # Kein interaktiver Flow im Cron. Hinweis zurückgeben.
        raise RuntimeError(
            "Kein gültiges Delegated-Token im Cache. Bitte Device-Code-Login ausführen und Cache setzen (siehe scripts/authorize_device.py & scripts/seed_cache.py)."
        )
    if "access_token" not in result:
        raise RuntimeError(f"Tokenfehler: {result.get('error_description')}")
    return result


# =========================
# ====== GRAPH STATE ======
# =========================
class AppState(TypedDict, total=False):
    messages: List[Any]
    msal_cache_b64: Optional[str]
    last_seen_iso: Optional[str]
    new_emails: List[Dict[str, Any]]
    drafted_count: int
    drafted_subjects: List[str]


def _graph_get(url: str, token: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    r = requests.get(url, headers={"Authorization": f"Bearer {token}"}, params=params or {}, timeout=20)
    r.raise_for_status()
    return r.json()


def _graph_post(url: str, token: str, json_body: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.post(url, headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"}, json=json_body, timeout=20)
    r.raise_for_status()
    return r.json()


def _graph_patch(url: str, token: str, json_body: Dict[str, Any]) -> None:
    r = requests.patch(url, headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"}, json=json_body, timeout=20)
    r.raise_for_status()


def node_fetch_new_emails(state: AppState) -> AppState:
    cache = _load_cache_from_state(state.get("msal_cache_b64"))
    token_res = _acquire_token(cache)
    # Cache-Update zurück in State schreiben (Refresh)
    cache_b64 = _dump_cache_to_b64(cache)

    at = token_res["access_token"]
    params = {"$top": "10", "$select": "id,subject,receivedDateTime,from,bodyPreview", "$orderby": "receivedDateTime desc"}
    since = state.get("last_seen_iso")
    if since:
        params["$filter"] = f"receivedDateTime gt {since}"

    url = f"{GRAPH_BASE}/users/{SHARED_MAILBOX}/mailFolders/Inbox/messages"
    data = _graph_get(url, at, params=params)
    msgs = data.get("value", [])

    # max timestamp ermitteln
    max_dt = since
    for m in msgs:
        rcv = m.get("receivedDateTime")
        if rcv and (max_dt is None or rcv > max_dt):
            max_dt = rcv

    out: AppState = {"new_emails": msgs, "last_seen_iso": max_dt or since}
    if cache_b64:
        out["msal_cache_b64"] = cache_b64
    return out


def _get_msg_html(token: str, msg_id: str) -> str:
    url = f"{GRAPH_BASE}/users/{SHARED_MAILBOX}/messages/{msg_id}?$select=body"
    data = _graph_get(url, token)
    body = data.get("body", {})
    return body.get("content", "") or ""


def _create_reply_draft(token: str, original_id: str, html_body: str) -> str:
    # 1) Draft erzeugen
    d1 = _graph_post(f"{GRAPH_BASE}/users/{SHARED_MAILBOX}/messages/{original_id}/createReply", token, {})
    draft_id = d1["id"]
    # 2) Body patchen
    _graph_patch(f"{GRAPH_BASE}/users/{SHARED_MAILBOX}/messages/{draft_id}", token, {"body": {"contentType": "HTML", "content": html_body}})
    return draft_id


def node_generate_drafts(state: AppState) -> AppState:
    cache = _load_cache_from_state(state.get("msal_cache_b64"))
    token_res = _acquire_token(cache)
    cache_b64 = _dump_cache_to_b64(cache)
    at = token_res["access_token"]

    drafted = 0
    subjects: List[str] = []
    for m in state.get("new_emails", []):
        msg_id = m["id"]
        subject = m.get("subject") or "(ohne Betreff)"
        preview = (m.get("bodyPreview") or "").strip()
        sender = ((m.get("from") or {}).get("emailAddress") or {}).get("address", "")

        full_html = _get_msg_html(at, msg_id)
        user_text = (
            f"Beantworte diese Kundenmail höflich und hilfreich. "
            f"Absender: {sender}\n"
            f"Betreff: {subject}\n"
            f"Mail (Auszug): {preview}\n"
            f"Mail (HTML-Volltext folgt):\n{full_html}\n"
            f"Erstelle bitte eine kurze, konkrete Antwort als HTML (ohne Signatur)."
        )
        reply_html = run_agent_with_tools(user_text) or "<p>Vielen Dank für Ihre Nachricht.</p>"
        try:
            _create_reply_draft(at, msg_id, reply_html)
            drafted += 1
            subjects.append(subject)
        except Exception as e:
            subjects.append(f"{subject} [Draft-Fehler: {e}]")

    out: AppState = {"drafted_count": drafted, "drafted_subjects": subjects}
    if cache_b64:
        out["msal_cache_b64"] = cache_b64
    return out


def node_summarize(state: AppState) -> AppState:
    drafted = state.get("drafted_count", 0)
    subjects = state.get("drafted_subjects", [])
    text = f"Erstellte Entwürfe: {drafted}\n" + "\n".join(f"- {s}" for s in subjects)
    return {"messages": [AIMessage(content=text)]}

# Graph bauen
builder = StateGraph(AppState)
builder.add_node("fetch_new_emails", node_fetch_new_emails)
builder.add_node("generate_drafts", node_generate_drafts)
builder.add_node("summarize", node_summarize)

builder.add_edge(START, "fetch_new_emails")
builder.add_edge("fetch_new_emails", "generate_drafts")
builder.add_edge("generate_drafts", "summarize")
builder.add_edge("summarize", END)

graph = builder.compile()
