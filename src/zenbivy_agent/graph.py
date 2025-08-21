# graph.py — Overkill Email Drafts
from __future__ import annotations
import os, time, random, re
import datetime as dt
from typing import TypedDict, List, Optional, Dict, Any
from pathlib import Path

import requests
from msal import ConfidentialClientApplication

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic

# =========================
# ====== ENV & CONSTS =====
# =========================
TENANT_ID = os.environ["MS_TENANT_ID"]           # Azure AD Tenant ID
CLIENT_ID = os.environ["MS_CLIENT_ID"]           # App (client) ID
CLIENT_SECRET = os.environ["MS_CLIENT_SECRET"]   # Client secret
SHARED_MAILBOX = os.environ["MS_SHARED_MAILBOX"] # z.B. "friends@zenbivy.eu"
GRAPH_BASE = "https://graph.microsoft.com/v1.0"

# --- Modell/Token-Config (OVERKILL) ---
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
# Default: Sonnet 3.7 (größtes Output-Limit) + 128k-Beta aktiv
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-7-sonnet-latest")
ANTHROPIC_MAX_TOKENS = int(os.getenv("ANTHROPIC_MAX_TOKENS", "90000"))  # groß, aber unter 128k
ANTHROPIC_OUTPUT_128K = os.getenv("ANTHROPIC_OUTPUT_128K", "1") == "1"
ANTHROPIC_FALLBACK_MODEL = os.getenv("ANTHROPIC_FALLBACK_MODEL", "claude-sonnet-4-20250514")
# Multi-Entwurf + Auswahl (Self-critique)
ANTHROPIC_N_CANDIDATES = int(os.getenv("ANTHROPIC_N_CANDIDATES", "2"))   # 2–3 ist sinnvoll
CRITIQUE_STRICT = os.getenv("CRITIQUE_STRICT", "1") == "1"

# Wie weit zurück (in Minuten) E-Mails geholt werden sollen
LOOKBACK_MINUTES = int(os.getenv("LOOKBACK_MINUTES", "5"))

# System-Prompt für beide Graphen
SYSTEM = (
    "Du bist sleepingbagREBEL, ein präziser, netter Mitarbeiter von Zenbivy. "
    "Antworte in der Sprache der Eingabe, kurz, konkret und korrekt. "
    "Nutze nur die gelieferten Informationen bzw. Tools. "
    "Wenn Informationen fehlen, stelle gezielte, knappe Rückfragen. "
    "Rendere Antworten als sauberes HTML ohne <html>/<head>/<body>-Wrapper."
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
    resp = requests.get(url, timeout=timeout, headers={"User-Agent":"ZenbivyAgent/1.0"})
    resp.raise_for_status()
    return resp.text

@tool("bedingungen")
def bedingungen(kategorie: str) -> str:
    """
    Shop-Bedingungen als kompakter Text.
    Mögliche Kategorien: Rabattcode | Rückgabe- & Umtauschbedingungen | Versandbedingungen
    """
    k = kategorie.strip().lower()
    if "rabatt" in k:
        return "Rabattcode erhältst du im Newsletter (https://zenbivy.eu)."
    if "rückgabe" in k or "umtausch" in k:
        return "Rückgabe/Umtausch: 14 Tage ab Erhalt; Artikel unbenutzt. Details auf der Website."
    if "versand" in k:
        return "Versand: EU-weit; Laufzeiten 2–7 Werktage. Genaues auf https://zenbivy.eu."
    return "Unbekannte Kategorie. Verfügbar: Rabattcode | Rückgabe- & Umtauschbedingungen | Versandbedingungen."

@tool("gear_guide")
def gear_guide(name: str) -> dict:
    """
    Lädt vordefinierte Zenbivy-Seiteninhalte grob. Verfügbar:
    - Größentabelle, Gebrauchsanweisung, Accessory Guide, Kontakt
    """
    if name not in _SOURCES:
        return {"error": f"'{name}' nicht verfügbar. Options: {', '.join(_SOURCES.keys())}"}
    url = _SOURCES[name]
    try:
        html = _http_get(url)
    except Exception as e:
        return {"error": f"Fehler beim Laden: {e}", "url": url}
    text = html
    if len(text) > 5000:
        text = text[:5000] + " …"
    return {"source": name, "url": url, "text": text}

_DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).resolve().parents[2] / "data"))

@tool("wieder_verfuegbar")
def wieder_verfuegbar(datei: str) -> str:
    """
    Liest eine Textdatei aus dem data/-Ordner (z. B. 'Light Quilt -4°C.txt') und
    gibt den gesamten Inhalt zurück (Termine/Verfügbarkeit).
    """
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
    """
    Platzhalter-Websuche (minimal). Für produktiv: Tavily verwenden.
    Hier nur Rückgabe eines Links, damit das Tool prinzipiell funktioniert.
    """
    base = "https://zenbivy.eu" if restrict_to_zenbivy else "https://duckduckgo.com/?q="
    return {"query": query, "note": "Demo-Suche", "url": base}

TOOLS = [bedingungen, gear_guide, wieder_verfuegbar, search_web]
_TOOL_MAP = {t.name: t for t in TOOLS}

# =========================
# ========= LLM ===========
# =========================
_extra_headers = {}
if ANTHROPIC_OUTPUT_128K:
    _extra_headers["anthropic-beta"] = "output-128k-2025-02-19"

llm = ChatAnthropic(
    model=ANTHROPIC_MODEL,
    api_key=ANTHROPIC_API_KEY,
    temperature=0.2,
    max_tokens=ANTHROPIC_MAX_TOKENS,
    extra_headers=_extra_headers or None,
)
llm_with_tools = llm.bind_tools(TOOLS)


def _is_retryable_error(exc: Exception) -> bool:
    s = str(exc).lower()
    return any(k in s for k in [
        "overloaded", "rate_limit", "timeout", "temporarily", "unavailable",
        "gateway", "service unavailable", "529", "429", "502", "503", "504"
    ])


def _invoke_with_retry(_llm, msgs, attempts: int = 7, base: float = 0.6, cap: float = 30.0):
    for i in range(attempts):
        try:
            return _llm.invoke(msgs, config=RunnableConfig())
        except Exception as e:
            if i == attempts - 1 or not _is_retryable_error(e):
                raise
            sleep_s = min(cap, base * (2 ** i)) + random.uniform(0, 0.6)
            time.sleep(sleep_s)


def _try_invoke_with_fallback(msgs):
    try:
        return _invoke_with_retry(llm_with_tools, msgs)
    except Exception as e:
        if _is_retryable_error(e):
            alt_llm = ChatAnthropic(
                model=ANTHROPIC_FALLBACK_MODEL,
                api_key=ANTHROPIC_API_KEY,
                temperature=0.2,
                max_tokens=min(ANTHROPIC_MAX_TOKENS, 64000),
            ).bind_tools(TOOLS)
            return _invoke_with_retry(alt_llm, msgs)
        raise


def run_agent_with_tools(user_text: str) -> str:
    """
    Eine Tool-Schleife für einfache Prompts -> Textantwort (max. 3 Tool-Runden).
    """
    msgs: List[Any] = [SystemMessage(content=SYSTEM), HumanMessage(content=user_text)]
    for _ in range(3):
        ai: AIMessage = _try_invoke_with_fallback(msgs)
        msgs.append(ai)
        tool_calls = getattr(ai, "tool_calls", None) or []
        if not tool_calls:
            return ai.content or ""
        for call in tool_calls:
            name = call.get("name")
            args = call.get("args") or {}
            tool = _TOOL_MAP.get(name)
            if not tool:
                msgs.append(ToolMessage(content=f"[FEHLER] Tool '{name}' nicht gefunden.", name=name, tool_call_id=call.get("id")))
                continue
            try:
                res = tool.invoke(args)
            except Exception as e:
                res = {"error": str(e)}
            msgs.append(ToolMessage(content=str(res), name=name, tool_call_id=call.get("id")))
    return "Ich konnte die Anfrage nicht abschließen. Bitte schreibe an friends@zenbivy.eu."


# ==============
# === QUALITY ===
# ==============
_SANITIZE_TAGS_RE = re.compile(r"<(script|style|iframe|link|meta|object|embed)[\s\S]*?>[\s\S]*?</\\1>", re.IGNORECASE)
_ON_EVENT_ATTR_RE = re.compile(r"\son[a-z]+=\"[^\"]*\"", re.IGNORECASE)
_JS_URL_RE = re.compile(r"(javascript:)[^\"']*", re.IGNORECASE)


def _sanitize_email_html(html: str) -> str:
    if not html:
        return html
    # Entferne gefährliche Tags (block)
    html = _SANITIZE_TAGS_RE.sub("", html)
    # Entferne on* Handler
    html = _ON_EVENT_ATTR_RE.sub("", html)
    # Entferne javascript: URLs
    html = _JS_URL_RE.sub("", html)
    # Trim
    return html.strip()


def _candidate_prompt(body_html: str, variant_id: int) -> str:
    return (
        "Erstelle eine höfliche, hilfreiche und konkrete Antwort als sauberes HTML (ohne <html>/<head>/<body>). "
        "Schließe mit ‘— sleepingbagREBEL’. "
        "Antworte ausschließlich basierend auf folgendem E-Mail-Body. "
        "Wenn Informationen fehlen, stelle am Ende maximal 3 prägnante Rückfragen als <ul>. "
        "Achte auf: korrekte Sprache (automatisch erkennen), klare Struktur (<p>, <ul>, <ol>, <strong>), "
        "präzise Produkthinweise. Nutze Links auf Zenbivy-Seiten nur, wenn sie explizit im Tool 'gear_guide' vorkommen. "
        "Gib ausschließlich den E-Mail-Body (HTML) zurück, keine Erklärungen.\n\n"
        "EMAIL_BODY_HTML_START\n"
        f"{body_html}\n"
        "EMAIL_BODY_HTML_END\n\n"
        f"VARIANT: {variant_id}"
    )


def _critique_prompt(candidates: List[str], body_html: str) -> str:
    rubric = (
        "Bewerte die Kandidaten nach: (1) Korrektheit/Bezug zum Body, (2) Vollständigkeit, (3) Klarheit/Struktur, "
        "(4) Ton/Markenstimme, (5) Sprachqualität. Korrigiere Mängel. Gib NUR das finale, verbessertes HTML zurück "
        "(ohne <html>/<head>/<body>), mit Abschluss ‘— sleepingbagREBEL’. Keine Kommentare/Erklärungen."
    )
    joined = "\n\n".join([f"CANDIDATE_{i+1}_START\n{c}\nCANDIDATE_{i+1}_END" for i, c in enumerate(candidates)])
    return (
        f"{rubric}\n\nEMAIL_BODY_HTML_START\n{body_html}\nEMAIL_BODY_HTML_END\n\n{joined}"
    )


def generate_overkill_reply(body_html: str) -> str:
    # 1) Kandidaten erstellen
    n = max(1, min(ANTHROPIC_N_CANDIDATES, 5))
    candidates: List[str] = []
    for i in range(n):
        user_text = _candidate_prompt(body_html, i + 1)
        cand_html = run_agent_with_tools(user_text).strip()
        candidates.append(cand_html)

    final_html = candidates[-1]

    # 2) Kritische Auswahl/Politur (zweiter Pass)
    if n > 1 or CRITIQUE_STRICT:
        critique_text = _critique_prompt(candidates, body_html)
        final_html = run_agent_with_tools(critique_text).strip() or final_html

    # 3) Sanitize
    return _sanitize_email_html(final_html)


# =========================
# ==== MS GRAPH CLIENT ====
# =========================
class GraphClient:
    def __init__(self):
        self.app = ConfidentialClientApplication(
            CLIENT_ID, authority=f"https://login.microsoftonline.com/{TENANT_ID}",
            client_credential=CLIENT_SECRET
        )

    def token(self) -> str:
        res = self.app.acquire_token_for_client(scopes=["https://graph.microsoft.com/.default"])
        if "access_token" not in res:
            raise RuntimeError(f"Tokenfehler: {res.get('error_description')}")
        return res["access_token"]

    def list_messages_since(self, since_iso: str, max_count: int = 50) -> List[Dict[str, Any]]:
        at = self.token()
        headers = {"Authorization": f"Bearer {at}"}
        params = {
            "$top": str(max_count),
            "$select": "id,receivedDateTime",
            "$orderby": "receivedDateTime desc",
            "$filter": f"receivedDateTime ge {since_iso}",
        }
        url = f"{GRAPH_BASE}/users/{SHARED_MAILBOX}/mailFolders/Inbox/messages"
        r = requests.get(url, headers=headers, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        return data.get("value", [])

    def get_message_body(self, msg_id: str) -> str:
        at = self.token()
        headers = {"Authorization": f"Bearer {at}"}
        url = f"{GRAPH_BASE}/users/{SHARED_MAILBOX}/messages/{msg_id}?$select=body"
        r = requests.get(url, headers=headers, timeout=20)
        r.raise_for_status()
        body = r.json().get("body", {})
        return body.get("content", "") or ""

    def create_reply_draft(self, original_id: str, html_body: str) -> str:
        at = self.token()
        headers = {"Authorization": f"Bearer {at}", "Content-Type": "application/json"}

        # 1) Reply-Entwurf anlegen
        url_create = f"{GRAPH_BASE}/users/{SHARED_MAILBOX}/messages/{original_id}/createReply"
        r = requests.post(url_create, headers=headers, timeout=20)
        r.raise_for_status()
        draft = r.json()
        draft_id = draft["id"]

        # 2) Body setzen (HTML)
        url_patch = f"{GRAPH_BASE}/users/{SHARED_MAILBOX}/messages/{draft_id}"
        patch = {"body": {"contentType": "HTML", "content": html_body}}
        r2 = requests.patch(url_patch, headers=headers, json=patch, timeout=20)
        r2.raise_for_status()
        return draft_id

# =========================
# ======= HELPERS =========
# =========================

def utc_iso_now_minus_minutes(minutes: int) -> str:
    return (dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=minutes)).strftime("%Y-%m-%dT%H:%M:%SZ")

# =========================
# ====== AUTODRAFT GRAPH ==
# =========================
class AppState(TypedDict, total=False):
    messages: List[Any]
    lookback_iso: str
    new_emails: List[Dict[str, Any]]
    drafted_count: int
    drafted_ids: List[str]


def node_fetch_recent_emails(state: AppState) -> AppState:
    client = GraphClient()
    since_iso = utc_iso_now_minus_minutes(LOOKBACK_MINUTES)
    msgs = client.list_messages_since(since_iso=since_iso, max_count=50)
    return {
        "lookback_iso": since_iso,
        "new_emails": msgs,
    }


def node_generate_drafts_body_only(state: AppState) -> AppState:
    client = GraphClient()
    drafted = 0
    draft_ids: List[str] = []

    for m in state.get("new_emails", []):
        msg_id = m["id"]
        body_html = client.get_message_body(msg_id)

        # OVERKILL: mehrere Kandidaten -> beste Version -> Sanitizer
        reply_html = generate_overkill_reply(body_html)

        try:
            draft_id = client.create_reply_draft(original_id=msg_id, html_body=reply_html)
            drafted += 1
            draft_ids.append(draft_id)
        except Exception as e:
            draft_ids.append(f"[Draft-Fehler für {msg_id}: {e}]")

    return {"drafted_count": drafted, "drafted_ids": draft_ids}


def node_summarize(state: AppState) -> AppState:
    drafted = state.get("drafted_count", 0)
    ids = state.get("drafted_ids", [])
    lookback = state.get("lookback_iso", "")
    summary_lines = [
        f"Zeitraum: seit {lookback}",
        f"Erstellte Entwürfe: {drafted}",
    ]
    if ids:
        summary_lines.append("Draft-IDs / Meldungen:")
        summary_lines.extend(f"- {x}" for x in ids)
    text = "\n".join(summary_lines)
    return {"messages": [AIMessage(content=text)]}

# Graph bauen (Autodraft)
builder_autodraft = StateGraph(AppState)
builder_autodraft.add_node("fetch_recent_emails", node_fetch_recent_emails)
builder_autodraft.add_node("generate_drafts_body_only", node_generate_drafts_body_only)
builder_autodraft.add_node("summarize", node_summarize)

builder_autodraft.add_edge(START, "fetch_recent_emails")
builder_autodraft.add_edge("fetch_recent_emails", "generate_drafts_body_only")
builder_autodraft.add_edge("generate_drafts_body_only", "summarize")
builder_autodraft.add_edge("summarize", END)

graph_autodraft = builder_autodraft.compile()

# =================================
# ====== CHAT GRAPH (CONVERSATION)
# =================================

def call_model(state: MessagesState) -> Dict[str, Any]:
    msgs = [SystemMessage(content=SYSTEM)] + state["messages"]
    ai = _try_invoke_with_fallback(msgs)
    return {"messages": [ai]}

# Tool-Knoten für automatische Toolausführung
tool_node = ToolNode(TOOLS)

# Graph bauen (Chat)
builder_chat = StateGraph(MessagesState)
builder_chat.add_node("call_model", call_model)
builder_chat.add_node("tools", tool_node)

builder_chat.add_edge(START, "call_model")
builder_chat.add_conditional_edges(
    "call_model",
    tools_condition,
    {
        "tools": "tools",
        END: END,
    },
)
builder_chat.add_edge("tools", "call_model")

graph_chat = builder_chat.compile()

# =================================
# ===== Default-Export (optional) ==
# =================================
# Für bestehende Deployments zeigt der Default auf Autodraft
graph = graph_autodraft
