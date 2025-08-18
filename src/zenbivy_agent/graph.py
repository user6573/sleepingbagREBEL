from __future__ import annotations
import os
import datetime as dt
from typing import TypedDict, List, Optional, Dict, Any

import requests
from msal import ConfidentialClientApplication

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic

# =========================
# ====== ENV & CONSTS =====
# =========================
TENANT_ID = os.environ["MS_TENANT_ID"]           # <-- HIER EINTRAGEN (oder als Secret setzen)
CLIENT_ID = os.environ["MS_CLIENT_ID"]           # <-- HIER EINTRAGEN
CLIENT_SECRET = os.environ["MS_CLIENT_SECRET"]   # <-- HIER EINTRAGEN
SHARED_MAILBOX = os.environ["MS_SHARED_MAILBOX"] # z.B. "[email protected]"  <-- HIER EINTRAGEN
GRAPH_BASE = "https://graph.microsoft.com/v1.0"

ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]  # <-- HIER EINTRAGEN

# System-Prompt (aus deiner Beschreibung, leicht gestrafft)
SYSTEM = (
    "Du bist sleepingbagREBEL, ein präziser, netter Mitarbeiter von Zenbivy. "
    "Zeige nicht, dass du eine KI bist. "
    "Antworte in der Sprache der Anfrage und kompakt. "
    "Gib nur die Informationen, nach denen gefragt wurde. "
    "Weise dezent auf den Newsletter hin (https://zenbivy.eu – rechts unten). "
    "Falls du die Frage nicht beantworten kannst, sage das und verweise auf friends@zenbivy.eu. "
    "Tool-Nutzung: "
    "• Für Größen/Anleitungen/Zubehör: 'gear_guide'. "
    "• Für Versand/Rückgabe/Rabatt: 'bedingungen'. "
    "• Für Verfügbarkeiten: 'wieder_verfuegbar'. "
    "• Nutze 'search_web' (Zenbivy-Domain) nur bei Bedarf. "
    "• RAG nur als letztes Mittel (hier nicht aktiv). "
    "Verwende keine Sternchen (*)."
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
    # Vereinfachte Extraktion (ohne Bilder), bewusst kurz gehalten
    text = html
    if len(text) > 5000:
        text = text[:5000] + " …"
    return {"source": name, "url": url, "text": text}

from pathlib import Path
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

# =========================
# ========= LLM ===========
# =========================
llm = ChatAnthropic(
    model=ANTHROPIC_MODEL,
    api_key=ANTHROPIC_API_KEY,
    temperature=0.2,
    max_tokens=2000,
)
llm_with_tools = llm.bind_tools(TOOLS)
_TOOL_MAP = {t.name: t for t in TOOLS}

def run_agent_with_tools(user_text: str) -> str:
    """Einmalige Tool-Schleife: LLM aufrufen, evtl. Tools ausführen, finalen Text zurückgeben."""
    msgs: List[Any] = [SystemMessage(content=SYSTEM), HumanMessage(content=user_text)]
    for _ in range(4):  # bis zu 4 Tool-Runden
        ai: AIMessage = llm_with_tools.invoke(msgs, config=RunnableConfig())
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
    # Falls keine toolfreien Antworten kamen:
    return "Ich konnte die Anfrage nicht abschließen. Bitte schreibe an friends@zenbivy.eu."

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

    def list_new_messages(self, since_iso: Optional[str], max_count: int = 10) -> List[Dict[str, Any]]:
        """
        Holt neue Mails aus Inbox des Shared Mailbox.
        Wenn since_iso gesetzt ist, filtert nach receivedDateTime > since_iso.
        """
        at = self.token()
        headers = {"Authorization": f"Bearer {at}"}
        # Filter zusammenbauen
        params = {
            "$top": str(max_count),
            "$select": "id,subject,receivedDateTime,from,bodyPreview,conversationId",
            "$orderby": "receivedDateTime desc",
        }
        if since_iso:
            params["$filter"] = f"receivedDateTime gt {since_iso}"
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
        """
        Erzeugt einen Antwort-ENTWURF zu einer bestehenden Nachricht.
        Gibt die Draft-ID zurück. NICHT senden.
        """
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
# ====== GRAPH STATE ======
# =========================
class AppState(TypedDict, total=False):
    # für LangGraph
    messages: List[Any]
    # eigene Felder
    last_seen_iso: Optional[str]
    new_emails: List[Dict[str, Any]]
    drafted_count: int
    drafted_subjects: List[str]

def node_fetch_new_emails(state: AppState) -> AppState:
    client = GraphClient()
    since = state.get("last_seen_iso")
    msgs = client.list_new_messages(since_iso=since, max_count=10)
    # Max timestamp merken
    max_dt = since
    for m in msgs:
        rcv = m.get("receivedDateTime")
        if rcv and (max_dt is None or rcv > max_dt):
            max_dt = rcv
    out: AppState = {
        "new_emails": msgs,
        "last_seen_iso": max_dt or since,
    }
    return out

def node_generate_drafts(state: AppState) -> AppState:
    client = GraphClient()
    drafted = 0
    subjects: List[str] = []
    for m in state.get("new_emails", []):
        msg_id = m["id"]
        subject = m.get("subject") or "(ohne Betreff)"
        # Volltext nachladen:
        body_html = client.get_message_body(msg_id)
        preview = (m.get("bodyPreview") or "").strip()
        sender = (m.get("from", {}) or {}).get("emailAddress", {}).get("address", "")
        # Prompt für die Antwort
        user_text = (
            f"Beantworte diese Kundenmail höflich und hilfreich. "
            f"Absender: {sender}\n"
            f"Betreff: {subject}\n"
            f"Mail (Auszug): {preview}\n"
            f"Mail (HTML-Volltext folgt):\n{body_html}\n"
            f"Erstelle bitte eine kurze, konkrete Antwort als HTML (ohne Signatur)."
        )
        reply_html = run_agent_with_tools(user_text)
        # Entwurf erzeugen
        try:
            draft_id = client.create_reply_draft(original_id=msg_id, html_body=reply_html)
            drafted += 1
            subjects.append(subject)
        except Exception as e:
            subjects.append(f"{subject} [Draft-Fehler: {e}]")
    return {"drafted_count": drafted, "drafted_subjects": subjects}

def node_summarize(state: AppState) -> AppState:
    drafted = state.get("drafted_count", 0)
    subjects = state.get("drafted_subjects", [])
    text = f"Erstellte Entwürfe: {drafted}\n" + "\n".join(f"- {s}" for s in subjects)
    # Eine Abschlussnachricht in den State schreiben (optional nützlich fürs Studio)
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
