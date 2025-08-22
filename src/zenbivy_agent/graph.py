# graph.py
from __future__ import annotations
import os, time, random, json, re
import datetime as dt
from typing import TypedDict, List, Optional, Dict, Any
from pathlib import Path
from urllib.parse import urljoin

import requests
from msal import ConfidentialClientApplication

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic

# Optionales HTML-Parsing (schönere Ergebnisse, aber nicht zwingend)
try:
    from bs4 import BeautifulSoup  # pip install beautifulsoup4
except Exception:
    BeautifulSoup = None

# =========================
# ====== ENV & CONSTS =====
# =========================
TENANT_ID = os.environ["MS_TENANT_ID"]           # Azure AD Tenant ID
CLIENT_ID = os.environ["MS_CLIENT_ID"]           # App (client) ID
CLIENT_SECRET = os.environ["MS_CLIENT_SECRET"]   # Client secret
SHARED_MAILBOX = os.environ["MS_SHARED_MAILBOX"] # z.B. "friends@zenbivy.eu"
GRAPH_BASE = "https://graph.microsoft.com/v1.0"

# --- Modell/Token-Config (robust) ---
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-7-sonnet-20250219")
REQ_MAX_TOKENS = int(os.getenv("ANTHROPIC_MAX_TOKENS", "8000"))  # sicherer Default
ANTHROPIC_OUTPUT_128K = os.getenv("ANTHROPIC_OUTPUT_128K", "0") == "1"
ANTHROPIC_FALLBACK_MODEL = os.getenv("ANTHROPIC_FALLBACK_MODEL", "claude-3-5-sonnet-20241022")

# Wie weit zurück (in Minuten) E-Mails geholt werden sollen
LOOKBACK_MINUTES = int(os.getenv("LOOKBACK_MINUTES", "5"))

# System-Prompt (aktiv, tool-freundlich, bildbewusst)
SYSTEM = (
    "Du bist sleepingbagREBEL, ein präziser, freundlicher Mitarbeiter von Zenbivy. "
    "Antworte kurz, konkret und in der Sprache des Nutzers. "
    "Dir stehen Tools zur Verfügung (Webseiten laden, Guides, Bedingungen, Verfügbarkeiten). "
    "Nutze die Tools **proaktiv** und **lieber zu viel als zu wenig**, sobald externe Fakten, Website-Inhalte, "
    "Produktdetails, Größen/Verfügbarkeiten oder Richtlinien relevant sein könnten. "
    "Wenn du Webseiten-Inhalte nutzt, berücksichtige auch Bilder (Alt-Text, Bildunterschriften, og:image) "
    "und integriere deren **beschreibende Information** in deine Antwort, falls hilfreich. "
    "Zitiere keine riesigen HTML-Blöcke; fasse Inhalte präzise zusammen. "
    "Wenn du unsicher bist, recherchiere mit den Tools statt zu spekulieren."
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

def _clean_text(text: str, limit: int = 8000) -> str:
    t = re.sub(r"\s+", " ", (text or "")).strip()
    if len(t) > limit:
        t = t[:limit] + " …"
    return t

def _extract_with_bs4(html: str, base_url: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    # Titel
    title = (soup.title.string if soup.title else "") or ""
    # Sichtbarer Text (rudimentär)
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    text = soup.get_text(separator=" ", strip=True)

    # Bilder
    imgs: List[Dict[str, str]] = []

    # og:image / twitter:image
    for m in soup.find_all("meta"):
        prop = (m.get("property") or m.get("name") or "").lower()
        if prop in {"og:image", "twitter:image", "image"}:
            content = m.get("content")
            if content:
                imgs.append({
                    "url": urljoin(base_url, content),
                    "alt": "Seitenbild (Meta)",
                    "caption": "",
                    "source": "meta"
                })

    # figure/figcaption
    for fig in soup.find_all("figure"):
        img = fig.find("img")
        if not img:
            continue
        src = img.get("src") or img.get("data-src") or ""
        if not src:
            continue
        alt = img.get("alt") or img.get("title") or ""
        cap_tag = fig.find("figcaption")
        caption = cap_tag.get_text(" ", strip=True) if cap_tag else ""
        imgs.append({
            "url": urljoin(base_url, src),
            "alt": alt,
            "caption": caption,
            "source": "figure"
        })

    # allgemeine <img>
    blacklist = {"sprite", "icon", "logo", "placeholder", "tracking", "pixel", "badge", "spinner"}
    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src") or ""
        if not src:
            continue
        low = src.lower()
        if any(b in low for b in blacklist):
            continue
        alt = img.get("alt") or img.get("title") or ""
        if not alt and not any(w in low for w in ("product", "size", "chart", "manual", "guide", "bed", "quilt")):
            # wenig hilfreiche, rein dekorative Bilder weglassen
            continue
        imgs.append({
            "url": urljoin(base_url, src),
            "alt": alt,
            "caption": "",
            "source": "img"
        })

    # Duplikate filtern (nach URL)
    seen = set()
    deduped = []
    for it in imgs:
        u = it["url"]
        if u in seen:
            continue
        seen.add(u)
        deduped.append(it)

    return {
        "title": (title or "").strip(),
        "text": _clean_text(text, limit=10000),
        "images": deduped[:12],  # begrenzen
    }

def _extract_without_bs4(html: str, base_url: str) -> Dict[str, Any]:
    # sehr einfache Fallback-Extraktion
    text = re.sub(r"(?s)<(script|style|noscript).*?>.*?</\1>", " ", html, flags=re.I)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = _clean_text(text, limit=10000)
    img_urls = re.findall(r'<img[^>]+(?:src|data-src)\s*=\s*["\']([^"\']+)["\']', html, flags=re.I)
    imgs = []
    for u in img_urls:
        u_abs = urljoin(base_url, u)
        if any(b in u_abs.lower() for b in ("sprite", "icon", "logo", "placeholder", "tracking", "pixel", "badge", "spinner")):
            continue
        imgs.append({"url": u_abs, "alt": "", "caption": "", "source": "img"})
    return {"title": "", "text": text, "images": imgs[:12]}

def _fetch_and_extract(url: str) -> Dict[str, Any]:
    html = _http_get(url)
    if BeautifulSoup is not None:
        data = _extract_with_bs4(html, base_url=url)
    else:
        data = _extract_without_bs4(html, base_url=url)
    data["url"] = url
    return data

@tool("bedingungen")
def bedingungen(kategorie: str) -> str:
    """
    [Aktiv nutzen] Shop-Bedingungen in Kurzform.
    Verwende dieses Tool proaktiv bei Fragen zu: Rabattcodes, Rückgabe/Umtausch, Versand.
    Eingabe: Freitext mit einer Kategorie (z.B. "Rückgabe", "Versand", "Rabatt").
    Ausgabe: kompakte, kundentaugliche Zusammenfassung.
    """
    k = kategorie.strip().lower()
    if "rabatt" in k:
        return "Rabattcode: Meist im Newsletter (https://zenbivy.eu)."
    if "rückgabe" in k or "umtausch" in k:
        return "Rückgabe/Umtausch: 14 Tage ab Erhalt; Artikel unbenutzt. Details auf der Website."
    if "versand" in k:
        return "Versand: EU-weit; Laufzeiten 2–7 Werktage. Genaues auf https://zenbivy.eu."
    return "Unbekannte Kategorie. Verfügbar: Rabattcode | Rückgabe- & Umtauschbedingungen | Versandbedingungen."

@tool("gear_guide")
def gear_guide(name: str) -> str:
    """
    [Aktiv nutzen] Lädt vordefinierte Zenbivy-Seiten (Size Guide, Manual, Accessory Guide, Kontakt)
    und extrahiert **Text + Bildinfos** (Alt-Text/Caption/URL).
    Verwende dieses Tool, wenn es um Größen, Handbücher, Zubehör oder Kontaktinfos geht.
    Input: Einer der Namen: Größentabelle | Gebrauchsanweisung | Accessory Guide | Kontakt
    Output: JSON-ähnlicher Text mit 'title', 'text', 'images' (Liste mit url, alt, caption).
    """
    if name not in _SOURCES:
        return json.dumps({"error": f"'{name}' nicht verfügbar. Options: {', '.join(_SOURCES.keys())}"}, ensure_ascii=False)
    url = _SOURCES[name]
    try:
        data = _fetch_and_extract(url)
        # Text etwas begrenzen für den Kontext
        data["text"] = _clean_text(data.get("text",""), limit=6000)
        return json.dumps(data, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"Fehler beim Laden: {e}", "url": url}, ensure_ascii=False)

_DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).resolve().parents[2] / "data"))

@tool("wieder_verfuegbar")
def wieder_verfuegbar(datei: str) -> str:
    """
    [Aktiv nutzen] Liest Verfügbarkeits-/Termininfos aus einer Datei im data/-Ordner.
    Verwende dies, wenn es um Liefertermine/Back-in-Stock geht.
    Input: Dateiname ohne/mit .txt (z.B. 'Light Quilt -4°C').
    Output: Klartext-Inhalt der Datei.
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
    [Aktiv nutzen] Einfache Websuche (Platzhalter).
    Verwende dieses Tool, um **schnell** eine relevante URL zu bestimmen.
    Danach direkt mit 'fetch_page' die Seite laden, um **Text + Bildinfos** zu bekommen.
    Output: {query, note, url}
    """
    base = "https://zenbivy.eu" if restrict_to_zenbivy else "https://duckduckgo.com/?q="
    return {"query": query, "note": "Demo-Suche – nutze anschließend fetch_page(url)", "url": base}

@tool("fetch_page")
def fetch_page(url: str) -> str:
    """
    [Aktiv nutzen] Lädt **beliebige** Website-URL und extrahiert **Text + Bildinfos** (Alt-Text/Caption/URL).
    Verwende dieses Tool proaktiv, wenn externe Fakten, Produktseiten, Maßtabellen oder Handbücher relevant sind.
    Input: url (http/https)
    Output: JSON-ähnlicher Text mit 'title', 'text', 'images' (Liste mit url, alt, caption).
    """
    try:
        data = _fetch_and_extract(url)
        data["text"] = _clean_text(data.get("text",""), limit=6000)
        return json.dumps(data, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"Fehler beim Laden: {e}", "url": url}, ensure_ascii=False)

TOOLS = [bedingungen, gear_guide, wieder_verfuegbar, search_web, fetch_page]
_TOOL_MAP = {t.name: t for t in TOOLS}

# =========================
# ========= LLM ===========
# =========================
_extra_headers = {}
if ANTHROPIC_OUTPUT_128K:
    _extra_headers["anthropic-beta"] = "output-128k-2025-02-19"
SAFE_MAX_TOKENS = REQ_MAX_TOKENS if ANTHROPIC_OUTPUT_128K else min(REQ_MAX_TOKENS, 8000)

llm = ChatAnthropic(
    model=ANTHROPIC_MODEL,
    api_key=ANTHROPIC_API_KEY,
    temperature=0.2,
    max_tokens=SAFE_MAX_TOKENS,
    extra_headers=_extra_headers or None,
)
llm_with_tools = llm.bind_tools(TOOLS)

def _is_retryable_error(exc: Exception) -> bool:
    s = str(exc).lower()
    return any(k in s for k in [
        "overloaded", "rate_limit", "timeout", "temporarily", "unavailable",
        "gateway", "service unavailable", "529", "429", "502", "503", "504"
    ])

def _invoke_with_retry(_llm, msgs, attempts: int = 6, base: float = 0.5, cap: float = 20.0):
    for i in range(attempts):
        try:
            return _llm.invoke(msgs, config=RunnableConfig())
        except Exception as e:
            if i == attempts - 1 or not _is_retryable_error(e):
                raise
            sleep_s = min(cap, base * (2 ** i)) + random.uniform(0, 0.5)
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
                max_tokens=min(SAFE_MAX_TOKENS, 8000),
            ).bind_tools(TOOLS)
            return _invoke_with_retry(alt_llm, msgs)
        raise

def run_agent_with_tools(user_text: str) -> str:
    """
    Single-turn mit Tools (max 3 Runden). Mutig Tools nutzen (durch SYSTEM).
    """
    msgs: List[Any] = [SystemMessage(content=SYSTEM), HumanMessage(content=user_text)]
    for _ in range(3):
        ai: AIMessage = _try_invoke_with_fallback(msgs)
        msgs.append(ai)
        tool_calls = getattr(ai, "tool_calls", None) or []
        if not tool_calls:
            return (ai.content or "").strip()
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
            # ToolMessage erwartet String – sauber serialisieren
            if isinstance(res, (dict, list)):
                content = json.dumps(res, ensure_ascii=False)
            else:
                content = str(res)
            msgs.append(ToolMessage(content=content, name=name, tool_call_id=call.get("id")))
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

    def _auth_headers(self, prefer_html: bool = False) -> Dict[str, str]:
        h = {"Authorization": f"Bearer {self.token()}"}
        if prefer_html:
            h["Prefer"] = 'outlook.body-content-type="html"'
        return h

    def list_messages_since(self, since_iso: str, max_count: int = 50) -> List[Dict[str, Any]]:
        headers = self._auth_headers()
        params = {
            "$top": str(max_count),
            "$select": "id,receivedDateTime,subject,from",
            "$orderby": "receivedDateTime desc",
            "$filter": f"receivedDateTime ge {since_iso}",
        }
        url = f"{GRAPH_BASE}/users/{SHARED_MAILBOX}/mailFolders/Inbox/messages"
        r = requests.get(url, headers=headers, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        return data.get("value", [])

    def get_message_core(self, msg_id: str) -> Dict[str, Any]:
        headers = self._auth_headers(prefer_html=True)
        url = f"{GRAPH_BASE}/users/{SHARED_MAILBOX}/messages/{msg_id}"
        params = {
            "$select": "id,subject,from,sentDateTime,receivedDateTime,body",
        }
        r = requests.get(url, headers=headers, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        body = data.get("body", {}) or {}
        return {
            "id": data.get("id"),
            "subject": data.get("subject") or "",
            "from": ((data.get("from") or {}).get("emailAddress") or {}).get("name") or "",
            "from_addr": ((data.get("from") or {}).get("emailAddress") or {}).get("address") or "",
            "sentDateTime": data.get("sentDateTime"),
            "receivedDateTime": data.get("receivedDateTime"),
            "body_html": body.get("content", "") or "",
        }

    def create_reply_draft(self, original_id: str, html_body: str) -> str:
        headers = self._auth_headers()
        headers["Content-Type"] = "application/json"

        # 1) Reply-Entwurf anlegen -> behält Threading-Header (In-Reply-To/References)
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

def _format_dt_for_quote(iso: Optional[str]) -> str:
    if not iso:
        return ""
    try:
        d = dt.datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone(dt.timezone(dt.timedelta(hours=0)))
        return d.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return iso or ""

def build_reply_with_history(reply_html: str, original_html: str,
                             from_name: str = "", sent_iso: str = "", subject: str = "") -> str:
    when = _format_dt_for_quote(sent_iso)
    header_line = ""
    if when or from_name or subject:
        header_line = (
            f'<div style="margin-top:16px;margin-bottom:8px;font-size:12px;color:#555;">'
            f'----- Original Message -----<br>'
            f'Von: {from_name or "Unbekannt"}<br>'
            f'Gesendet: {when or "Unbekannt"}<br>'
            f'Betreff: {subject or "(kein Betreff)"}'
            f'</div>'
        )
    quoted = (
        '<blockquote style="margin:0;padding-left:.8em;border-left:2px solid #ccc;">'
        f'{original_html}'
        '</blockquote>'
    )
    final_html = f'{reply_html}<br><br>{header_line}{quoted}'
    return final_html

def sanitize_llm_html(s: str) -> str:
    t = (s or "").strip()
    if t.startswith("```"):
        t = t.strip("`").strip()
        if t.lower().startswith("html"):
            t = t[4:].lstrip()
    return t

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
    return {"lookback_iso": since_iso, "new_emails": msgs}

def node_generate_drafts_body_only(state: AppState) -> AppState:
    client = GraphClient()
    drafted = 0
    draft_ids: List[str] = []

    for m in state.get("new_emails", []):
        msg_id = m["id"]

        # 1) Original inkl. HTML-Body, From, Betreff, Datum
        core = client.get_message_core(msg_id)
        body_html = core["body_html"] or "<div>(Kein Inhalt erkannt)</div>"
        from_name = core["from"]
        sent_iso = core["sentDateTime"]
        subject = core["subject"]

        # 2) LLM nur mit E-Mail-Body füttern (Antwort-HTML als reinen Body zurückgeben)
        user_text = (
            "Erstelle eine höfliche, hilfreiche und konkrete Antwort als HTML und unterschreibe mit 'sleepingbagREBEL'. "
            "Antworte ausschließlich basierend auf folgendem E-Mail-Body. "
            "Gib NUR den Email-Body der Antwort zurück (keinen Betreff, keine Meta-Zeilen, kein Codeblock). "
            "Antworte in der Sprache des folgenden Inhalts.\n\n"
            "EMAIL_BODY_HTML_START\n"
            f"{body_html}\n"
            "EMAIL_BODY_HTML_END"
        )

        reply_html_raw = run_agent_with_tools(user_text)
        reply_html = sanitize_llm_html(reply_html_raw)
        if not reply_html:
            reply_html = (
                "<p>Vielen Dank für Ihre Nachricht! "
                "Wir prüfen Ihr Anliegen und melden uns in Kürze.</p>"
                "<p>Beste Grüße<br>sleepingbagREBEL</p>"
            )

        # 3) Antwort + zitiertes Original kombinieren (wie Outlook)
        combined_html = build_reply_with_history(
            reply_html=reply_html,
            original_html=body_html,
            from_name=from_name,
            sent_iso=sent_iso,
            subject=subject,
        )

        # 4) Draft im Thread erstellen & Body patchen
        try:
            draft_id = client.create_reply_draft(original_id=msg_id, html_body=combined_html)
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

tool_node = ToolNode(TOOLS)

builder_chat = StateGraph(MessagesState)
builder_chat.add_node("call_model", call_model)
builder_chat.add_node("tools", tool_node)

builder_chat.add_edge(START, "call_model")
builder_chat.add_conditional_edges(
    "call_model",
    tools_condition,
    {"tools": "tools", END: END},
)
builder_chat.add_edge("tools", "call_model")

graph_chat = builder_chat.compile()

# =================================
# ===== Default-Export (run) ======
# =================================
graph = graph_autodraft
