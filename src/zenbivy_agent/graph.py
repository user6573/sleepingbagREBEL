# graph.py
from __future__ import annotations

import uuid, os, re, json, time, random, datetime as dt
from typing import Annotated, Literal, List, Dict, Any, Optional, Tuple
from typing_extensions import TypedDict
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# =========================
# ===== LangChain/LG ======
# =========================
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# LLM (Claude)
from langchain_anthropic import ChatAnthropic

# --- RAG-Dependencies ---
import chromadb
from chromadb.config import Settings
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None
try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None

# --- HTTP / Parsing ---
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# --- Optional Tavily ---
try:
    from tavily import TavilyClient
except Exception:
    TavilyClient = None

# --- MS Graph (Koch-Tracking + AutoDraft) ---
try:
    from msal import ConfidentialClientApplication
except Exception:
    ConfidentialClientApplication = None

# =========================
# ====== LLM & PROMPT =====
# =========================
SYSTEM = (
    "Du bist sleepingbagREBEL, ein präziser, netter Mitarbeiter von Zenbivy. "
    "Zeige nicht, dass du eine KI bist. "
    "Antworte stets in der Sprache der Anfrage und in kompaktem Text. "
    "Weise dezent auf den Newsletter hin (https://zenbivy.eu – rechts unten). "
    "Für Bestelländerungen, Reklamationen oder Fälle für Menschen: bitte um E-Mail an friends@zenbivy.eu.\n"
    "\n"
    "Tool-Nutzung (proaktiv, lieber zu oft als zu selten):\n"
    "• Für Größen, Anleitung, Füllgewicht, Zubehör: nutze 'gear_guide' (lädt Seiten & berücksichtigt Bilder via Alt-Text/Captions).\n"
    "• Für Versand/Rückgabe/Rabatt: nutze 'bedingungen'.\n"
    "• Für Verfügbarkeiten/Termine: nutze 'wieder_verfuegbar'.\n"
    "• Für E-Mail-Antworten mit vorhandenen internen Infos: nutze 'rag' (Kontext holen, dann verallgemeinern, keine Namen/Datumsangaben übernehmen).\n"
    "• Für Web-Recherche oder direkte URL-Inhalte: nutze 'search_web' (liefert Seiten-Text + Bild-Hinweise).\n"
    "• Für Sendungs-Status-Links: nutze 'finde_lieferung' (sucht im Shared Mailbox die gesendete Versandmail und extrahiert den Tracking-Link).\n"
    "\n"
    "Wenn du Webseiten nutzt, integriere relevante Bildinformationen (Alt-Text, Bildunterschrift) inhaltlich in deine Antwort, aber füge keine großen HTML-Blöcke ein."
)

# Anthropic / Modell-Config
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-7-sonnet-20250219")
REQ_MAX_TOKENS = int(os.getenv("ANTHROPIC_MAX_TOKENS", "8000"))
ANTHROPIC_OUTPUT_128K = os.getenv("ANTHROPIC_OUTPUT_128K", "0") == "1"
ANTHROPIC_FALLBACK_MODEL = os.getenv("ANTHROPIC_FALLBACK_MODEL", "claude-3-5-sonnet-20241022")
SAFE_MAX_TOKENS = REQ_MAX_TOKENS if ANTHROPIC_OUTPUT_128K else min(REQ_MAX_TOKENS, 8000)
_extra_headers = {"anthropic-beta": "output-128k-2025-02-19"} if ANTHROPIC_OUTPUT_128K else None

llm = ChatAnthropic(
    model=ANTHROPIC_MODEL,
    api_key=ANTHROPIC_API_KEY,
    temperature=0.2,
    max_tokens=SAFE_MAX_TOKENS,
    extra_headers=_extra_headers,
)

# =========================
# ====== PFAD/DATEIEN =====
# =========================
_BASE_DIR = os.getenv("ZENBIVY_DATA_DIR", r"C:\Users\Fritz\Desktop\Python\sleepingbagREBEL_Infos\geschliffen")

DateiAuswahl = Literal[
    "Compression Caps",
    "Core  Quilt Double  -4°C",
    "Core Quilt -12°C",
    "Core Quilt -4°C",
    "Core Quilt -4°C Synthetic",
    "Core Sheet Double -4°C",
    "Core Sheet Down",
    "Core Sheet Synthetic",
    "Core Sheet Uninsulated",
    "Coupon EUR 50",
    "Ditty Dry Sack",
    "Double Flex 3D Mattress",
    "Double Luxe Sheet -4°C",
    "Double Quilt -4°C",
    "Down Pillow Topper",
    "Dry Sack",
    "Flex 3D Mattress",
    "Flex Air Mattress",
    "Flex Mattress",
    "Inflation Dry Sack",
    "Light Mattress",
    "Light Quilt +4°C Synthetic",
    "Light Quilt -12°C",
    "Light Quilt -20°C",
    "Light Quilt -4°C",
    "Light Quilt 2025 +4°C Synthetic",
    "Light Quilt 2025 -12°C",
    "Light Quilt 2025 -4°C",
    "Light Quilt Double  -4°C",
    "Light Sheet -12°C",
    "Light Sheet -20°C",
    "Light Sheet -4°C",
    "Light Sheet Double -4°C",
    "Light Sheet Uninsulated",
    "Mattress Repair Kit",
    "Max Pump 2 Pro",
    "Pillow Bladder",
    "Pillowcase",
    "Sonstiges",
    "Titan Bivy Mug Lid",
    "Ultralight Mattress",
    "Ultralight Muscovy Quilt -12°C",
    "Ultralight Muscovy Quilt -4°C",
    "Ultralight Muscovy Sheet -12°C",
    "Ultralight Muscovy Sheet -4°C",
    "Ultralight Quilt -12°C",
    "Ultralight Quilt -4°C",
    "Ultralight Sheet -12°C",
    "Ultralight Sheet -4°C",
    "Ultralight Sheet Uninsulated",
    "Zenbivy Bed -12°C",
    "Zenbivy Bed -4°C",
    "ZENBIVY VOUCHER",
    "Zip Sack",
    "Zipbed Overland -4°C Down",
    "Zipbed Overland -4°C Synthetic",
]

PolicyKey = Literal[
    "Rabattcode",
    "Rückgabe- & Umtauschbedingungen",
    "Versandbedingungen",
]

GuideKey = Literal[
    "Baue dein Schlafsystem",
    "Größentabelle",
    "Gebrauchsanweisung",
    "Füllgewicht",
    "Besitzerhandbuch",
    "Reparaturanleitung",
    "Putzanleitung",
    "Patents",
    "Kontakt",
    "Accessory Guide",
    "Give Away",
]

_SOURCES = {
    "Baue dein Schlafsystem": "https://zenbivy.eu/pages/build-your-sleeping-bag-system",
    "Größentabelle": "https://zenbivy.eu/pages/size-guide",
    "Gebrauchsanweisung": "https://zenbivy.eu/pages/owners-manual-support-document",
    "Füllgewicht": "https://zenbivy.eu/pages/down-fill-weights",
    "Besitzerhandbuch": "https://zenbivy.com/pages/owners",
    "Reparaturanleitung": "https://zenbivy.com/pages/mattress-repair-guide",
    "Putzanleitung": "https://zenbivy.com/pages/washing-instructions",
    "Patents": "https://zenbivy.com/pages/patents",
    "Kontakt": "https://zenbivy.eu/pages/kontakt",
    "Accessory Guide": "https://zenbivy.eu/pages/accessory-guide",
    "Give Away": "https://zenbivy.eu/pages/giveaway",
}

POLICIES: Dict[PolicyKey, str] = {
    "Rabattcode": "Rabattcodes erhältst du im Newsletter (https://zenbivy.eu).",
    "Rückgabe- & Umtauschbedingungen": "Rückgabe/Umtausch: 14 Tage ab Erhalt; Artikel unbenutzt. Details auf der Website.",
    "Versandbedingungen": "Versand: EU-weit; Laufzeiten 2–7 Werktage. Genaues auf https://zenbivy.eu.",
}

_USER_AGENT = "Mozilla/5.0 (compatible; ZenbivyAgent/1.0)"
_HEADERS = {"User-Agent": _USER_AGENT}

# =========================
# === TEXT + BILD PARSER ==
# =========================
def _extract_text_and_images(html: str, base_url: str):
    """
    Kompaktes Extrahieren von Text + bis zu 12 Bildern.
    Berücksichtigt og:image & figcaption; dedupliziert Bilder.
    Rückgabe: (title, text, images=[{src,alt}])
    """
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    main = soup.find("main") or soup.find("article") or soup.body or soup

    parts = []
    for el in main.find_all(["h1","h2","h3","h4","h5","h6","p","li"], recursive=True):
        t = " ".join(el.get_text(" ", strip=True).split())
        if t:
            parts.append(t)
    text = "\n".join(parts)
    if len(text) > 8000:
        text = text[:8000] + " … [gekürzt]"

    images: List[Dict[str, str]] = []

    # figures
    for fig in main.find_all("figure"):
        img = fig.find("img")
        if not img:
            continue
        src = img.get("src") or img.get("data-src") or img.get("data-original")
        if not src:
            continue
        src = urljoin(base_url, src.strip())
        if src.lower().endswith(".svg"):
            continue
        alt = (img.get("alt") or img.get("title") or "").strip()
        cap_tag = fig.find("figcaption")
        if cap_tag and not alt:
            alt = " ".join(cap_tag.get_text(" ", strip=True).split())
        images.append({"src": src, "alt": alt})

    # og:image / twitter:image
    for m in soup.find_all("meta"):
        prop = (m.get("property") or m.get("name") or "").lower()
        if prop in {"og:image", "twitter:image", "image"}:
            content = (m.get("content") or "").strip()
            if content:
                src = urljoin(base_url, content)
                if not src.lower().endswith(".svg"):
                    images.append({"src": src, "alt": ""})

    # generic <img>
    for img in main.find_all("img"):
        src = img.get("src") or img.get("data-src") or img.get("data-original")
        if not src:
            continue
        src = urljoin(base_url, src.strip())
        if src.lower().endswith(".svg"):
            continue
        alt = (img.get("alt") or img.get("title") or "").strip()
        images.append({"src": src, "alt": alt})

    blacklist = ("sprite", "icon", "logo", "placeholder", "tracking", "pixel", "badge", "spinner")
    seen = set()
    deduped = []
    for im in images:
        u = im["src"]
        if any(b in u.lower() for b in blacklist):
            continue
        if u in seen:
            continue
        seen.add(u)
        deduped.append(im)
        if len(deduped) >= 12:
            break

    title = soup.title.get_text(strip=True) if soup.title else ""
    return title, text, deduped

def _fetch_page(url: str) -> dict:
    try:
        r = requests.get(url, timeout=20, headers=_HEADERS)
        r.raise_for_status()
        title, text, images = _extract_text_and_images(r.text, url)
        return {"url": url, "title": title, "text": text, "images": images}
    except Exception as e:
        return {"url": url, "error": f"Fehler beim Laden: {e}"}

def _looks_like_url(s: str) -> bool:
    return bool(re.match(r"^https?://", s.strip(), flags=re.I))

# =========================
# ======== TOOLS ==========
# =========================
@tool("search_web")
def search_web(
    query: str,
    max_results: int = 5,
    restrict_to_zenbivy: bool = True,
) -> dict:
    """
    Websuche via Tavily ODER Direkt-URL:
      - Bei DIREKTER URL: lädt die Seite und gibt Text + Bilder zurück.
      - Bei SUCHE: findet relevante Seiten (optional auf zenbivy.com/.eu beschränkt) und lädt jede Seite.
    """
    if _looks_like_url(query):
        page = _fetch_page(query.strip())
        return {
            "query": query,
            "restricted": restrict_to_zenbivy,
            "results": [{
                "title": page.get("title", ""),
                "url": page.get("url", query),
                "snippet": "",
                "score": None,
                "page": page,
            }],
        }

    if TavilyClient is None:
        return {"query": query, "error": "tavily-python nicht installiert. Bitte 'pip install tavily-python'."}

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return {"query": query, "error": "TAVILY_API_KEY fehlt in der Umgebung."}

    client = TavilyClient(api_key=api_key)
    include_domains = ["zenbivy.com", "zenbivy.eu"] if restrict_to_zenbivy else None
    try:
        search = client.search(
            query=query,
            max_results=max(1, min(int(max_results), 10)),
            include_domains=include_domains,
            search_depth="basic",
            include_images=False,
            include_answer=False,
        )
        raw_results = search.get("results", [])
    except Exception as e:
        return {"query": query, "restricted": bool(restrict_to_zenbivy), "error": f"Fehler bei Tavily: {e}"}

    out = {"query": query, "restricted": bool(restrict_to_zenbivy), "results": []}
    for res in raw_results:
        url = res.get("url", "")
        snippet = res.get("content", "") or res.get("snippet", "")
        title = res.get("title", "")
        score = res.get("score")
        page = _fetch_page(url) if url else {"url": url, "error": "Kein URL im Suchtreffer."}
        out["results"].append({
            "title": title,
            "url": url,
            "snippet": snippet,
            "score": score,
            "page": page,
        })
    return out

@tool("gear_guide")
def gear_guide(name: GuideKey) -> dict:
    """
    Lädt vordefinierte Zenbivy-Seiten und gibt strukturierte Infos inkl. Bildhinweisen zurück.
    """
    url = _SOURCES[name]
    try:
        resp = requests.get(url, timeout=20, headers={"User-Agent": _USER_AGENT})
        resp.raise_for_status()
    except Exception as e:
        return {"source": name, "url": url, "error": f"Fehler beim Laden: {e}"}
    title, text, images = _extract_text_and_images(resp.text, url)
    return {"source": name, "url": url, "title": title, "text": text, "images": images}

@tool("bedingungen")
def bedingungen(kategorie: PolicyKey) -> str:
    return POLICIES[kategorie]

_DATA_DIR = Path(os.getenv("DATA_DIR", Path.cwd() / "data"))
@tool("wieder_verfuegbar")
def wieder_verfuegbar(datei: DateiAuswahl) -> str:
    """
    Öffnet '{datei}.txt' aus dem Datenordner und liefert den Inhalt (Verfügbarkeiten/Termine).
    """
    filename = f"{datei}.txt"
    path = _DATA_DIR / filename
    if not path.exists():
        return f"[FEHLER] Datei nicht gefunden: {filename}"
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return path.read_text(encoding="latin-1", errors="ignore")

# =========================
# === MS GRAPH TOOL: finde_lieferung ===
# =========================
def _normalize_email(e: str) -> str:
    return (e or "").strip().lower()

def _extract_tracking_links_from_html(html_or_text: str) -> Tuple[List[str], List[str]]:
    content = html_or_text or ""
    try:
        soup = BeautifulSoup(content, "html.parser")
        hrefs = [a.get("href") for a in soup.find_all("a") if a.get("href")]
        text = soup.get_text("\n", strip=True)
        candidates = hrefs + re.findall(r"https?://[^\s<>\"]+", text)
    except Exception:
        text = re.sub(r"<[^>]+>", " ", content)
        candidates = re.findall(r"https?://[^\s<>\"]+", text)

    post_links = [u for u in candidates if re.search(r"(^|://)(www\.)?post\.at/", u)]
    seen = set(); uniq_links = []
    for u in post_links:
        if u not in seen:
            seen.add(u); uniq_links.append(u)

    ids = []
    for u in uniq_links:
        m = re.search(r"(?:\?|&|/)(?:pnum1|barcodelist|barcode|pnum)=([0-9]{10,})", u)
        if m:
            ids.append(m.group(1))
    for m in re.finditer(r"\b([0-9]{18,30})\b", text):
        if m.group(1) not in ids:
            ids.append(m.group(1))

    return uniq_links, ids

class _GraphKochClient:
    GRAPH_BASE = "https://graph.microsoft.com/v1.0"
    def __init__(self):
        if ConfidentialClientApplication is None:
            raise RuntimeError("msal ist nicht installiert. Bitte 'pip install msal' und erneut versuchen.")
        self.tenant = os.getenv("MS_TENANT_ID")
        self.client_id = os.getenv("MS_CLIENT_ID")
        self.client_secret = os.getenv("MS_CLIENT_SECRET")
        self.mailbox = os.getenv("MS_SHARED_MAILBOX_KOCH")  # separate Mailbox für Tracking-Suche
        if not (self.tenant and self.client_id and self.client_secret and self.mailbox):
            raise RuntimeError("Fehlende ENV Variablen: MS_TENANT_ID, MS_CLIENT_ID, MS_CLIENT_SECRET, MS_SHARED_MAILBOX_KOCH")

        self.app = ConfidentialClientApplication(
            self.client_id,
            authority=f"https://login.microsoftonline.com/{self.tenant}",
            client_credential=self.client_secret
        )

    def _token(self) -> str:
        res = self.app.acquire_token_for_client(scopes=["https://graph.microsoft.com/.default"])
        if "access_token" not in res:
            raise RuntimeError(f"Tokenfehler: {res.get('error_description')}")
        return res["access_token"]

    def _headers(self, prefer_text: bool = True) -> Dict[str, str]:
        h = {"Authorization": f"Bearer {self._token()}"}
        if prefer_text:
            h["Prefer"] = 'outlook.body-content-type="text"'
        return h

    def list_sent_messages_top(self, top: int = 400) -> List[Dict[str, Any]]:
        url = f"{self.GRAPH_BASE}/users/{self.mailbox}/mailFolders/SentItems/messages"
        params = {
            "$top": str(max(1, min(int(top), 400))),
            "$orderby": "receivedDateTime desc",
            "$select": "id,subject,sentDateTime,receivedDateTime,toRecipients,ccRecipients,bccRecipients",
        }
        r = requests.get(url, headers=self._headers(), params=params, timeout=20)
        r.raise_for_status()
        return r.json().get("value", [])

    def get_message_body(self, msg_id: str) -> Dict[str, Any]:
        url = f"{self.GRAPH_BASE}/users/{self.mailbox}/messages/{msg_id}"
        params = {"$select": "id,subject,sentDateTime,receivedDateTime,body"}
        r = requests.get(url, headers=self._headers(prefer_text=True), params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        body = (data.get("body") or {}).get("content", "") or ""
        return {
            "id": data.get("id"),
            "subject": data.get("subject") or "",
            "sentDateTime": data.get("sentDateTime"),
            "receivedDateTime": data.get("receivedDateTime"),
            "body": body,
        }

def _addresses(lst) -> List[str]:
    out = []
    for x in (lst or []):
        ema = ((x.get("emailAddress") or {}).get("address") or "").strip().lower()
        if ema:
            out.append(ema)
    return out

@tool("finde_lieferung")
def finde_lieferung(email: str) -> dict:
    """
    Durchsucht 'Gesendete Elemente' der Koch-Mailbox und extrahiert post.at-Trackinglink.
    """
    try:
        client = _GraphKochClient()
    except Exception as e:
        return {"error": str(e), "email": email}

    target = _normalize_email(email)
    if not target:
        return {"error": "Ungültige E-Mail-Adresse.", "email": email}

    try:
        msgs = client.list_sent_messages_top(top=400)
    except Exception as e:
        return {"error": f"Fehler beim Laden aus SentItems: {e}", "email": email}

    matched_meta = None
    for m in msgs:
        to_l = _addresses(m.get("toRecipients"))
        cc_l = _addresses(m.get("ccRecipients"))
        bcc_l = _addresses(m.get("bccRecipients"))
        all_rcpts = set(to_l + cc_l + bcc_l)
        if target in all_rcpts:
            matched_meta = m
            break

    if not matched_meta:
        return {
            "email": email, "matched": False, "message": None,
            "status_link": None, "versand_ids": [],
            "checked_count": len(msgs),
            "note": "Keine passende gesendete Versandmail unter den neuesten 400 gefunden."
        }

    try:
        full = client.get_message_body(matched_meta["id"])
    except Exception as e:
        return {"email": email, "matched": True, "message": matched_meta, "error": f"Body-Fehler: {e}"}

    links, ids = _extract_tracking_links_from_html(full.get("body",""))
    status_link = next((u for u in links if "post.at" in u), None)

    return {
        "email": email,
        "matched": True,
        "message": {
            "id": full.get("id"),
            "subject": full.get("subject"),
            "sentDateTime": full.get("sentDateTime"),
            "receivedDateTime": full.get("receivedDateTime"),
        },
        "status_link": status_link,
        "versand_ids": ids,
        "checked_count": len(msgs),
        "note": "Erste passende Nachricht aus den neuesten 400 'Gesendete Elemente' ausgewertet."
    }

# =========================
# ========= RAG ===========
# =========================
_EMBED_LOCAL_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

class _LocalEmbedder:
    def __init__(self, model_name: str = _EMBED_LOCAL_MODEL):
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers fehlt. `pip install sentence-transformers`")
        self.model = SentenceTransformer(model_name)
    def embed(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()

class _OpenAIEmbedder:
    def __init__(self, model: str = "text-embedding-3-small", api_key: Optional[str] = None):
        from openai import OpenAI
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY fehlt (.env)")
        self.client = OpenAI(api_key=key)
        self.model = model
    def embed(self, texts: List[str]) -> List[List[float]]:
        res = self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in res.data]

def _tokenize(texts: List[str]) -> List[List[str]]:
    toks = []
    for t in texts:
        t = re.sub(r"[^a-zA-Z0-9äöüÄÖÜß]+", " ", t or "")
        toks.append([w.lower() for w in t.split() if w])
    return toks

def _build_context(docs: List[str], metas: List[Dict[str,Any]], max_chars: int = 12000):
    parts, used, refs, items = [], 0, [], []
    for d, m in zip(docs, metas):
        ref = f"{m.get('conv_id')}#chunk{int(m.get('chunk',0))+1}/{m.get('chunks_total')}"
        header = f"[DOC {ref} | {m.get('subject')}]"
        blk = header + "\n" + (d or "")
        if used + len(blk) > max_chars:
            break
        parts.append(blk); used += len(blk); refs.append(ref)
        items.append({
            "ref": ref,
            "subject": m.get("subject"),
            "first_time": m.get("first_time"),
            "last_time": m.get("last_time"),
            "message_count": m.get("message_count"),
            "chunk": int(m.get("chunk",0))+1,
            "chunks_total": m.get("chunks_total"),
        })
    return "\n\n".join(parts), refs, items

@tool("rag")
def rag(query: str, top_k: int = 5) -> dict:
    """
    Outlook-RAG-Index (Chroma) durchsuchen und kompakten Kontext zurückgeben.
    """
    index_dir = os.getenv("CHROMA_PATH") or "./rag_index"
    client = chromadb.PersistentClient(path=index_dir, settings=Settings(allow_reset=False))
    coll = client.get_or_create_collection("outlook_rag")
    if coll.count() == 0:
        return {"error": f"Leerer Index unter {index_dir}. Bitte Index kopieren/erstellen."}

    emb_type = (os.getenv("RAG_EMBEDDING") or "local").lower()
    if emb_type == "openai":
        embedder = _OpenAIEmbedder(model=os.getenv("OPENAI_EMBED_MODEL","text-embedding-3-small"))
    else:
        embedder = _LocalEmbedder()

    emb = embedder.embed([query])[0]
    pool_n = max(top_k, int(os.getenv("BM25_POOL", "20")))
    res = coll.query(query_embeddings=[emb], n_results=pool_n, include=["documents","metadatas","distances"])
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    hybrid = (os.getenv("HYBRID", "true").lower() != "false") and (BM25Okapi is not None)
    alpha = float(os.getenv("HYBRID_ALPHA", "0.5"))
    if hybrid and docs:
        corpus_tokens = _tokenize(docs)
        bm25 = BM25Okapi(corpus_tokens)
        q_tokens = _tokenize([query])[0]
        bm_scores = bm25.get_scores(q_tokens)
        if dists:
            max_d, min_d = max(dists), min(dists); rng = max(1e-9, max_d - min_d)
            vec_scores = [1.0 - ((d - min_d) / rng) for d in dists]
        else:
            vec_scores = [0.0]*len(docs)
        max_b = max(bm_scores) if bm_scores is not None and len(bm_scores) else 1.0
        bm_norm = [(s/max_b) if max_b else 0.0 for s in (bm_scores or [])]
        scored = []
        for i in range(len(docs)):
            score = alpha*bm_norm[i] + (1-alpha)*vec_scores[i]
            scored.append((score, docs[i], metas[i]))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_k]
        docs = [t[1] for t in top]; metas = [t[2] for t in top]
    else:
        pairs = list(zip(docs, metas, dists))
        pairs.sort(key=lambda x: x[2]); pairs = pairs[:top_k]
        docs = [p[0] for p in pairs]; metas = [p[1] for p in pairs]

    context, refs, items = _build_context(docs, metas, max_chars=12000)
    return {"context": context, "sources": refs, "items": items}

# =========================
# ======== LLM BIND =======
# =========================
TOOLS = [wieder_verfuegbar, bedingungen, gear_guide, rag, search_web, finde_lieferung]
llm_with_tools = llm.bind_tools(TOOLS)
_TOOL_MAP = {t.name: t for t in TOOLS}

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
    Einmalige Tool-Schleife für einfache Prompts -> Textantwort.
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
            msgs.append(ToolMessage(content=str(res), name=name, tool_call_id=call.get("id")))
    return "Ich konnte die Anfrage nicht abschließen. Bitte schreibe an friends@zenbivy.eu."

# =========================
# ======= CHAT GRAPH ======
# =========================
def _call_model(state: MessagesState) -> Dict[str, Any]:
    msgs = [SystemMessage(content=SYSTEM)] + state["messages"]
    ai = _try_invoke_with_fallback(msgs)
    return {"messages": [ai]}

tool_node = ToolNode(TOOLS)

builder_chat = StateGraph(MessagesState)
builder_chat.add_node("call_model", _call_model)
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

# =================================================
# ====== MAIL AUTO-DRAFT GRAPH (Nodes, keine Tools)
# =================================================
TENANT_ID = os.environ.get("MS_TENANT_ID")
CLIENT_ID = os.environ.get("MS_CLIENT_ID")
CLIENT_SECRET = os.environ.get("MS_CLIENT_SECRET")
SHARED_MAILBOX = os.environ.get("MS_SHARED_MAILBOX")  # z.B. "friends@zenbivy.eu"
GRAPH_BASE = "https://graph.microsoft.com/v1.0"
LOOKBACK_MINUTES = int(os.getenv("LOOKBACK_MINUTES", "5"))

class GraphClient:
    def __init__(self):
        if ConfidentialClientApplication is None:
            raise RuntimeError("msal ist nicht installiert. Bitte 'pip install msal' ausführen.")
        missing = [k for k, v in {
            "MS_TENANT_ID": TENANT_ID, "MS_CLIENT_ID": CLIENT_ID,
            "MS_CLIENT_SECRET": CLIENT_SECRET, "MS_SHARED_MAILBOX": SHARED_MAILBOX
        }.items() if not v]
        if missing:
            raise RuntimeError(f"Fehlende ENV Variablen: {', '.join(missing)}")

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
        params = {"$select": "id,subject,from,sentDateTime,receivedDateTime,body"}
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

        # 1) Reply-Entwurf anlegen
        url_create = f"{GRAPH_BASE}/users/{SHARED_MAILBOX}/messages/{original_id}/createReply"
        r = requests.post(url_create, headers=headers, timeout=20)
        r.raise_for_status()
        draft = r.json()
        draft_id = draft["id"]

        # 2) Body (HTML) setzen
        url_patch = f"{GRAPH_BASE}/users/{SHARED_MAILBOX}/messages/{draft_id}"
        patch = {"body": {"contentType": "HTML", "content": html_body}}
        r2 = requests.patch(url_patch, headers=headers, json=patch, timeout=20)
        r2.raise_for_status()
        return draft_id

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
    return f'{reply_html}<br><br>{header_line}{quoted}'

def sanitize_llm_html(s: str) -> str:
    t = (s or "").strip()
    if t.startswith("```"):
        t = t.strip("`").strip()
        if t.lower().startswith("html"):
            t = t[4:].lstrip()
    return t

def _run_agent_with_tools__email(user_text: str) -> str:
    """
    Nutzt dein llm_with_tools (inkl. Tools) und führt bis zu 3 Tool-Runden aus.
    """
    msgs: List[Any] = [SystemMessage(content=SYSTEM), HumanMessage(content=user_text)]
    for _ in range(3):
        ai: AIMessage = _try_invoke_with_fallback(msgs)
        msgs.append(ai)
        tcs = getattr(ai, "tool_calls", None) or []
        if not tcs:
            return (ai.content or "").strip()
        for call in tcs:
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
    return "<p>Vielen Dank für Ihre Nachricht! Wir melden uns in Kürze.</p><p>Beste Grüße<br>sleepingbagREBEL</p>"

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

        core = client.get_message_core(msg_id)
        body_html = core["body_html"] or "<div>(Kein Inhalt erkannt)</div>"
        from_name = core["from"]
        sent_iso = core["sentDateTime"]
        subject = core["subject"]

        user_text = (
            "Erstelle eine höfliche, hilfreiche und konkrete Antwort als HTML und unterschreibe mit 'sleepingbagREBEL'. "
            "Antworte ausschließlich basierend auf folgendem E-Mail-Body. "
            "Gib NUR den Email-Body der Antwort zurück (keinen Betreff, keine Meta-Zeilen, kein Codeblock). "
            "Antworte in der Sprache des folgenden Inhalts.\n\n"
            "EMAIL_BODY_HTML_START\n"
            f"{body_html}\n"
            "EMAIL_BODY_HTML_END"
        )

        reply_html_raw = _run_agent_with_tools__email(user_text)
        reply_html = sanitize_llm_html(reply_html_raw)
        if not reply_html:
            reply_html = (
                "<p>Vielen Dank für Ihre Nachricht! "
                "Wir prüfen Ihr Anliegen und melden uns in Kürze.</p>"
                "<p>Beste Grüße<br>sleepingbagREBEL</p>"
            )

        combined_html = build_reply_with_history(
            reply_html=reply_html,
            original_html=body_html,
            from_name=from_name,
            sent_iso=sent_iso,
            subject=subject,
        )

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
# ===== Default-Export (run) ======
# =================================
# Für den Cron-Job:
graph = graph_autodraft
# Dein Chat bleibt parallel verfügbar:
# graph_chat (bereits oben kompiliert)

if __name__ == "__main__":
    # Demo: AutoDraft ausführen
    out = graph.invoke({})
    print(out.get("messages", [AIMessage(content="")])[-1].content)
    # Demo: Chat-Aufruf
    demo = graph_chat.invoke({"messages": [HumanMessage(content="Hi! Welche Größe brauche ich bei 180 cm?")]})
    print("CHAT:", demo["messages"][-1].content[:500])
