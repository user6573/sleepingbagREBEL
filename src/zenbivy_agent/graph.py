from __future__ import annotations

import uuid, os, re, json, datetime as dt
from typing import Annotated, Literal, List, Dict, Any, Optional, Tuple
from typing_extensions import TypedDict

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver

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

# --- MS Graph (für Outlook) ---
try:
    from msal import ConfidentialClientApplication
except Exception:
    ConfidentialClientApplication = None

# =========================
# ====== LLM & PROMPT =====
# =========================
MODEL = "claude-opus-4-1-20250805"  # ggf. anpassen (siehe Hinweis in der Chat-Nachricht)

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

# =========================
# ====== PFAD/DATEIEN =====
# =========================
_BASE_DIR = r"C:\Users\Fritz\Desktop\Python\sleepingbagREBEL_Infos\geschliffen"

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

_USER_AGENT = "Mozilla/5.0 (compatible; ZenbivyAgent/1.0)"
_HEADERS = {"User-Agent": _USER_AGENT}

# =========================
# === TEXT + BILD PARSER ==
# =========================
def _extract_text_and_images(html: str, base_url: str):
    """
    Kompaktes Extrahieren von Text + bis zu 12 Bildern.
    Berücksichtigt og:image & figcaption; dedupliziert Bilder.
    Rückgabe: images = [{ "src", "alt" }, ...]
    """
    soup = BeautifulSoup(html, "html.parser")

    # irrelevante Tags entfernen
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    main = soup.find("main") or soup.find("article") or soup.body or soup

    # sichtbaren Text
    parts = []
    for el in main.find_all(["h1","h2","h3","h4","h5","h6","p","li"], recursive=True):
        t = " ".join(el.get_text(" ", strip=True).split())
        if t:
            parts.append(t)
    text = "\n".join(parts)
    if len(text) > 8000:
        text = text[:8000] + " … [gekürzt]"

    # Bilder sammeln
    images: List[Dict[str, str]] = []

    # 1) figure + figcaption
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

    # 2) og:image / twitter:image (ergänzend)
    for m in soup.find_all("meta"):
        prop = (m.get("property") or m.get("name") or "").lower()
        if prop in {"og:image", "twitter:image", "image"}:
            content = (m.get("content") or "").strip()
            if content:
                src = urljoin(base_url, content)
                if not src.lower().endswith(".svg"):
                    images.append({"src": src, "alt": ""})

    # 3) generische <img>
    for img in main.find_all("img"):
        src = img.get("src") or img.get("data-src") or img.get("data-original")
        if not src:
            continue
        src = urljoin(base_url, src.strip())
        if src.lower().endswith(".svg"):
            continue
        alt = (img.get("alt") or img.get("title") or "").strip()
        images.append({"src": src, "alt": alt})

    # Deduplizieren nach URL, triviale Assets ausfiltern
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
    [Aktiv nutzen] Websuche via Tavily ODER Direkt-URL:
      - Bei DIREKTER URL: lädt die Seite sofort und gibt Text + Bilder (Alt/Captions) zurück.
      - Bei SUCHE: findet relevante Seiten (optional auf zenbivy.com/.eu beschränkt) und lädt jede Seite.
    """
    # Direkter URL-Fetch ohne Suche
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

    # Tavily verfügbar?
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
    POLICIES = {
        "Rückgabe- & Umtauschbedingungen": (
            """             
                Rückgabe- und Umtauschanweisungen
                ...
                Kundenservice: friends@zenbivy.eu
                Rücksendeadresse: Koch alpin GmbH
                                   Dr-Franz-Werner-Str.13
                                   A-6020 Innsbruck
                                   Tyrol, Austria, EU
                """
        ),
        "Versandbedingungen": (
            """
            Versandbedingungen
            ...
            """
        ),
        "Rabattcode": "Man kann einen Rabattcode im Newsletter finden",
    }
    return POLICIES[kategorie]

@tool("wieder_verfuegbar")
def wieder_verfuegbar(datei: DateiAuswahl) -> str:
    filename = f"{datei}.txt"
    path = os.path.join(_BASE_DIR, filename)
    if not os.path.isfile(path):
        return f"[FEHLER] Datei nicht gefunden: {filename}"
    for enc in ("utf-8-sig","utf-8","cp1252","latin-1"):
        try:
            with open(path, "r", encoding=enc, errors="strict") as f:
                return f.read()
        except Exception:
            continue
    with open(path, "rb") as f:
        return f.read().decode("utf-8", errors="ignore")

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
        self.mailbox = os.getenv("MS_SHARED_MAILBOX_KOCH")  # << Shared Mailbox UPN/SMTP
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
            "status_link": None, "versand_ids": [], "checked_count": len(msgs),
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
        max_b = max(bm_scores) if bm_scores else 1.0
        bm_norm = [(s/max_b) if max_b else 0.0 for s in bm_scores]
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
llm = ChatAnthropic(
    model=MODEL,
    temperature=0,
    max_tokens=50000,  # ggf. reduzieren (siehe Hinweis)
)

TOOLS = [wieder_verfuegbar, bedingungen, gear_guide, rag, search_web, finde_lieferung]
_TOOL_MAP = {t.name: t for t in TOOLS}
llm_with_tools = llm.bind_tools(TOOLS)

class State(MessagesState):
    pass

# =========================
# === Anthropic-Guards ====
# =========================
def _has_nonempty_content(msg) -> bool:
    c = getattr(msg, "content", None)
    if c is None:
        return False
    if isinstance(c, str):
        return c.strip() != ""
    if isinstance(c, list):
        return len(c) > 0
    return True

def _normalized_msgs_for_anthropic(msgs, system_text: str):
    if not msgs or (hasattr(msgs[0], "type") and msgs[0].type != "system"):
        msgs = [SystemMessage(content=system_text)] + msgs

    cleaned = []
    for m in msgs:
        if getattr(m, "type", None) in ("human", "ai", "system"):
            if not _has_nonempty_content(m):
                continue
        cleaned.append(m)

    if cleaned and cleaned[0].type == "system":
        tail = cleaned[1:]
    else:
        tail = cleaned

    if not tail:
        return cleaned

    if tail[0].type != "human":
        return cleaned

    return cleaned

def agent_node(state: State, config: RunnableConfig):
    msgs = state["messages"]
    msgs = _normalized_msgs_for_anthropic(msgs, SYSTEM)

    only_system = len(msgs) == 1 and msgs[0].type == "system"
    first_is_human = (len(msgs) > 1 and msgs[0].type == "system" and msgs[1].type == "human") or (len(msgs) > 0 and msgs[0].type == "human")

    if only_system or not first_is_human:
        return {"messages": []}

    ai = llm_with_tools.invoke(msgs, config=config)
    return {"messages": [ai]}

tool_node = ToolNode(TOOLS)

builder_chat = StateGraph(State)
builder_chat.add_node("agent", agent_node)
builder_chat.add_node("tools", tool_node)

builder_chat.add_edge(START, "agent")
builder_chat.add_conditional_edges("agent", tools_condition)
builder_chat.add_edge("tools", "agent")

checkpointer = InMemorySaver()
graph_chat = builder_chat.compile(checkpointer=checkpointer)

# =====================================================================
# ============ OUTLOOK AUTODRAFT (Entwürfe im Thread speichern) =======
# =====================================================================

GRAPH_BASE = "https://graph.microsoft.com/v1.0"
TENANT_ID = os.getenv("MS_TENANT_ID")
CLIENT_ID = os.getenv("MS_CLIENT_ID")
CLIENT_SECRET = os.getenv("MS_CLIENT_SECRET")
SHARED_MAILBOX = os.getenv("MS_SHARED_MAILBOX")
LOOKBACK_MINUTES = int(os.getenv("LOOKBACK_MINUTES", "5"))

def _ensure(val: Optional[str], name: str):
    if not val:
        raise RuntimeError(f"ENV {name} fehlt.")

class GraphClient:
    def __init__(self):
        if ConfidentialClientApplication is None:
            raise RuntimeError("msal ist nicht installiert. Bitte 'pip install msal'")
        _ensure(TENANT_ID, "MS_TENANT_ID")
        _ensure(CLIENT_ID, "MS_CLIENT_ID")
        _ensure(CLIENT_SECRET, "MS_CLIENT_SECRET")
        _ensure(SHARED_MAILBOX, "MS_SHARED_MAILBOX")
        self.app = ConfidentialClientApplication(
            CLIENT_ID,
            authority=f"https://login.microsoftonline.com/{TENANT_ID}",
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
        params = {
            "$top": str(max_count),
            "$select": "id,receivedDateTime,subject,from",
            "$orderby": "receivedDateTime desc",
            "$filter": f"receivedDateTime ge {since_iso}",
        }
        url = f"{GRAPH_BASE}/users/{SHARED_MAILBOX}/mailFolders/Inbox/messages"
        r = requests.get(url, headers=self._auth_headers(), params=params, timeout=20)
        r.raise_for_status()
        return r.json().get("value", [])

    def get_message_core(self, msg_id: str) -> Dict[str, Any]:
        url = f"{GRAPH_BASE}/users/{SHARED_MAILBOX}/messages/{msg_id}"
        params = {"$select": "id,subject,from,sentDateTime,receivedDateTime,body"}
        r = requests.get(url, headers=self._auth_headers(prefer_html=True), params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        body = (data.get("body") or {}) or {}
        email = (data.get("from") or {}).get("emailAddress") or {}
        return {
            "id": data.get("id"),
            "subject": data.get("subject") or "",
            "from": email.get("name") or "",
            "from_addr": email.get("address") or "",
            "sentDateTime": data.get("sentDateTime"),
            "receivedDateTime": data.get("receivedDateTime"),
            "body_html": body.get("content", "") or "",
        }

    def create_reply_draft(self, original_id: str, html_body: str) -> str:
        headers = self._auth_headers()
        headers["Content-Type"] = "application/json"
        url_create = f"{GRAPH_BASE}/users/{SHARED_MAILBOX}/messages/{original_id}/createReply"
        r = requests.post(url_create, headers=headers, timeout=20)
        r.raise_for_status()
        draft = r.json()
        draft_id = draft["id"]

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

def build_reply_with_history(reply_html: str, original_html: str, from_name: str = "", sent_iso: str = "", subject: str = "") -> str:
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
    return f'{reply_html}<br><br>{header_line}{quoted}'

def sanitize_llm_html(s: str) -> str:
    t = (s or "").strip()
    # Codefences entfernen
    if t.startswith("```"):
        t = t.strip("`").strip()
        # evtl. führendes "html" entfernen
        if t.lower().startswith("html"):
            t = t[4:].lstrip()
    return t

def _invoke_llm_with_tools(msgs: List[Any]) -> AIMessage:
    # Einfache direkte Nutzung (du kannst hier bei Bedarf Retry/Fallback ergänzen)
    return llm_with_tools.invoke(msgs, config=RunnableConfig())

def run_agent_with_tools(user_text: str) -> str:
    msgs: List[Any] = [SystemMessage(content=SYSTEM), HumanMessage(content=user_text)]
    # bis zu 3 Tool-Runden
    for _ in range(3):
        ai: AIMessage = _invoke_llm_with_tools(msgs)
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
            msgs.append(ToolMessage(content=json.dumps(res) if isinstance(res, (dict, list)) else str(res),
                                    name=name, tool_call_id=call.get("id")))
    return "Ich konnte die Anfrage nicht abschließen. Bitte schreibe an friends@zenbivy.eu."

# ---- Autodraft-State & Nodes ----
class AutoDraftState(TypedDict, total=False):
    messages: List[Any]
    lookback_iso: str
    new_emails: List[Dict[str, Any]]
    drafted_count: int
    drafted_ids: List[str]

def node_fetch_recent_emails(state: AutoDraftState) -> AutoDraftState:
    client = GraphClient()
    since_iso = utc_iso_now_minus_minutes(LOOKBACK_MINUTES)
    msgs = client.list_messages_since(since_iso=since_iso, max_count=50)
    return {"lookback_iso": since_iso, "new_emails": msgs}

def node_generate_drafts_body_only(state: AutoDraftState) -> AutoDraftState:
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
        reply_html_raw = run_agent_with_tools(user_text)
        reply_html = sanitize_llm_html(reply_html_raw) or (
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

def node_summarize(state: AutoDraftState) -> AutoDraftState:
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
builder_autodraft = StateGraph(AutoDraftState)
builder_autodraft.add_node("fetch_recent_emails", node_fetch_recent_emails)
builder_autodraft.add_node("generate_drafts_body_only", node_generate_drafts_body_only)
builder_autodraft.add_node("summarize", node_summarize)

builder_autodraft.add_edge(START, "fetch_recent_emails")
builder_autodraft.add_edge("fetch_recent_emails", "generate_drafts_body_only")
builder_autodraft.add_edge("generate_drafts_body_only", "summarize")
builder_autodraft.add_edge("summarize", END)

graph_autodraft = builder_autodraft.compile()

# =========================
# ===== Export Auswahl ====
# =========================
_DEFAULT = (os.getenv("DEFAULT_GRAPH") or "chat").lower().strip()
graph = graph_autodraft if _DEFAULT == "autodraft" else graph_chat

if __name__ == "__main__":
    # Beispiel: Chat
    thread = {"configurable": {"thread_id": str(uuid.uuid4())}}
    q = {"role":"user","content":"Nutze 'bedingungen' und sag mir, wie der Versand läuft"}
    out = graph_chat.invoke({"messages":[q]}, config=thread)
    print("ASSISTANT (Chat):", out["messages"][-1].content[:500] if out.get("messages") else "<no reply>")

    # Beispiel: Autodraft (ruft Inbox der letzten LOOKBACK_MINUTES Minuten ab und erstellt Antwort-Entwürfe)
    out2 = graph_autodraft.invoke({})
    print("AUTODRAFT:", out2.get("messages",[AIMessage(content='<no summary>')])[-1].content)
