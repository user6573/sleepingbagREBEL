from __future__ import annotations

import uuid
from typing import Annotated, Literal
from typing_extensions import TypedDict

import os
# .env nur lokal laden; in der Cloud sind ENV-Variablen gesetzt
try:
    from dotenv import load_dotenv  # optional
    load_dotenv()
except Exception:
    pass


from langchain_core.messages import AnyMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver

# LLM (Claude)
from langchain_anthropic import ChatAnthropic

# --- NEU: RAG-Dependencies ---
import re
from typing import List, Dict, Any, Optional, Tuple
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

MODEL = "claude-opus-4-1-20250805"   # dein Wunschmodell
SYSTEM = (
    "Du bist sleepingbagREBEL, ein präziser, netter Mitarbeiter von Zenbivy. "
    "Zeige nicht, dass du eine KI bist. "
    "Antworte stets in der Sprache der Anfrage und in kompaktem Text. "
    "Weise dezent auf den Newsletter hin (https://zenbivy.eu – rechts unten). "
    "Für Bestelländerungen, Reklamationen oder Fälle für Menschen: bitte um E-Mail an friends@zenbivy.eu. "
    "Tool-Nutzung: "
    "• Für Größen, Anleitung, Füllgewicht, Zubehör: nutze 'gear_guide'. "
    "• Für Versand/Rückgabe/Rabatt: nutze 'bedingungen'. "
    "• Für Verfügbarkeiten: nutze 'wieder_verfuegbar'. "
    "• Für E-Mail-Antworten auf Kundennachrichten: nutze 'rag', wenn verfügbar, um Kontext zu holen."
)




# nach dieser Zeile ...
_BASE_DIR = r"C:\Users\Fritz\Desktop\Python\sleepingbagREBEL_Infos\geschliffen"
# ... sofort hinzufügen:
_BASE_DIR = os.getenv("BASE_DIR", _BASE_DIR)  # Cloud: per ENV setzen, lokal: Windows-Pfad

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

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# --- NEU: Tool 'search_web' --------------------------------------------------
import os
import re
import requests
from typing import Optional
from urllib.parse import urljoin
from langchain_core.tools import tool

# Optional: BeautifulSoup für HTML-Parsing
from bs4 import BeautifulSoup

# Tavily SDK (sauberer als direkter REST-Call)
try:
    from tavily import TavilyClient
except Exception:
    TavilyClient = None

_USER_AGENT = "Mozilla/5.0 (compatible; ZenbivyAgent/1.0)"
_HEADERS = {"User-Agent": _USER_AGENT}

def _extract_text_and_images(html: str, base_url: str):
    """Kompaktes Extrahieren von Text + bis zu 12 Bildern (wie im gear_guide)."""
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

    images = []
    for img in main.find_all("img"):
        src = img.get("src") or img.get("data-src") or img.get("data-original")
        if not src:
            continue
        src = urljoin(base_url, src.strip())
        if src.lower().endswith(".svg"):
            continue
        alt = (img.get("alt") or "").strip()
        images.append({"src": src, "alt": alt})
        if len(images) >= 12:
            break

    title = soup.title.get_text(strip=True) if soup.title else ""
    return title, text, images

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

@tool("search_web")
def search_web(
    query: str,
    max_results: int = 5,
    restrict_to_zenbivy: bool = True,
) -> dict:
    """
    Websuche via Tavily, um beliebige (v. a. Zenbivy-)Seiten zu finden und Inhalte zu holen.
    Parameter:
      - query: Suchbegriff ODER direkte URL.
      - max_results: Anzahl Treffer (1–10 sinnvoll).
      - restrict_to_zenbivy: Wenn True, nur Domains 'zenbivy.com' & 'zenbivy.eu'.
    Rückgabe:
      {
        "query": str,
        "restricted": bool,
        "results": [
          {
            "title": str,
            "url": str,
            "snippet": str,
            "score": float|None,
            "page": {"url": str, "title": str, "text": str, "images": [{"src","alt"}]} | {"url": str, "error": str}
          },
          ...
        ]
      }
    Hinweise:
      - Bei direkter URL lädt das Tool die Seite ohne Suche.
      - Bilder können wichtig sein; bis zu 12 pro Seite werden zurückgegeben.
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
            include_domains=include_domains,  # None => Web-weit
            search_depth="basic",  # schnell & ausreichend für gezielte Seiten
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
# ---------------------------------------------------------------------------

@tool("gear_guide")
def gear_guide(name: GuideKey) -> dict:
    """Lädt eine vordefinierte Zenbivy-Seite und gibt strukturierte Infos zurück."""
    url = _SOURCES[name]
    try:
        resp = requests.get(url, timeout=20, headers={"User-Agent":"Mozilla/5.0 (compatible; GearGuideBot/1.0)"})
        resp.raise_for_status()
    except Exception as e:
        return {"source": name, "url": url, "error": f"Fehler beim Laden: {e}"}
    title, text, images = _extract_text_and_images(resp.text, url)
    return {"source": name, "url": url, "title": title, "text": text, "images": images}

@tool("bedingungen")
def bedingungen(kategorie: PolicyKey) -> str:
    """Gibt eure Shop-Bedingungen als Text zurück – je nach Kategorie."""
    POLICIES = {
        "Rückgabe- & Umtauschbedingungen": (
            "Rückgabe- und Umtauschanweisungen\n"
            "Wir möchten, dass Sie Ihr Zenbivy-Schlafsacksystem lieben! ... (gekürzt für Beispiel) ..."
        ),
        "Versandbedingungen": (
            "Versandbedingungen\n"
            "Lieferungen nach: Österreich, Belgien, ... (gekürzt) ..."
        ),
        "Rabattcode": ("Man kann einen Rabattcode im Newsletter finden"),
    }
    return POLICIES[kategorie]

@tool("wieder_verfuegbar")
def wieder_verfuegbar(datei: DateiAuswahl) -> str:
    """
    Öffnet '{datei}.txt' im Ordner und gibt den Dateiinhalt zurück.
    Der Dateiinhalt gibt Informationen zu den Verfügbarkeiten bestimmter Produkte
    In einer Datei stehen mehrere Produkte
    """
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
    Durchsuche den Outlook-RAG-Index (Chroma) und liefere kompakten Kontext + Quellen.
    Die Daten sind nicht verallgemeinert und müssen nach dem nutzen des Tool verallgemeinert werden.
    - Namen in den Daten sollen nicht beachtet werden.
    - Datum und Uhrzeit sollen nicht beachtet werden.
    Parameter:
      - query: Suchfrage
      - top_k: Anzahl der Snippets (Default 5)
    Rückgabe:
      {
        "context": str,           # textfertiger Kontext für LLM
        "sources": [str],         # z.B. ["abc#chunk1/3", ...]
        "items": [                # strukturierte Quelleninfos
           {"ref": str, "subject": str, "first_time": str, "last_time": str,
            "message_count": int, "chunk": int, "chunks_total": int}
        ]
      }
    """

    index_dir = os.getenv("CHROMA_PATH") or "./rag_index"
    client = chromadb.PersistentClient(path=index_dir, settings=Settings(allow_reset=False))
    coll = client.get_or_create_collection("outlook_rag")
    if coll.count() == 0:
        return {"error": f"Leerer Index unter {index_dir}. Bitte Index kopieren/erstellen."}

    # Embedding-Provider
    emb_type = (os.getenv("RAG_EMBEDDING") or "local").lower()
    if emb_type == "openai":
        embedder = _OpenAIEmbedder(model=os.getenv("OPENAI_EMBED_MODEL","text-embedding-3-small"))
    else:
        embedder = _LocalEmbedder()

    # 1) Vektor-Pool
    emb = embedder.embed([query])[0]
    pool_n = max(top_k, int(os.getenv("BM25_POOL", "20")))
    res = coll.query(query_embeddings=[emb], n_results=pool_n, include=["documents","metadatas","distances"])
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    # 2) Hybrid-Rerank (optional)
    hybrid = (os.getenv("HYBRID", "true").lower() != "false") and (BM25Okapi is not None)
    alpha = float(os.getenv("HYBRID_ALPHA", "0.5"))
    if hybrid and docs:
        corpus_tokens = _tokenize(docs)
        bm25 = BM25Okapi(corpus_tokens)
        q_tokens = _tokenize([query])[0]
        bm_scores = bm25.get_scores(q_tokens)
        # Distanz -> Similarität [0..1]
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


llm = ChatAnthropic(
    model=MODEL,
    temperature=0.2,
    max_tokens=20000,
)

TOOLS = [wieder_verfuegbar, bedingungen, gear_guide, rag, search_web]
llm_with_tools = llm.bind_tools(TOOLS)

class State(MessagesState):
    pass

def agent_node(state: State, config: RunnableConfig):
    """
    Ruft das Modell mit der bisherigen Message-Historie auf
    und gibt die neue AIMessage in den State zurück.
    """
    msgs = state["messages"]
    if not msgs or msgs[0].type != "system":
        msgs = [SystemMessage(content=SYSTEM)] + msgs

    ai = llm_with_tools.invoke(msgs, config=config)
    return {"messages": [ai]}

tool_node = ToolNode(TOOLS)

builder = StateGraph(State)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)  # ruft Tools, wenn vom LLM angefordert
builder.add_edge("tools", "agent")

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    thread = {"configurable": {"thread_id": str(uuid.uuid4())}}

    # Beispiel: explizit RAG nutzen
    q1 = {
        "role": "user",
        "content": "Bitte nutze 'rag' und beantworte: Wie reklamiere ich defektes Zubehör?"
    }
    out1 = graph.invoke({"messages": [q1]}, config=thread)
    print("ASSISTANT (RAG):", out1["messages"][-1].content[:1200])

    # Bestehende Tools funktionieren weiter
    q2 = {"role":"user", "content":"Nutze 'rag' und sag mir wann meine bestellung ankommt"}
    out2 = graph.invoke({"messages":[q2]}, config=thread)
    print("ASSISTANT (Bedingungen):", out2["messages"][-1].content[:800])
