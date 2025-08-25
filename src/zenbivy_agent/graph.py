from __future__ import annotations

import uuid, os, re, json
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

# --- MS Graph (für finde_lieferung) ---
try:
    from msal import ConfidentialClientApplication
except Exception:
    ConfidentialClientApplication = None

# =========================
# ====== LLM & PROMPT =====
# =========================
MODEL = "claude-3-7-sonnet-20250219"  # dein Wunschmodell

SYSTEM = (
    "Du bist sleepingbagREBEL, ein präziser, netter Mitarbeiter von Zenbivy. "
    "Zeige nicht, dass du eine KI bist. "
    "Antworte stets in der Sprache der Anfrage und in kompaktem Text. "
    "Verwende smileys"
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
    Parameter:
      - query: Suchbegriff ODER direkte URL (https://...).
      - max_results: 1–10 (Standard 5).
      - restrict_to_zenbivy: True => nur zenbivy.com/zenbivy.eu durchsuchen.
    Rückgabe:
      {
        "query": str,
        "restricted": bool,
        "results": [
          { "title": str, "url": str, "snippet": str, "score": float|None,
            "page": {"url": str, "title": str, "text": str, "images": [{"src","alt"}] } | {"url": str, "error": str}
          },
          ...
        ]
      }
    Hinweise:
      - Bilder sind wichtig für Größen/Anleitungen/Produktdetails; bis zu 12 werden geliefert.
      - Nutze dieses Tool proaktiv, wenn externe Fakten/Seiteninhalte gebraucht werden.
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
    """
    [Aktiv nutzen] Lädt vordefinierte Zenbivy-Seiten (Größen, Anleitung, Füllgewicht, Zubehör, Kontakt …)
    und gibt strukturierte Infos inkl. Bildhinweisen zurück.
    Input: einer der festen Schlüssel (z. B. 'Größentabelle', 'Gebrauchsanweisung').
    Output: { source, url, title, text, images:[{src,alt}] }
    Hinweis: Bilder/Alt-Texte können relevante Maßtabellen oder Diagramme andeuten – inhaltlich erwähnen, nicht einbetten.
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
    """
    [Aktiv nutzen] Shop-Bedingungen kompakt:
      - 'Rabattcode'  → wie erhalten?
      - 'Rückgabe- & Umtauschbedingungen' → Fristen/Voraussetzungen
      - 'Versandbedingungen' → Länder/Laufzeiten
    Nutze dieses Tool proaktiv bei allen Politik-/Bedingungsfragen.
    """
    POLICIES = {
        "Rückgabe- & Umtauschbedingungen": (
            """             
                Rückgabe- und Umtauschanweisungen
                
                Allgemeine Bedingungen:
                - Rückgabe innerhalb von 14 Tagen nach Lieferung möglich
                - Vollständige Rückerstattung des Kaufpreises inkl. ursprünglicher Versandkosten
                - Größe und Komfort können zu Hause getestet werden
                
                Voraussetzungen für Rückgabe:
                - Artikel muss unbenutzt sein
                - In Originalverpackung zurücksenden (Ausnahme: Matten)
                - Im gleichen Zustand wie bei Erhalt (inkl. aller Etiketten und Labels)
                - Ausgefülltes Rücksendeformular beilegen
                - Artikel dürfen nicht schmutzig oder mit Tierhaaren bedeckt sein
                
                Strafabzüge bei nicht ordnungsgemäßer Rückgabe:
                - Fehlendes Etikett: 10€ Abzug
                - Fehlendes eingenähtes Label (Law Tag): 50% Abzug
                
                Umtauschprozess:
                1. Neue Bestellung für gewünschten Artikel aufgeben
                2. Ursprünglichen Artikel zur Rückerstattung zurücksenden
                Hinweis: Beide Bestellungen werden temporär belastet
                
                Bearbeitungszeiten:
                - Rücksendebearbeitung: 3-6 Werktage (max. 2 Wochen)
                - Rückerstattung: automatisch innerhalb 10 Werktagen auf ursprüngliche Zahlungsmethode
                
                Rücksendekosten und Verantwortung:
                - Kunde trägt Rücksendekosten
                - Empfehlung: Versand mit Sendungsverfolgung verwenden
                - Zenbivy haftet nicht für verlorene oder beschädigte Pakete
                - Keine Rücksendenummer ausstellbar
                
                Kontakt und Adresse:
                - Kundenservice: friends@zenbivy.eu
                - Rücksendeadresse: Koch alpin GmbH
                                   Dr-Franz-Werner-Str.13
                                   A-6020 Innsbruck
                                   Tyrol, Austria, EU
                """
        ),
        "Versandbedingungen": (
            """
            Versandbedingungen
            
            Lieferungen nach: Österreich, Belgien, Tschechien, Dänemark, Finnland, Frankreich, 
            Deutschland, Irland, Italien, Niederlande, Polen, Portugal, Spanien, Schweden, 
            Schweiz, Bulgarien, Kroatien, Zypern, Estland, Griechenland, Ungarn, Lettland, 
            Litauen, Luxemburg, Malta, Rumänien, Slowakei, Slowenien - ausgenommen Überseegebiete.
            
            EU-Versand:
            - Lieferung mit DPD
            - Lieferzeit: 2 Tage (Österreich, Deutschland), bis zu 1 Woche (andere EU-Länder)
            - Versandkosten: EUR 20,00 für Bestellungen bis EUR 300,00
            - Kostenloser Versand für Bestellungen über EUR 300,00
            - Ausnahme Zypern/Malta: Lieferung mit TNT für pauschal EUR 80,00
            
            Kleinbestellungen (bis EUR 150,00):
            - Nach Deutschland und Österreich: nur EUR 6,00 Versandkosten
            
            Nicht-EU-Länder:
            - Lieferung durch Post.at
            - Lieferzeit: bis zu 1 Woche
            - Preise ohne 20% Umsatzsteuer ausgewiesen
            - Einfuhrumsatzsteuer und Zollgebühren bei Zustellung zu bezahlen
            - Versandkosten:
              * Schweiz: EUR 26,00
              * UK, Island, Norwegen und andere Staaten: EUR 60,00
            """
        ),
        "Rabattcode": "Man kann einen Rabattcode im Newsletter finden",
    }
    return POLICIES[kategorie]

@tool("wieder_verfuegbar")
def wieder_verfuegbar(datei: DateiAuswahl) -> str:
    """
    Findet heraus wann ein Produkt wieder erhältlich ist, wann es wieder auf Lager ist.
    """
    return "in 2 Monaten"
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
    """
    Extrahiert Tracking-Links (post.at) und ggf. Versandeinheits-IDs (lange Ziffernfolgen).
    Gibt (links, ids) zurück.
    """
    content = html_or_text or ""
    # HTML -> Text grob: Entities & <br> -> newline
    try:
        soup = BeautifulSoup(content, "html.parser")
        # Links direkt aus <a href>
        hrefs = [a.get("href") for a in soup.find_all("a") if a.get("href")]
        text = soup.get_text("\n", strip=True)
        candidates = hrefs + re.findall(r"https?://[^\s<>\"]+", text)
    except Exception:
        text = re.sub(r"<[^>]+>", " ", content)
        candidates = re.findall(r"https?://[^\s<>\"]+", text)

    # Nur post.at (robust)
    post_links = [u for u in candidates if re.search(r"(^|://)(www\.)?post\.at/", u)]
    # Duplikate entfernen, Reihenfolge halten
    seen = set(); uniq_links = []
    for u in post_links:
        if u not in seen:
            seen.add(u); uniq_links.append(u)

    # IDs: aus URL param pnum1=... ODER aus Text (lange Ziffern)
    ids = []
    for u in uniq_links:
        m = re.search(r"(?:\?|&|/)(?:pnum1|barcodelist|barcode|pnum)=([0-9]{10,})", u)
        if m:
            ids.append(m.group(1))
    # zusätzlich aus Text
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
        """
        Holt die neuesten 'top' Nachrichten aus Gesendete Elemente.
        """
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
        """
        Holt Body als Text (per Prefer-Header), plus ein paar Metadaten.
        """
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
    Durchsucht das **Shared Mailbox** 'Koch Alpin GmbH - Service' → Ordner **Gesendete Elemente** (SentItems),
    max. die **400 neuesten** Mails, nach einer Mail, die an die gegebene **E-Mail-Adresse** gesendet wurde
    (To/Cc/Bcc). Findet die Mail und extrahiert den **Sendungs-Status-Link** (post.at).

    Input: email (str)
    Output:
      {
        "email": "...",
        "matched": true|false,
        "message": {
           "id": "...", "subject": "...",
           "sentDateTime": "...", "receivedDateTime": "..."
        } | null,
        "status_link": "http://www.post.at/tnt_query.php?pnum1=...",  # wenn gefunden
        "versand_ids": ["..."],  # falls extrahierbar
        "checked_count": 123,
        "note": "..."
      }

    Erfordert ENV: MS_TENANT_ID, MS_CLIENT_ID, MS_CLIENT_SECRET, MS_SHARED_MAILBOX_KOCH
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
    [Aktiv nutzen] Durchsuche den Outlook-RAG-Index (Chroma) und liefere kompakten Kontext + Quellen.
    - Verwende dies proaktiv für E-Mail-Antworten, um bestehende Wissens-Snippets zu holen.
    - Nach Nutzung: Inhalte verallgemeinern; keine Personen-/Datumsdetails übernehmen.
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

# =========================
# ======== LLM BIND =======
# =========================
llm = ChatAnthropic(
    model=MODEL,
    temperature=0,
    max_tokens=30000,
    timeout=1000,                       # großzügiger Timeout
    max_retries=50,  
)

TOOLS = [wieder_verfuegbar, bedingungen, gear_guide, rag, search_web, finde_lieferung]
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
    return True  # konservativ

def _normalized_msgs_for_anthropic(msgs, system_text: str):
    """Sichert: System vorn, keine leeren Messages, erster Nicht-System ist Human."""
    # System vorne
    if not msgs or (hasattr(msgs[0], "type") and msgs[0].type != "system"):
        msgs = [SystemMessage(content=system_text)] + msgs

    # Leere Human/AI/System entfernen
    cleaned = []
    for m in msgs:
        if getattr(m, "type", None) in ("human", "ai", "system"):
            if not _has_nonempty_content(m):
                continue
        cleaned.append(m)

    # Nach System muss ein Human kommen; sonst kein LLM-Call
    if cleaned and cleaned[0].type == "system":
        tail = cleaned[1:]
    else:
        tail = cleaned

    if not tail:
        return cleaned  # nur System → no-op später

    if tail[0].type != "human":
        # Historie beginnt nicht mit Human (z. B. Tool/AI) → lieber no-op
        return cleaned

    return cleaned

def agent_node(state: State, config: RunnableConfig):
    """
    Ruft das Modell mit der bisherigen Message-Historie auf
    und gibt die neue AIMessage in den State zurück.
    Verhindert 400er bei Background-Runs (leere/fehlende User-Message).
    """
    msgs = state["messages"]
    msgs = _normalized_msgs_for_anthropic(msgs, SYSTEM)

    only_system = len(msgs) == 1 and msgs[0].type == "system"
    first_is_human = (len(msgs) > 1 and msgs[0].type == "system" and msgs[1].type == "human") or (len(msgs) > 0 and msgs[0].type == "human")

    if only_system or not first_is_human:
        # Kein valider Human-Turn → kein Model-Call (fixes Anthropic 400)
        return {"messages": []}

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

# Alias für Cloud-Configs, die 'graph_chat' erwarten
graph_chat = graph

if __name__ == "__main__":
    thread = {"configurable": {"thread_id": str(uuid.uuid4())}}

    # Beispiel: Tracking-Link suchen
    q0 = {"role":"user","content":"finde_lieferung für max.mustermann@example.com"}
    out0 = graph.invoke({"messages":[q0]}, config=thread)
    print("ASSISTANT (finde_lieferung):", out0["messages"][-1].content[:800] if out0["messages"] else "<no reply>")

    # Beispiel: RAG
    q1 = {"role":"user","content":"Bitte nutze 'rag' und beantworte: Wie reklamiere ich defektes Zubehör?"}
    out1 = graph.invoke({"messages": [q1]}, config=thread)
    print("ASSISTANT (RAG):", out1["messages"][-1].content[:800] if out1["messages"] else "<no reply>")

    # Beispiel: Bedingungen
    q2 = {"role":"user","content":"Nutze 'bedingungen' und sag mir, wie der Versand läuft"}
    out2 = graph.invoke({"messages":[q2]}, config=thread)
    print("ASSISTANT (Bedingungen):", out2["messages"][-1].content[:800] if out2["messages"] else "<no reply>")












