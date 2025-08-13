# Zenbivy Agent (LangGraph, LangChain, Claude)

Minimal lauffähiges Repo für deinen Agenten mit Tools (`gear_guide`, `wieder_verfuegbar`, `bedingungen`, `rag`, `search_web`)
und LangGraph-Deploy.

## 1) Setup (lokal)

```bash
python -m venv .venv
# Windows PowerShell:
. .venv/Scripts/Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
copy .env.example .env   # oder manuell .env anlegen
# .env mit Keys befüllen: ANTHROPIC_API_KEY, optional TAVILY_API_KEY, OPENAI_API_KEY etc.
```

## 2) Dev-Server starten

```bash
langgraph dev
```

Studio-Link erscheint im Terminal. Alternativ Test per SDK:

```python
from langgraph_sdk import get_sync_client
client = get_sync_client(url="http://localhost:2024")
out = client.runs.invoke(None, "agent", input={"messages":[{"role":"human","content":"Sag Hallo"}]})
print(out["messages"][-1]["content"])
```

## 3) Cloud-Deploy (LangGraph Platform / Pro)

1. Repo zu GitHub pushen.
2. In LangSmith → Deployments → **New Deployment** → Repo wählen.
3. In der Deployment-Ansicht **ENV** setzen:
   - `ANTHROPIC_API_KEY` (Pflicht)
   - `TAVILY_API_KEY` (für `search_web`)
   - Optional: `OPENAI_API_KEY`, `CHROMA_PATH` etc.
4. Deploy → API-URL & Studio-Link nutzen.

### Remote aufrufen (SDK)

```python
from langgraph_sdk import get_sync_client
client = get_sync_client(url="https://<deployment-url>", api_key="<LANGSMITH_API_KEY>")
for ev in client.runs.stream(None, "agent", input={"messages":[{"role":"human","content":"Sag Hallo"}]}, stream_mode="messages-tuple"):
    pass
```

## 4) Hinweise
- `rag` nutzt standardmäßig lokale Sentence-Transformers. Für Cloud ohne Model-Download `RAG_EMBEDDING=openai` setzen.
- `search_web` benötigt Tavily-API-Key.
- `wieder_verfuegbar` erwartet die .txt-Dateien exakt wie benannt im Windows-Pfad (lokal). Für Cloud-Use ggf. Dateien als Assets einbinden.

Viel Erfolg! 🎒
