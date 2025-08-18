# Zenbivy Agent (Outlook Auto-Draft)

## Lokal starten
```bash
python -m venv .venv && . .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # .env mit deinen Keys füllen
langgraph dev
