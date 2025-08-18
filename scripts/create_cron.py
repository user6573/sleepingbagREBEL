import os
import asyncio
from langgraph_sdk import get_client

"""
Nutzung:
  setze LANGGRAPH_URL und LANGSMITH_API_KEY als Env-Variablen (in der Cloud nicht nötig,
  wenn du die Studio-Konsole verwendest).
  Dann:
    python scripts/create_cron.py
"""

ASSISTANT_ID = os.getenv("ASSISTANT_ID", "zenbivy")  # muss zu langgraph.json passen
CRON = os.getenv("CRON_EXPR", "*/10 * * * *")        # alle 10 Minuten
TIMEZONE = os.getenv("CRON_TZ", "Europe/Vienna")

URL = os.environ["LANGGRAPH_URL"]        # z.B. "https://<your-deployment>.langgraph.run"
LS_API_KEY = os.environ["LANGSMITH_API_KEY"]

async def main():
    client = get_client(url=URL, api_key=LS_API_KEY)

    # Thread anlegen (persistenter Kontext, z.B. last_seen_iso)
    thread = await client.threads.create()
    thread_id = thread["thread_id"]

    # Cron für diesen Thread einrichten
    cron = await client.crons.create_for_thread(
        thread_id=thread_id,
        assistant_id=ASSISTANT_ID,
        schedule=CRON,
        timezone=TIMEZONE,
        input={"messages": [{"role": "user", "content": "Checke neue E-Mails und lege Antwortentwürfe an."}]},
    )
    print("Cron angelegt:", cron)

if __name__ == "__main__":
    asyncio.run(main())
