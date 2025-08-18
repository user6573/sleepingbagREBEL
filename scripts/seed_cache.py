import os, asyncio
from langgraph_sdk import get_client

LANGGRAPH_URL = os.environ.get("LANGGRAPH_URL")
LANGSMITH_API_KEY = os.environ.get("LANGSMITH_API_KEY")
ASSISTANT_ID = os.environ.get("ASSISTANT_ID", "zenbivy")
MSAL_CACHE_B64 = os.environ.get("MSAL_CACHE_B64")

async def main():
    if not (LANGGRAPH_URL and LANGSMITH_API_KEY and MSAL_CACHE_B64):
        raise SystemExit("Bitte LANGGRAPH_URL, LANGSMITH_API_KEY, MSAL_CACHE_B64 setzen.")

    client = get_client(url=LANGGRAPH_URL, api_key=LANGSMITH_API_KEY)

    # 1) Thread anlegen
    thread = await client.threads.create()
    thread_id = thread["thread_id"]
    print("Thread:", thread_id)

    # 2) Initial-Run, der den Cache in den State schreibt
    run = await client.runs.create(
        assistant_id=ASSISTANT_ID,
        thread_id=thread_id,
        input={
            "msal_cache_b64": MSAL_CACHE_B64,
            "messages": [{"role": "user", "content": "Init OAuth Cache"}],
        },
    )
    print("Run:", run["run_id"])

    print("\nJetzt Cron für diesen Thread anlegen (siehe README).")

if __name__ == "__main__":
    asyncio.run(main())
