import os, time, base64
import msal

TENANT_ID = os.environ.get("MS_TENANT_ID")
CLIENT_ID = os.environ.get("MS_PUBLIC_CLIENT_ID")
AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID}"
SCOPES = [
    "Mail.Read",
    "Mail.ReadWrite",
    "Mail.Read.Shared",
    "Mail.ReadWrite.Shared",
    "offline_access",
    "email",
    "openid",
    "profile",
]

def main():
    if not TENANT_ID or not CLIENT_ID:
        raise SystemExit("Bitte MS_TENANT_ID und MS_PUBLIC_CLIENT_ID in der Umgebung setzen.")

    cache = msal.SerializableTokenCache()
    app = msal.PublicClientApplication(client_id=CLIENT_ID, authority=AUTHORITY, token_cache=cache)

    flow = app.initiate_device_flow(scopes=SCOPES)
    if "user_code" not in flow:
        raise SystemExit(f"Device Flow Fehler: {flow}")

    print("\n=== DEVICE LOGIN ===")
    print("Öffne:", flow["verification_uri"])
    print("Gib diesen Code ein:", flow["user_code"])
    print("(Warte auf Bestätigung …)\n")

    result = app.acquire_token_by_device_flow(flow)
    if "access_token" not in result:
        raise SystemExit(f"Login fehlgeschlagen: {result}")

    b64 = base64.b64encode(cache.serialize().encode("utf-8")).decode("ascii")
    print("\n=== MSAL_CACHE_B64 (kopieren) ===\n")
    print(b64)
    print("\nHinweis: Diesen Wert gleich mit scripts/seed_cache.py in den Thread-State schreiben.")

if __name__ == "__main__":
    main()
