"""
LinkedIn OAuth 2.0 authentication flow.
Opens a browser window, starts a local callback server, exchanges the code
for an access token, and persists the token in the .env file.

Fallback: if the automatic callback is not received within the timeout,
the user is prompted to paste the redirect URL from the browser manually.
"""

import socket
import threading
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import requests
from dotenv import set_key

from config.settings import get_settings, Settings

LINKEDIN_AUTH_URL = "https://www.linkedin.com/oauth/v2/authorization"
LINKEDIN_TOKEN_URL = "https://www.linkedin.com/oauth/v2/accessToken"

# LinkedIn now uses OpenID Connect scopes (r_liteprofile is deprecated).
# openid + profile  →  Sign In with LinkedIn using OpenID Connect product
# w_member_social   →  Share on LinkedIn product
_SCOPES = "openid profile email w_member_social"
_CALLBACK_TIMEOUT = 300   # seconds to wait for the browser callback

# Shared state between the HTTP server thread and the main thread
_auth_code: str | None = None
_auth_error: str | None = None
_auth_event = threading.Event()


class _CallbackHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler that captures the OAuth authorization code."""

    def do_GET(self):
        global _auth_code, _auth_error
        query = urllib.parse.urlparse(self.path).query
        params = urllib.parse.parse_qs(query)

        if "code" in params:
            _auth_code = params["code"][0]
            body = b"<html><body><h2>Authentication successful! You can close this window.</h2></body></html>"
            self.send_response(200)
        elif "error" in params:
            err = params["error"][0]
            desc = params.get("error_description", [err])[0]
            _auth_error = f"{err}: {desc}"
            body = f"<html><body><h2>LinkedIn error: {_auth_error}</h2></body></html>".encode()
            self.send_response(400)
        else:
            body = b"<html><body><h2>No code or error received.</h2></body></html>"
            self.send_response(400)

        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(body)
        _auth_event.set()

    def log_message(self, fmt, *args):  # suppress server log output
        pass


def get_access_token() -> str:
    """Return the stored access token, or run the OAuth flow to obtain one."""
    settings = get_settings()
    if settings.linkedin_access_token:
        return settings.linkedin_access_token
    return _run_oauth_flow(settings)


def _is_port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) != 0


def _run_oauth_flow(settings: Settings) -> str:
    global _auth_code, _auth_event
    _auth_code = None
    _auth_event = threading.Event()

    # ── Check port availability ───────────────────────────────────────────────
    redirect_uri = settings.linkedin_redirect_uri
    port = int(urllib.parse.urlparse(redirect_uri).port or 8000)

    if not _is_port_free(port):
        print(
            f"\nPort {port} is already in use by another process.\n"
            f"To free it on Windows, run:\n"
            f"  netstat -ano | findstr :{port}\n"
            f"  taskkill /PID <PID> /F\n"
            f"Then try again."
        )
        raise RuntimeError(f"Port {port} is busy. Free it and retry.")

    # ── Build authorization URL ───────────────────────────────────────────────
    params = {
        "response_type": "code",
        "client_id": settings.linkedin_client_id,
        "redirect_uri": redirect_uri,
        "scope": _SCOPES,
        "state": "linkedin_agent_csrf",
    }
    auth_url = f"{LINKEDIN_AUTH_URL}?{urllib.parse.urlencode(params)}"

    # ── Start local callback server (serve_forever for robustness) ───────────
    import socketserver
    class _ThreadedServer(socketserver.ThreadingMixIn, HTTPServer):
        daemon_threads = True

    server = _ThreadedServer(("localhost", port), _CallbackHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    print("\n" + "=" * 60)
    print("  LinkedIn Login Required — Follow these 4 steps")
    print("=" * 60)
    print("""
  STEP 1: Your browser will open (or open it manually using
          the URL printed below).

  STEP 2: Log in to LinkedIn with your personal account.

  STEP 3: Click the blue 'Allow' button on the permissions page.

  STEP 4: LinkedIn will redirect your browser to localhost.
          The page may show an error — that is NORMAL.
          The agent captures the code automatically.
""")
    print(f"Login URL (open this in your browser if it doesn't open):\n  {auth_url}\n")

    browser_opened = webbrowser.open(auth_url)
    if not browser_opened:
        print("Could not open browser automatically. Please copy the URL above and open it manually.\n")

    print(f"Waiting up to {_CALLBACK_TIMEOUT // 60} minutes for you to complete the login...")
    _auth_event.wait(timeout=_CALLBACK_TIMEOUT)
    server.shutdown()

    # ── LinkedIn returned an error ─────────────────────────────────────────────
    if _auth_error:
        raise RuntimeError(
            f"\nLinkedIn returned an error during login: {_auth_error}\n\n"
            "Most likely fix: your LinkedIn app is missing a required Product.\n"
            "Steps to fix:\n"
            "  1. Go to https://www.linkedin.com/developers/apps\n"
            "  2. Open your app → click the 'Products' tab\n"
            "  3. Click 'Request access' on 'Share on LinkedIn'\n"
            "  4. Click 'Request access' on 'Sign In with LinkedIn using OpenID Connect'\n"
            "  5. Wait 30 seconds, refresh the page, confirm both show 'Added'\n"
            "  6. Run python main.py again\n"
        )

    # ── Fallback: manual code entry ───────────────────────────────────────────
    if not _auth_code:
        print("\n" + "=" * 60)
        print("  Automatic capture did not work — Manual entry needed")
        print("=" * 60)
        print("""
  After clicking 'Allow', your browser address bar should show
  a URL that looks like this:

    http://localhost:8000/callback?code=AQT1x...long_code...&state=...
                                        ^^^^^^^^^^^^^^^^^^^^^^^^^
                                        Copy only THIS part (the code value)

  If the page shows an error, that is fine — just look at the
  address bar and copy the value after "?code=" up to "&state".
""")
        code_input = input("  Paste just the 'code' value here (the long string after ?code=): ").strip()

        # Accept either a raw code or the full URL
        if code_input.startswith("http"):
            if "linkedin.com/oauth" in code_input:
                raise RuntimeError(
                    "\nYou pasted the LOGIN URL — that is the URL you open to start the login.\n"
                    "You need to:\n"
                    "  1. Open that URL in your browser\n"
                    "  2. Log in to LinkedIn\n"
                    "  3. Click 'Allow'\n"
                    "  4. THEN copy the URL from the address bar (it starts with http://localhost:8000/callback?code=...)\n"
                    "Run python main.py again and follow the steps."
                )
            parsed_params = urllib.parse.parse_qs(urllib.parse.urlparse(code_input).query)
            if "code" not in parsed_params:
                raise RuntimeError("Could not find 'code' in the pasted URL. See instructions above.")
            _auth_code = parsed_params["code"][0]
        elif code_input:
            _auth_code = code_input  # user pasted just the code value directly
        else:
            raise RuntimeError("No code provided. Authentication cancelled.")

    # ── Exchange code for access token ────────────────────────────────────────
    resp = requests.post(
        LINKEDIN_TOKEN_URL,
        data={
            "grant_type": "authorization_code",
            "code": _auth_code,
            "redirect_uri": redirect_uri,
            "client_id": settings.linkedin_client_id,
            "client_secret": settings.linkedin_client_secret,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    resp.raise_for_status()
    token = resp.json()["access_token"]

    _persist_token(token)
    print("\nLinkedIn authentication successful. Token saved to .env.")
    return token


def _persist_token(token: str) -> None:
    env_path = Path(".env")
    if not env_path.exists():
        env_path.write_text("")
    set_key(str(env_path), "LINKEDIN_ACCESS_TOKEN", token)
    get_settings.cache_clear()
