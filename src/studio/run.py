"""
Launcher for Praxis Studio with Dynamic Free-Port Allocation (`src/studio/run.py`).
Automatically detects available ports to avoid collisions with other local web services
(e.g., Artsylometrie, Nizaba, Jupyter), launches the browser, and runs the FastAPI backend.
"""

import os
import sys
import socket
import time
import threading
import webbrowser
import uvicorn

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def find_available_port(start_port: int = 8055, max_attempts: int = 100) -> int:
    """
    Finds the first available TCP port on localhost starting at start_port.
    Avoids occupied ports like 8000, 8080, etc.
    """
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.15)
            # connect_ex returns 0 if port is open (occupied), non-zero if free
            if s.connect_ex(("127.0.0.1", port)) != 0:
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as bind_sock:
                        bind_sock.bind(("127.0.0.1", port))
                        return port
                except OSError:
                    continue
    return start_port


def open_browser_delayed(url: str, delay_sec: float = 1.0):
    """Opens default browser in a background thread once the server starts."""
    def _open():
        time.sleep(delay_sec)
        print(f"[Praxis Studio] Opening {url} in your default browser...")
        try:
            webbrowser.open(url)
        except Exception as e:
            print(f"[Praxis Studio] Notice: could not automatically open browser ({e}). Please navigate to {url}")
    t = threading.Thread(target=_open, daemon=True)
    t.start()


def main():
    preferred_port = 8055
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        preferred_port = int(sys.argv[1])

    port = find_available_port(start_port=preferred_port)
    url = f"http://127.0.0.1:{port}"

    print("=" * 68, flush=True)
    print("  PRAXIS STUDIO — 3D Video-to-Physical Digital Twin for Humanoids", flush=True)
    print("=" * 68, flush=True)
    print(f"  * Detected Available Port : {port} (Conflict-free)", flush=True)
    print(f"  * Local Web Studio URL    : {url}", flush=True)
    print(f"  * Target Embodiments      : Human (27 DoF), Unitree H1, Tesla Optimus", flush=True)
    print(f"  * Physics Verifier        : MuJoCo 3.x Native C-Bindings", flush=True)
    print(f"  * Press Ctrl+C in this console to terminate the server.", flush=True)
    print("=" * 68, flush=True)

    # Launch browser automatically
    open_browser_delayed(url)

    # Run FastAPI app
    from src.studio.server import app
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")


if __name__ == "__main__":
    main()
