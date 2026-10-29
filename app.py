"""Local launcher for the Green Grid Allocator web app.

    python app.py            -> http://127.0.0.1:8000  (opens your browser)

The app itself lives in api/index.py (the same file Vercel runs).
"""
import socket
import sys
import threading
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "api"))


def free_port(start=8000):
    for port in range(start, start + 20):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return start


def main():
    import uvicorn
    from index import app  # api/index.py

    port = free_port()
    url = f"http://127.0.0.1:{port}"
    print(f"\n  Green Grid Allocator running at {url}   (Ctrl+C to stop)\n")
    threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
