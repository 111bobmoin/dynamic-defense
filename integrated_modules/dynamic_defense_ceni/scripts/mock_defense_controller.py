from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from datetime import datetime
import argparse
import json


class Handler(BaseHTTPRequestHandler):
    log_file = Path("reports/controller_actions.jsonl")

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8") if length else "{}"

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"raw": raw}

        record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "path": self.path,
            "payload": payload,
        }

        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with self.log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        response = {
            "status": "OK",
            "path": self.path,
            "received": payload,
        }

        data = json.dumps(response, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        print("[%s] %s" % (datetime.now().isoformat(timespec="seconds"), fmt % args))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    server = HTTPServer((args.host, args.port), Handler)
    print(f"mock defense controller listening on http://{args.host}:{args.port}")
    print("logs: reports/controller_actions.jsonl")
    server.serve_forever()


if __name__ == "__main__":
    main()
