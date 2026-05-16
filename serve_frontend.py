#!/usr/bin/env python3

"""
Simple HTTP server for serving the React/JavaScript frontend
Usage: python3 serve_frontend.py
Then open http://localhost:8000 in your browser
"""

import http.server
import socketserver
import os
from pathlib import Path

PORT = 8000
FRONTEND_DIR = Path(__file__).parent / "frontend"

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)
    
    def end_headers(self):
        # Add headers to prevent caching during development
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

if __name__ == "__main__":
    os.chdir(FRONTEND_DIR)
    
    with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
        print(f"Frontend server running at http://localhost:{PORT}")
        print(f"Serving files from: {FRONTEND_DIR}")
        print(f"Backend should be running at http://localhost:5000")
        print("\nPress Ctrl+C to stop")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")
