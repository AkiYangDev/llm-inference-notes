"""Preview GitHub Pages with its real project prefix on localhost."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

BASE = '/llm-inference-notes/'
ROOT = Path(__file__).resolve().parents[1] / '_site'


class Handler(SimpleHTTPRequestHandler):
    def translate_path(self, path):
        path = urlsplit(path).path
        if path.startswith(BASE):
            path = '/' + path[len(BASE):]
        return super().translate_path(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8876)
    args = parser.parse_args()
    if not (ROOT / 'index.html').exists():
        parser.error('Build first: python site/build.py')
    server = ThreadingHTTPServer(('127.0.0.1', args.port), partial(Handler, directory=str(ROOT)))
    print(f'Preview: http://127.0.0.1:{args.port}{BASE}', flush=True)
    server.serve_forever()
