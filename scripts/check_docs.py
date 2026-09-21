"""Check UTF-8 Markdown and ordinary inline relative file links."""
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def check(root, strict_style=False):
    errors = []
    warnings = []

    def style_problem(message):
        (errors if strict_style else warnings).append(message)
    for path in sorted(root.rglob("*.md")):
        if any(part in {".git", ".venv", "node_modules"} for part in path.relative_to(root).parts):
            continue
        name = path.relative_to(root)
        raw = path.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            errors.append(f"{name}: invalid UTF-8")
            continue
        if b"\r" in raw:
            style_problem(f"{name}: use LF line endings")
        if raw and not raw.endswith(b"\n"):
            style_problem(f"{name}: missing final newline")
        fence = None
        for number, line in enumerate(text.splitlines(), 1):
            if re.match(r"^(<{7}|={7}|>{7})(?: |$)", line):
                errors.append(f"{name}:{number}: possible conflict marker")
            marker = re.match(r"^\s*(`{3,}|~{3,})", line)
            if marker:
                token = marker.group(1)
                if fence is None:
                    fence = token
                elif token[0] == fence[0] and len(token) >= len(fence):
                    fence = None
                continue
            if fence:
                continue
            line = re.sub(r"`[^`]*`", "", line)
            for match in re.finditer(r"!?\[[^\]]*\]\(([^\s)]+)(?:\s+\"[^\"]*\")?\)", line):
                target = match.group(1).strip("<>")
                parsed = urlsplit(target)
                if parsed.scheme or parsed.netloc or not parsed.path:
                    continue
                target_path = unquote(parsed.path)
                destination = (root / target_path.lstrip("/") if target_path.startswith("/") else path.parent / target_path).resolve()
                if not destination.is_relative_to(root.resolve()):
                    errors.append(f"{name}:{number}: link leaves repository: {target}")
                elif not destination.exists():
                    errors.append(f"{name}:{number}: missing local target: {target}")
    return errors, warnings


if __name__ == "__main__":
    strict_style = "--strict-style" in sys.argv[1:]
    problems, warnings = check(ROOT, strict_style=strict_style)
    if warnings:
        print("\n".join(f"warning: {item}" for item in warnings), file=sys.stderr)
    if problems:
        print("\n".join(problems), file=sys.stderr)
        sys.exit(1)
    mode = "strict" if strict_style else "publish"
    print(f"Documentation checks passed ({mode} mode; text hygiene and inline local file links).")
