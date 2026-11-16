"""Turn files into (title, markdown-ish text). Markdown, text, HTML and PDF."""
from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path

_BLOCK = {"p", "div", "section", "article", "li", "tr", "br", "table", "ul", "ol", "pre"}
_SKIP = {"script", "style", "nav", "footer"}


class _HtmlToMarkdown(HTMLParser):
    """Keeps heading structure (h1-h6 -> #) so the chunker can use it."""

    def __init__(self):
        super().__init__()
        self.out: list[str] = []
        self.title = ""
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP:
            self._skip += 1
        if tag == "title":
            self._in_title = True
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self.out.append("\n\n" + "#" * int(tag[1]) + " ")
        elif tag == "li":
            self.out.append("\n- ")
        elif tag in _BLOCK:
            self.out.append("\n\n")

    def handle_endtag(self, tag):
        if tag in _SKIP:
            self._skip = max(0, self._skip - 1)
        if tag == "title":
            self._in_title = False
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6") or tag in _BLOCK:
            self.out.append("\n\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data.strip()
        elif not self._skip:
            self.out.append(data)


def html_to_text(html: str) -> tuple[str, str]:
    p = _HtmlToMarkdown()
    p.feed(html)
    lines = [" ".join(line.split()) for line in "".join(p.out).splitlines()]
    text, blank = [], False
    for line in lines:
        if line or not blank:
            text.append(line)
        blank = not line
    return p.title, "\n".join(text).strip()


def _md_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def load_path(path: Path) -> tuple[str, str]:
    """Return (title, text) for a supported file."""
    suffix = path.suffix.lower()
    fallback = path.stem.replace("-", " ").replace("_", " ").title()
    if suffix in (".md", ".markdown", ".txt"):
        text = path.read_text(encoding="utf-8")
        return _md_title(text, fallback), text
    if suffix in (".html", ".htm"):
        title, text = html_to_text(path.read_text(encoding="utf-8"))
        return title or _md_title(text, fallback), text
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = [f"## Page {i}\n\n{(pg.extract_text() or '').strip()}" for i, pg in enumerate(reader.pages, 1)]
        meta_title = (reader.metadata.title if reader.metadata else None) or fallback
        return meta_title, "\n\n".join(pages)
    raise ValueError(f"unsupported file type: {path.suffix}")
