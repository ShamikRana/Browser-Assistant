"""Web page text extraction.

Extraction runs as a cascade because no single extractor handles every layout:

1. ``trafilatura`` in markdown mode - best for articles, keeps headings/tables.
2. ``readability-lxml`` - good on pages trafilatura considers boilerplate-only.
3. ``BeautifulSoup`` - last resort, strips known boilerplate tags.

The HTML itself preferably comes from the browser extension (the *rendered* DOM),
which is the only reliable way to read JavaScript-rendered, logged-in or
paywalled pages. A server-side fetch is used only when the extension can't
supply it.
"""

import logging
import re

import requests
import trafilatura
from bs4 import BeautifulSoup
from readability import Document

from config import REQUEST_TIMEOUT, USER_AGENT

logger = logging.getLogger(__name__)

# Below this, an extractor is treated as having failed and the next one runs.
_MIN_USEFUL_CHARS = 200

_BOILERPLATE_TAGS = (
    "script", "style", "noscript", "iframe", "svg", "canvas",
    "nav", "header", "footer", "aside", "form", "button",
)

_session = requests.Session()
_session.headers.update({"User-Agent": USER_AGENT})


def _collapse_whitespace(text: str) -> str:
    text = re.sub(r"[ \t\u00a0]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


def _extract_with_trafilatura(html: str) -> str:
    """Preferred extractor. Markdown output preserves heading structure."""
    try:
        # Fallback algorithms stay enabled (trafilatura's default) so that
        # pages with unusual markup still produce text.
        text = trafilatura.extract(
            html,
            output_format="markdown",
            include_comments=False,
            include_tables=True,
            include_formatting=True,
            favor_recall=True,
        )
        return text or ""
    except Exception as exc:
        logger.debug("trafilatura extraction failed: %s", exc)
        return ""


def _extract_with_readability(html: str) -> str:
    try:
        summary_html = Document(html).summary(html_partial=True)
        soup = BeautifulSoup(summary_html, "lxml")
        return soup.get_text("\n")
    except Exception as exc:
        logger.debug("readability extraction failed: %s", exc)
        return ""


def _extract_with_soup(html: str) -> str:
    try:
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(_BOILERPLATE_TAGS):
            tag.decompose()
        main = soup.find("main") or soup.find("article") or soup.body or soup
        return main.get_text("\n")
    except Exception as exc:
        logger.debug("soup extraction failed: %s", exc)
        return ""


def page_title(html: str) -> str:
    try:
        soup = BeautifulSoup(html, "lxml")
        if soup.title and soup.title.string:
            return soup.title.string.strip()
        heading = soup.find("h1")
        if heading:
            return heading.get_text(" ").strip()
    except Exception:
        pass
    return ""


def extract_from_html(html: str) -> str:
    """Run the extractor cascade and return the best result."""
    if not html:
        return ""

    best = ""
    for extractor in (_extract_with_trafilatura, _extract_with_readability, _extract_with_soup):
        text = _collapse_whitespace(extractor(html))
        if len(text) > len(best):
            best = text
        if len(best) >= _MIN_USEFUL_CHARS:
            break
    return best


def fetch_html(url: str) -> str:
    """Server-side fetch, used only when the extension can't supply the DOM."""
    try:
        response = _session.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.text
    except Exception as exc:
        logger.warning("Failed to fetch %s: %s", url, exc)
        return ""


def get_page_content(url: str, html: str | None = None, text: str | None = None) -> tuple[str, str]:
    """Resolve page content, returning ``(title, text)``.

    Precedence: caller-supplied text, then caller-supplied HTML (the rendered
    DOM from the extension), then a server-side fetch.
    """
    if text and text.strip():
        return "", _collapse_whitespace(text)

    if html and html.strip():
        extracted = extract_from_html(html)
        if len(extracted) >= _MIN_USEFUL_CHARS:
            return page_title(html), extracted

    fetched = fetch_html(url)
    if fetched:
        extracted = extract_from_html(fetched)
        if extracted:
            return page_title(fetched), extracted

    # The rendered DOM may still be the only thing available even if it was short.
    if html:
        return page_title(html), extract_from_html(html)
    return "", ""
