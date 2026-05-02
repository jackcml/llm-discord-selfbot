import asyncio
import json
import logging
import urllib.parse
import urllib.request
from html.parser import HTMLParser

logger = logging.getLogger(__name__)

DUCKDUCKGO_LITE_ENDPOINT = "https://lite.duckduckgo.com/lite/"


class _ReadableHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.title = ""
        self.text = []
        self._capture_title = False
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
        elif tag == "title":
            self._capture_title = True

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
        elif tag == "title":
            self._capture_title = False

    def handle_data(self, data):
        text = " ".join(data.split())
        if not text:
            return
        if self._capture_title:
            self.title = self._clean_text(f"{self.title} {text}")
        elif not self._skip_depth:
            self.text.append(text)

    @staticmethod
    def _clean_text(text: str) -> str:
        return " ".join(text.split())

    def readable_text(self) -> str:
        return self._clean_text(" ".join(self.text))


class _DuckDuckGoHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.results = []
        self._current = None
        self._capture = None
        self._chunks = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = set(attrs.get("class", "").split())

        if tag == "a" and "result-link" in classes:
            self._finish_current()
            self._current = {
                "title": "",
                "url": self._clean_url(attrs.get("href", "")),
            }
            self._capture = "title"
            self._chunks = []
            return

        if self._current is not None and "result-snippet" in classes:
            self._capture = "snippet"
            self._chunks = []

    def handle_data(self, data):
        if self._capture:
            self._chunks.append(data)

    def handle_endtag(self, tag):
        if self._capture == "title" and tag == "a":
            self._current["title"] = self._clean_text("".join(self._chunks))
            self._capture = None
            self._chunks = []
            return

        if self._capture == "snippet" and tag in {"a", "div", "td", "span"}:
            self._current["snippet"] = self._clean_text("".join(self._chunks))
            self._finish_current()

    def close(self):
        self._finish_current()
        super().close()

    def _finish_current(self):
        if self._current is None:
            return
        if self._current.get("title") and self._current.get("url"):
            self._current.setdefault("snippet", "")
            self.results.append(self._current)
        self._current = None
        self._capture = None
        self._chunks = []

    @staticmethod
    def _clean_text(text: str) -> str:
        return " ".join(text.split())

    @staticmethod
    def _clean_url(url: str) -> str:
        if url.startswith("//"):
            url = f"https:{url}"
        parsed = urllib.parse.urlparse(url)
        query = urllib.parse.parse_qs(parsed.query)
        if "uddg" in query:
            return query["uddg"][0]
        return url


def _search_duckduckgo_sync(query: str, max_results: int) -> list[dict]:
    params = urllib.parse.urlencode({"q": query})
    html, status = _fetch_duckduckgo_html(DUCKDUCKGO_LITE_ENDPOINT, params)
    results = _parse_duckduckgo_results(html, max_results)
    logger.debug(
        "DuckDuckGo lite search parsed status=%s html_chars=%s results=%s query=%r",
        status,
        len(html),
        len(results),
        query,
    )
    if not results:
        diagnostics = _html_search_diagnostics(html, status)
        logger.warning(
            "DuckDuckGo lite search returned zero parsed results: %s",
            diagnostics,
        )
        if status == 202:
            raise RuntimeError(
                "DuckDuckGo lite returned HTTP 202 without search results; "
                f"{diagnostics}"
            )

    return results


def _fetch_duckduckgo_html(endpoint_url: str, params: str) -> tuple[str, int | None]:
    request = urllib.request.Request(
        f"{endpoint_url}?{params}",
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) "
                "Gecko/20100101 Firefox/125.0"
            )
        },
    )

    with urllib.request.urlopen(request, timeout=10) as response:
        html = response.read().decode("utf-8", errors="replace")
        status = getattr(response, "status", None)
    return html, status


def _parse_duckduckgo_results(html: str, max_results: int) -> list[dict]:
    parser = _DuckDuckGoHTMLParser()
    parser.feed(html)
    parser.close()
    return parser.results[:max_results]


def _html_search_diagnostics(html: str, status: int | None) -> dict:
    lower_html = html.lower()
    title = ""
    title_start = lower_html.find("<title")
    if title_start != -1:
        title_text_start = lower_html.find(">", title_start)
        title_end = lower_html.find("</title>", title_text_start)
        if title_text_start != -1 and title_end != -1:
            title = " ".join(html[title_text_start + 1 : title_end].split())

    return {
        "status": status,
        "html_chars": len(html),
        "title": title[:200],
        "result_link_markers": lower_html.count("result-link"),
        "result_snippet_markers": lower_html.count("result-snippet"),
        "captcha_markers": lower_html.count("captcha"),
        "no_results_markers": lower_html.count("no results"),
    }


def _valid_fetch_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _decode_response_body(body: bytes, content_type: str) -> str:
    encoding = "utf-8"
    for part in content_type.split(";"):
        part = part.strip()
        if part.lower().startswith("charset="):
            encoding = part.split("=", 1)[1].strip() or encoding
            break
    return body.decode(encoding, errors="replace")


def _fetch_url_sync(url: str, max_chars: int) -> dict:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (compatible; llm-discord-selfbot/1.0; "
                "+https://duckduckgo.com/)"
            )
        },
    )

    max_bytes = max_chars * 4
    with urllib.request.urlopen(request, timeout=10) as response:
        body = response.read(max_bytes + 1)
        truncated = len(body) > max_bytes
        body = body[:max_bytes]
        final_url = response.geturl()
        status = getattr(response, "status", None)
        content_type = response.headers.get("Content-Type", "")

    decoded = _decode_response_body(body, content_type)
    title = ""
    text = decoded
    if "html" in content_type.lower() or "<html" in decoded[:500].lower():
        parser = _ReadableHTMLParser()
        parser.feed(decoded)
        title = parser.title
        text = parser.readable_text()
    else:
        text = " ".join(decoded.split())

    if len(text) > max_chars:
        text = text[:max_chars].rstrip()
        truncated = True

    return {
        "url": url,
        "final_url": final_url,
        "status": status,
        "content_type": content_type,
        "title": title,
        "text": text,
        "truncated": truncated,
    }


async def web_search(query: str, max_results: int = 5) -> str:
    """Run a web search and return compact JSON for an LLM tool response."""
    query = query.strip()
    if not query:
        logger.info("web_search rejected empty query")
        return json.dumps({"query": query, "results": [], "error": "empty query"})

    max_results = max(1, min(max_results, 10))
    logger.info("web_search start query=%r max_results=%s", query, max_results)
    try:
        search_task = asyncio.to_thread(_search_duckduckgo_sync, query, max_results)
        results = await asyncio.wait_for(search_task, timeout=12)
        logger.info("web_search done query=%r results=%s", query, len(results))
        return json.dumps({"query": query, "results": results}, ensure_ascii=True)
    except Exception as e:
        logger.warning(
            "web_search failed query=%r error=%s: %s",
            query,
            type(e).__name__,
            e,
        )
        return json.dumps(
            {"query": query, "results": [], "error": f"{type(e).__name__}: {e}"},
            ensure_ascii=True,
        )


async def web_fetch(url: str, max_chars: int = 4000) -> str:
    """Fetch a URL and return compact readable JSON for an LLM tool response."""
    url = url.strip()
    if not url:
        logger.info("web_fetch rejected empty url")
        return json.dumps({"url": url, "error": "empty url"}, ensure_ascii=True)
    if not _valid_fetch_url(url):
        logger.info("web_fetch rejected invalid url=%r", url)
        return json.dumps(
            {"url": url, "error": "url must be absolute http or https"},
            ensure_ascii=True,
        )

    max_chars = max(500, max_chars)
    logger.info("web_fetch start url=%r max_chars=%s", url, max_chars)
    try:
        fetch_task = asyncio.to_thread(_fetch_url_sync, url, max_chars)
        result = await asyncio.wait_for(fetch_task, timeout=12)
        logger.info(
            "web_fetch done url=%r final_url=%r status=%s text_chars=%s truncated=%s",
            url,
            result.get("final_url"),
            result.get("status"),
            len(result.get("text") or ""),
            result.get("truncated"),
        )
        return json.dumps(result, ensure_ascii=True)
    except Exception as e:
        logger.warning(
            "web_fetch failed url=%r error=%s: %s",
            url,
            type(e).__name__,
            e,
        )
        return json.dumps(
            {"url": url, "error": f"{type(e).__name__}: {e}"},
            ensure_ascii=True,
        )
