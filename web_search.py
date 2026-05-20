import asyncio
import json
import logging
import os
import urllib.parse
import urllib.request
from html.parser import HTMLParser
import yaml

logger = logging.getLogger(__name__)


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


BRAVE_SEARCH_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"


def _search_brave_sync(query: str, max_results: int, api_key: str) -> list[dict]:
    params = urllib.parse.urlencode({"q": query, "count": max_results})
    url = f"{BRAVE_SEARCH_ENDPOINT}?{params}"

    request = urllib.request.Request(
        url,
        headers={
            "X-Subscription-Token": api_key,
            "Accept": "application/json",
            "User-Agent": "llm-discord-selfbot/1.0",
        },
    )

    with urllib.request.urlopen(request, timeout=10) as response:
        body = response.read()

    data = json.loads(body.decode("utf-8", errors="replace"))
    results = []
    web_data = data.get("web", {})
    raw_results = web_data.get("results", [])

    for r in raw_results:
        results.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("description", ""),
        })

    return results[:max_results]



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


async def web_search(
    query: str, max_results: int = 5, api_key: str | None = None
) -> str:
    """Run a web search and return compact JSON for an LLM tool response."""
    query = query.strip()
    if not query:
        logger.info("web_search rejected empty query")
        return json.dumps({"query": query, "results": [], "error": "empty query"})

    max_results = max(1, min(max_results, 10))
    logger.info("web_search start query=%r max_results=%s", query, max_results)

    if not api_key:
        api_key = os.environ.get("BRAVE_API_KEY")

    if not api_key:
        try:
            if os.path.exists("config.yaml"):
                with open("config.yaml", "r") as f:
                    config = yaml.safe_load(f)
                    api_key = (
                        config.get("llm", {})
                        .get("web_search", {})
                        .get("brave_api_key")
                    )
        except Exception:
            pass

    try:
        if not api_key:
            raise ValueError(
                "Brave API key is not configured. Please set the brave_api_key in config.yaml "
                "or the BRAVE_API_KEY environment variable."
            )

        search_task = asyncio.to_thread(
            _search_brave_sync, query, max_results, api_key
        )
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
