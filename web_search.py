import asyncio
import json
import urllib.parse
import urllib.request
from html.parser import HTMLParser


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

        if tag == "a" and "result__a" in classes:
            self._current = {"title": "", "url": self._clean_url(attrs.get("href", ""))}
            self._capture = "title"
            self._chunks = []
            return

        if self._current is not None and "result__snippet" in classes:
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

        if self._capture == "snippet" and tag in {"a", "div"}:
            self._current["snippet"] = self._clean_text("".join(self._chunks))
            self.results.append(self._current)
            self._current = None
            self._capture = None
            self._chunks = []

    @staticmethod
    def _clean_text(text: str) -> str:
        return " ".join(text.split())

    @staticmethod
    def _clean_url(url: str) -> str:
        parsed = urllib.parse.urlparse(url)
        query = urllib.parse.parse_qs(parsed.query)
        if "uddg" in query:
            return query["uddg"][0]
        return url


def _search_duckduckgo_sync(query: str, max_results: int) -> list[dict]:
    params = urllib.parse.urlencode({"q": query})
    request = urllib.request.Request(
        f"https://html.duckduckgo.com/html/?{params}",
        headers={
            "User-Agent": (
                "Mozilla/5.0 (compatible; llm-discord-selfbot/1.0; "
                "+https://duckduckgo.com/)"
            )
        },
    )

    with urllib.request.urlopen(request, timeout=10) as response:
        html = response.read().decode("utf-8", errors="replace")

    parser = _DuckDuckGoHTMLParser()
    parser.feed(html)
    return parser.results[:max_results]


async def web_search(query: str, max_results: int = 5) -> str:
    """Run a web search and return compact JSON for an LLM tool response."""
    query = query.strip()
    if not query:
        return json.dumps({"query": query, "results": [], "error": "empty query"})

    max_results = max(1, min(max_results, 10))
    try:
        search_task = asyncio.to_thread(_search_duckduckgo_sync, query, max_results)
        results = await asyncio.wait_for(search_task, timeout=12)
        return json.dumps({"query": query, "results": results}, ensure_ascii=True)
    except Exception as e:
        return json.dumps(
            {"query": query, "results": [], "error": f"{type(e).__name__}: {e}"},
            ensure_ascii=True,
        )
