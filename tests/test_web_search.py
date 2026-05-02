import asyncio
import json

import web_search as web_search_module


async def _run_without_thread(func, *args, **kwargs):
    return func(*args, **kwargs)


class _FakeResponse:
    def __init__(
        self,
        body: str,
        *,
        url: str = "https://example.com",
        status: int = 200,
        headers: dict | None = None,
    ):
        self._body = body.encode("utf-8")
        self._url = url
        self.status = status
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, amt=None):
        if amt is None:
            return self._body
        return self._body[:amt]

    def geturl(self):
        return self._url


def test_empty_query_returns_error_without_searching(monkeypatch):
    def fake_search(query, max_results):
        raise AssertionError("search should not be called for an empty query")

    monkeypatch.setattr(web_search_module, "_search_duckduckgo_sync", fake_search)

    payload = json.loads(asyncio.run(web_search_module.web_search("   ")))

    assert payload == {"query": "", "results": [], "error": "empty query"}


def test_web_search_clamps_max_results_and_returns_json(monkeypatch):
    calls = []

    def fake_search(query, max_results):
        calls.append((query, max_results))
        return [{"title": "Result", "url": "https://example.com", "snippet": "Text"}]

    monkeypatch.setattr(web_search_module, "_search_duckduckgo_sync", fake_search)
    monkeypatch.setattr(web_search_module.asyncio, "to_thread", _run_without_thread)

    payload = json.loads(
        asyncio.run(
            web_search_module.web_search("  latest python release  ", max_results=99)
        )
    )

    assert calls == [("latest python release", 10)]
    assert payload["query"] == "latest python release"
    assert payload["results"] == [
        {"title": "Result", "url": "https://example.com", "snippet": "Text"}
    ]


def test_web_search_serializes_search_errors(monkeypatch):
    def fake_search(query, max_results):
        raise TimeoutError("search took too long")

    monkeypatch.setattr(web_search_module, "_search_duckduckgo_sync", fake_search)
    monkeypatch.setattr(web_search_module.asyncio, "to_thread", _run_without_thread)

    payload = json.loads(
        asyncio.run(web_search_module.web_search("news", max_results=3))
    )

    assert payload["query"] == "news"
    assert payload["results"] == []
    assert payload["error"] == "TimeoutError: search took too long"


def test_duckduckgo_parser_decodes_lite_results():
    lite_results = """
    <html>
      <body>
        <a class="result-link" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fstory">
          Fallback Story
        </a>
        <td class="result-snippet">Fallback snippet.</td>
      </body>
    </html>
    """

    results = web_search_module._parse_duckduckgo_results(lite_results, 5)

    assert results == [
        {
            "title": "Fallback Story",
            "url": "https://example.com/story",
            "snippet": "Fallback snippet.",
        }
    ]


def test_duckduckgo_search_uses_lite_endpoint(monkeypatch):
    lite_results = """
    <html>
      <body>
        <a class="result-link" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fstory">
          Fallback Story
        </a>
        <td class="result-snippet">Fallback snippet.</td>
      </body>
    </html>
    """
    requested_urls = []

    def fake_urlopen(request, timeout):
        requested_urls.append(request.full_url)
        return _FakeResponse(lite_results)

    monkeypatch.setattr(web_search_module.urllib.request, "urlopen", fake_urlopen)

    results = web_search_module._search_duckduckgo_sync("met gala sponsors", 5)

    assert requested_urls == [
        "https://lite.duckduckgo.com/lite/?q=met+gala+sponsors",
    ]
    assert results == [
        {
            "title": "Fallback Story",
            "url": "https://example.com/story",
            "snippet": "Fallback snippet.",
        }
    ]


def test_duckduckgo_search_reports_blocked_lite_page(monkeypatch):
    blocked = "<html><head><title>DuckDuckGo</title></head><body></body></html>"

    def fake_urlopen(request, timeout):
        return _FakeResponse(blocked, status=202)

    monkeypatch.setattr(web_search_module.urllib.request, "urlopen", fake_urlopen)

    try:
        web_search_module._search_duckduckgo_sync("met gala sponsors", 5)
    except RuntimeError as e:
        assert "DuckDuckGo lite returned HTTP 202 without search results" in str(e)
    else:
        raise AssertionError("expected blocked DuckDuckGo lite page to raise")


def test_duckduckgo_lite_diagnostics_summarize_zero_result_pages():
    diagnostics = web_search_module._html_search_diagnostics(
        "<html><head><title>Blocked</title></head><body>captcha</body></html>",
        202,
    )

    assert diagnostics == {
        "status": 202,
        "html_chars": 68,
        "title": "Blocked",
        "result_link_markers": 0,
        "result_snippet_markers": 0,
        "captcha_markers": 1,
        "no_results_markers": 0,
    }


def test_web_fetch_rejects_empty_and_non_http_urls():
    empty = json.loads(asyncio.run(web_search_module.web_fetch("   ")))
    invalid = json.loads(asyncio.run(web_search_module.web_fetch("file:///etc/passwd")))

    assert empty == {"url": "", "error": "empty url"}
    assert invalid == {
        "url": "file:///etc/passwd",
        "error": "url must be absolute http or https",
    }


def test_web_fetch_extracts_readable_html(monkeypatch):
    html = """
    <html>
      <head>
        <title> Example Page </title>
        <style>body { display: none }</style>
      </head>
      <body>
        <h1>Hello</h1>
        <script>ignored()</script>
        <p>Readable   text.</p>
      </body>
    </html>
    """
    captured_requests = []

    def fake_urlopen(request, timeout):
        captured_requests.append((request, timeout))
        return _FakeResponse(
            html,
            url="https://example.com/final",
            headers={"Content-Type": "text/html; charset=utf-8"},
        )

    monkeypatch.setattr(web_search_module.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(web_search_module.asyncio, "to_thread", _run_without_thread)

    payload = json.loads(
        asyncio.run(web_search_module.web_fetch("https://example.com/page"))
    )

    assert payload == {
        "url": "https://example.com/page",
        "final_url": "https://example.com/final",
        "status": 200,
        "content_type": "text/html; charset=utf-8",
        "title": "Example Page",
        "text": "Hello Readable text.",
        "truncated": False,
    }
    request, timeout = captured_requests[0]
    assert timeout == 10
    assert request.full_url == "https://example.com/page"
    assert "llm-discord-selfbot" in request.headers["User-agent"]


def test_web_fetch_enforces_min_chars_and_truncates_plain_text(monkeypatch):
    def fake_urlopen(request, timeout):
        return _FakeResponse("x" * 800, headers={"Content-Type": "text/plain"})

    monkeypatch.setattr(web_search_module.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(web_search_module.asyncio, "to_thread", _run_without_thread)

    payload = json.loads(
        asyncio.run(web_search_module.web_fetch("https://example.com/text", max_chars=10))
    )

    assert payload["text"] == "x" * 500
    assert payload["truncated"] is True


def test_web_fetch_allows_large_config_clamped_char_counts(monkeypatch):
    def fake_urlopen(request, timeout):
        return _FakeResponse("x" * 13000, headers={"Content-Type": "text/plain"})

    monkeypatch.setattr(web_search_module.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(web_search_module.asyncio, "to_thread", _run_without_thread)

    payload = json.loads(
        asyncio.run(
            web_search_module.web_fetch("https://example.com/log", max_chars=12000)
        )
    )

    assert len(payload["text"]) == 12000
    assert payload["truncated"] is True
