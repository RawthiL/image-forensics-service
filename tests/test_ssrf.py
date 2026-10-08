"""URL fetching is off by default, and guarded when it is on.

Fetching a caller-supplied URL makes the node issue requests from its own
network position. These tests pin both halves of the mitigation: the default,
and the address guard behind it.
"""

import asyncio

import pytest

from ifs.media.ssrf import UnsafeUrlError, validate_url

# IP literals throughout: the suite must not depend on DNS.
NON_PUBLIC = [
    "http://127.0.0.1/x.png",            # loopback
    "http://10.0.0.5/x.png",             # RFC1918
    "http://192.168.1.1/x.png",          # RFC1918
    "http://172.16.0.1/x.png",           # RFC1918
    "http://169.254.169.254/latest/",    # cloud metadata
    "http://100.64.0.1/x.png",           # CGNAT
    "http://0.0.0.0/x.png",              # unspecified
    "http://[::1]/x.png",                # IPv6 loopback
    "http://[::ffff:127.0.0.1]/x.png",   # IPv4-mapped loopback
    "http://[fe80::1]/x.png",            # IPv6 link-local
    "http://[fc00::1]/x.png",            # IPv6 unique-local
]


def test_url_fetching_is_off_by_default():
    from ifs.config import LimitsConfig

    assert LimitsConfig().allow_url_fetch is False
    assert LimitsConfig().allow_private_urls is False


@pytest.mark.parametrize("url", NON_PUBLIC)
def test_non_public_addresses_are_rejected(url):
    with pytest.raises(UnsafeUrlError, match="non-public address"):
        validate_url(url)


def test_public_address_passes():
    validate_url("http://93.184.216.34/x.png")  # IANA example range, no DNS needed


@pytest.mark.parametrize(
    "url,match",
    [
        ("file:///etc/passwd", "http or https"),
        ("gopher://127.0.0.1/", "http or https"),
        ("ftp://example.com/x.png", "http or https"),
        ("http://user:pw@93.184.216.34/x.png", "credentials"),
        ("http:///x.png", "no host"),
    ],
)
def test_malformed_or_dangerous_urls_are_rejected(url, match):
    with pytest.raises(UnsafeUrlError, match=match):
        validate_url(url)


def test_guard_can_be_disabled_for_development():
    validate_url("http://127.0.0.1/x.png", allow_private=True)


# -- the loader path ------------------------------------------------------


def _settings(**limits):
    """Just the `limits` section -- the only part `from_url` reads.

    Deliberately not a full `Settings`: that would pick up the operator's
    config.yaml and make these tests depend on local configuration.
    """
    from types import SimpleNamespace

    from ifs.config import LimitsConfig

    return SimpleNamespace(limits=LimitsConfig(**limits))


def test_loader_refuses_when_fetching_is_disabled():
    from ifs.media.loader import MediaError, from_url

    with pytest.raises(MediaError) as exc:
        asyncio.run(from_url("http://93.184.216.34/x.png", _settings()))
    assert exc.value.code == "url_fetch_disabled"


@pytest.mark.parametrize("url", NON_PUBLIC)
def test_loader_blocks_non_public_even_when_fetching_is_enabled(url):
    from ifs.media.loader import MediaError, from_url

    with pytest.raises(MediaError) as exc:
        asyncio.run(from_url(url, _settings(allow_url_fetch=True)))
    assert exc.value.code == "url_not_allowed"


def test_redirect_to_an_internal_address_is_blocked(monkeypatch):
    """The classic bypass: a public host that 302s somewhere internal.

    Redirects are followed manually for exactly this reason -- httpx's own
    follow_redirects would validate only the URL the caller supplied.
    """
    import http.server
    import threading

    class Redirector(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(302)
            self.send_header("Location", "http://169.254.169.254/latest/meta-data/")
            self.end_headers()

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), Redirector)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    try:
        # Let the first hop through (it is loopback only because a test cannot
        # host a real public server) and keep real validation for every hop
        # after it. If the redirect target were trusted, this would fetch the
        # metadata endpoint instead of raising.
        import ifs.media.loader as loader
        from ifs.media.loader import MediaError, from_url

        real = loader.validate_url
        hops: list[str] = []

        def validate_after_first(url, *, allow_private=False):
            hops.append(url)
            if len(hops) == 1:
                return
            return real(url, allow_private=allow_private)

        monkeypatch.setattr(loader, "validate_url", validate_after_first)

        with pytest.raises(MediaError) as exc:
            asyncio.run(
                from_url(f"http://127.0.0.1:{port}/", _settings(allow_url_fetch=True))
            )
        assert exc.value.code == "url_not_allowed"
        assert len(hops) == 2, "the redirect target was not re-validated"
        assert hops[1].startswith("http://169.254.169.254/")
    finally:
        srv.shutdown()


def test_redirect_loop_is_capped():
    import http.server
    import threading

    class Looper(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(302)
            self.send_header("Location", self.path)
            self.end_headers()

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), Looper)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    try:
        from ifs.media.loader import MediaError, from_url

        with pytest.raises(MediaError) as exc:
            asyncio.run(
                from_url(
                    f"http://127.0.0.1:{port}/",
                    _settings(allow_url_fetch=True, allow_private_urls=True),
                )
            )
        assert exc.value.code == "url_fetch_failed"
        assert "redirect" in str(exc.value)
    finally:
        srv.shutdown()
