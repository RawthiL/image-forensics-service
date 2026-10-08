"""Server-side request forgery guard for URL fetching.

Fetching a caller-supplied URL makes this node issue requests from its own
network position. Without a guard that reaches loopback, RFC1918, link-local
and cloud metadata addresses -- so a caller can probe and read internal
services that are not exposed to the internet.

Every hop is checked: the initial URL and each redirect target, by resolved IP
rather than by hostname, so `internal.example.com -> 10.0.0.5` is rejected the
same as `http://10.0.0.5`.

**Residual risk.** There is a window between resolving a name and connecting to
it in which DNS can change (rebinding). Closing it needs a transport that pins
the validated address, which httpx does not expose cleanly. The guard narrows
the attack surface a great deal but is not a substitute for keeping URL
fetching off, which is the default.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from urllib.parse import urlsplit

log = logging.getLogger(__name__)

MAX_REDIRECTS = 3
ALLOWED_SCHEMES = ("http", "https")


class UnsafeUrlError(ValueError):
    """The URL resolves somewhere a public fetch must not reach."""


def validate_url(url: str, *, allow_private: bool = False) -> None:
    """Raise `UnsafeUrlError` unless `url` is a plain public http(s) URL."""
    parts = urlsplit(url)

    if parts.scheme.lower() not in ALLOWED_SCHEMES:
        raise UnsafeUrlError("url must be http or https")
    if parts.username or parts.password:
        # Credentials in a URL are never needed for a public image and are a
        # common way to smuggle a different host past naive parsers.
        raise UnsafeUrlError("credentials in the url are not accepted")
    host = parts.hostname
    if not host:
        raise UnsafeUrlError("url has no host")

    if allow_private:
        log.warning("SSRF guard disabled: fetching %s without address checks", host)
        return

    for ip in _resolve(host, parts.port):
        if not _is_public(ip):
            raise UnsafeUrlError(
                f"url resolves to a non-public address ({ip}); "
                "only publicly routable hosts may be fetched"
            )


def _resolve(host: str, port: int | None) -> list[ipaddress._BaseAddress]:
    """Every address the host resolves to. All of them must pass."""
    try:
        infos = socket.getaddrinfo(host, port or 0, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise UnsafeUrlError(f"could not resolve host: {exc}") from exc

    out: list[ipaddress._BaseAddress] = []
    for info in infos:
        try:
            out.append(ipaddress.ip_address(info[4][0]))
        except ValueError:  # pragma: no cover - getaddrinfo shouldn't do this
            continue
    if not out:
        raise UnsafeUrlError("host did not resolve to any address")
    return out


def _is_public(ip: ipaddress._BaseAddress) -> bool:
    """True only for a globally routable unicast address.

    IPv6 is unwrapped first: an address that merely *embeds* a private IPv4 one
    (v4-mapped, 6to4, Teredo) must be judged on the address it reaches.
    """
    ip = _unwrap_v6(ip)

    if (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    ):
        return False

    if isinstance(ip, ipaddress.IPv4Address):
        # Shared Address Space (CGNAT). Covered by is_private on current
        # CPython, checked explicitly so the guard does not depend on that.
        if ip in ipaddress.ip_network("100.64.0.0/10"):
            return False
        # Cloud metadata lives at 169.254.169.254, already caught by
        # is_link_local; spelled out because it is the payload that matters.
        if ip in ipaddress.ip_network("169.254.0.0/16"):
            return False

    return True


def _unwrap_v6(ip: ipaddress._BaseAddress) -> ipaddress._BaseAddress:
    if not isinstance(ip, ipaddress.IPv6Address):
        return ip
    for embedded in (ip.ipv4_mapped, ip.sixtofour, (ip.teredo or (None, None))[1]):
        if embedded is not None:
            return embedded
    return ip
