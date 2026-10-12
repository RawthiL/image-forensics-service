"""What an inspection returns.

A detection verdict is one number, because "is this AI-generated" has one
answer. An injection verdict is not: an operator deciding whether to feed an
image to an agent needs to know *what* was found and *where*, because the
location is most of the evidence. Text that appears only after a downscale is
an attack; the same text at native resolution is a photograph of a sign.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Severity = Literal["none", "low", "medium", "high"]
Verdict = Literal["clean", "suspicious", "malicious"]

_RANK: dict[str, int] = {"none": 0, "low": 1, "medium": 2, "high": 3}


@dataclass(frozen=True)
class Finding:
    """One thing worth reporting, and where it was found."""

    kind: str
    """Machine-readable class, e.g. `rendered_text`, `scaling_attack`,
    `metadata_text`, `unicode_tag_block`, `trailing_data`."""

    severity: Severity
    score: float
    """Confidence in [0, 1]. Deterministic checks report 1.0."""

    where: str
    """`exif:UserComment`, `view:rescale:384:bicubic`, `png:tEXt:Comment`, ..."""

    detail: str = ""
    excerpt: str | None = None
    """A short quotation of the offending text. Present only when
    `inspect.include_text` is on -- see InspectionService for why it is off by
    default."""

    def as_dict(self) -> dict:
        out = {
            "kind": self.kind,
            "severity": self.severity,
            "score": self.score,
            "where": self.where,
        }
        if self.detail:
            out["detail"] = self.detail
        if self.excerpt is not None:
            out["excerpt"] = self.excerpt
        return out


@dataclass(frozen=True)
class Upstream:
    """One composed service, and whether it actually ran.

    `status` is not decoration. A report that silently omitted the classifier
    because it was unreachable would read exactly like a clean one, and a
    caller would act on a check that never happened.
    """

    status: Literal["ok", "disabled", "skipped", "unreachable"]
    fingerprint: str | None = None
    detail: str = ""

    def as_dict(self) -> dict:
        out: dict = {"status": self.status}
        if self.fingerprint:
            out["fingerprint"] = self.fingerprint
        if self.detail:
            out["detail"] = self.detail
        return out


@dataclass
class InspectionReport:
    scope: str = "image"
    """How much of the file was examined: `container` (metadata, smuggled text
    and structure -- the pixels were never looked at) or `image` (that, plus
    everything rendered in the pixels). A `clean` at container scope is the
    far weaker statement, and a caller must be able to tell them apart."""

    findings: list[Finding] = field(default_factory=list)
    upstreams: dict[str, Upstream] = field(default_factory=dict)
    views_rendered: int = 0
    elapsed_ms: float = 0.0

    def add(self, finding: Finding) -> None:
        self.findings.append(finding)

    @property
    def score(self) -> float:
        return max((f.score for f in self.findings), default=0.0)

    @property
    def severity(self) -> Severity:
        worst = max((_RANK[f.severity] for f in self.findings), default=0)
        return next(k for k, v in _RANK.items() if v == worst)  # type: ignore[return-value]

    def verdict(self, threshold: float) -> Verdict:
        """`high` severity is an attack signature, not a probability.

        Those checks -- a tag-block payload, text that appears only under
        downscaling -- have no benign explanation, so they decide the verdict
        on their own rather than being averaged into a score.
        """
        if any(f.severity == "high" for f in self.findings):
            return "malicious"
        if any(f.severity == "medium" for f in self.findings) or self.score >= threshold:
            return "suspicious"
        return "clean"

    def sorted_findings(self) -> list[Finding]:
        return sorted(self.findings, key=lambda f: (-_RANK[f.severity], -f.score, f.where))


@dataclass(frozen=True)
class ExtractedText:
    """A string pulled out of the image, and where it came from.

    Metadata parsing and OCR both produce these, so the checks that run over
    text -- unicode, rules, the classifier -- are written once and do not care
    which carrier a payload arrived on.
    """

    where: str
    text: str
    source: Literal["metadata", "view"]
    confidence: float = 1.0
