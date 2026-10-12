"""The seam for `/v1/inspect`.

`DetectionService` composes one model. This composes two *services* and three
local checks, and its job is to keep them honest about what ran:

    metadata + unicode + rules   local, deterministic, always run
    views -> OCR service         image-inference-server
    text  -> classifier service  language-forensics-service

Nothing here loads a model or imports a backend. The pixels-to-text step and
the text-to-verdict step live in the services that already do those jobs; what
this owns is *forensic policy* -- which views to render, what counts as
evidence, and how the pieces combine into one verdict.

A note on severity. `malicious` is reserved for deterministic attack
signatures: a Tags-block payload, a bidi override, text that appears only
after a transformation the victim would apply. Those have no benign
explanation. A classifier score is a probability, however high, so it raises
`suspicious` and reports its number rather than claiming certainty it does not
have.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Literal

import httpx
from PIL import Image

from ..config import Settings
from ..media import views as view_policy
from . import classifier, metadata, ocr, rules, unicode_scan
from .errors import UpstreamError
from .ocr import ViewRegions
from .report import ExtractedText, Finding, InspectionReport, Upstream

log = logging.getLogger(__name__)

Scope = Literal["container", "image"]
"""How much of the file an inspection looked at.

`container` is everything that is not the picture: metadata, smuggled
codepoints, bytes past the end of the image. No model, no network.

`image` is that plus the pixels, which needs both sibling services -- an OCR
service to turn views into text and boxes, and a classifier to judge it.
"""

# How much unexplained text a derived view must carry before it is evidence
# rather than OCR noise on a rescaled photograph.
_MIN_NOVEL_CHARS = 12
_MIN_NOVEL_RATIO = 0.6

_WORD = re.compile(r"[\w']{3,}", re.UNICODE)

_UNICODE_SCORE = {"high": 1.0, "medium": 0.7, "low": 0.3}


class InspectionService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.cfg = settings.endpoints.inspect
        self._client: httpx.AsyncClient | None = None
        self._ocr_fingerprint: str | None = None
        self._classifier_fingerprint: str | None = None

    async def start(self) -> None:
        # trust_env=False for the same reason media/loader.py sets it: an
        # HTTP_PROXY in the environment would route these calls through a host
        # nobody configured. Upstream URLs are operator config, never
        # caller-supplied, so they sit outside the SSRF guard by design.
        self._client = httpx.AsyncClient(trust_env=False, follow_redirects=False)
        # Both identities are resolved once, here. Reading them from each
        # response instead would make the fingerprint this service reports
        # depend on what other requests were doing at the time.
        if self.cfg.ocr.url:
            self._ocr_fingerprint = await ocr.fingerprint(self._client, self.cfg.ocr)
        if self.cfg.classifier.url:
            self._classifier_fingerprint = await classifier.fingerprint(
                self._client, self.cfg.classifier
            )
        log.info("inspection ready: %s", self.pipeline_fingerprint("image"))

    async def stop(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def pipeline_fingerprint(self, scope: Scope) -> str:
        """Everything that decides the answer at this scope.

        A container screen depends only on the rule table, so that is all it
        names -- the view policy and the two upstreams are not hops it
        declined to use, they are not part of that pipeline at all.
        """
        if scope == "container":
            return f"screen/{rules.fingerprint()}"
        return "+".join(
            [
                f"inspect/{rules.fingerprint()}",
                view_policy.policy_fingerprint(self.cfg.views),
                f"ocr:{self._ocr_fingerprint or _unknown(self.cfg.ocr.url)}",
                f"cls:{self._classifier_fingerprint or _unknown(self.cfg.classifier.url)}",
            ]
        )

    def describe_screen(self) -> dict:
        cfg = self.settings.endpoints.screen
        return {
            "checks": ["metadata", "unicode", "rules", "trailing_data"],
            "flag_threshold": cfg.flag_threshold,
            "include_text": cfg.include_text,
            "pipeline_fingerprint": self.pipeline_fingerprint("container"),
        }

    def describe(self) -> dict:
        return {
            "views": view_policy.describe(self.cfg.views),
            "ocr": {"url": self.cfg.ocr.url, "required": self.cfg.ocr.required},
            "classifier": {
                "url": self.cfg.classifier.url,
                "required": self.cfg.classifier.required,
            },
            "flag_threshold": self.cfg.flag_threshold,
            "include_text": self.cfg.include_text,
            "pipeline_fingerprint": self.pipeline_fingerprint("image"),
        }

    async def screen(self, image: Image.Image, raw: bytes) -> InspectionReport:
        """Everything in the file that is not the picture.

        No model, no sibling service, no network. This is the whole of the
        screening endpoint, and it is deliberately the same code the full
        inspection runs first -- so the two never disagree about a container.
        """
        cfg = self.settings.endpoints.screen
        started = time.perf_counter()
        report = InspectionReport(scope="container")

        texts, structural = metadata.extract(image, raw)
        for finding in structural:
            report.add(finding)
        self._scan_texts(texts, report, include_text=cfg.include_text)

        report.elapsed_ms = (time.perf_counter() - started) * 1000
        return report

    async def inspect(self, image: Image.Image, raw: bytes) -> InspectionReport:
        started = time.perf_counter()
        report = InspectionReport(scope="image")
        include = self.cfg.include_text

        # 1. The container, exactly as /v1/screen reads it.
        texts, structural = metadata.extract(image, raw)
        for finding in structural:
            report.add(finding)

        # 2. The pixels, via the OCR service: boxes and their contents.
        view_texts, regions = await self._read_views(image, report)
        texts.extend(view_texts)

        # 3. Checks over every extracted string, whatever carried it.
        worst = self._scan_texts(texts, report, include_text=include)

        # 4. The classifier's opinion on the same strings.
        await self._classify(texts, report, worst)

        # 5. The two differentials: what a derived view reveals that the
        #    native one does not. Text where recognition read some, boxes
        #    where it did not.
        self._differential(view_texts, report, worst)
        self._region_differential(regions, view_texts, report)

        report.elapsed_ms = (time.perf_counter() - started) * 1000
        return report

    async def _read_views(
        self, image: Image.Image, report: InspectionReport
    ) -> tuple[list[ExtractedText], list[ViewRegions]]:
        cfg = self.cfg.ocr
        if not cfg.url:
            report.upstreams["ocr"] = Upstream("disabled", detail="no inspect.ocr.url configured")
            return [], []

        # Rendering is only worth its CPU if something will read the result.
        rendered = await asyncio.to_thread(view_policy.expand, image, self.cfg.views)
        report.views_rendered = len(rendered)

        assert self._client is not None
        try:
            # Boxes and their contents: recognition needs detection anyway,
            # so the regions come back free and cover the views where
            # recognition read nothing.
            outcome = await ocr.read(self._client, cfg, rendered, recognize=True)
        except UpstreamError as exc:
            if cfg.required:
                raise
            log.warning("OCR unavailable, continuing without it: %s", exc)
            report.upstreams["ocr"] = Upstream("unreachable", detail=str(exc))
            return [], []

        detail = "; ".join(outcome.errors[:3]) if outcome.errors else ""
        report.upstreams["ocr"] = Upstream(
            "ok", fingerprint=self._ocr_fingerprint, detail=detail
        )
        return outcome.texts, outcome.regions

    def _scan_texts(
        self, texts: list[ExtractedText], report: InspectionReport, *, include_text: bool
    ) -> dict[str, float]:
        """Unicode and rules over every string. Returns the worst score per location."""
        worst: dict[str, float] = {}

        for extracted in texts:
            for hit in unicode_scan.scan(extracted.text):
                score = _UNICODE_SCORE[hit.severity]
                detail = f"{hit.count} codepoint(s)"
                if hit.decoded:
                    detail += "; payload decodes to readable text"
                report.add(
                    Finding(
                        kind=hit.kind,
                        severity=hit.severity,  # type: ignore[arg-type]
                        score=score,
                        where=extracted.where,
                        detail=detail,
                        excerpt=self._excerpt(hit.decoded, include_text),
                    )
                )
                worst[extracted.where] = max(worst.get(extracted.where, 0.0), score)

            # Rules run on normalized text so a zero-width space inside
            # `ig<ZWSP>nore previous instructions` cannot step around them.
            normalized = unicode_scan.normalize(extracted.text)
            for match in rules.apply(normalized):
                report.add(
                    Finding(
                        kind=match.rule.kind,
                        severity=match.rule.severity,  # type: ignore[arg-type]
                        score=match.rule.score,
                        where=extracted.where,
                        detail=match.rule.detail,
                        excerpt=self._excerpt(match.excerpt, include_text),
                    )
                )
                worst[extracted.where] = max(
                    worst.get(extracted.where, 0.0), match.rule.score
                )

            # Metadata carrying prose at all is worth a line in the report:
            # it is the field an EXIF-forwarding pipeline would paste into a
            # prompt, whether or not a rule fired on it.
            if extracted.source == "metadata" and len(normalized) >= 40:
                report.add(
                    Finding(
                        kind="metadata_text",
                        severity="low",
                        score=0.2,
                        where=extracted.where,
                        detail=f"{len(normalized)} characters of free text in container metadata",
                        excerpt=self._excerpt(normalized[:200], include_text),
                    )
                )

        return worst

    async def _classify(
        self, texts: list[ExtractedText], report: InspectionReport, worst: dict[str, float]
    ) -> None:
        cfg = self.cfg.classifier
        if not cfg.url:
            report.upstreams["classifier"] = Upstream(
                "disabled", detail="no inspect.classifier.url configured"
            )
            return

        candidates = [t for t in texts if unicode_scan.normalize(t.text).strip()]
        if not candidates and cfg.skip_when_no_text:
            # The common case: an image with no text anywhere. Nothing to
            # classify, so nothing is sent and nothing is charged.
            report.upstreams["classifier"] = Upstream("skipped", detail="no text extracted")
            return

        assert self._client is not None
        try:
            outcome = await classifier.classify(self._client, cfg, candidates)
        except UpstreamError as exc:
            if cfg.required:
                raise
            log.warning("classifier unavailable, continuing without it: %s", exc)
            report.upstreams["classifier"] = Upstream("unreachable", detail=str(exc))
            return

        if (
            outcome.fingerprint
            and self._classifier_fingerprint
            and outcome.fingerprint != self._classifier_fingerprint
        ):
            # The upstream was redeployed under us. Say so loudly: every
            # fingerprint this service has reported since startup named the
            # old model.
            log.warning(
                "classifier fingerprint changed under us: started with %s, now %s",
                self._classifier_fingerprint,
                outcome.fingerprint,
            )
        report.upstreams["classifier"] = Upstream(
            "ok", fingerprint=outcome.fingerprint or self._classifier_fingerprint
        )

        for judgement in outcome.judgements:
            where = judgement.text.where
            worst[where] = max(worst.get(where, 0.0), judgement.score)
            if judgement.flagged:
                report.add(
                    Finding(
                        kind="classifier_flagged",
                        severity="medium",
                        score=round(judgement.score, self.cfg.score_decimals),
                        where=where,
                        detail=(
                            f"prompt-injection classifier flagged this text "
                            f"(threshold {outcome.threshold})"
                        ),
                        excerpt=self._excerpt(judgement.text.text[:200], self.cfg.include_text),
                    )
                )

    def _differential(
        self,
        view_texts: list[ExtractedText],
        report: InspectionReport,
        worst: dict[str, float],
    ) -> None:
        """Text a derived view reveals that the native view does not.

        This is the sharpest signal the service has, and it exists only
        because view rendering stays local. A scaling attack is defined by
        this asymmetry: benign at full resolution, instructions once the
        victim has resized. Nothing legitimate produces it -- a real sign in a
        photograph reads *better* at native resolution, not worse.

        OCR on a rescaled photograph does hallucinate the odd word, so novel
        text on its own is reported as an anomaly. It is only called an attack
        when the novel text is also something a reader should not obey.
        """
        native = next((t.text for t in view_texts if t.where == "view:native"), "")
        native_words = _words(native)

        for extracted in view_texts:
            if extracted.where == "view:native":
                continue
            novel = _words(extracted.text) - native_words
            novel_chars = sum(len(w) for w in novel)
            if novel_chars < _MIN_NOVEL_CHARS:
                continue
            total = sum(len(w) for w in _words(extracted.text)) or 1
            if novel_chars / total < _MIN_NOVEL_RATIO:
                continue

            suspicious = worst.get(extracted.where, 0.0) >= self.cfg.flag_threshold
            is_rescale = extracted.where.startswith("view:rescale:")
            if suspicious:
                kind = "scaling_attack" if is_rescale else "hidden_text"
                severity, score = "high", 1.0
                detail = (
                    "text that is absent at native resolution appears in this view, "
                    "and reads as an injection"
                )
            else:
                kind = "view_differential"
                severity, score = "medium", 0.5
                detail = (
                    "text appears in this view but not at native resolution; "
                    "no injection pattern matched it"
                )
            report.add(
                Finding(
                    kind=kind,
                    severity=severity,  # type: ignore[arg-type]
                    score=score,
                    where=extracted.where,
                    detail=detail,
                    excerpt=self._excerpt(" ".join(sorted(novel))[:200], self.cfg.include_text),
                )
            )

    def _region_differential(
        self,
        regions: list[ViewRegions],
        view_texts: list[ExtractedText],
        report: InspectionReport,
    ) -> None:
        """Text *boxes* a derived view has that the native view does not.

        This is the whole of what a screening pass can conclude, and it is
        most of the value: you do not need to read the text to know it is
        hidden. A downscale legitimately finds *less* text than the native
        frame, never more, so the reverse is the signature.

        Views that produced recognized text are skipped -- the text
        differential already covered them, and reporting both would say the
        same thing twice. What is left is the genuinely informative case:
        boxes with nothing readable in them.
        """
        if not regions:
            return
        cfg = self.cfg.regions
        native = next((r for r in regions if r.where == "view:native"), None)
        # Native already sees text, so "more text over here" is not news.
        if native and native.count >= cfg.min_regions:
            return

        read_elsewhere = {t.where for t in view_texts}
        for found in regions:
            if found.where == "view:native" or found.where in read_elsewhere:
                continue
            if found.count < cfg.min_regions or found.area_fraction < cfg.min_area_fraction:
                continue
            report.add(
                Finding(
                    kind="view_differential",
                    severity="medium",
                    score=0.5,
                    where=found.where,
                    detail=(
                        f"{found.count} text region(s) covering "
                        f"{found.area_fraction:.2%} of this view, and none at native "
                        "resolution; the text was not read, so what it says is unknown"
                    ),
                )
            )

    def _excerpt(self, text: str | None, include: bool) -> str | None:
        """Quote the offending text only when the operator asked for it.

        Two suppliers running different OCR builds return different strings
        for the same image, so the excerpt is the least reproducible part of
        the report. It is genuinely useful for a human triaging a hit, and
        unhelpful in anything comparing answers across suppliers, so it is
        opt-in rather than default.
        """
        if not include or not text:
            return None
        return text


def _unknown(url: str | None) -> str:
    """`none` means not configured; `unknown` means configured but unidentified.

    The difference matters to anyone comparing fingerprints: one is a
    deployment choice, the other is a hop whose identity we failed to read.
    """
    return "unknown" if url else "none"


def _words(text: str) -> set[str]:
    return {m.group().lower() for m in _WORD.finditer(unicode_scan.normalize(text))}
