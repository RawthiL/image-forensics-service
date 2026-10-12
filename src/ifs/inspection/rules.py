"""Deterministic patterns over extracted text.

This runs before -- and without -- the classifier, for three reasons. Most
images carry no text at all, so the common case should cost nothing. A
regex is reproducible across suppliers in a way a model is not. And when the
classifier is disabled or unreachable, these still answer.

They are a filter, not a defense. An attacker who reads this file can write
around it in a minute. The classifier exists because of that; these exist
because the classifier is a network hop away and is wrong in different ways.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Rule:
    kind: str
    severity: str
    score: float
    pattern: re.Pattern[str]
    detail: str


def _p(expr: str) -> re.Pattern[str]:
    return re.compile(expr, re.IGNORECASE)


RULES: list[Rule] = [
    Rule(
        "instruction_override",
        "high",
        0.95,
        _p(r"\b(ignore|disregard|forget|override)\b[^.\n]{0,40}\b"
           r"(previous|prior|above|earlier|all|your)\b[^.\n]{0,20}\b"
           r"(instruction|prompt|rule|direction|context|message)s?\b"),
        "classic instruction-override phrasing",
    ),
    Rule(
        "role_hijack",
        "high",
        0.9,
        _p(r"\b(you are now|act as|pretend to be|from now on you|"
           r"new instructions?|updated instructions?|system prompt|developer mode)\b"),
        "attempts to redefine the assistant's role or inject a new system prompt",
    ),
    Rule(
        "turn_injection",
        "high",
        0.85,
        _p(r"(^|\n)\s*(system|assistant|user|human)\s*:|"
           r"<\|?(im_start|im_end|system|endoftext)\|?>|\[/?INST\]"),
        "forged conversation turn or chat-template control token",
    ),
    Rule(
        "exfiltration",
        "high",
        0.95,
        _p(r"!\[[^\]]*\]\(\s*https?://[^)]*[?&=][^)]*\)|"
           r"\b(send|post|upload|exfiltrate|forward|leak)\b[^.\n]{0,40}\b"
           r"(to|at)\b\s+https?://"),
        "markdown image beacon or an instruction to send data to a URL",
    ),
    Rule(
        "tool_invocation",
        "high",
        0.85,
        _p(r"<\s*/?\s*(tool_call|function_call|invoke|antml:invoke)\b|"
           r"\b(call|invoke|run|execute)\b[^.\n]{0,25}\b(tool|function|command|shell)\b"),
        "attempts to drive tool use directly",
    ),
    Rule(
        "confirmation_bypass",
        "high",
        0.9,
        _p(r"\btrust\s*=\s*true\b|--yolo\b|--dangerously[\w-]*|"
           r"\b(without|skip|bypass|no need (for|to))\b[^.\n]{0,30}\b"
           r"(confirm\w*|approval|asking|permission)\b"),
        "tries to suppress the human-in-the-loop step",
    ),
    Rule(
        "credential_access",
        "medium",
        0.7,
        _p(r"\b(api[_ -]?key|secret[_ -]?key|access[_ -]?token|password|credentials?|"
           r"private[_ -]?key)\b|\.env\b|~/\.ssh|\bid_rsa\b|\bAWS_SECRET\w*"),
        "names credential material",
    ),
    Rule(
        "command_execution",
        "medium",
        0.7,
        _p(r"\b(curl|wget|rm\s+-rf|chmod\s+\+x|base64\s+-d|powershell|Invoke-WebRequest)\b|"
           r"\bos\.system\b|\bsubprocess\.\w+|\beval\s*\("),
        "shell or code execution primitives",
    ),
    Rule(
        "urgency_framing",
        "low",
        0.4,
        _p(r"\b(urgent|immediately|important|do not tell|don'?t mention|"
           r"keep this (secret|hidden)|without telling)\b"),
        "social-engineering framing often paired with an injection",
    ),
]


def fingerprint() -> str:
    """Digest of the rule table.

    These patterns decide what this service calls an injection, so editing one
    changes the answers. Two suppliers running different rule tables are not
    comparable, and this is what makes that visible.
    """
    joined = "|".join(f"{rule.kind}:{rule.pattern.pattern}" for rule in RULES)
    return "rules@" + hashlib.sha256(joined.encode()).hexdigest()[:12]


@dataclass(frozen=True)
class Match:
    rule: Rule
    excerpt: str


def apply(text: str, *, excerpt_chars: int = 120) -> list[Match]:
    """Every rule that fires, with the span that fired it."""
    if not text or not text.strip():
        return []
    matches = []
    for rule in RULES:
        found = rule.pattern.search(text)
        if found:
            matches.append(Match(rule, _excerpt(text, found, excerpt_chars)))
    return matches


def score(matches: list[Match]) -> float:
    return max((m.rule.score for m in matches), default=0.0)


def _excerpt(text: str, match: re.Match[str], width: int) -> str:
    start = max(0, match.start() - width // 4)
    end = min(len(text), match.end() + width // 4)
    body = " ".join(text[start:end].split())
    return f"{'…' if start else ''}{body}{'…' if end < len(text) else ''}"
