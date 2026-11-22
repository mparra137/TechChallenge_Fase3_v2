"""Deterministic input, evidence and presentation safeguards."""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

from src.anonymizer import DEFAULT_PATTERNS, NAME_AFTER_LABEL

_CHART = re.compile(r"\b(?:pct|prt)[ -]?\d+\b", re.I)
_CONTROL = re.compile(r"<\|[^>\r\n]+\|>|[\x00-\x08\x0b-\x1f\x7f-\x9f]|[\u202a-\u202e\u2066-\u2069]")
_LOCATION = re.compile(
    r"\b(?:endere[cç]o|address|bairro|cidade|munic[ií]pio|city|neighborhood|residente em)\s*:?\s*[^.;\r\n]+"
    r"|(?:^|(?<=[.;\n ]))(?:rua|avenida|travessa|alameda)\s+[^.;\r\n]+", re.I
)
_BIRTH = re.compile(
    r"\b(?:data de nascimento|nascimento|date of birth|birth date|dob|nascid[oa] em|born on)\s*:?\s*"
    r"(?:\d{4}-\d\d-\d\d|\d\d?[/.-]\d\d?[/.-]\d{2,4}|\d{1,2}\s+de\s+\w+\s+de\s+\d{4})", re.I
)
_EXTRA_ID = re.compile(
    r"\b(?:cpf|rg|cns|sus|cep|telefone|phone|passaporte|passport)\s*:?\s*(?=\d)[\d./ -]{5,}[\dXx]\b"
    r"|\b\d{3}[-.\s]\d{3}[-.\s]\d{3}[-.\s]\d{2}\b"
    r"|\b(?:\+55[ -]?)?\(?[1-9]\d\)?[ -]?(?:9\d{4}|\d{4})[ -]?\d{4}\b", re.I
)
_INJECTION = re.compile(
    r"\b(?:ignore|desconsidere|esque[cç]a)\b.{0,70}\b(?:instru[cç][õo]es|instructions?|regras|rules?|prompt)\b"
    r"|\b(?:system prompt|developer message|chame a ferramenta|call the tool|exfiltrat\w*)\b", re.I
)
_UNSAFE_OUTPUT = re.compile(
    r"\b(?:eu diagnostico|diagn[oó]stico definitivo|definitive diagnosis|i diagnose|prescrevo|i prescribe"
    r"|tome \d|take \d|administre \d|administer \d)\b", re.I
)
_PRIVATE_REQUEST = re.compile(
    r"\b(?:qual|quais|mostr\w*|inform\w*|revele|tell|show|give|what|where|list\w*)\b.{0,60}"
    r"\b(?:cpf|rg|cns|cep|birth date|date of birth|data de nascimento|address|endere[cç]o|"
    r"telefones?|phone|e-mails?|bairro|munic[ií]pio|cidade|city|onde mora|where .{0,20} live)\b", re.I
)
_EMERGENCY = re.compile(
    r"(?:\b(?:agora|acabei de|estou com|right now|currently|just had)\b.{0,45}"
    r"\b(?:dor (?:no|de) peito|falta de ar|chest pain|shortness of breath|difficulty breathing|"
    r"hemorragia|bleeding|suic[ií]d\w*)\b)|"
    r"\b(?:n[aã]o (?:est[aá] respirando|consigo respirar)|not breathing|can't breathe|quero me matar|"
    r"want to kill myself|est[aá] inconsciente|is unconscious)\b", re.I
)
_CLINICAL_LIMIT = re.compile(
    r"\b(?:diagn[oó]stico definitivo|definitive diagnosis|prescreva|prescribe|"
    r"(?:qual|que) (?:dose|medicamento) (?:devo|deve)|what dose should)\b", re.I
)


def input_safety(question: str) -> str | None:
    """Classify explicit acute, disclosure and clinical boundary requests."""
    if _EMERGENCY.search(question):
        return "emergency"
    if _PRIVATE_REQUEST.search(question) or _INJECTION.search(question):
        return "privacy"
    if _CLINICAL_LIMIT.search(question):
        return "clinical"
    return None


def unsafe_output(text: str) -> bool:
    """Detect instructions and explicit prescriptions in output."""
    return bool(_UNSAFE_OUTPUT.search(text) or _INJECTION.search(text))


def _fold(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text.casefold()) if not unicodedata.combining(c))


def _chart_key(code: str) -> str:
    return code.replace("-", "").replace(" ", "").casefold()


class Redactor:
    """Remove known identities and Brazilian identifiers while preserving clinical data."""

    def __init__(self, names: Iterable[str], cities: Iterable[str] = ()) -> None:
        known = {_fold(value) for value in [*names, *cities] if value}
        variants = {"a": "[aáàâãä]", "e": "[eéèêë]", "i": "[iíìîï]",
                    "o": "[oóòôõö]", "u": "[uúùûü]", "c": "[cç]"}
        terms = [
            "".join(variants.get(char, r"\s+" if char == " " else re.escape(char)) for char in term)
            for term in sorted(known, key=len, reverse=True)
        ]
        self.known = re.compile(r"\b(?:" + "|".join(terms) + r")\b", re.I) if terms else None

    @staticmethod
    def mask_charts(text: str, *, preserve_chart: str | None = None) -> str:
        """Mask chart codes other than the selected chart when one is supplied."""
        allowed = _chart_key(preserve_chart) if preserve_chart else None
        return _CHART.sub(
            lambda match: match[0] if _chart_key(match[0]) == allowed else "[PRONTUÁRIO]", text
        )

    def clean(self, text: str, *, preserve_chart: str | None = None) -> str:
        """Return a minimized copy for prompts, checkpoints or presentation."""
        text = _CONTROL.sub(" ", unicodedata.normalize("NFKC", text))
        text = _BIRTH.sub("[NASCIMENTO]", text)
        text = _LOCATION.sub("[LOCAL]", text)
        if self.known:
            text = self.known.sub("[IDENTIDADE]", text)
        text = NAME_AFTER_LABEL.sub(lambda match: match[1] + " [NOME]", text)
        text = _EXTRA_ID.sub("[IDENTIFICADOR]", text)
        for kind, pattern in DEFAULT_PATTERNS:
            if kind != "DATE":
                text = pattern.sub(f"[{kind}]", text)
        return self.mask_charts(text, preserve_chart=preserve_chart).strip()

    def evidence(self, text: str, *, preserve_chart: str | None = None) -> str:
        """Accept clinical prose after excluding clear embedded instructions."""
        return "" if _INJECTION.search(text) else self.clean(text, preserve_chart=preserve_chart)
