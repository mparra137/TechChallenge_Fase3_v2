"""Hybrid classification and language hints for the conversational graph."""
from __future__ import annotations

import re
from typing import Literal

from src.retrieval.patient_resolver import normalize_patient_name
from src.retrieval.structured import requests_record_document

Intent = Literal["GENERAL_MEDICAL", "PATIENT_RECORD", "CLARIFICATION", "SAFETY"]
Language = Literal["pt", "en"]
ROUTE_LABELS: dict[int, Intent] = {
    0: "CLARIFICATION", 1: "GENERAL_MEDICAL", 2: "PATIENT_RECORD", 3: "SAFETY",
}
_FOLLOW_UP = re.compile(
    r"\b(?:dele|dela|desse paciente|dessa paciente|esses resultados|resultados anteriores|os anteriores|"
    r"(?:nesse|neste|esse) prontuario|his|hers|her|their|this patient|that patient|"
    r"(?:this|that) chart|the chart we just discussed|these results|previous results|the previous ones)\b"
)
_HYPOTHETICAL = re.compile(r"^(?:se (?:um|uma)|if (?:a|an)|hipoteticamente|hypothetically)\b")
_GENERAL = re.compile(
    r"^(?:o que e|o que significa|quais (?:sao )?(?:as )?causas|qual a diferenca|qual o tratamento (?:indicado )?para|"
    r"what is|what does|what causes|what are the causes|how is)\b"
)
_CHART_CUE = re.compile(r"\b(?:prontuario|paciente|patient|chart|medical record|exames? de|records? for)\b")
_RECORD_REFERENCE = re.compile(r"\b(?:d[eo] paciente|da paciente|do prontuario|of the patient|(?:this|that|the) chart)\b")
_ENGLISH = re.compile(
    r"\b(?:what|when|where|which|how|does|is|were|was|the|if|latest|previous|"
    r"show|please|patient|help|and|his|her|their|blood|results|creatinine|can|should)\b"
)


def language_for(question: str, previous: Language = "pt") -> Language:
    """Infer the response language for the supported languages and short follow-ups."""
    normalized = normalize_patient_name(question)
    if _ENGLISH.search(normalized):
        return "en"
    if re.search(r"\b(?:o|a|e|qual|quais|mostre|paciente|exames?|dele|mudou|creatinina)\b", normalized):
        return "pt"
    return previous


def is_follow_up(question: str) -> bool:
    """Require a contextual pronoun or explicit reference to previous results."""
    return bool(_FOLLOW_UP.search(normalize_patient_name(question)))


def deterministic_route(question: str, explicit: bool, pending: bool) -> Intent | None:
    """Recognize only cases with a clear record scope or general intent."""
    normalized = normalize_patient_name(question)
    if _HYPOTHETICAL.search(normalized) and not explicit:
        return "GENERAL_MEDICAL"
    if explicit or pending or is_follow_up(question) or _RECORD_REFERENCE.search(normalized):
        return "PATIENT_RECORD"
    if requests_record_document(normalized):
        return "PATIENT_RECORD"
    if _GENERAL.search(normalized):
        return "GENERAL_MEDICAL"
    if _CHART_CUE.search(normalized):
        return "PATIENT_RECORD"
    return None


def route_for_label(label: int) -> Intent:
    """Map a validated label to a route, or require clarification."""
    return ROUTE_LABELS.get(label, "CLARIFICATION") if type(label) is int else "CLARIFICATION"
