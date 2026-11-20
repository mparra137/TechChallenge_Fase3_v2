"""Consultas determinísticas sobre o acervo sintético integral."""

import re
from datetime import date
from statistics import mean

from src.ingestion.document_loader import HospitalDocumentLoader
from src.models.schemas import DocumentSource, PatientResolution, RetrievedDocument
from src.retrieval.patient_resolver import normalize_patient_name


def _glycemia_summary(query: str, rows: list[dict]) -> tuple[str, list[RetrievedDocument]]:
    """Aggregate fasting glucose observations with the same recorded unit."""
    observations = []
    for row in rows:
        for result in row.get("metadata", {}).get("results", []):
            if result["name"] == "Glicemia de jejum" and result["unit"] == "mg/dL":
                observations.append((row, result))
    if not observations:
        return "", []
    values = [result["value"] for _, result in observations]
    if "media" in query:
        answer = f"Média das {len(values)} medições de glicemia de jejum: {mean(values):.2f} mg/dL."
    elif "maxim" in query or "maior" in query:
        maximum = max(values)
        observations = [(row, r) for row, r in observations if r["value"] == maximum]
        answer = f"Maior glicemia de jejum registrada: {maximum} mg/dL."
    else:
        selected = []
        for row, result in observations:
            interval = re.fullmatch(r"\s*\d+(?:,\d+)?\s*[–-]\s*(\d+(?:,\d+)?) mg/dL", result["reference"])
            if not interval:
                raise ValueError("Intervalo de referência documental não comparável.")
            if result["value"] > float(interval[1].replace(",", ".")):
                selected.append((row, result))
        observations = selected
        identifiers = sorted({row["patient_id"] for row, _ in observations})
        answer = f"Pacientes com ao menos uma glicemia acima da referência registrada ({len(identifiers)}): "
        answer += ", ".join(identifiers) or "nenhum"
    docs = [RetrievedDocument(source=DocumentSource.model_validate(row), content=row["content"])
            for row, _ in observations]
    return answer, docs


def exact_query(question: str, patient: PatientResolution, document_type: str | None,
                loader: HospitalDocumentLoader) -> tuple[str, list[RetrievedDocument]] | None:
    """Count patients or select records by date without making top-K population claims."""
    query = normalize_patient_name(question).strip(" ?.")
    if re.fullmatch(r"quantos pacientes(?: (?:existem|ha|estao cadastrados|cadastrados|na base))*", query):
        return f"Total no cadastro sintético: {len(loader.load_patients())} pacientes.", []
    if document_type in (None, "exame_laboratorial") and re.fullmatch(
        r"(?:qual (?:e|foi) a (?:media da|maior) glicemia(?: de jejum)?(?: na base)?"
        r"|quais pacientes (?:possuem registros de|tem) glicemia elevada)", query
    ):
        rows = [r for r in loader.load_documents() if not patient.patient_id or r["patient_id"] == patient.patient_id]
        return _glycemia_summary(query, rows)
    if re.search(r"\b(quais pacientes|quantos|quantas|media|maximo|maior|menor|todos os pacientes)\b", query):
        raise ValueError("Essa agregação requer critério e consulta estruturada específicos.")
    dates = re.findall(r"\b(?:\d{4}-\d\d-\d\d|\d\d/\d\d/\d{4})\b", query)
    latest = bool(re.search(r"\b(?:ultim[oa]s?|mais recente)\b", query))
    if not dates and not latest:
        return None
    if not patient.patient_id or len(dates) > 1:
        raise ValueError("Informe um prontuário e no máximo uma data específica.")
    if not document_type:
        if re.search(r"glic|hemoglobina|hemograma|laborator", query):
            document_type = "exame_laboratorial"
        elif "anamnese" in query:
            document_type = "anamnese"
        elif "imagem" in query:
            document_type = "laudo_imagem"
    rows = [r for r in loader.load_documents() if r["patient_id"] == patient.patient_id
            and (not document_type or r["document_type"] == document_type)]
    if not document_type and "exame" in query:
        rows = [r for r in rows if r["document_type"] in {"laudo_imagem", "exame_laboratorial"}]
    if dates:
        day = "-".join(reversed(dates[0].split("/"))) if "/" in dates[0] else dates[0]
        rows = [r for r in rows if r["document_date"] == date.fromisoformat(day).isoformat()]
    if latest and rows:
        day = max(r["document_date"] for r in rows)
        rows = [r for r in rows if r["document_date"] == day]
    docs = [RetrievedDocument(source=DocumentSource.model_validate(row), content=row["content"]) for row in rows]
    answer = "\n\n".join(f"[{d.source.document_id}] {d.content}" for d in docs)
    return answer, docs
