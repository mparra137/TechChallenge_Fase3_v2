"""Consultas determinísticas sobre o acervo sintético integral."""

import re
from datetime import date
from decimal import Decimal
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


_ANALYTES = {
    "Creatinina": r"creatinin[ae]",
    "Glicemia de jejum": r"glicemia|glicose|glucose|blood sugar",
    "Hemoglobina glicada": r"hemoglobin[ae] glicada|hba1c|a1c|glycated hemoglobin",
    "Hemoglobina": r"hemoglobin[ae]",
    "Leucócitos": r"leucocitos|white blood cells",
    "Plaquetas": r"plaquetas|platelets",
    "TGO (AST)": r"tgo|ast",
    "TGP (ALT)": r"tgp|alt",
    "Colesterol LDL": r"ldl",
    "Triglicerídeos": r"triglicerideos|triglycerides",
}
_LATEST = r"\b(?:ultim[oa]s?|mais recent(?:e|es)|latest|most recent|last)\b"
_ANAMNESIS = r"\b(?:anamnese|anamnesis|medical history)\b"
_REGISTRY = r"\b(?:cadastro|patient registration)\b"
_ABDOMINAL_ULTRASOUND = r"\b(?:ultrassom|ultrassonografia)(?: d[eo])? (?:abdominal|abdome[n]?)\b|\babdominal ultrasound\b"
_RECORD_TITLE_REQUEST = re.compile(
    r"^(?:e )?(?:(?:qual(?: (?:e|foi))?|quero (?:saber|ver)|mostre|recupere|consulte|me traga|show) )?"
    r"(?:a |o |the )?(?:anamnese|anamnesis|cadastro|patient registration)\b"
)
_LATEST_ANAMNESIS = re.compile(
    r"\b(?:(?:ultima|mais recente) anamnese|anamnese mais recente|"
    r"(?:latest|most recent|last) (?:anamnesis|medical history))\b"
)
_ANAMNESIS_REFERENCE = re.compile(
    r"\b(?:d[ao]|n[ao]|pel[ao]|(?:de|em|sobre)(?: a)?|"
    r"(?:of|in|from|about|according to)(?: the)?)\s+$"
)
_DOCUMENT_ANALYSIS = re.compile(
    r"\b(?:resum\w*|sintetiz\w*|summary|summari[sz]\w*|interpret\w*|explique|explicar|explain|"
    r"analis\w*|analy[sz]\w*|compare|comparar)\b"
)
_TREND = r"\b(?:evolucao|tendencia|compar[\w]*|melhor[\w]*|pior[\w]*|aument[\w]*|diminui[\w]*|mudou|trend|changed|improv[\w]*|worsen[\w]*|increas[\w]*|decreas[\w]*)\b"
_DATES = r"\b(?:\d{4}-\d\d-\d\d|\d\d/\d\d/\d{4})\b"
_EXAM_TYPES = ("exame_laboratorial", "laudo_imagem")
_EXAM_LIST = re.compile(
    r"^(?:quais (?:foram |sao )?|(?:mostre|liste|me traga) )(?:os )?exames"
    r"(?: (?:foram )?realizados)?(?: (?:pelo|pela|do|da) (?:paciente )?.+)?$"
)
_RECENT_EXAM_COUNT = re.compile(
    r"\b(?:ultimos ([1-9]\d*) exames|([1-9]\d*) ultimos exames|"
    r"([1-9]\d*) exames mais recentes|(?:last|latest|most recent) ([1-9]\d*) tests)\b"
)
_EXAM_MENTION = re.compile(r"\b(?:exames|tests)\b")
_RECORD_RESULT = re.compile(
    r"^(?:e )?(?:qual(?: (?:foi|e))?|quais(?: (?:foram|sao))?|mostre|recupere|me traga|what (?:is|was|were)|show)"
    r" (?:o |os |the )?(?:resultado[s]?|results?)\b"
)


def requests_record_document(question: str) -> bool:
    """Recognize a standalone request for documented data, without a patient name."""
    query = normalize_patient_name(question).strip(" ?.")
    if _RECENT_EXAM_COUNT.search(query) or _EXAM_LIST.fullmatch(query):
        return True
    if _RECORD_TITLE_REQUEST.search(query):
        return True
    if re.search(_LATEST, query) and (
        re.search(_ANAMNESIS, query) or re.search(_ABDOMINAL_ULTRASOUND, query) or clinical_analytes(query)
    ):
        return True
    if _RECORD_RESULT.search(query) and re.search(_ABDOMINAL_ULTRASOUND, query):
        return True
    return bool(
        re.match(r"^(?:resuma|sintetize|interprete|analise|compare|summari[sz]e|interpret|analy[sz]e)\b", query)
        and (re.search(_ANAMNESIS, query) or re.search(_ABDOMINAL_ULTRASOUND, query))
    )


def requests_document_analysis(question: str) -> bool:
    """Identify explicit synthesis, comparison or interpretation of evidence."""
    return bool(_DOCUMENT_ANALYSIS.search(normalize_patient_name(question)))


def recent_exam_count(question: str) -> int | None:
    """Return the requested number of recent examinations when explicit."""
    match = _RECENT_EXAM_COUNT.search(normalize_patient_name(question))
    return int(next(count for count in match.groups() if count)) if match else None


def _requests_latest_anamnesis_document(query: str) -> bool:
    """Recognize direct requests regardless of their introductory wording."""
    document = _LATEST_ANAMNESIS.search(query)
    if document is None:
        return False
    return not (
        _ANAMNESIS_REFERENCE.search(query[:document.start()])
        or _DOCUMENT_ANALYSIS.search(query)
        or re.search(r"\b(?:apenas|so|somente|only)\b", query[document.end():])
    )


def clinical_analytes(question: str) -> list[str]:
    """Recognize lab analytes in the synthetic panels in Portuguese and English."""
    normalized = normalize_patient_name(question)
    normalized = re.sub(r"\[(?:identidade|nome|prontuario)\]", " ", normalized)
    matches = []
    for analyte, pattern in _ANALYTES.items():
        normalized, count = re.subn(r"\b(?:" + pattern + r")\b", " ", normalized)
        if count:
            matches.append(analyte)
    return matches


def _patient_measurements(query: str, patient_id: str, loader: HospitalDocumentLoader
                          ) -> tuple[str, list[RetrievedDocument]] | None:
    """Select recorded measurements instead of estimating dates or trends using top-K."""
    analytes = clinical_analytes(query)
    dates = re.findall(_DATES, query)
    trend = bool(re.search(_TREND, query))
    if not analytes:
        return ("", []) if trend else None
    if len(analytes) != 1 or len(dates) > 1 or re.search(
        r"\b(?:entre|desde|ate|meses|semanas|between|since|until|months|weeks)\b", query
    ):
        return "", []
    day = None
    if dates:
        day = date.fromisoformat("-".join(reversed(dates[0].split("/"))) if "/" in dates[0] else dates[0])
    rows = loader.load_documents()
    observations = [
        (row, result)
        for row in rows if row["patient_id"] == patient_id and row["document_type"] == "exame_laboratorial"
        and (day is None or date.fromisoformat(row["document_date"]) == day)
        for result in row["metadata"].get("results", [])
        if result["name"] == analytes[0]
    ]
    if not observations:
        return "", []
    observations.sort(key=lambda pair: (pair[0]["document_date"], pair[0]["document_id"]))
    if not trend:
        newest = observations[-1][0]["document_date"]
        observations = [(row, result) for row, result in observations if row["document_date"] == newest]
    documents = [
        RetrievedDocument(source=DocumentSource.model_validate(row),
                          content=f'{result["name"]}: {result["value"]} {result["unit"]}')
        for row, result in observations
    ]
    conclusion = ""
    if trend:
        units = {result["unit"] for _, result in observations}
        days = {row["document_date"] for row, _ in observations}
        if len(units) == 1 and len(days) == len(observations) and len(days) >= 2:
            change = Decimal(str(observations[-1][1]["value"])) - Decimal(str(observations[0][1]["value"]))
            conclusion = f"{change:+f} {observations[0][1]['unit']}"
    return conclusion, documents


def _exam_documents(patient_id: str, loader: HospitalDocumentLoader, *, listing: bool) -> list[RetrievedDocument]:
    rows = sorted(
        (row for row in loader.load_documents()
         if row["patient_id"] == patient_id and row["document_type"] in _EXAM_TYPES),
        key=lambda row: (row["document_date"], row["document_id"]),
        reverse=True,
    )
    return [
        RetrievedDocument(
            source=DocumentSource.model_validate(row),
            content=(row.get("metadata", {}).get("exam_type") or row["document_type"].replace("_", " "))
            if listing else row["content"],
        )
        for row in rows
    ]


def _selected_documents(
    query: str, patient_id: str, loader: HospitalDocumentLoader
) -> list[RetrievedDocument] | None:
    """Select registry or abdominal ultrasound records by identity, type and date."""
    if re.search(_ABDOMINAL_ULTRASOUND, query):
        document_type = "laudo_imagem"
    elif re.search(_REGISTRY, query):
        document_type = "cadastro"
    else:
        return None
    dates = re.findall(_DATES, query)
    if len(dates) > 1:
        raise ValueError("Informe no máximo uma data específica.")
    rows = [
        row for row in loader.load_documents()
        if row["patient_id"] == patient_id and row["document_type"] == document_type
        and (document_type == "cadastro" or re.search(
            _ABDOMINAL_ULTRASOUND, normalize_patient_name(row.get("metadata", {}).get("exam_type") or "")
        ))
    ]
    if dates:
        day = "-".join(reversed(dates[0].split("/"))) if "/" in dates[0] else dates[0]
        day = date.fromisoformat(day).isoformat()
        rows = [row for row in rows if row["document_date"] == day]
    elif rows and (re.search(_LATEST, query) or not _DOCUMENT_ANALYSIS.search(query)):
        latest = max(row["document_date"] for row in rows)
        rows = [row for row in rows if row["document_date"] == latest]
    return [RetrievedDocument(source=DocumentSource.model_validate(row), content=row["content"]) for row in rows]


def exact_query(question: str, patient: PatientResolution, document_type: str | None,
                loader: HospitalDocumentLoader, *, scoped: bool = False
                ) -> tuple[str, list[RetrievedDocument]] | None:
    """Count patients or select records by date without making top-K population claims."""
    query = normalize_patient_name(question).strip(" ?.")
    if scoped:
        if not patient.patient_id or re.search(
            r"\b(?:quais pacientes|todos os pacientes|all patients|which patients)\b", query
        ):
            return "", []
        analysis = bool(_DOCUMENT_ANALYSIS.search(query))
        analytes = clinical_analytes(query)
        if not analytes:
            selected = _selected_documents(query, patient.patient_id, loader)
            if selected is not None:
                return "", selected
        limit = recent_exam_count(query)
        if limit is not None and not re.search(_DATES, query) and not analytes:
            exam_type = document_type
            if exam_type is None:
                if re.search(r"\b(?:laboratoriais|laboratory)\b", query):
                    exam_type = "exame_laboratorial"
                elif re.search(r"\b(?:de imagem|imaging)\b", query):
                    exam_type = "laudo_imagem"
            docs = _exam_documents(patient.patient_id, loader, listing=not analysis)
            if exam_type:
                docs = [doc for doc in docs if doc.source.document_type == exam_type]
            return "", docs[:limit]
        if not analysis and _EXAM_LIST.fullmatch(query) and not re.search(_DATES, query):
            return "", _exam_documents(patient.patient_id, loader, listing=True)
        if analysis and _EXAM_MENTION.search(query) and not analytes and not re.search(_DATES, query):
            docs = _exam_documents(patient.patient_id, loader, listing=False)
            return "\n\n".join(f"[{doc.source.document_id}] {doc.content}" for doc in docs), docs
        measurements = _patient_measurements(query, patient.patient_id, loader)
        if measurements is not None:
            return measurements
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
    dates = re.findall(_DATES, query)
    latest = bool(re.search(_LATEST, query))
    if not dates and not latest:
        return None
    if not patient.patient_id or len(dates) > 1:
        raise ValueError("Informe um prontuário e no máximo uma data específica.")
    if not document_type:
        if re.search(r"glic|hemoglobina|hemograma|laborator", query):
            document_type = "exame_laboratorial"
        elif re.search(_ANAMNESIS, query):
            document_type = "anamnese"
        elif "imagem" in query:
            document_type = "laudo_imagem"
    rows = [r for r in loader.load_documents() if r["patient_id"] == patient.patient_id
            and (not document_type or r["document_type"] == document_type)]
    if not document_type and re.search(r"\b(?:exames?|tests?)\b", query):
        rows = [r for r in rows if r["document_type"] in {"laudo_imagem", "exame_laboratorial"}]
    if dates:
        day = "-".join(reversed(dates[0].split("/"))) if "/" in dates[0] else dates[0]
        rows = [r for r in rows if r["document_date"] == date.fromisoformat(day).isoformat()]
    if latest and rows:
        day = max(r["document_date"] for r in rows)
        rows = [r for r in rows if r["document_date"] == day]
    docs = [RetrievedDocument(source=DocumentSource.model_validate(row), content=row["content"]) for row in rows]
    if scoped and document_type == "anamnese" and _requests_latest_anamnesis_document(query):
        # The agent can render the selected records with its redaction and citation checks.
        return "", docs
    answer = "\n\n".join(f"[{d.source.document_id}] {d.content}" for d in docs)
    return answer, docs
