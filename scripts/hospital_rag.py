"""
Gerador de base hospitalar sintética + ingestão RAG com LangChain.

Uso:
  python scripts/hospital_rag.py generate
  python scripts/hospital_rag.py ingest
  python scripts/hospital_rag.py query "Qual foi a evolução da glicemia?" --patient-id PCT-000001

Os dados são inteiramente sintéticos e destinados a fins acadêmicos/testes.
Não usar para decisões clínicas, treinamento clínico sem revisão ou produção.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import uuid
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

# Garante que a raiz do projeto esteja no sys.path para execução direta do script
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src import config

SEED = config.RAG_DEFAULT_SEED
N_PATIENTS = config.RAG_DEFAULT_N_PATIENTS
OUTPUT_DIR = config.RAG_DATA_DIR
JSONL_PATH = config.RAG_JSONL_PATH
PATIENTS_PATH = config.RAG_PATIENTS_PATH
CHROMA_DIR = config.RAG_CHROMA_DIR
COLLECTION_NAME = config.RAG_COLLECTION_NAME
EMBEDDING_MODEL = config.RAG_EMBEDDING_MODEL
SCHEMA_VERSION = "1.0"

# Valores ilustrativos para simulação. Intervalos de referência podem variar
# conforme laboratório, método, idade, sexo e contexto clínico.
FIRST_NAMES = [
    "Ana", "Bruno", "Carla", "Daniel", "Eduarda", "Felipe", "Gabriela", "Henrique",
    "Isabela", "João", "Larissa", "Marcos", "Natália", "Otávio", "Patrícia", "Rafael",
    "Sofia", "Tiago", "Vanessa", "William", "Aline", "Caio", "Débora", "Eduardo",
    "Fernanda", "Gustavo", "Helena", "Igor", "Juliana", "Lucas", "Mariana", "Nicolas",
    "Beatriz", "Bernardo", "Camila", "Davi", "Elisa", "Enzo", "Flávia", "Heitor",
    "Iara", "Júlia", "Leonardo", "Manuela", "Murilo", "Noemi", "Otto", "Paula",
    "Renata", "Samuel", "Talita", "Vitor", "Yasmin", "Adriana", "Alexandre", "Bianca",
    "Cecília", "Diego", "Emanuel", "Fábio", "Giovana", "Hugo", "Ítalo", "Joana",
    "Kaique", "Lívia", "Maitê", "Nelson", "Olívia", "Pedro", "Raquel", "Sérgio",
    "Tainá", "Valentina", "Wesley", "Zélia", "André", "Bárbara", "César", "Diana",
    "Ester", "Francisco", "Geraldo", "Isadora", "José", "Leandro", "Mônica", "Rita",
    "Roberto", "Simone", "Tatiane", "Ubirajara", "Vicente", "Alexia", "Cristiano",
    "Danilo", "Elaine", "Milena", "Priscila", "Rodrigo", "Sara", "Tomás", "Vinícius",
]
LAST_NAMES = [
    "Silva", "Santos", "Oliveira", "Souza", "Pereira", "Costa", "Rodrigues", "Almeida",
    "Nascimento", "Lima", "Araújo", "Fernandes", "Carvalho", "Gomes", "Martins",
    "Ribeiro", "Alves", "Monteiro", "Mendes", "Barros", "Freitas", "Pinto", "Moura",
    "Cavalcanti", "Dias", "Castro", "Cardoso", "Campos", "Teixeira", "Moreira",
    "Correia", "Vieira", "Rocha", "Reis", "Machado", "Andrade", "Batista", "Nunes",
    "Melo", "Ramos", "Cunha", "Azevedo", "Farias", "Tavares", "Lopes", "Miranda",
    "Gonçalves", "Duarte", "Rezende", "Peixoto", "Borges", "Macedo", "Queiroz",
    "Coelho", "Vasconcelos", "Moraes", "Medeiros", "Brito", "Leal", "Pacheco",
    "Siqueira", "Amaral", "Assis", "Abreu", "Figueiredo", "Pires", "Coutinho",
    "Fonseca", "Garcia", "Mota", "Paiva", "Porto", "Lacerda", "Dantas",
    "Bittencourt", "Viana", "Peçanha", "Menezes", "Goulart", "Seabra", "Falcão",
    "Müller", "Bernardes", "Cordeiro", "Xavier", "Bastos", "Matos", "Aguiar",
    "Câmara", "Brandão", "Esteves", "Franco", "Saldanha", "Sampaio",
    "Loureiro", "Trindade", "Valente", "Prado", "Serrano", "Rangel", "Noronha",
]

CITIES = [
    "São Paulo", "Santo André", "São Bernardo do Campo", "São Caetano do Sul",
    "Diadema", "Santos", "Campinas", "Sorocaba", "Jundiaí", "São José dos Campos",
]
SPECIALTIES = [
    "clínica médica", "cardiologia", "endocrinologia", "gastroenterologia",
    "nefrologia", "pneumologia", "ortopedia", "neurologia",
]
CONDITIONS = [
    "hipertensão arterial", "diabetes tipo 2", "dislipidemia",
    "esteatose hepática", "doença renal crônica estágio 2",
    "asma leve", "lombalgia mecânica", "refluxo gastroesofágico",
]
CONDITION_WEIGHTS = [20, 17, 15, 12, 8, 8, 10, 10]
BLOOD_TYPES = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
MEDICATIONS = {
    "hipertensão arterial": ["losartana 50 mg", "anlodipino 5 mg"],
    "diabetes tipo 2": ["metformina 500 mg", "metformina 850 mg"],
    "dislipidemia": ["sinvastatina 20 mg", "atorvastatina 10 mg"],
    "esteatose hepática": ["orientação de mudança de estilo de vida"],
    "doença renal crônica estágio 2": ["orientação de acompanhamento nefrológico"],
    "asma leve": ["salbutamol spray, se necessário"],
    "lombalgia mecânica": ["analgesia conforme avaliação médica"],
    "refluxo gastroesofágico": ["omeprazol 20 mg"],
}

def iso(d: date) -> str:
    return d.isoformat()

def age_on(birth: date, at: date) -> int:
    return at.year - birth.year - ((at.month, at.day) < (birth.month, birth.day))

def rand_date(rng: random.Random, start: date, end: date) -> date:
    return start + timedelta(days=rng.randint(0, (end - start).days))

def jsonl_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

def normalize_name(name: str) -> str:
    """Normaliza acentos, caixa e espaços para comparação de nomes."""
    import unicodedata
    normalized = unicodedata.normalize("NFKD", name)
    normalized = "".join(c for c in normalized if not unicodedata.combining(c))
    return " ".join(normalized.casefold().split())


def make_unique_name(rng: random.Random, used_names: set[str]) -> str:
    """Gera nome completo único dentro da base."""
    for _ in range(10000):
        first = rng.choice(FIRST_NAMES)
        last1, last2 = rng.sample(LAST_NAMES, 2)
        candidate = f"{first} {last1}" if rng.random() < 0.45 else f"{first} {last1} {last2}"
        key = normalize_name(candidate)
        if key not in used_names:
            used_names.add(key)
            return candidate
    raise RuntimeError("Não foi possível gerar um nome único.")


def make_patient(
    rng: random.Random, i: int, today: date, used_names: set[str]
) -> dict[str, Any]:
    birth = rand_date(rng, date(1940, 1, 1), date(2005, 12, 31))
    sex = rng.choice(["Feminino", "Masculino"])
    condition = rng.choices(CONDITIONS, weights=CONDITION_WEIGHTS, k=1)[0]
    return {
        "patient_id": f"PCT-{i:06d}",
        "full_name": make_unique_name(rng, used_names),
        "birth_date": iso(birth),
        "age_at_generation": age_on(birth, today),
        "sex": sex,
        "blood_type": rng.choice(BLOOD_TYPES),
        "city": rng.choice(CITIES),
        "allergies": rng.choice([[], [], ["nenhuma alergia conhecida"], ["penicilina"], ["dipirona"]]),
        "family_history": rng.choice([
            "histórico familiar de hipertensão arterial",
            "histórico familiar de diabetes tipo 2",
            "histórico familiar de doença cardiovascular",
            "sem histórico familiar relevante informado",
        ]),
        "primary_condition": condition,
        "created_at": iso(today),
    }

def patient_profile_document(p: dict[str, Any]) -> dict[str, Any]:
    content = (
        f"Cadastro sintético do paciente {p['patient_id']}. "
        f"Nome fictício: {p['full_name']}. Data de nascimento: {p['birth_date']}. "
        f"Idade na geração da base: {p['age_at_generation']} anos. Sexo: {p['sex']}. "
        f"Tipo sanguíneo informado: {p['blood_type']}. Município: {p['city']}. "
        f"Alergias registradas: {', '.join(p['allergies']) if p['allergies'] else 'não informado'}. "
        f"Histórico familiar: {p['family_history']}. "
        f"Condição principal simulada para este cenário: {p['primary_condition']}."
    )
    return {
        "document_id": f"DOC-{p['patient_id']}-CAD",
        "patient_id": p["patient_id"],
        "document_type": "cadastro",
        "document_date": p["created_at"],
        "source": f"Prontuário {p['patient_id']}",
        "content": content,
        "metadata": {"schema_version": SCHEMA_VERSION, "record_id": p["patient_id"]},
    }

def lab_values(rng: random.Random, condition: str) -> list[dict[str, Any]]:
    # A condição influencia alguns valores para que exista coerência temática,
    # sem pretender reproduzir relações clínicas reais com validade diagnóstica.
    glucose = rng.randint(88, 105)
    hba1c = round(rng.uniform(5.1, 5.7), 1)
    ldl = rng.randint(90, 135)
    triglycerides = rng.randint(90, 160)
    alt = rng.randint(18, 42)
    creatinine = round(rng.uniform(0.7, 1.1), 2)

    if condition == "diabetes tipo 2":
        glucose = rng.randint(125, 185)
        hba1c = round(rng.uniform(6.6, 8.4), 1)
    elif condition == "dislipidemia":
        ldl = rng.randint(145, 205)
        triglycerides = rng.randint(170, 280)
    elif condition == "esteatose hepática":
        alt = rng.randint(45, 82)
    elif condition == "doença renal crônica estágio 2":
        creatinine = round(rng.uniform(1.15, 1.45), 2)

    return [
        {"name": "Hemoglobina", "value": round(rng.uniform(12.5, 16.5), 1), "unit": "g/dL", "reference": "12,0–17,5 g/dL"},
        {"name": "Leucócitos", "value": rng.randint(4500, 10500), "unit": "/mm³", "reference": "4.000–11.000 /mm³"},
        {"name": "Plaquetas", "value": rng.randint(160000, 390000), "unit": "/mm³", "reference": "150.000–450.000 /mm³"},
        {"name": "TGO (AST)", "value": rng.randint(18, 39), "unit": "U/L", "reference": "10–40 U/L"},
        {"name": "TGP (ALT)", "value": alt, "unit": "U/L", "reference": "7–56 U/L"},
        {"name": "Creatinina", "value": creatinine, "unit": "mg/dL", "reference": "0,7–1,3 mg/dL"},
        {"name": "Glicemia de jejum", "value": glucose, "unit": "mg/dL", "reference": "70–99 mg/dL"},
        {"name": "Hemoglobina glicada", "value": hba1c, "unit": "%", "reference": "4,0–5,6%"},
        {"name": "Colesterol LDL", "value": ldl, "unit": "mg/dL", "reference": "alvo depende do risco cardiovascular"},
        {"name": "Triglicerídeos", "value": triglycerides, "unit": "mg/dL", "reference": "inferior a 150 mg/dL"},
    ]

def lab_document(rng: random.Random, p: dict[str, Any], n: int, when: date) -> dict[str, Any]:
    values = lab_values(rng, p["primary_condition"])
    lines = [f"{x['name']}: {x['value']} {x['unit']} (referência informativa: {x['reference']})" for x in values]
    content = (
        f"Painel laboratorial sintético do paciente {p['patient_id']}, coletado em {iso(when)}. "
        + "; ".join(lines)
        + ". Resultados fictícios para teste de recuperação documental; não representam avaliação clínica."
    )
    return {
        "document_id": f"LAB-{p['patient_id']}-{n:02d}",
        "patient_id": p["patient_id"],
        "document_type": "exame_laboratorial",
        "document_date": iso(when),
        "source": f"Prontuário {p['patient_id']} / Exame LAB-{p['patient_id']}-{n:02d}",
        "content": content,
        "metadata": {
            "exam_type": "painel laboratorial",
            "results": values,
            "schema_version": SCHEMA_VERSION,
        },
    }

def encounter_document(rng: random.Random, p: dict[str, Any], n: int, when: date) -> dict[str, Any]:
    specialty = {
        "hipertensão arterial": "cardiologia",
        "diabetes tipo 2": "endocrinologia",
        "dislipidemia": "clínica médica",
        "esteatose hepática": "gastroenterologia",
        "doença renal crônica estágio 2": "nefrologia",
        "asma leve": "pneumologia",
        "lombalgia mecânica": "ortopedia",
        "refluxo gastroesofágico": "gastroenterologia",
    }[p["primary_condition"]]
    complaint = {
        "hipertensão arterial": "retorno para acompanhamento da pressão arterial",
        "diabetes tipo 2": "acompanhamento do controle glicêmico",
        "dislipidemia": "avaliação de perfil lipídico",
        "esteatose hepática": "desconforto abdominal inespecífico e revisão de exames",
        "doença renal crônica estágio 2": "acompanhamento da função renal",
        "asma leve": "revisão de sintomas respiratórios intermitentes",
        "lombalgia mecânica": "dor lombar relacionada a esforço físico",
        "refluxo gastroesofágico": "episódios de azia e regurgitação",
    }[p["primary_condition"]]
    history = (
        f"Paciente comparece para {complaint}. "
        f"Condição acompanhada neste cenário sintético: {p['primary_condition']}. "
        f"Histórico familiar registrado: {p['family_history']}. "
        "Relata adesão variável às orientações de estilo de vida. "
        "Nega sinais de alarme no roteiro fictício deste atendimento."
    )
    content = (
        f"Anamnese do atendimento ATD-{p['patient_id']}-{n:02d}, paciente {p['patient_id']}, "
        f"data {iso(when)}, especialidade {specialty}. Queixa principal: {complaint}. "
        f"História da doença atual: {history} "
        "Exame físico: registro narrativo simulado, sem achados agudos relevantes no cenário. "
        "Avaliação: acompanhamento da condição previamente registrada, com necessidade de correlação "
        "com avaliação presencial e exames. Plano: revisar exames, reforçar orientações e programar retorno."
    )
    return {
        "document_id": f"ATD-{p['patient_id']}-{n:02d}",
        "patient_id": p["patient_id"],
        "document_type": "anamnese",
        "document_date": iso(when),
        "source": f"Prontuário {p['patient_id']} / Atendimento ATD-{p['patient_id']}-{n:02d}",
        "content": content,
        "metadata": {
            "specialty": specialty,
            "appointment_id": f"ATD-{p['patient_id']}-{n:02d}",
            "schema_version": SCHEMA_VERSION,
        },
    }

def imaging_document(rng: random.Random, p: dict[str, Any], n: int, when: date) -> dict[str, Any]:
    options = [
        ("ultrassonografia abdominal", "Fígado com dimensões preservadas e discreto aumento difuso da ecogenicidade. Não são descritas lesões focais no cenário fictício.", "Discretos sinais compatíveis com esteatose hepática leve no contexto simulado."),
        ("radiografia de tórax", "Campos pulmonares sem opacidades focais descritas. Silhueta cardíaca sem aumento evidente no texto sintético.", "Sem alteração radiográfica aguda descrita no laudo fictício."),
        ("ressonância magnética de coluna lombar", "Discreta desidratação discal em níveis lombares inferiores, sem descrição de compressão significativa de estruturas neurais.", "Alterações degenerativas leves no cenário simulado."),
        ("tomografia computadorizada de crânio", "Parênquima encefálico sem alteração aguda descrita. Sistema ventricular com dimensões preservadas no texto fictício.", "Sem achado intracraniano agudo descrito no laudo sintético."),
    ]
    exam, description, conclusion = rng.choice(options)
    content = (
        f"Laudo sintético de {exam}, documento IMG-{p['patient_id']}-{n:02d}, "
        f"paciente {p['patient_id']}, realizado em {iso(when)}. "
        f"Indicação clínica: investigação complementar no contexto do acompanhamento de {p['primary_condition']}. "
        f"Descrição: {description} Conclusão: {conclusion} "
        "Este laudo foi gerado artificialmente e não corresponde a exame real."
    )
    return {
        "document_id": f"IMG-{p['patient_id']}-{n:02d}",
        "patient_id": p["patient_id"],
        "document_type": "laudo_imagem",
        "document_date": iso(when),
        "source": f"Prontuário {p['patient_id']} / Imagem IMG-{p['patient_id']}-{n:02d}",
        "content": content,
        "metadata": {"exam_type": exam, "schema_version": SCHEMA_VERSION},
    }

def plan_document(p: dict[str, Any], n: int, when: date) -> dict[str, Any]:
    medication = MEDICATIONS[p["primary_condition"]][0]
    content = (
        f"Registro de plano e evolução EVOL-{p['patient_id']}-{n:02d}, paciente {p['patient_id']}, "
        f"data {iso(when)}. Condição em acompanhamento: {p['primary_condition']}. "
        f"Medicação ou orientação registrada no cenário fictício: {medication}. "
        "Conduta: manter acompanhamento médico, revisar resposta clínica e resultados de exames, "
        "e reavaliar a estratégia no retorno. Esta informação é sintética e não constitui prescrição."
    )
    return {
        "document_id": f"EVOL-{p['patient_id']}-{n:02d}",
        "patient_id": p["patient_id"],
        "document_type": "evolucao_plano",
        "document_date": iso(when),
        "source": f"Prontuário {p['patient_id']} / Evolução EVOL-{p['patient_id']}-{n:02d}",
        "content": content,
        "metadata": {"schema_version": SCHEMA_VERSION, "plan_type": "orientação simulada"},
    }

def generate_dataset(n_patients: int = N_PATIENTS, seed: int = SEED, output_dir: Path = OUTPUT_DIR):
    rng = random.Random(seed)
    today = date.today()
    start = today - timedelta(days=365 * 2)
    patients, documents = [], []
    used_names: set[str] = set()

    for i in range(1, n_patients + 1):
        p = make_patient(rng, i, today, used_names)
        patients.append(p)
        documents.append(patient_profile_document(p))

        # Densidade média: 6 a 12 documentos por paciente, incluindo cadastro.
        target = rng.randint(6, 12)
        remaining = target - 1
        encounter_count = max(2, min(3, remaining // 3))
        lab_count = max(2, min(4, remaining // 3))
        imaging_count = 1 if remaining < 8 else 2
        plan_count = max(0, remaining - encounter_count - lab_count - imaging_count)

        # Datas distribuídas e ordenadas para permitir consultas longitudinais.
        dates = sorted(rand_date(rng, start, today) for _ in range(
            encounter_count + lab_count + imaging_count + plan_count
        ))
        k = 0
        for j in range(1, encounter_count + 1):
            documents.append(encounter_document(rng, p, j, dates[k])); k += 1
        for j in range(1, lab_count + 1):
            documents.append(lab_document(rng, p, j, dates[k])); k += 1
        for j in range(1, imaging_count + 1):
            documents.append(imaging_document(rng, p, j, dates[k])); k += 1
        for j in range(1, plan_count + 1):
            documents.append(plan_document(p, j, dates[k])); k += 1

    # Nome denormalizado em cada documento para filtros e rastreabilidade.
    patient_names = {p["patient_id"]: p["full_name"] for p in patients}
    for document in documents:
        document["patient_name"] = patient_names[document["patient_id"]]

    documents.sort(key=lambda d: (d["patient_id"], d["document_date"], d["document_id"]))
    jsonl_write(output_dir / "patients.jsonl", patients)
    jsonl_write(output_dir / "medical_documents.jsonl", documents)
    report = validate_dataset(patients, documents)
    (output_dir / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return patients, documents, report

def validate_dataset(patients: list[dict[str, Any]], documents: list[dict[str, Any]]) -> dict[str, Any]:
    errors = []
    patient_ids = [p.get("patient_id") for p in patients]
    doc_ids = [d.get("document_id") for d in documents]

    if len(patient_ids) != len(set(patient_ids)):
        errors.append("Há patient_id duplicado.")
    if len(doc_ids) != len(set(doc_ids)):
        errors.append("Há document_id duplicado.")
    normalized_names = [normalize_name(p.get("full_name", "")) for p in patients]
    if len(normalized_names) != len(set(normalized_names)):
        errors.append("Há nomes completos duplicados na base.")

    known_patients = set(patient_ids)
    counts = Counter()
    for d in documents:
        required = ["document_id", "patient_id", "document_type", "document_date", "source", "content"]
        for field in required:
            if not d.get(field):
                errors.append(f"Documento sem campo obrigatório {field}: {d.get('document_id')}")
        if d.get("patient_id") not in known_patients:
            errors.append(f"Documento aponta para paciente inexistente: {d.get('document_id')}")
        else:
            expected_name = next(p["full_name"] for p in patients if p["patient_id"] == d["patient_id"])
            if d.get("patient_name") != expected_name:
                errors.append(f"patient_name inconsistente: {d.get('document_id')}")
        if len(d.get("content", "")) < 30:
            errors.append(f"Conteúdo muito curto: {d.get('document_id')}")
        try:
            date.fromisoformat(d["document_date"])
        except Exception:
            errors.append(f"Data inválida: {d.get('document_id')}")
        counts[d.get("patient_id")] += 1

    outliers = {pid: n for pid, n in counts.items() if not 6 <= n <= 12}
    if outliers:
        errors.append(f"Quantidade de documentos fora da faixa média (6–12): {outliers}")

    return {
        "valid": not errors,
        "patients": len(patients),
        "documents": len(documents),
        "documents_per_patient_min": min(counts.values()) if counts else 0,
        "documents_per_patient_max": max(counts.values()) if counts else 0,
        "document_types": dict(Counter(d["document_type"] for d in documents)),
        "errors": errors,
    }

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"JSON inválido em {path}, linha {line_no}: {exc}") from exc
    return rows

def ingest_langchain(
    jsonl_path: Path = JSONL_PATH,
    persist_dir: Path = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
    embedding_model: str = EMBEDDING_MODEL,
):
    """Cria/atualiza uma coleção Chroma local usando embeddings multilíngues."""
    try:
        from langchain_core.documents import Document
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_chroma import Chroma
    except ImportError as exc:
        raise SystemExit(
            "Dependências RAG ausentes. Instale com: "
            "pip install -r requirements.txt"
        ) from exc

    records = load_jsonl(jsonl_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=850, chunk_overlap=120,
        separators=["\n\n", "\n", ". ", "; ", " ", ""],
    )
    docs = []
    for record in records:
        metadata = {
            "document_id": str(record["document_id"]),
            "patient_id": str(record["patient_id"]),
            "patient_name": str(record["patient_name"]),
            "document_type": str(record["document_type"]),
            "document_date": str(record["document_date"]),
            "source": str(record["source"]),
        }
        for key, value in record.get("metadata", {}).items():
            # Chroma aceita metadados escalares; listas/objetos devem ser serializados.
            if isinstance(value, (str, int, float, bool)):
                metadata[key] = value
            elif value is not None:
                metadata[key] = json.dumps(value, ensure_ascii=False)
        docs.append(Document(page_content=record["content"], metadata=metadata))

    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )

    # IDs determinísticos tornam a reingestão idempotente.
    ids = []
    for i, chunk in enumerate(chunks):
        raw = f"{chunk.metadata['document_id']}::{i}::{chunk.page_content}"
        ids.append(str(uuid.uuid5(uuid.NAMESPACE_URL, raw)))
    vectorstore.add_documents(documents=chunks, ids=ids)
    return vectorstore, len(chunks)

def resolve_patient_by_name(
    patient_name: str,
    patients_path: Path = PATIENTS_PATH,
) -> list[dict[str, Any]]:
    """
    Busca primeiro nome completo exato (sem diferenciar acentos/caixa).
    Se não encontrar, tenta trecho parcial. Retorna todos os candidatos para
    que a aplicação possa solicitar desambiguação, sem escolher por conta própria.
    """
    query = normalize_name(patient_name)
    if not query:
        return []
    patients = load_jsonl(patients_path)
    exact = [p for p in patients if normalize_name(p["full_name"]) == query]
    if exact:
        return exact
    return [p for p in patients if query in normalize_name(p["full_name"])]


def search_rag(
    question: str,
    patient_id: str | None = None,
    patient_name: str | None = None,
    document_type: str | None = None,
    k: int = 5,
    patients_path: Path = PATIENTS_PATH,
    persist_dir: Path = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
    embedding_model: str = EMBEDDING_MODEL,
):
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_chroma import Chroma
    except ImportError as exc:
        raise SystemExit("Instale as dependências com: pip install -r requirements.txt") from exc

    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )
    if patient_name:
        matches = resolve_patient_by_name(patient_name, patients_path)
        if not matches:
            raise ValueError(f"Nenhum paciente encontrado para o nome: {patient_name}")
        if len(matches) > 1:
            options = ", ".join(f"{p['full_name']} ({p['patient_id']})" for p in matches)
            raise ValueError(
                "Mais de um paciente corresponde ao nome informado. "
                f"Solicite desambiguação: {options}"
            )
        resolved_id = matches[0]["patient_id"]
        if patient_id and patient_id != resolved_id:
            raise ValueError("patient_id e patient_name identificam pacientes diferentes.")
        patient_id = resolved_id

    filters = []
    if patient_id:
        filters.append({"patient_id": patient_id})
    if document_type:
        filters.append({"document_type": document_type})
    where = None
    if len(filters) == 1:
        where = filters[0]
    elif len(filters) > 1:
        where = {"$and": filters}

    results = vectorstore.similarity_search_with_score(question, k=k, filter=where)
    for rank, (doc, score) in enumerate(results, 1):
        print(f"\n[{rank}] Similaridade/distância: {score:.4f}")
        print(f"Paciente: {doc.metadata.get('patient_id')}")
        print(f"Documento: {doc.metadata.get('document_id')}")
        print(f"Tipo: {doc.metadata.get('document_type')}")
        print(f"Data: {doc.metadata.get('document_date')}")
        print(f"Fonte: {doc.metadata.get('source')}")
        print(f"Trecho: {doc.page_content}")
    return results

def main():
    parser = argparse.ArgumentParser(description="Base hospitalar sintética para RAG")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Gerar pacientes e documentos JSONL")
    gen.add_argument("--patients", type=int, default=N_PATIENTS)
    gen.add_argument("--seed", type=int, default=SEED)
    gen.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)

    ing = sub.add_parser("ingest", help="Indexar documentos no Chroma")
    ing.add_argument("--jsonl", type=Path, default=JSONL_PATH)
    ing.add_argument("--persist-dir", type=Path, default=CHROMA_DIR)

    qry = sub.add_parser("query", help="Pesquisar documentos no índice")
    qry.add_argument("question")
    qry.add_argument("--patient-id")
    qry.add_argument("--patient-name", help="Nome completo ou trecho único do nome")
    qry.add_argument("--patients-jsonl", type=Path, default=PATIENTS_PATH)
    qry.add_argument("--document-type")
    qry.add_argument("-k", type=int, default=5)
    qry.add_argument("--persist-dir", type=Path, default=CHROMA_DIR)

    args = parser.parse_args()
    if args.command == "generate":
        patients, docs, report = generate_dataset(args.patients, args.seed, args.output_dir)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not report["valid"]:
            raise SystemExit("Validação falhou; consulte validation_report.json")
        print(f"\nArquivos gerados em: {args.output_dir.resolve()}")
    elif args.command == "ingest":
        vectorstore, count = ingest_langchain(args.jsonl, args.persist_dir)
        print(f"Ingestão concluída: {count} chunks indexados.")
    elif args.command == "query":
        search_rag(
            question=args.question,
            patient_id=args.patient_id,
            patient_name=args.patient_name,
            document_type=args.document_type,
            k=args.k,
            patients_path=args.patients_jsonl,
            persist_dir=args.persist_dir,
        )

if __name__ == "__main__":
    main()
