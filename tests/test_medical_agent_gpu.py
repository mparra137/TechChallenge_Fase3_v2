"""Opt-in Windows/CUDA smoke and endurance benchmark for the local agent."""
from __future__ import annotations

import logging
import os
from statistics import median
from time import perf_counter

import pytest


@pytest.mark.skipif("TEST_MEDICAL_AGENT_GPU_TURNS" not in os.environ, reason="benchmark CUDA opcional")
def test_medical_agent_gpu_benchmark(caplog: pytest.LogCaptureFixture) -> None:
    """Measure one process, one model load and repeated turns on cached synthetic data."""
    import torch

    from inference.inference import LocalInference, load_medical_model
    from src import config
    from src.agent.graph import MedicalAgent
    from src.llm.local_chat import LocalChat
    from src.rag.components import build_rag_components

    assert torch.cuda.is_available(), "O benchmark exige CUDA."
    psutil = pytest.importorskip("psutil")
    caplog.set_level(logging.INFO)
    process = psutil.Process()
    count = int(os.environ["TEST_MEDICAL_AGENT_GPU_TURNS"])
    assert count > 0
    started = perf_counter()
    inference = LocalInference(*load_medical_model())
    initialization = perf_counter() - started
    _, store, retriever = build_rag_components(create_if_missing=False)
    assert store.get_collection_stats()["vector_count"] == config.RAG_EXPECTED_DOCUMENT_COUNT
    agent = MedicalAgent(LocalChat(inference, config.AGENT_ANSWER_TOKENS, config.AGENT_CONTEXT_TOKENS),
                         LocalChat(inference, config.AGENT_CLASSIFICATION_TOKENS, config.AGENT_CONTEXT_TOKENS),
                         retriever)
    general = (
        ("O que é diabetes tipo 2?", "GENERAL_MEDICAL"),
        ("What is asthma?", "GENERAL_MEDICAL"),
    )
    records = (
        ("Última creatinina do prontuário PCT-000001", "PATIENT_RECORD"),
        ("E a glicemia dele?", "PATIENT_RECORD"),
        ("Quais os exames realizados pelo paciente PCT-000001?", "PATIENT_RECORD"),
    )
    # Exercise both languages and exact retrieval without making cloud requests.
    if count <= 5:
        cases = [general[0], *records, general[1]][:count]
    else:
        cases = [general[0], *(records * count)[:count - 2], general[1]]
    latencies: list[float] = []
    ram: list[int] = []
    statuses: dict[str, int] = {}
    try:
        for question, route in cases:
            started = perf_counter()
            answer = agent.ask(question, "benchmark")
            latencies.append(perf_counter() - started)
            ram.append(process.memory_info().rss)
            assert answer.intent == route
            assert answer.status == "ok"
            if route == "PATIENT_RECORD":
                assert answer.sources and all("PCT-000001" in source.document_id for source in answer.sources)
            statuses[answer.status] = statuses.get(answer.status, 0) + 1
    finally:
        agent.clear("benchmark")
    p95 = sorted(latencies)[(95 * count + 99) // 100 - 1]
    print(f"carga_s={initialization:.2f}, turnos={count}, p50_s={median(latencies):.2f}, "
          f"p95_s={p95:.2f}, pico_rss_amostrado_mib={max(ram) / 2**20:.0f}, "
          f"delta_rss_mib={(ram[-1] - ram[0]) / 2**20:.0f}, estados={statuses}")
    for record in caplog.records:
        if record.name in {"src.agent.graph", "inference.inference"}:
            print(record.getMessage())
